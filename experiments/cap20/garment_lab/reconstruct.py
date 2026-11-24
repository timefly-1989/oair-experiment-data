"""Image-mask-conditioned open-surface baseline and optional trained refinement.

Unobserved depth is a category prior, not a measured 3D reconstruction. A calibrated
NLF cache can supply metric body context; otherwise output scale is explicitly arbitrary.
"""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from .mesh import Mesh, boundary_edges, boundary_loops
from .templates import panel, row_intervals


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def clean_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.sum() < 32:
        raise ValueError('Garment mask must be a nonempty HxW array with at least 32 pixels.')
    components, count = ndimage.label(mask)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    keep = sizes >= max(16, sizes.max()*.015)
    mask = keep[components]
    return ndimage.binary_closing(mask, iterations=1) | mask


def intervals_in_row(mask, y, xmin, xmax):
    y = int(np.clip(round(y), 0, len(mask)-1))
    row = mask[y].copy()
    if row.sum() < 2:
        for delta in range(1, min(12, len(mask))):
            candidates = [yy for yy in (y-delta, y+delta) if 0 <= yy < len(mask)]
            if any(mask[yy].sum() >= 2 for yy in candidates):
                row = mask[next(yy for yy in candidates if mask[yy].sum() >= 2)]
                break
    row = ndimage.binary_closing(row, iterations=2)
    indices = np.where(row)[0]
    if len(indices) < 2:
        return [(float(xmin), float(xmax))]
    starts = np.r_[0, np.where(np.diff(indices) > 1)[0]+1]
    ends = np.r_[starts[1:]-1, len(indices)-1]
    intervals = [(float(indices[a]),float(indices[b])) for a,b in zip(starts,ends) if indices[b]-indices[a] >= 2]
    return intervals or [(float(indices[0]),float(indices[-1]))]


def fit_panel_to_mask(template, mask):
    yy, xx = np.where(mask)
    xmin,xmax,ymin,ymax = xx.min(),xx.max(),yy.min(),yy.max()
    uv = template['panel_uv']
    fitted = np.empty_like(uv)
    ambiguous = 0
    for i,(u,v) in enumerate(uv):
        image_y = ymin + v*(ymax-ymin)
        source = row_intervals(template['polygon'], float(v))
        target = intervals_in_row(mask, image_y, xmin, xmax)
        if not source:
            source = [(0.,1.)]
        part = min(range(len(source)), key=lambda j: max(source[j][0]-u, u-source[j][1], 0))
        if len(source) == len(target):
            a,b = source[part]
            c,d = target[part]
        else:
            ambiguous += 1
            a,b,c,d = source[0][0],source[-1][1],target[0][0],target[-1][1]
        fraction = np.clip((u-a)/max(b-a,1e-5), 0., 1.)
        fitted[i] = [c+fraction*(d-c),image_y]
    return fitted, {'bbox_xyxy':[int(xmin),int(ymin),int(xmax),int(ymax)],
                    'ambiguous_row_fraction': ambiguous/len(uv)}


def read_body_cache(path, image_shape):
    with np.load(path, allow_pickle=False) as raw:
        required = {'initial_vertices','intrinsics','image_size'}
        if not required.issubset(raw.files):
            raise ValueError('NLF cache needs initial_vertices, intrinsics and image_size.')
        body, k, size = [np.asarray(raw[key]).copy() for key in ('initial_vertices','intrinsics','image_size')]
        faces = np.asarray(raw['faces']).copy() if 'faces' in raw.files else None
    if body.ndim != 2 or body.shape[1] != 3 or not np.isfinite(body).all() or k.shape != (3,3):
        raise ValueError('Invalid NLF body/camera cache.')
    if not np.array_equal(size.astype(int), image_shape[:2]):
        raise ValueError('Image must be the SAME calibrated crop as NLF cache image_size; no implicit resizing.')
    if not np.isfinite(k).all() or np.linalg.det(k) <= 0 or np.any(body[:,2] <= 0):
        raise ValueError('Require finite perspective calibration and positive-depth camera_xyz_meters body.')
    return {'vertices':body.astype(np.float32),'faces':faces,'intrinsics':k.astype(np.float32)}


def reconstruct(image, mask, category='dress', resolution=24, depth_ratio=.28, body=None, inverse_depth=None):
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != np.shape(mask):
        raise ValueError('RGB image and binary garment mask must have the same H,W.')
    if not .02 <= depth_ratio <= .8:
        raise ValueError('depth_ratio must be between .02 and .8.')
    mask = clean_mask(mask)
    template = panel(category, resolution)
    pixels, fitting = fit_panel_to_mask(template, mask)
    source, side = template['source_indices'],template['side']
    height,width = image.shape[:2]
    scale = float(max(width,height))
    # Distance to sewn side edges supplies an explicit, deterministic thickness prior.
    seam_distance = cKDTree(pixels[template['seam_vertices']]).query(pixels)[0]/scale
    half_depth = np.minimum(seam_distance*depth_ratio, depth_ratio*.15)
    half_depth[template['seam_vertices']] = 0
    xy = pixels[source]
    # Relative monocular predictions are never treated as metric measurements.
    # Modulate the visible sheet by at most 65% of its prior distance to the
    # center sheet. This preserves front/back order and welded side seams.
    front_factor = np.ones(len(xy))
    depth_range = None
    if inverse_depth is not None:
        inverse_depth = np.asarray(inverse_depth,dtype=np.float32)
        if inverse_depth.shape != mask.shape or not np.isfinite(inverse_depth).all():
            raise ValueError('Relative inverse depth must be finite and match the RGB image dimensions.')
        smoothed = ndimage.gaussian_filter(inverse_depth,1.5)
        low,high = np.percentile(smoothed[mask],[5,95])
        normalized = np.zeros_like(smoothed) if high-low < 1e-6 else np.clip(2*(smoothed-low)/(high-low)-1,-1,1)
        sampled = ndimage.map_coordinates(normalized,[xy[:,1],xy[:,0]],order=1,mode='nearest')
        front_factor[side==0] += .65*sampled[side==0]
        depth_range = [float(low),float(high)]
    zoffset = np.where(side == 0, -front_factor,1.)*half_depth[source]
    if body is None:
        vertices = np.c_[(xy[:,0]-(width-1)/2)/scale, -(xy[:,1]-(height-1)/2)/scale, zoffset]
        units,coords = 'arbitrary_image_scale','image_x_right_y_up_z_away'
    else:
        bv,k = body['vertices'],body['intrinsics']
        proj = bv @ k.T
        projected = proj[:,:2]/proj[:,2:]
        _,neighbors = cKDTree(projected).query(xy, k=min(48,len(bv)))
        depth = bv[neighbors,2]
        near,far = np.quantile(depth,.12,axis=1),np.quantile(depth,.88,axis=1)
        center = (near+far)/2
        # Scale the inferred image prior into metric depth using the observed body distance.
        offset = half_depth[source]*center*scale/max(k[0,0],k[1,1])
        z = center + np.where(side==0,-front_factor,1)*(np.maximum((far-near)/2,offset)+.01)
        z[template['is_seam']] = center[template['is_seam']]
        rays = np.c_[xy,np.ones(len(xy))] @ np.linalg.inv(k).T
        vertices = rays*z[:,None]
        units,coords = 'meters','camera_xyz_meters'
    uv = np.c_[(xy[:,0]+.5)/width,(xy[:,1]+.5)/height].astype(np.float32)
    ix = np.clip(np.rint(xy[:,0]).astype(int),0,width-1)
    iy = np.clip(np.rint(xy[:,1]).astype(int),0,height-1)
    colors = image[iy,ix].copy()
    # Back appearance is not observed: use a neutral garment color rather than claim recovery.
    colors[side==1] = np.median(image[mask],axis=0).astype(np.uint8)
    mesh = Mesh(vertices.astype(np.float32),template['faces'].copy(),uv,colors).validate()
    loops = boundary_loops(mesh.faces)
    if len(loops) != template['expected_openings']:
        raise ValueError('Template boundary topology changed unexpectedly.')
    boundary = np.zeros(len(vertices), dtype=np.float32)
    boundary[np.unique(boundary_edges(mesh.faces))] = 1
    metadata = {'method':'mask_conditioned_open_panel_baseline','category':category,
                'units':units,'coordinate_system':coords,'body_condition_used':body is not None,
                'depth_source':'inferred_category_prior' if body is None else 'NLF_conditioned_depth_prior',
                'front_relative_depth_used':inverse_depth is not None,
                'front_depth_robust_range':depth_range,
                'unobserved_back':'inferred_shape_and_neutral_color_not_measured',
                'geometry_accuracy_measured':False,'expected_openings':template['expected_openings'],**fitting}
    return mesh, boundary, metadata, mask
