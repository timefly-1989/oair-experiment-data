"""Category-specific sewn open panels; boundaries are not closed by volume meshing."""
from functools import lru_cache
import numpy as np
import triangle

# Segment marker 0 = seam, positive = natural opening.
PATTERNS = {
    'top': ([(.15,0),(.4,0),(.42,.05),(.5,.09),(.58,.05),(.6,0),(.85,0),
             (1,.2),(.96,.3),(.75,.23),(.7,1),(.3,1),(.25,.23),(.04,.3),(0,.2)],
            {1:1,2:1,3:1,4:1,7:3,10:2,13:4}, 4),
    'dress': ([(.28,0),(.4,0),(.42,.035),(.5,.06),(.58,.035),(.6,0),(.72,0),
               (.66,.18),(.61,.30),(.95,1),(.05,1),(.39,.30),(.34,.18)],
              {1:1,2:1,3:1,4:1,6:3,7:3,9:2,11:4,12:4}, 4),
    'skirt': ([(.3,0),(.7,0),(1,1),(0,1)], {0:1,2:2}, 2),
    'pants': ([(.2,0),(.8,0),(.98,1),(.56,1),(.5,.42),(.44,1),(.02,1)],
              {0:1,2:2,5:3}, 3),
}


@lru_cache(maxsize=24)
def panel(category, resolution=24):
    if category not in PATTERNS:
        raise ValueError(f'Unsupported garment category: {category}; choose {list(PATTERNS)}')
    if not 8 <= resolution <= 128:
        raise ValueError('Resolution must be between 8 and 128.')
    polygon, openings, expected = PATTERNS[category]
    polygon = np.array(polygon, dtype=np.float64)
    points,labels = [],[]
    for i,(a,b) in enumerate(zip(polygon,np.roll(polygon,-1,axis=0))):
        # Every opening needs interior edge vertices: otherwise its two seam endpoints
        # weld the front/back edges together and silently close a low-resolution cuff.
        count=max(int(np.ceil(np.linalg.norm(b-a)*resolution)),3 if i in openings else 1)
        for j in range(count):
            points.append(a+(b-a)*j/count)
            labels.append(openings.get(i,0)+1)
    points=np.asarray(points)
    segments=np.stack([np.arange(len(points)),np.roll(np.arange(len(points)),-1)],axis=1)
    markers=np.asarray(labels)[:,None]
    tri = triangle.triangulate({'vertices':points, 'segments':segments, 'segment_markers':markers},
                               f'pq25a{1/(resolution*resolution):.10f}')
    verts, faces = tri['vertices'].astype(np.float32), tri['triangles'].astype(np.int64)
    segments = tri['segments'].astype(np.int64)
    markers = tri['segment_markers'].reshape(-1).astype(np.int64)-1
    seam_vertices = np.unique(segments[markers == 0])
    boundary_vertices = np.unique(segments)
    # Front has N vertices; only seam vertices are shared with the back panel.
    back = np.arange(len(verts), 2*len(verts))
    back[seam_vertices] = seam_vertices
    raw_faces = np.concatenate([faces, back[faces[:, ::-1]]], axis=0)
    used, inverse = np.unique(raw_faces, return_inverse=True)
    source = used % len(verts)
    side = (used >= len(verts)).astype(np.int64)
    full_boundary = np.isin(source, boundary_vertices).astype(np.float32)
    full_seam = np.isin(source, seam_vertices)
    return {'panel_uv':verts, 'polygon':polygon, 'panel_segments':segments,
            'segment_labels':markers, 'seam_vertices':seam_vertices,
            'source_indices':source, 'side':side, 'is_seam':full_seam,
            'boundary_feature':full_boundary,
            'faces':inverse.reshape(-1,3).astype(np.int64), 'expected_openings':expected}


def row_intervals(polygon, y):
    intersections = []
    y = np.clip(y, 1e-6, 1-1e-6)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (a[1] <= y < b[1]) or (b[1] <= y < a[1]):
            intersections.append(float(a[0]+(y-a[1])*(b[0]-a[0])/(b[1]-a[1])))
    intersections.sort()
    return list(zip(intersections[::2], intersections[1::2]))
