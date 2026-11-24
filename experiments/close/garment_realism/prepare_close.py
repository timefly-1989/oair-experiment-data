"""Scan-grounded renderings with explicitly partial garment surface supervision.

Input RGB is a rendering of a real scan, never described as a captured photo.
Semantic-patch boundaries are not asserted to be physical cuffs or hems.
"""
import argparse,hashlib,json,math,os,time,traceback
from collections import Counter
from pathlib import Path
os.environ.setdefault('PYOPENGL_PLATFORM','egl');os.environ.setdefault('EGL_DEVICE_ID','1')
import numpy as np
from PIL import Image,ImageFilter
import torch,torchvision,pyrender,trimesh
from scipy import ndimage
from garment_lab.reconstruct import reconstruct
from garment_lab.templates import panel
from garment_study.topology import topology_features,projection_support
from .common import ROOT,STUDY,write,digest

LABELS={2:('shirt','top'),3:('tshirt','top'),4:('vest','top'),5:('coat','top'),6:('dress','dress'),7:('skirt','skirt'),8:('pants','pants'),9:('shorts','pants'),11:('hoodie','top'),17:('jacket','top')}

def seed_for(text):return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)

def camera(size,yaw,fov,elevation=0.):
    theta=math.radians(yaw);phi=math.radians(elevation)
    # Scan is centered. Fit the recovered scan extent with margin; distance
    # varies with FOV so framing is comparable without changing metric geometry.
    distance=1.25/math.tan(math.radians(fov/2))
    eye=np.array([math.sin(theta)*math.cos(phi),math.sin(phi),math.cos(theta)*math.cos(phi)])*distance
    forward=-eye/np.linalg.norm(eye);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right)
    down=np.cross(forward,right);rotation=np.stack([right,down,forward])
    focal=size/(2*math.tan(math.radians(fov/2)))
    k=np.array([[focal,0,size/2],[0,focal,size/2],[0,0,1]],dtype=np.float32)
    return k,rotation.astype(np.float32),eye.astype(np.float32)

def surface_samples(vertices,faces,n,seed):
    rng=np.random.default_rng(seed);tri=vertices[faces]
    cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)
    good=area>1e-12
    if good.sum()<32:raise ValueError('too_few_nondegenerate_patch_faces')
    ids=rng.choice(np.where(good)[0],size=n,p=area[good]/area[good].sum())
    uv=rng.random((n,2));a=np.sqrt(uv[:,0]);b=np.c_[1-a,a*(1-uv[:,1]),a*uv[:,1]]
    points=(tri[ids]*b[:,:,None]).sum(1)
    normals=cross[ids]/area[ids,None]
    return points.astype(np.float32),normals.astype(np.float32)

def make_selection(source,out,scans_per_group,engineering=False):
    files=sorted(source.glob('*.npz'));groups=sorted({p.stem.split('_')[0] for p in files},key=lambda x:seed_for('close-v6-group:'+x))
    # 10001 was previously visually inspected; keep it outside fresh evaluation.
    groups=[g for g in groups if g!='10001']
    assignment={g:'test' if i<12 else 'val' if i<22 else 'train' for i,g in enumerate(groups)}
    assignment['10001']='train'
    selected=[]
    for group in sorted(assignment):
        group_files=sorted([p for p in files if p.stem.startswith(group+'_')],key=lambda p:seed_for('close-v6-scan:'+p.stem))
        count=1 if engineering else (scans_per_group if assignment[group]=='train' else min(scans_per_group,2))
        for p in group_files[:count]:selected.append({'scan_id':p.stem,'group_id':group,'split':assignment[group],'source':str(p)})
    if engineering:selected=[r for r in selected if r['group_id']=='10001'][:1]
    plan={'source_scans':len(files),'prefix_groups':len(assignment),'prefix_group_counts':dict(Counter(assignment.values())),
          'selected_scans':selected,'split_assignment':assignment,'selection_uses_model_error':False,
          'grouping_caveat':'Filename prefix is a conservative grouping proxy; global physical-subject identity not independently verified.',
          'historically_inspected_group_10001_reserved_for_training':True,
          'geometry_target':'author semantic observed outer patches, not complete physical garments',
          'input_domain':'real_scan_vertex_color_rerender_not_captured_photograph',
          'units':'source coordinates after dividing by author scale; artificial centering changes translation only',
          'author_normalization_source':'https://github.com/anticdimi/CloSe/blob/main/prep_scan.py'}
    path=out/'locked_selection.json'
    if path.exists():
        if json.loads(path.read_text())!=plan:raise ValueError('Selection changed; use a new preparation directory')
    else:write(path,plan)
    return selected

def scene_for(vertices,faces,colors,k):
    scene=pyrender.Scene(bg_color=[.8,.82,.85,1],ambient_light=[.8,.8,.8])
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,vertex_colors=np.clip(colors*255,0,255).astype(np.uint8),process=False)
    node=scene.add(pyrender.Mesh.from_trimesh(mesh,smooth=False))
    scene.add(pyrender.IntrinsicsCamera(k[0,0],k[1,1],k[0,2],k[1,2]),pose=np.diag([1,-1,-1,1]))
    return scene,node

def render_case(renderer,points,faces,colors,labels,k,label=None):
    scene,node=scene_for(points,faces,colors,k)
    rgb,depth=renderer.render(scene,flags=pyrender.RenderFlags.RGBA)
    foreground=depth>0;rgb=rgb[:,:,:3]
    if label is None:return rgb,foreground
    selected=np.all(labels[faces]==label,axis=1)
    target=faces[selected];other=faces[~selected]
    segmentation=pyrender.Scene(bg_color=[0,0,0,0])
    n=segmentation.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(points,target,process=False),smooth=False))
    rest=segmentation.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(points,other,process=False),smooth=False))
    segmentation.add(pyrender.IntrinsicsCamera(k[0,0],k[1,1],k[0,2],k[1,2]),pose=np.diag([1,-1,-1,1]))
    visible,_=renderer.render(segmentation,flags=pyrender.RenderFlags.SEG,seg_node_map={n:np.array([255,255,255]),rest:np.array([0,0,0])})
    return visible[:,:,0]>127,target

def run(args):
    start=time.time();out=Path(args.out);out.mkdir(parents=True,exist_ok=True);(out/'cache').mkdir(exist_ok=True);(out/'previews').mkdir(exist_ok=True)
    selected=make_selection(Path(args.scans),out,args.scans_per_group,args.engineering)
    if args.only_split:selected=[row for row in selected if row['split']==args.only_split]
    torch.set_num_threads(2);torch._C._jit_set_profiling_executor(False);torch._C._jit_set_profiling_mode(False)
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.085,torch.device(args.device))
    nlf=torch.jit.load(args.nlf,map_location=args.device).eval();renderer=pyrender.OffscreenRenderer(args.size,args.size)
    rows=[];failures=[]
    completed_file=out/'prepared_records.jsonl'
    if completed_file.exists():rows=[json.loads(s) for s in completed_file.read_text().splitlines() if s.strip()]
    existing={r['sample_id'] for r in rows}
    views=[(0,35,0),(35,50,5),(-35,60,12)]
    view_offset=0
    if args.extra_views:views=[(180,45,-5),(70,40,10)];view_offset=3
    if args.engineering:views=views[:1]
    for scan_index,record in enumerate(selected):
        path=Path(record['source'])
        with np.load(path,allow_pickle=False) as raw:
            points=raw['points'].astype(np.float32)/float(raw['scale']);faces=raw['faces'].astype(np.int32)
            colors=raw['colors'].astype(np.float32);labels=raw['labels'].astype(np.int32)
            # Coordinates are centered already. No registered SMPL or scan depth
            # is given to the NLF-conditioned clothing initializer.
            counts=Counter(labels.tolist());candidates=[label for label in LABELS if counts[label]>=5000]
            candidates=sorted(candidates,key=lambda label:(-counts[label],label))[:2]
            # Keep near-zero triangles out of renderer normals without repairing holes.
            triangles=points[faces];area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1)
            faces=faces[area>1e-12]
        for view_index,(yaw,fov,elev) in enumerate(views,start=view_offset):
            if all(record['scan_id']+f'_v{view_index}_l{label}' in existing for label in candidates):continue
            k,r,eye=camera(args.size,yaw,fov,elev);camera_points=(points-eye)@r.T
            try:
                image,foreground=render_case(renderer,camera_points,faces,colors,labels,k)
                yy,xx=np.where(foreground)
                if not len(xx):raise ValueError('empty_person_render')
                box=torch.tensor([[xx.min(),yy.min(),xx.max()-xx.min()+1,yy.max()-yy.min()+1,1]],dtype=torch.float32,device=args.device)
                with torch.inference_mode():
                    prediction=nlf._estimate_parametric_batched(torch.from_numpy(image.copy()).permute(2,0,1)[None].to(args.device),[box],
                        intrinsic_matrix=torch.from_numpy(k)[None].to(args.device),world_up_vector=torch.from_numpy(r@np.array([0.,1.,0.],dtype=np.float32)).to(args.device),
                        num_aug=1,internal_batch_size=1,suppress_implausible_poses=False)
                body=prediction['vertices3d'][0][0].cpu().numpy().astype(np.float32)/1000
                if not np.isfinite(body).all():raise ValueError('nonfinite_NLF_body')
            except Exception as error:
                failures.append({'scan_id':record['scan_id'],'view':view_index,'stage':'RGB_NLF','type':type(error).__name__,'error':str(error)[:240]})
                if isinstance(error,torch.OutOfMemoryError):raise
                continue
            for label in candidates:
                sample_id=record['scan_id']+f'_v{view_index}_l{label}'
                if sample_id in existing:continue
                try:
                    mask,target_faces=render_case(renderer,camera_points,faces,colors,labels,k,label)
                    if mask.sum()<100:raise ValueError('tiny_or_fully_occluded_visible_garment')
                    if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any():raise ValueError('truncated_visible_garment')
                    category=LABELS[label][1]
                    initial,boundary,meta,cleaned=reconstruct(image,mask,category,args.resolution,body={'vertices':body,'intrinsics':k})
                    target_points,target_normals=surface_samples(camera_points,target_faces,8192,seed_for(sample_id))
                    arrays={'image':image,'mask':cleaned.astype(np.uint8),'initial_vertices':initial.vertices,'faces':initial.faces,'uv':initial.uv,
                            'body_vertices':body,'intrinsics':k,'visible_proxy':(panel(category,args.resolution)['side']==0).astype(np.float32),
                            'patch_points':target_points,'patch_normals':target_normals,'boundary':boundary,
                            **topology_features(initial.vertices,initial.faces)}
                    dest=out/'cache'/(sample_id+'.npz');np.savez_compressed(dest,**arrays)
                    row={**record,'sample_id':sample_id,'subject_id':'close_prefix_'+record['group_id'],'garment_id':record['scan_id']+'_label'+str(label),
                         'category':category,'semantic_label':label,'cache':str(dest),'cache_sha256':digest(dest),
                         'target_kind':'observed_semantic_garment_patch','input_domain':'real_scan_rerender','units':'source_metric_after_inverse_author_scale',
                         'view_index':view_index,'yaw_degrees':yaw,'fov_degrees':fov,'elevation_degrees':elev,
                         'mask_source':'visible_semantic_scan_render','body_source':'frozen_NLF_prediction_on_RGB',
                         'source_sha256':digest(path),'source_scale':float(1/np.ptp(points,axis=0).max()),'patch_target_vertices':len(target_points),
                         'patch_target_boundary_supervised':False,'initialization_metadata':meta}
                    rows.append(row);existing.add(sample_id)
                    with completed_file.open('a') as stream:stream.write(json.dumps(row)+'\n')
                    if scan_index<3:
                        Image.fromarray(image).save(out/'previews'/(sample_id+'_rgb.png'))
                        Image.fromarray(mask.astype(np.uint8)*255).save(out/'previews'/(sample_id+'_mask.png'))
                except Exception as error:
                    failures.append({'sample_id':sample_id,'stage':'garment_prepare','type':type(error).__name__,'error':str(error)[:240]})
                    if isinstance(error,torch.OutOfMemoryError):raise
        for split in ['train','val','test']:
            (out/(split+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows if row['split']==split))
        write(out/'progress.json',{'status':'preparing','completed_scans':scan_index+1,'selected_scans':len(selected),'samples':dict(Counter(row['split'] for row in rows)),
                                  'failures':len(failures),'seconds':time.time()-start})
        write(out/'failures.json',failures)
        print(json.dumps({'scan':scan_index+1,'total':len(selected),'samples':len(rows),'failures':len(failures)}),flush=True)
    renderer.delete()
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not {r['group_id'] for r in rows if r['split']==a}&{r['group_id'] for r in rows if r['split']==b}
    write(out/'receipt.json',{'status':'engineering_completed' if args.engineering else 'completed','scans':len(selected),'counts':dict(Counter(r['split'] for r in rows)),
                              'failures':len(failures),'source_scan_count':1444,'prefix_group_disjoint':True,
                              'domain':'real_scan_rerender_not_captured_photos','target':'observed_outer_semantic_patch_not_complete_garment',
                              'seconds':time.time()-start,'nlf_sha256':digest(args.nlf)})

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--scans',default=str(ROOT/'data/external_public_20261103/CloSe-D/CloSe-Di'))
    p.add_argument('--nlf',default=str(ROOT/'models/nlf_l_multi_0.3.2.patch5.torchscript'));p.add_argument('--device',default='cuda:1')
    p.add_argument('--size',type=int,default=256);p.add_argument('--resolution',type=int,default=24);p.add_argument('--scans-per-group',type=int,default=3);p.add_argument('--engineering',action='store_true')
    p.add_argument('--only-split',choices=['train','val','test']);p.add_argument('--extra-views',action='store_true')
    run(p.parse_args())
if __name__=='__main__':main()
