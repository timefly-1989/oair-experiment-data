"""Prepare original, calibrated GarmentCodeData images with independent garment targets.

The supplied garment mask is an amodal silhouette rendered from the annotation.
This is an explicit strong 2D input condition, not predicted segmentation.
NLF sees only the source RGB, its alpha foreground box, and the known camera.
No target depth, target topology or body ground truth enters initialization.
"""
import argparse,hashlib,json,os,time
from collections import Counter
from pathlib import Path
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('EGL_DEVICE_ID','1')
import numpy as np
from PIL import Image
import torch
import torchvision
import trimesh
import pyrender
import yaml
from garment_lab.mesh import Mesh,boundary_loops
from garment_lab.reconstruct import reconstruct,sha256
from garment_lab.templates import panel
from .topology import topology_features


def camera(size,location=(0,.97,4.15),side='front'):
    x,y=np.deg2rad([-15,20+(180 if side=='back' else 0)])
    rx=np.array([[1,0,0],[0,np.cos(x),-np.sin(x)],[0,np.sin(x),np.cos(x)]])
    ry=np.array([[np.cos(y),0,np.sin(y)],[0,1,0],[-np.sin(y),0,np.cos(y)]])
    rotation=ry@rx; translation=rotation@np.asarray(location)
    transform=np.diag([1,-1,-1])@rotation.T
    f=size/(2*np.tan(np.pi/12))
    k=np.array([[f,0,size/2],[0,f,size/2],[0,0,1]],dtype=np.float32)
    return k,transform,translation


def infer_category(design):
    design=design['design']; meta=design['meta']
    upper,bottom=meta['upper']['v'],meta['bottom']['v']
    if not upper:
        if not bottom: raise ValueError('unsupported_no_garment')
        return 'pants' if bottom=='Pants' else 'skirt'
    if design.get('shirt',{}).get('strapless',{}).get('v',False): raise ValueError('unsupported_strapless_topology')
    if not bottom: return 'top'
    if bottom=='Pants': raise ValueError('unsupported_jumpsuit')
    if not design.get('sleeve',{}).get('sleeveless',{}).get('v',False): raise ValueError('unsupported_sleeved_dress')
    return 'dress'


def load_target(path,rotation,translation):
    raw=trimesh.load_mesh(path,process=False)
    raw.vertices=np.asarray(raw.vertices)/100 # Upstream documents geometry in centimeters.
    raw.merge_vertices(merge_tex=True,merge_norm=True,digits_vertex=7)
    raw.update_faces(raw.nondegenerate_faces(height=1e-8)); raw.remove_unreferenced_vertices()
    vertices=(np.asarray(raw.vertices)-translation)@rotation.T
    return Mesh(vertices.astype(np.float32),np.asarray(raw.faces,dtype=np.int64)).validate()


def mask_renderer(renderer,mesh,k):
    scene=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[1,1,1])
    material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[1,1,1,1],doubleSided=True)
    node=scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(mesh.vertices,mesh.faces,process=False),material=material,smooth=False))
    scene.add(pyrender.IntrinsicsCamera(k[0,0],k[1,1],k[0,2],k[1,2]),pose=np.diag([1,-1,-1,1]))
    color,depth=renderer.render(scene,flags=pyrender.RenderFlags.SEG,seg_node_map={node:np.array([255,255,255])})
    return depth>0


def run(args):
    root=Path(args.root).resolve(); out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'train.jsonl').exists(): raise FileExistsError('Use a new preparation directory to preserve locked manifests.')
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction,torch.device(args.device))
    torch._C._jit_set_profiling_executor(False); torch._C._jit_set_profiling_mode(False)
    model=torch.jit.load(args.nlf,map_location=args.device).eval()
    artifact_hash=sha256(args.nlf)
    renderer=pyrender.OffscreenRenderer(args.image_size,args.image_size)
    official=json.loads((root/'gcd_v2/GarmentCodeData_v2_official_train_valid_test_data_split.json').read_text())
    splits={}
    for upstream,split in [('training','train'),('validation','val'),('test','test')]:
        for value in official[upstream]:
            parts=Path(value).parts
            key=(parts[-3],parts[-1])
            if key in splits and splits[key]!=split: raise ValueError('Upstream design split overlap.')
            splits[key]=split
    records=[]; excluded=[]; counts=Counter(); start=time.time()
    props={}
    cases=[]
    for batch in args.batches:
        for body in args.body_types:
            path=root/f'extracted/garments_5000_{batch}/{body}_body'
            if not path.exists(): continue
            props[(batch,body)]=yaml.safe_load((root/f'gcd_v2/GarmentCodeData_v2/garments_5000_{batch}/{body}_body/dataset_properties_{body}_body.yaml').read_text())
            for case in sorted(path.iterdir()):
                if case.is_dir() and (case/(case.name+'_vertex_labels.yaml')).exists(): cases.append((batch,body,case))
    # Official held-out groups are assigned before training groups; only metadata
    # identities are consulted here, never performance or target-driven selection.
    order={'test':0,'val':1,'train':2}
    cases.sort(key=lambda item:(order.get(splits.get((f'garments_5000_{item[0]}',item[2].name),'none'),9),item[0],item[1],item[2].name))
    identity_split={}
    limits={'train':args.train_limit,'val':args.val_limit,'test':args.test_limit}
    for batch,body,case in cases:
        name=case.name; split=splits.get((f'garments_5000_{batch}',name))
        if split is None: excluded.append({'sample':name,'reason':'not_in_official_paired_success_split'}); continue
        if counts[split]>=limits[split]: continue
        identifier=f'b{batch}_{body}_{name}'
        try:
            prop=props[(batch,body)]
            failures=prop.get('sim',{}).get('stats',{}).get('fails',{})
            if any(name in values for values in failures.values() if isinstance(values,list)):
                raise ValueError('upstream_simulation_failure')
            design=yaml.safe_load((case/(name+'_design_params.yaml')).read_text())
            category=infer_category(design)
            measures=yaml.safe_load((case/(name+'_body_measurements.yaml')).read_text())
            body_sample=measures.get('body',{}).get('body_sample')
            identity='gcd_body_'+str(body_sample).split('_')[0] if body_sample else 'body_'+hashlib.sha256(json.dumps(measures,sort_keys=True).encode()).hexdigest()[:24]
            if identity in identity_split and identity_split[identity]!=split: raise ValueError('body_identity_cross_split_excluded')
            render_config=prop['render']['config']
            k,rotation,translation=camera(args.image_size,render_config['front_camera_location'])
            target=load_target(case/(name+'_sim.ply'),rotation,translation)
            gt_loops=boundary_loops(target.faces)
            if len(gt_loops)!=panel(category,args.resolution)['expected_openings']:
                raise ValueError('unsupported_boundary_topology')
            rgba=Image.open(case/(name+'_render_front.png')).convert('RGBA').resize((args.image_size,args.image_size),Image.Resampling.LANCZOS)
            white=Image.new('RGBA',rgba.size,'white'); white.alpha_composite(rgba)
            image=np.array(white.convert('RGB'))
            foreground=np.array(rgba)[:,:,3]>127
            mask=mask_renderer(renderer,target,k)
            if mask.sum()<80: raise ValueError('garment_mask_too_small')
            if mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any(): raise ValueError('garment_truncated_by_frame')
            yy,xx=np.where(foreground)
            box=torch.tensor([[xx.min(),yy.min(),xx.max()-xx.min()+1,yy.max()-yy.min()+1,1]],dtype=torch.float32,device=args.device)
            inputs=torch.from_numpy(image.copy()).permute(2,0,1)[None].to(args.device)
            gravity=torch.from_numpy((rotation@np.array([0.,1.,0.])).astype(np.float32)).to(args.device)
            with torch.inference_mode():
                predicted=model._estimate_parametric_batched(inputs,[box],intrinsic_matrix=torch.from_numpy(k)[None].to(args.device),
                    world_up_vector=gravity,num_aug=1,internal_batch_size=1,suppress_implausible_poses=False)
            if len(predicted['vertices3d'][0])!=1: raise ValueError('NLF_returned_wrong_person_count')
            body_vertices=predicted['vertices3d'][0][0].cpu().numpy()/1000
            if not np.isfinite(body_vertices).all() or body_vertices[:,2].min()<=0: raise ValueError('invalid_NLF_prediction')
            initial,boundary,meta,mask=reconstruct(image,mask,category,args.resolution,body={'vertices':body_vertices,'intrinsics':k})
            geo=topology_features(initial.vertices,initial.faces)
            template=panel(category,args.resolution)
            visible_proxy=(template['side']==0).astype(np.float32)
            cache=out/(identifier+'.npz')
            np.savez_compressed(cache,image=image,mask=mask.astype(np.uint8),initial_vertices=initial.vertices,
                faces=initial.faces,uv=initial.uv,target_vertices=target.vertices,target_faces=target.faces,
                body_vertices=body_vertices.astype(np.float32),intrinsics=k,visible_proxy=visible_proxy,
                target_boundary_edges=np.concatenate([np.c_[loop,np.roll(loop,-1)] for loop in gt_loops]),**geo)
            record={'sample_id':identifier,'subject_id':identity,'garment_id':f'garments_5000_{batch}/{name}',
                'split':split,'cache':cache.name,'cache_sha256':sha256(cache),'data_kind':'garment_benchmark',
                'domain':'GarmentCodeData_v2_synthetic_original_render','target_kind':'independent_garment_surface',
                'units':'meters','category':category,'body_source':'frozen_NLF_prediction_with_known_intrinsics',
                'mask_source':'annotation_rendered_amodal_garment_silhouette','known_camera':True,
                'source_mesh_sha256':sha256(case/(name+'_sim.ply')),'source_image_sha256':sha256(case/(name+'_render_front.png')),
                'body_variant':body,'vertices':len(initial.vertices),'target_vertices':len(target.vertices),'target_openings':len(gt_loops),
                'initialization':meta}
            records.append(record); counts[split]+=1; identity_split[identity]=split
            if split=='train' and counts['train']<=4:
                sample_dir=out/'previews'/identifier; sample_dir.mkdir(parents=True,exist_ok=True)
                Image.fromarray(image).save(sample_dir/'image.png'); Image.fromarray(mask.astype(np.uint8)*255).save(sample_dir/'mask.png')
                from garment_lab.mesh import export_mesh
                from garment_lab.visualize import preview
                export_mesh(initial,sample_dir)
                export_mesh(target,sample_dir,name='target')
                preview(initial,image,mask,sample_dir/'initial.png')
            print(json.dumps({'prepared':len(records),'counts':dict(counts),'sample':identifier,'category':category,
                              'elapsed':round(time.time()-start,1)}),flush=True)
        except (ValueError,RuntimeError) as error:
            excluded.append({'sample':identifier,'reason':str(error)[:250],'type':type(error).__name__})
            if len(excluded)%30==0: print(json.dumps({'excluded':len(excluded),'reason':str(error)[:180]}),flush=True)
            if isinstance(error,RuntimeError):
                (out/'runtime_failure.json').write_text(json.dumps({'sample':identifier,'type':type(error).__name__,
                    'message':str(error),'completed_samples':len(records)},indent=2))
                raise # A model/runtime failure must not silently filter benchmark samples.
    renderer.delete()
    if not records:
        (out/'failed_preparation.json').write_text(json.dumps({'excluded':excluded,'counts':dict(Counter(row['reason'] for row in excluded))},indent=2))
        raise RuntimeError('No eligible independent garments were prepared; inspect failed_preparation.json.')
    for split in ('train','val','test'):
        rows=[row for row in records if row['split']==split]
        (out/(split+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report={'dataset':'GarmentCodeData_v2','upstream_doi':'10.3929/ethz-b-000690432','license':'CC-BY-4.0',
            'counts':dict(counts),'source_cases_scanned':len(cases),'excluded':excluded,
            'exclusion_counts':dict(Counter(row['reason'] for row in excluded)),'args':vars(args),
            'nlf_sha256':artifact_hash,'domain':'synthetic_original_dataset_renders_not_real_photos',
            'strong_input_condition':'amodal_garment_mask_and_garment_category_and_known_camera',
            'ground_truth_depth_used_in_initializer':False,'ground_truth_body_used_in_initializer':False,
            'target_camera_conversion':'centimeters_to_meters_then_original_pyrender_pose_to_camera_xyz',
            'test_used_for_selection':False,'grouping':'official_garment_split_and_body_sample_identity',
            'seconds':time.time()-start}
    (out/'preparation_receipt.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({key:value for key,value in report.items() if key!='excluded'},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='data');p.add_argument('--out',required=True)
    p.add_argument('--nlf',required=True)
    p.add_argument('--batches',nargs='+',type=int,default=[0]);p.add_argument('--body-types',nargs='+',choices=['random','default'],default=['random'])
    p.add_argument('--image-size',type=int,default=256);p.add_argument('--resolution',type=int,default=24)
    p.add_argument('--train-limit',type=int,default=600);p.add_argument('--val-limit',type=int,default=60);p.add_argument('--test-limit',type=int,default=60)
    p.add_argument('--device',default='cuda:1');p.add_argument('--threads',type=int,default=3);p.add_argument('--memory-fraction',type=float,default=.085)
    run(p.parse_args())


if __name__=='__main__':main()
