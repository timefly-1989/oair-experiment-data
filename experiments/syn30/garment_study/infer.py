"""Use a trained independent-clothing study checkpoint on an RGB person photograph."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision
from garment_lab.mesh import Mesh,export_mesh
from garment_lab.reconstruct import reconstruct,read_body_cache,sha256
from garment_lab.segmentation import parse_garment,load_mask
from garment_lab.data import sample_tensors
from garment_lab.templates import panel
from garment_lab.visualize import preview,viewer
from .model import Refiner
from .topology import topology_features,dense_geodesic,projection_support


def body_from_rgb(args,image):
    h,w=image.shape[:2]
    if args.intrinsics:
        k=np.load(args.intrinsics,allow_pickle=False).astype(np.float32)
        if k.shape!=(3,3):raise ValueError('Intrinsics must be a 3x3 NPY matrix for this exact image.')
        camera_source='user_supplied_intrinsics'
    else:
        f=max(h,w)/(2*np.tan(np.deg2rad(args.fov)/2))
        k=np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1]],dtype=np.float32)
        camera_source=f'assumed_fov_{args.fov}_degrees_not_calibrated'
    torch._C._jit_set_profiling_executor(False);torch._C._jit_set_profiling_mode(False)
    model=torch.jit.load(args.nlf,map_location=args.device).eval()
    inputs=torch.from_numpy(image.copy()).permute(2,0,1)[None].to(args.device)
    with torch.inference_mode():
        if args.person_box:
            box=torch.tensor([[*args.person_box,1]],device=args.device,dtype=torch.float32)
            output=model._estimate_parametric_batched(inputs,[box],intrinsic_matrix=torch.from_numpy(k)[None].to(args.device),
                       num_aug=1,internal_batch_size=1,suppress_implausible_poses=False)
        else:
            output=model.detect_smpl_batched(inputs,intrinsic_matrix=torch.from_numpy(k)[None].to(args.device),
                       num_aug=1,internal_batch_size=1,max_detections=1,suppress_implausible_poses=False)
    if not len(output['vertices3d'][0]):raise ValueError('NLF found no person; supply --person-box x y width height.')
    vertices=output['vertices3d'][0][0].cpu().numpy()/1000
    del model;gc.collect()
    if args.device.startswith('cuda'):torch.cuda.empty_cache()
    return {'vertices':vertices.astype(np.float32),'intrinsics':k,'faces':None},camera_source


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('image','checkpoint','out'):p.add_argument('--'+key,required=True)
    p.add_argument('--mask');p.add_argument('--category',choices=['auto','top','dress','skirt','pants'],default='auto')
    p.add_argument('--body-cache');p.add_argument('--body-assets');p.add_argument('--intrinsics')
    p.add_argument('--nlf',required=True)
    p.add_argument('--fov',type=float,default=55);p.add_argument('--person-box',nargs=4,type=float)
    p.add_argument('--device',default='cuda:1');p.add_argument('--threads',type=int,default=3)
    p.add_argument('--resolution',type=int,default=24);p.add_argument('--offline',action='store_true')
    p.add_argument('--model-cache',default='models');args=p.parse_args()
    out=Path(args.out)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output directory.')
    out.mkdir(parents=True,exist_ok=True);began=time.perf_counter();torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.085,torch.device(args.device))
    image=np.array(Image.open(args.image).convert('RGB'))
    if args.mask:
        if args.category=='auto':raise ValueError('An explicit mask requires a known garment category.')
        mask=load_mask(args.mask,image.shape);category=args.category;seg={'mask_kind':'user_supplied','sha256':sha256(args.mask)}
    else:
        mask,category,seg=parse_garment(image,args.category,args.device,args.model_cache,args.offline)
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    if args.body_cache:
        body=read_body_cache(args.body_cache,image.shape);camera_source='exact_body_cache_camera'
    else:body,camera_source=body_from_rgb(args,image)
    coarse,boundary,metadata,mask=reconstruct(image,mask,category,args.resolution,body=body)
    arrays={'initial_vertices':coarse.vertices,'faces':coarse.faces,'uv':coarse.uv,'image':image,'mask':mask,'body_vertices':body['vertices']}
    sample=sample_tensors(arrays)
    for key,value in topology_features(coarse.vertices,coarse.faces).items():
        if key in ('geodesic_distance','slice_support','vertex_mass','slot_distance'):sample[key]=torch.from_numpy(value)
    sample['visible_proxy']=torch.from_numpy((panel(category,args.resolution)['side']==0).astype(np.float32))
    sample['projection_support']=torch.from_numpy(projection_support(mask,coarse.uv))
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if checkpoint.get('task')!='independent_garment_surface_study_v1':raise ValueError('Use a trained garment_study checkpoint.')
    if checkpoint['model_config']['architecture']=='dense_geodesic':
        sample['full_geodesic_distance']=torch.from_numpy(dense_geodesic(coarse.vertices,coarse.faces))
    model=Refiner(checkpoint['model_config']).to(args.device).eval();model.load_state_dict(checkpoint['model'])
    with torch.inference_mode():vertices=model({key:value.to(args.device) for key,value in sample.items()})['vertices'].cpu().numpy()
    refined=Mesh(vertices,coarse.faces.copy(),coarse.uv,coarse.colors).validate()
    export_mesh(coarse,out,name='initial_garment');topology=export_mesh(refined,out)
    # Camera coordinates use y down; only the preview is converted to y up.
    display=Mesh(refined.vertices*np.array([1,-1,1],dtype=np.float32),refined.faces,refined.uv,refined.colors)
    preview(display,image,mask,out/'preview.png');viewer(display,out/'viewer.html')
    Image.fromarray(image).save(out/'input.png');Image.fromarray(mask.astype(np.uint8)*255).save(out/'garment_mask.png')
    np.savez_compressed(out/'body_context.npz',initial_vertices=body['vertices'],intrinsics=body['intrinsics'],image_size=np.array(image.shape[:2]))
    if args.body_assets:
        with np.load(args.body_assets,allow_pickle=False) as assets:
            export_mesh(Mesh(body['vertices'],assets['faces'].astype(np.int64)),out,name='body')
    metadata.update(task='independent_garment_surface_study_v1',model_config=model.model_config(),
        checkpoint_sha256=sha256(args.checkpoint),checkpoint_training_domain=checkpoint['data_kind'],
        checkpoint_source_sha256=checkpoint['source_sha256'],segmentation=seg,camera_source=camera_source,
        image_sha256=sha256(args.image),topology=topology,total_seconds=time.perf_counter()-began,
        target_depth_or_body_ground_truth_used=False,real_photo_3d_accuracy_measured=False,
        texture='image_vertex_colors_with_inferred_neutral_back',
        preview_coordinates='x_right_y_up_z_forward_display_only_mesh_files_remain_camera_xyz_meters')
    (out/'reconstruction.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps({'output':str(out.resolve()),'topology':topology,'seconds':metadata['total_seconds']},indent=2))


if __name__=='__main__':main()
