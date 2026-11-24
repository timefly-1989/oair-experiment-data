"""Locked real-scan evaluation; partial outer patches are not complete garments."""
import argparse,hashlib,io,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion,binary_dilation
import torch
from garment_lab.reconstruct import reconstruct
from garment_lab.templates import panel
from garment_study.topology import topology_features,projection_support
from garment_study.train import counterfactual
from garment_expansion.common import mesh_quality
from .common import write,digest
from .dataset import Dataset,move,sample_surface
from .models import ScaleRefiner
from .train import metric_scale

CONDITIONS=['clean','scale065','scale150','body_noise012','exposure050','jpeg25','blur2','occlusion','mask_erode2','mask_dilate2']

def changed(sample,row,condition):
    if condition=='clean':return sample
    if condition.startswith('scale'):return metric_scale(sample,{'scale065':.65,'scale150':1.5}[condition])
    if condition=='body_noise012':
        generator=torch.Generator(device=sample['initial_vertices'].device).manual_seed(int(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:8],16))
        return counterfactual(sample,.12,generator)
    if condition=='exposure050':return dict(sample,image=sample['image']*.5)
    if condition in ['jpeg25','blur2','occlusion']:
        image=Image.fromarray(np.uint8(sample['image'].cpu().permute(1,2,0).numpy()*255))
        if condition=='jpeg25':
            buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=25);image=Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')
        elif condition=='blur2':image=image.filter(ImageFilter.GaussianBlur(2))
        else:
            array=np.array(image);y,x=np.where(sample['mask'].cpu().numpy()>.5)
            if len(x):
                cx=int(np.median(x));cy=int(np.median(y));array[max(0,cy-20):cy+20,max(0,cx-20):cx+20]=[100,100,100]
            image=Image.fromarray(array)
        return dict(sample,image=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).to(sample['image']).div(255))
    if condition in ['mask_erode2','mask_dilate2']:
        mask=(binary_erosion if condition=='mask_erode2' else binary_dilation)(sample['mask'].cpu().numpy()>.5,iterations=2)
        rgb=np.uint8(sample['image'].cpu().permute(1,2,0).numpy()*255);body=sample['body_vertices'].cpu().numpy()
        with np.load(row['cache'],allow_pickle=False) as z:k=z['intrinsics']
        initial,boundary,meta,mask=reconstruct(rgb,mask,row['category'],24,body={'vertices':body,'intrinsics':k})
        bv=initial.vertices;dist,ids=cKDTree(body).query(bv);size=max(np.ptp(bv,axis=0).max(),1e-5)
        from garment_lab.mesh import edge_counts
        edges,_=edge_counts(initial.faces)
        arrays={'initial_vertices':bv,'faces':initial.faces.astype(np.int64),'uv':initial.uv,'mask':mask.astype(np.float32),
                'body_features':np.c_[(bv-body[ids])/size,dist/size].astype(np.float32),'edges':edges.astype(np.int64),
                'boundary':boundary,'visible_proxy':(panel(row['category'],24)['side']==0).astype(np.float32),
                'projection_support':projection_support(mask,initial.uv),**topology_features(bv,initial.faces)}
        return dict(sample,**{key:torch.from_numpy(value.copy()).to(sample['initial_vertices'].device) for key,value in arrays.items()})
    raise ValueError(condition)

def load_model(checkpoint,device):
    data=torch.load(checkpoint,map_location='cpu',weights_only=True);config=data['model_config']
    model=ScaleRefiner(config['architecture'],config.get('coordinate_mode','raw'),config.get('protect',config.get('protect_projection',True)),config.get('width',64)).to(device)
    state=data['model']
    (model if any(key.startswith('core.') for key in state) else model.core).load_state_dict(state)
    return model.eval()

@torch.inference_mode()
def run(args):
    lock=json.loads(Path(args.lock).read_text())
    if args.checkpoint:
        if digest(args.checkpoint) not in lock['checkpoint_sha256']:raise ValueError('Checkpoint not locked before test')
    if digest(args.manifest) not in lock['test_manifest_sha256']:raise ValueError('Test manifest not locked')
    device=args.device;torch.set_num_threads(2)
    if device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.028,torch.device(device))
    data=Dataset(args.manifest);model=load_model(args.checkpoint,device) if args.checkpoint else None
    rows=[];failures=[];start=time.time();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    for condition in args.conditions:
        for i,row in enumerate(data.records):
            sample=move(data[i],device)
            try:
                sample=changed(sample,row,condition)
                with torch.random.fork_rng(devices=[torch.device(device).index] if device.startswith('cuda') else []):
                    torch.manual_seed(int(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:8],16))
                    pred=model(sample)['vertices'] if model else sample['initial_vertices']
                vertices=pred.cpu().numpy();faces=sample['faces'].cpu().numpy();initial=sample['initial_vertices'].cpu().numpy()
                if not np.isfinite(vertices).all():raise ValueError('nonfinite prediction')
                points=sample_surface(vertices,faces,4096,61234)
                target=sample['target_points'].cpu().numpy()
                coverage=cKDTree(points).query(target)[0];precision=cKDTree(target).query(points)[0]
                extent=float(ScaleRefiner.extent(sample));recall=float((coverage<.01).mean());diagnostic_precision=float((precision<.01).mean())
                item={'sample_id':row['sample_id'],'group_id':row.get('group_id',row['subject_id']),'category':row['category'],'condition':condition,
                      'target_to_prediction':float(coverage.mean()),'prediction_to_target_diagnostic':float(precision.mean()),
                      'coverage_percent_body_extent':float(coverage.mean()/extent*100),'recall_10mm_source_unit':recall,
                      'precision_10mm_partial_truth_diagnostic':diagnostic_precision,
                      'fscore_10mm_partial_truth_diagnostic':2*recall*diagnostic_precision/max(recall+diagnostic_precision,1e-12),
                      **mesh_quality(vertices,initial,faces)}
                rows.append(item)
                if condition=='clean' and i<6:
                    import trimesh
                    trimesh.Trimesh(vertices,faces,process=False).export(out/(row['sample_id']+'.ply'))
            except (ValueError,RuntimeError) as error:
                failures.append({'sample_id':row['sample_id'],'condition':condition,'error':str(error)[:200]})
        write(out/'progress.json',{'status':'evaluating','condition':condition,'rows':len(rows),'failures':len(failures),'seconds':time.time()-start})
    summary={}
    for condition in args.conditions:
        values=[r for r in rows if r['condition']==condition]
        summary[condition]={'cases':len(values),'target_to_prediction':float(np.mean([r['target_to_prediction'] for r in values])) if values else None,
                            'coverage_percent_body_extent':float(np.mean([r['coverage_percent_body_extent'] for r in values])) if values else None}
    report={'status':'completed','rows':rows,'failures':failures,'summary':summary,'seconds':time.time()-start,
            'checkpoint_sha256':digest(args.checkpoint) if args.checkpoint else None,'manifest_sha256':digest(args.manifest),
            'input_domain':'real_scan_rerender_not_captured_photograph','target_kind':'observed_semantic_outer_patch',
            'distance_units':'source metric coordinates recovered with author scale; absolute scale not independently remeasured',
            'test_used_for_selection':False,'lock_sha256':digest(args.lock)}
    write(out/'report.json',report);write(out/'progress.json',{'status':'completed','rows':len(rows),'failures':len(failures)})
    print(json.dumps({'out':str(out),'status':'completed','rows':len(rows),'failures':len(failures)}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--lock',required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--conditions',nargs='+',default=CONDITIONS);run(p.parse_args())
if __name__=='__main__':main()
