"""Controlled training, validation selection and final locked evaluation.

All architectures share image encoder layout, geometry losses, paired body
perturbations and boundary projection protection. Parameter counts are reported.
"""
import argparse,csv,hashlib,json,random,time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from garment_lab.data import GarmentDataset,read_manifest,audit_splits
from garment_lab.mesh import Mesh,boundary_edges
from garment_lab.metrics import training_loss,surface_metrics
from garment_lab.reconstruct import sha256
from .model import Config,Refiner,ARCHITECTURES
from .topology import dense_geodesic,projection_support

EXTRA=('geodesic_distance','slice_support','vertex_mass','slot_distance','visible_proxy','body_vertices')


def source_hash():
    result=hashlib.sha256()
    root=Path(__file__).resolve().parents[1]
    for package in ('garment_study','garment_lab'):
        for path in sorted((root/package).glob('*.py')):
            result.update(str(path.relative_to(root)).encode()); result.update(path.read_bytes())
    return result.hexdigest()


class Dataset:
    def __init__(self,manifest,dense=False):
        self.base=GarmentDataset(manifest,allow_synthetic=False)
        self.records=self.base.records
        if any(row.get('domain')!='GarmentCodeData_v2_synthetic_original_render' for row in self.records):
            raise ValueError('This protocol requires provenance for original GarmentCodeData renders.')
        self.samples=[]
        for index,row in enumerate(self.records):
            sample=self.base[index]
            with np.load(row['cache'],allow_pickle=False) as raw:
                for key in EXTRA: sample[key]=torch.from_numpy(raw[key].copy())
                sample['target_boundary_vertices']=torch.from_numpy(np.unique(raw['target_boundary_edges']).astype(np.int64))
                sample['initial_boundary_vertices']=torch.where(sample['boundary']>0)[0]
                sample['projection_support']=torch.from_numpy(projection_support(raw['mask'],raw['uv']))
                if dense:sample['full_geodesic_distance']=torch.from_numpy(dense_geodesic(raw['initial_vertices'],raw['faces']))
            self.samples.append(sample)

    def __len__(self):return len(self.samples)
    def __getitem__(self,index):return self.samples[index]


def to_device(sample,device): return {key:value.to(device) for key,value in sample.items()}


def paired_predictions(model,sample,paired):
    """Use common stochastic operator draws for the same-image pair."""
    device=sample['initial_vertices'].device
    cpu_before=torch.get_rng_state();cuda_before=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
    first=model(sample)['vertices']
    cpu_after=torch.get_rng_state();cuda_after=torch.cuda.get_rng_state(device) if device.type=='cuda' else None
    torch.set_rng_state(cpu_before)
    if cuda_before is not None:torch.cuda.set_rng_state(cuda_before,device)
    second=model(paired)['vertices']
    torch.set_rng_state(cpu_after)
    if cuda_after is not None:torch.cuda.set_rng_state(cuda_after,device)
    return first,second


def predictable_prediction(model,sample,index):
    device=sample['initial_vertices'].device
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(47000+index)
        return model(sample)['vertices']


def counterfactual(sample,magnitude=.08,generator=None):
    """Same projected garment/image, different predicted body and coarse depth.

    These are controlled input corruptions, not new independent observations.
    The target is never consulted. The stochastic law is shared by every model.
    """
    if magnitude==0:return sample
    v=sample['initial_vertices']; body=sample['body_vertices']; device=v.device
    delta=(torch.rand((),device=device,generator=generator)*2-1)*magnitude
    z=(v[:,2]+delta).clamp_min(.1)
    changed=v*(z/v[:,2].clamp_min(.1))[:,None]
    scales=torch.exp(torch.randn(3,device=device,generator=generator)*magnitude*.35)
    center=body.mean(0,keepdim=True)
    warped=(body-center)*scales+center+torch.stack([delta*0,delta*0,delta])
    # No N x 6890 distance tensor is kept beyond nearest-point feature creation.
    with torch.no_grad():
        indices=[]; distances=[]
        for block in changed.split(256):
            distance,index=torch.cdist(block[None],warped[None])[0].min(1)
            indices.append(index); distances.append(distance)
        index=torch.cat(indices); distance=torch.cat(distances)
        scale=(changed.max(0).values-changed.min(0).values).max().clamp_min(1e-5)
        features=torch.cat([(changed-warped[index])/scale,distance[:,None]/scale],-1)
    return dict(sample,initial_vertices=changed,body_vertices=warped,body_features=features)


def geometry_loss(vertices,sample,points):
    # Same objective as garment_lab.metrics.training_loss, but cache topology
    # indices once. No per-update CUDA->CPU mesh transfer or np.unique sorting.
    def draw(v,f):
        tri=v[f]
        area=torch.linalg.vector_norm(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),dim=-1)
        ids=torch.multinomial(area.detach().clamp_min(1e-10),points,replacement=True)
        r=torch.rand(points,2,device=v.device);a=r[:,0].sqrt()
        weights=torch.stack([1-a,a*(1-r[:,1]),a*r[:,1]],-1)
        return (tri[ids]*weights[:,:,None]).sum(1)
    p=draw(vertices,sample['faces']);q=draw(sample['target_vertices'],sample['target_faces'])
    distance=torch.cdist(p[None],q[None])[0]
    surface=(distance.min(0).values.mean()+distance.min(1).values.mean())/2
    pi,qi=sample['initial_boundary_vertices'],sample['target_boundary_vertices']
    boundary=surface*0
    if len(pi) and len(qi):
        distance=torch.cdist(vertices[pi][None],sample['target_vertices'][qi][None])[0]
        boundary=(distance.min(0).values.mean()+distance.min(1).values.mean())/2
    edges=sample['edges'];initial=sample['initial_vertices']
    before=torch.linalg.vector_norm(initial[edges[:,0]]-initial[edges[:,1]],dim=-1)
    after=torch.linalg.vector_norm(vertices[edges[:,0]]-vertices[edges[:,1]],dim=-1)
    stretch=((after-before)/before.clamp_min(1e-5)).square().mean()
    displacement=(vertices-initial).square().mean()
    return surface+.35*boundary+.002*stretch+.01*displacement


@torch.inference_mode()
def evaluate(model,dataset,device='cuda:1',samples=2000,noise=0.,save_meshes=None):
    model.eval(); rows=[]; latencies=[]
    if save_meshes: Path(save_meshes).mkdir(parents=True,exist_ok=False)
    for index in range(len(dataset)):
        sample=to_device(dataset[index],device)
        generator=torch.Generator(device=device).manual_seed(39000+index)
        perturbed=counterfactual(sample,noise,generator)
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        began=time.perf_counter(); output=predictable_prediction(model,perturbed,index)
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        latencies.append(time.perf_counter()-began)
        pred=Mesh(output.cpu().numpy(),sample['faces'].cpu().numpy())
        target=Mesh(sample['target_vertices'].cpu().numpy(),sample['target_faces'].cpu().numpy())
        metrics=surface_metrics(pred,target,count=samples,threshold=.01)
        initial=Mesh(perturbed['initial_vertices'].cpu().numpy(),pred.faces)
        initial_metrics=surface_metrics(initial,target,count=samples,threshold=.01)
        row={'sample_id':dataset.records[index]['sample_id'],'garment_id':dataset.records[index]['garment_id'],
             'subject_id':dataset.records[index]['subject_id'],'category':dataset.records[index]['category'],
             **metrics,'initial_sampled_surface_cd':initial_metrics['sampled_surface_cd'],
             'initial_sampled_boundary_cd':initial_metrics['sampled_boundary_cd']}
        if noise:
            clean=predictable_prediction(model,sample,index)
            row['prior_sensitivity_m']=float(torch.linalg.vector_norm(output-clean,dim=-1).mean())
        rows.append(row)
        if save_meshes:
            from garment_lab.mesh import export_mesh
            dest=Path(save_meshes)/row['sample_id']; dest.mkdir()
            export_mesh(Mesh(pred.vertices,pred.faces,sample['uv'].cpu().numpy()),dest)
    fields=('sampled_surface_cd','sampled_boundary_cd','normal_consistency','fscore','initial_sampled_surface_cd','initial_sampled_boundary_cd','prior_sensitivity_m')
    means={key:float(np.mean([row[key] for row in rows if row.get(key) is not None])) for key in fields if any(row.get(key) is not None for row in rows)}
    means['validation_score']=means['sampled_surface_cd']+.35*means.get('sampled_boundary_cd',0)
    means['median_refiner_seconds']=float(np.median(latencies))
    return {'means':means,'samples':rows,'units':'meters','alignment':'none','noise_magnitude':noise,
            'domain':'synthetic_original_dataset_render_with_amodal_mask','timing_scope':'refiner_only_not_NLF_or_preparation'}


def train(args):
    if args.epochs<1 or args.validate_every<1: raise ValueError('Positive training and validation schedule required.')
    audit=audit_splits(args.train,args.val)
    if audit['data_kind']!='garment_benchmark' or audit['units']!='meters':raise ValueError('Use independent garment targets in meters.')
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction,torch.device(args.device))
        torch.cuda.reset_peak_memory_stats(args.device)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    source=source_hash(); out=Path(args.out)
    out.mkdir(parents=True,exist_ok=bool(args.resume))
    config=Config(architecture=args.architecture,width=args.width,depth=args.depth,max_displacement=args.max_displacement,
                  protect_projection=not args.no_projection_protection)
    dataset,validation=Dataset(args.train,args.architecture=='dense_geodesic'),Dataset(args.val,args.architecture=='dense_geodesic')
    model=Refiner(config).to(args.device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    start=0; best=float('inf'); updates=0; seconds_before=0.
    protocol={key:getattr(args,key) for key in ('lr','seed','loss_samples','eval_samples','validate_every','noise_magnitude','consistency_weight')}
    if args.resume:
        if Path(args.resume).resolve().parent!=out.resolve() or not (out/'best.pt').exists():
            raise ValueError('Resume in the original run directory containing the validation-selected best.pt.')
        checkpoint=torch.load(args.resume,map_location=args.device,weights_only=True)
        if checkpoint['source_sha256']!=source or checkpoint['protocol']!=protocol or checkpoint['model_config']!=model.model_config():
            raise ValueError('Resume code/config/protocol changed.')
        if checkpoint['train_sha256']!=sha256(args.train) or checkpoint['val_sha256']!=sha256(args.val):raise ValueError('Resume data changed.')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        if args.device.startswith('cuda'):torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu(),args.device)
        start=checkpoint['epoch']+1;best=checkpoint['best_score'];updates=checkpoint['updates'];seconds_before=checkpoint['training_seconds']
        if start>=args.epochs:raise ValueError('Resume total epochs must exceed completed epochs.')
    metadata={'task':'independent_garment_surface_study_v1','args':vars(args),'model_config':model.model_config(),
              'trainable_parameters':sum(p.numel() for p in model.parameters()),'source_sha256':source,
              'train_manifest_sha256':sha256(args.train),'val_manifest_sha256':sha256(args.val),**audit,
              'selection':'clean_validation_surface_cd_plus_0.35_boundary_cd','test_used_for_selection':False,
              'torch':torch.__version__,'cuda':torch.version.cuda,'paired_forward_passes_per_update':2,
              'mask_condition':'annotation_rendered_amodal_silhouette','domain':'synthetic_dataset_not_real_photos'}
    (out/'config.json').write_text(json.dumps(metadata,indent=2))
    began_run=time.perf_counter()
    for epoch in range(start,args.epochs):
        began_epoch=time.perf_counter();model.train();losses=[]
        order=np.random.default_rng(args.seed+epoch).permutation(len(dataset))
        for index in order:
            sample=to_device(dataset[int(index)],args.device)
            paired=counterfactual(sample,args.noise_magnitude)
            optimizer.zero_grad(set_to_none=True)
            original,corrupted=paired_predictions(model,sample,paired)
            loss=(geometry_loss(original,sample,args.loss_samples)+geometry_loss(corrupted,paired,args.loss_samples))/2
            visible=sample['visible_proxy']>.5
            consistency=torch.linalg.vector_norm(original[visible]-corrupted[visible],dim=-1).mean()
            loss=loss+args.consistency_weight*consistency
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()));updates+=1
        score=None;improved=False
        if (epoch+1)%args.validate_every==0 or epoch+1==args.epochs:
            report=evaluate(model,validation,args.device,args.eval_samples)
            score=report['means']['validation_score'];improved=score<best
            if improved: best=score
            report['epoch']=epoch
            (out/f'validation_epoch_{epoch+1:03d}.json').write_text(json.dumps(report))
        training_seconds=seconds_before+time.perf_counter()-began_run
        checkpoint={'task':'independent_garment_surface_study_v1','model':model.state_dict(),'model_config':model.model_config(),
                    'optimizer':optimizer.state_dict(),'epoch':epoch,'best_score':best,'source_sha256':source,'protocol':protocol,
                    'train_sha256':sha256(args.train),'val_sha256':sha256(args.val),'updates':updates,'training_seconds':training_seconds,
                    'rng_state':torch.get_rng_state(),'cuda_rng_state':torch.cuda.get_rng_state(args.device) if args.device.startswith('cuda') else None,
                    'units':'meters','data_kind':'GarmentCodeData_v2_synthetic_original_render'}
        torch.save(checkpoint,out/'latest.pt')
        if improved:torch.save(checkpoint,out/'best.pt')
        row={'epoch':epoch,'loss':float(np.mean(losses)),'validation_score':score,'best_score':best,
             'seconds':time.perf_counter()-began_epoch,'updates':updates}
        with (out/'history.csv').open('a' if epoch or args.resume else 'w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row))
            if epoch==0 and not args.resume:writer.writeheader()
            writer.writerow(row)
        (out/'progress.json').write_text(json.dumps({'status':'training',**row,'epochs':args.epochs}))
        print(json.dumps(row),flush=True)
    result={'status':'completed','best_score':best,'training_seconds':training_seconds,'updates':updates,
            'trainable_parameters':metadata['trainable_parameters'],'source_sha256':source,
            'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else 0,
            'checkpoint_sha256':sha256(out/'best.pt'),'test_used_for_selection':False,'global_sota_claim':False}
    (out/'result.json').write_text(json.dumps(result,indent=2))
    (out/'progress.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    t=sub.add_parser('train')
    for key in ('train','val','out'):t.add_argument('--'+key,required=True)
    t.add_argument('--architecture',choices=ARCHITECTURES,default='evidence_delta')
    t.add_argument('--width',type=int,default=64);t.add_argument('--depth',type=int,default=4)
    t.add_argument('--max-displacement',type=float,default=.20);t.add_argument('--epochs',type=int,default=60)
    t.add_argument('--validate-every',type=int,default=5);t.add_argument('--lr',type=float,default=.001)
    t.add_argument('--loss-samples',type=int,default=512);t.add_argument('--eval-samples',type=int,default=3000)
    t.add_argument('--noise-magnitude',type=float,default=.08);t.add_argument('--consistency-weight',type=float,default=.05)
    t.add_argument('--seed',type=int,default=2026);t.add_argument('--device',default='cuda:1')
    t.add_argument('--threads',type=int,default=3);t.add_argument('--memory-fraction',type=float,default=.055)
    t.add_argument('--no-projection-protection',action='store_true');t.add_argument('--resume')
    e=sub.add_parser('evaluate')
    for key in ('manifest','checkpoint','out'):e.add_argument('--'+key,required=True)
    e.add_argument('--device',default='cuda:1');e.add_argument('--samples',type=int,default=10000)
    e.add_argument('--noise',type=float,default=0);e.add_argument('--save-meshes')
    args=p.parse_args()
    if args.command=='train':train(args)
    else:
        torch.set_num_threads(3)
        if args.device.startswith('cuda'):torch.cuda.set_per_process_memory_fraction(.055,torch.device(args.device))
        checkpoint=torch.load(args.checkpoint,map_location=args.device,weights_only=True)
        if checkpoint['task']!='independent_garment_surface_study_v1':raise ValueError('Wrong checkpoint task')
        model=Refiner(checkpoint['model_config']).to(args.device);model.load_state_dict(checkpoint['model'])
        report=evaluate(model,Dataset(args.manifest,checkpoint['model_config']['architecture']=='dense_geodesic'),args.device,args.samples,args.noise,args.save_meshes)
        report.update(checkpoint_sha256=sha256(args.checkpoint),manifest_sha256=sha256(args.manifest),
                      selection=False,source_sha256=source_hash())
        path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2))
        print(json.dumps(report['means'],indent=2))


if __name__=='__main__':main()
