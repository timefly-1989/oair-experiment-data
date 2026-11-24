"""Matched-budget training on complete synthetic garments and partial real scans."""
import argparse,csv,json,random,time
from pathlib import Path
import numpy as np
import torch
from garment_study.train import counterfactual,paired_predictions
from .models import ScaleRefiner
from .dataset import Dataset,move,check_groups
from .common import write,digest,source_hash

def metric_scale(sample,factor):
    changed=dict(sample)
    for key in ['initial_vertices','body_vertices','target_points','target_boundary']:changed[key]=sample[key]*factor
    return changed

def draw_pred(vertices,faces,count):
    tri=vertices[faces];area=torch.linalg.vector_norm(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),dim=-1)
    ids=torch.multinomial(area.detach().clamp_min(1e-12),count,replacement=True)
    r=torch.rand(count,2,device=vertices.device);a=r[:,0].sqrt();weights=torch.stack([1-a,a*(1-r[:,1]),a*r[:,1]],-1)
    return (tri[ids]*weights[:,:,None]).sum(1)

def loss_for(vertices,sample,points=384,dimensionless=False):
    pred=draw_pred(vertices,sample['faces'],points)
    target=sample['target_points'][torch.randint(len(sample['target_points']),(points,),device=vertices.device)]
    distance=torch.cdist(pred[None],target[None])[0]
    # Complete garments use bidirectional distance. A scan annotation is partial;
    # no reverse term forces unscanned cloth onto a cropped semantic patch.
    surface=distance.min(0).values.mean() if sample['partial_target'] else (distance.min(0).values.mean()+distance.min(1).values.mean())/2
    boundary=surface*0
    if not sample['partial_target'] and len(sample['target_boundary']):
        pv=vertices[sample['boundary']>.5];tv=sample['target_boundary']
        d=torch.cdist(pv[None],tv[None])[0];boundary=(d.min(0).values.mean()+d.min(1).values.mean())/2
    edge=sample['edges'];v0=sample['initial_vertices']
    before=torch.linalg.vector_norm(v0[edge[:,0]]-v0[edge[:,1]],dim=-1).clamp_min(1e-5)
    after=torch.linalg.vector_norm(vertices[edge[:,0]]-vertices[edge[:,1]],dim=-1)
    stretch=((after-before)/before).square().mean();displacement=(vertices-v0).square().mean()
    extent=ScaleRefiner.extent(sample) if dimensionless else 1.
    return (surface+.35*boundary)/extent+.002*stretch+.01*displacement/(extent*extent)

@torch.inference_mode()
def evaluate(model,dataset,device,points=1536):
    model.eval();rows=[]
    for i in range(len(dataset)):
        sample=move(dataset[i],device)
        with torch.random.fork_rng(devices=[torch.device(device).index] if str(device).startswith('cuda') else []):
            torch.manual_seed(51000+i);vertices=model(sample)['vertices'];pred=draw_pred(vertices,sample['faces'],points)
        target=sample['target_points'][:points];dist=torch.cdist(pred[None],target[None])[0]
        coverage=dist.min(0).values.mean();precision=dist.min(1).values.mean()
        score=coverage if sample['partial_target'] else (coverage+precision)/2
        if not sample['partial_target'] and len(sample['target_boundary']):
            d=torch.cdist(vertices[sample['boundary']>.5][None],sample['target_boundary'][None])[0]
            score=score+.35*(d.min(0).values.mean()+d.min(1).values.mean())/2
        rows.append({'sample_id':dataset.records[i]['sample_id'],'subject_id':dataset.records[i]['subject_id'],
                     'score':float(score),'target_to_prediction':float(coverage),'prediction_to_target_diagnostic':float(precision),
                     'partial_target':sample['partial_target']})
    return {'score':float(np.mean([r['score'] for r in rows])),'rows':rows,
            'metric':'partial scans: target-to-predicted sampled surface; complete synthetic: CD+0.35BCD',
            'test_used_for_selection':False,'real_scans_are_partial_targets':True}

def train(args):
    torch.set_num_threads(2);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    device=args.device
    if device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.026,torch.device(device));torch.cuda.reset_peak_memory_stats(device)
    audit=check_groups(args.train,args.val);out=Path(args.out);out.mkdir(parents=True,exist_ok=args.resume)
    dataset=Dataset(args.train,args.limit);validation=Dataset(args.val,args.val_limit)
    synthetic=Dataset(args.synthetic_replay,args.replay_limit) if args.synthetic_replay else None
    model=ScaleRefiner(args.architecture,args.coordinate_mode,not args.no_projection).to(device)
    if args.init:
        checkpoint=torch.load(args.init,map_location='cpu',weights_only=True)
        state=checkpoint['model']
        if any(k.startswith('core.') for k in state):model.load_state_dict(state)
        else:model.core.load_state_dict(state)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    protocol=vars(args).copy();src=source_hash()
    write(out/'config.json',{'protocol':protocol,'model_config':model.model_config(),'source_sha256':src,
                            'train_manifest_sha256':digest(args.train),'val_manifest_sha256':digest(args.val),
                            'trainable_parameters':sum(p.numel() for p in model.parameters()),'audit':audit,
                            'partial_scan_targets_not_complete_garments':True,'test_used_for_selection':False})
    best=float('inf');updates=0;start=time.time();start_epoch=0;previous_seconds=0.
    if args.resume:
        checkpoint=torch.load(out/'latest.pt',map_location=device,weights_only=True)
        previous_protocol={k:v for k,v in checkpoint['protocol'].items() if k!='resume'}
        if previous_protocol!={k:v for k,v in protocol.items() if k!='resume'} or checkpoint['source_sha256']!=src:raise ValueError('Resume protocol/source changed')
        model.load_state_dict(checkpoint['model']);optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng_state'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu(),device)
        start_epoch=checkpoint['epoch'];updates=checkpoint['updates'];best=checkpoint['best_score'];previous_seconds=checkpoint.get('training_seconds',0.)
    for epoch in range(start_epoch,args.epochs):
        model.train();losses=[];began=time.time()
        indices=np.random.default_rng(args.seed+epoch).permutation(len(dataset))
        for j,index in enumerate(indices):
            cases=[dataset[int(index)]]
            if synthetic is not None:cases.append(synthetic[(epoch*len(dataset)+j)%len(synthetic)])
            for case in cases:
                sample=move(case,device)
                if args.appearance_augmentation:
                    brightness=.7+.6*torch.rand((),device=device);contrast=.75+.5*torch.rand((),device=device)
                    image=sample['image'];image=((image-image.mean((1,2),keepdim=True))*contrast+image.mean((1,2),keepdim=True))*brightness
                    image=(image+torch.randn_like(image)*.012).clamp(0,1)
                    if torch.rand((),device=device)<.25:
                        y=int(torch.randint(16,200,()).item());x=int(torch.randint(16,200,()).item())
                        image=image.clone();image[:,y:y+40,x:x+40]=image.mean((1,2),keepdim=True)
                    sample=dict(sample,image=image)
                if args.scale_augmentation:
                    factor=torch.rand((),device=device)*.6+.7
                    sample=metric_scale(sample,factor)
                paired=counterfactual(sample,.08)
                optimizer.zero_grad(set_to_none=True)
                original,corrupted=paired_predictions(model,sample,paired)
                loss=(loss_for(original,sample,args.points,args.dimensionless_loss)+loss_for(corrupted,paired,args.points,args.dimensionless_loss))/2
                visible=sample['visible_proxy']>.5
                normalization=ScaleRefiner.extent(sample) if args.dimensionless_loss else 1.
                loss=loss+.05*torch.linalg.vector_norm(original[visible]-corrupted[visible],dim=-1).mean()/normalization
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1;losses.append(float(loss.detach()))
        report=None;improved=False
        if (epoch+1)%args.validate_every==0 or epoch+1==args.epochs:
            report=evaluate(model,validation,device,args.validation_points);improved=report['score']<best
            if improved:best=report['score']
            write(out/f'validation_epoch_{epoch+1:03d}.json',report)
        checkpoint={'task':'garment_realism_v6','model':model.state_dict(),'optimizer':optimizer.state_dict(),
                    'model_config':model.model_config(),'protocol':protocol,'source_sha256':src,'epoch':epoch+1,
                    'updates':updates,'best_score':best,'rng_state':torch.get_rng_state(),'training_seconds':previous_seconds+time.time()-start,
                    'cuda_rng_state':torch.cuda.get_rng_state(device) if device.startswith('cuda') else None}
        torch.save(checkpoint,out/'latest.pt')
        if improved:torch.save(checkpoint,out/'best.pt')
        row={'epoch':epoch+1,'epochs':args.epochs,'loss':float(np.mean(losses)),'validation_score':report['score'] if report else None,
             'best_score':best,'updates':updates,'seconds':previous_seconds+time.time()-start,'epoch_seconds':time.time()-began}
        with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        write(out/'progress.json',{'status':'training',**row});print(json.dumps(row),flush=True)
    result={'status':'completed','updates':updates,'epochs':args.epochs,'seconds':previous_seconds+time.time()-start,'best_score':best,
            'source_sha256':src,'checkpoint_sha256':digest(out/'best.pt'),'test_used_for_selection':False,
            'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else 0}
    write(out/'result.json',result);write(out/'progress.json',result)

def main():
    p=argparse.ArgumentParser()
    for name in ['train','val','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--architecture',default='evidence_delta');p.add_argument('--coordinate-mode',default='raw',choices=['raw','body_scale'])
    p.add_argument('--dimensionless-loss',action='store_true');p.add_argument('--scale-augmentation',action='store_true');p.add_argument('--no-projection',action='store_true')
    p.add_argument('--init');p.add_argument('--synthetic-replay');p.add_argument('--replay-limit',type=int,default=600)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=1104);p.add_argument('--epochs',type=int,default=15)
    p.add_argument('--limit',type=int);p.add_argument('--val-limit',type=int);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--points',type=int,default=384);p.add_argument('--validation-points',type=int,default=1536);p.add_argument('--validate-every',type=int,default=5)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--appearance-augmentation',action='store_true')
    train(p.parse_args())
if __name__=='__main__':main()
