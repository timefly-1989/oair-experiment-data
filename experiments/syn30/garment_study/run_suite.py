"""Bounded concurrent experimental sweep; select on validation, never on test."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from garment_lab.reconstruct import sha256
from garment_lab.data import read_manifest
from .model import Config,Refiner
from .train import source_hash


def count(config):return sum(parameter.numel() for parameter in Refiner(config).parameters())


def prepare_jobs(args):
    proposed=Config(architecture='evidence_delta',width=args.width,depth=args.depth)
    target=count(proposed)
    candidates=range(16,161,4)
    matched=min(candidates,key=lambda width:abs(count(Config(architecture='graph',width=width,depth=args.depth))-target))
    variants=[('B00_graph','graph',args.width),('B01_graph_capacity','graph',matched),
              ('B02_adaptive_slice','adaptive_slice',args.width),('B03_ddl','ddl',args.width),('B04_dense_geodesic','dense_geodesic',args.width),
              ('B05_TransolverPlusCore','transolver_plus',args.width),
              ('A01_geodesic_slice','geodesic_slice',args.width),('A02_geodesic_ddl','geodesic_ddl',args.width),
              ('A03_state_target','evidence_state_target',args.width),('A04_evidence_delta','evidence_delta',args.width)]
    jobs=[]
    for seed in args.seeds:
        for identifier,architecture,width in variants:
            jobs.append({'id':f'{identifier}_s{seed}','variant':identifier,'architecture':architecture,'width':width,'seed':seed,
                         'parameters':count(Config(architecture=architecture,width=width,depth=args.depth))})
    return jobs


def split_audit(root):
    sets={split:read_manifest(root/(split+'.jsonl')) for split in ('train','val','test')}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        for field in ('sample_id','subject_id','garment_id'):
            if {row[field] for row in sets[a]}&{row[field] for row in sets[b]}:raise ValueError('Leakage '+field+' '+a+' '+b)
    return {split:{'count':len(rows),'manifest_sha256':sha256(root/(split+'.jsonl'))} for split,rows in sets.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--width',type=int,default=64);p.add_argument('--depth',type=int,default=4)
    p.add_argument('--seeds',nargs='+',type=int,default=[2026,2027,2028])
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:0','cuda:1','cuda:1'])
    p.add_argument('--loss-samples',type=int,default=512);p.add_argument('--eval-samples',type=int,default=3000)
    p.add_argument('--validate-every',type=int,default=5)
    args=p.parse_args(); data=Path(args.data).resolve();out=Path(args.out).resolve()
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    audit=split_audit(data);source=source_hash();jobs=prepare_jobs(args)
    snapshot={'source_sha256':source,'args':vars(args),'data_audit':audit,'jobs':jobs,
              'selection_metric':'mean_over_seeds_of_best_clean_validation_cd_plus_0.35_boundary_cd',
              'final_evaluation':{'samples_per_surface':10000,'noise_magnitudes_m':[0.,.04,.08,.12],'stochastic_operator_seed':'47000_plus_sample_index_common_for_clean_and_corrupted'},
              'test_influences_model_selection':False,'created_unix':time.time()}
    config_path=out/'protocol_snapshot.json'
    if config_path.exists():raise FileExistsError('Use a new sweep directory; no automatic reuse of historical outcomes.')
    config_path.write_text(json.dumps(snapshot,indent=2))
    pending=jobs.copy();active={};finished=[];failed=[]
    began=time.time()
    while pending or active:
        if source_hash()!=source:raise RuntimeError('Source changed during the locked sweep; preserve logs and stop.')
        for slot,device in enumerate(args.devices):
            if slot in active or not pending:continue
            job=pending.pop(0);run_dir=out/job['id'];log_path=out/'logs'/(job['id']+'.log')
            command=[sys.executable,'-m','garment_study.train','train','--train',str(data/'train.jsonl'),'--val',str(data/'val.jsonl'),
                     '--out',str(run_dir),'--architecture',job['architecture'],'--width',str(job['width']),
                     '--depth',str(args.depth),'--seed',str(job['seed']),'--epochs',str(args.epochs),
                     '--validate-every',str(args.validate_every),'--loss-samples',str(args.loss_samples),
                     '--eval-samples',str(args.eval_samples),'--device',device,'--memory-fraction','.02','--threads','2']
            stream=log_path.open('w');environment=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=environment)
            active[slot]={'job':job,'process':process,'stream':stream,'command':command,'device':device,'began':time.time()}
            print(json.dumps({'started':job['id'],'pid':process.pid,'device':device}),flush=True)
        for slot,state in list(active.items()):
            code=state['process'].poll()
            if code is None:continue
            state['stream'].close();job=state['job']
            row={**job,'exit_code':code,'seconds':time.time()-state['began'],'device':state['device']}
            result=out/job['id']/'result.json'
            if code==0 and result.exists():row.update(json.loads(result.read_text()));finished.append(row)
            else:failed.append(row)
            print(json.dumps({'finished':job['id'],'exit_code':code}),flush=True);del active[slot]
        progress={'status':'training','completed':len(finished),'failed':failed,'remaining':len(pending),
                  'active':[{'id':state['job']['id'],'pid':state['process'].pid,'device':state['device']} for state in active.values()],
                  'elapsed_seconds':time.time()-began}
        (out/'suite_progress.json').write_text(json.dumps(progress,indent=2))
        if pending or active:time.sleep(10)
    (out/'completed_runs.json').write_text(json.dumps(finished,indent=2))
    if failed:
        (out/'suite_failure.json').write_text(json.dumps(failed,indent=2));raise RuntimeError('One or more locked training runs failed. Do not drop failed configurations from the comparison.')
    groups={}
    for row in finished:groups.setdefault(row['variant'],[]).append(row)
    ranking=[]
    for identifier,rows in groups.items():
        scores=[row['best_score'] for row in rows]
        ranking.append({'variant':identifier,'validation_score_mean':float(np.mean(scores)),
                        'validation_score_std':float(np.std(scores,ddof=1)) if len(scores)>1 else 0,
                        'seeds':[row['seed'] for row in rows],'parameters':rows[0]['parameters'],
                        'runs':[row['id'] for row in rows]})
    ranking.sort(key=lambda row:row['validation_score_mean'])
    winner=ranking[0];chosen=min(groups[winner['variant']],key=lambda row:row['best_score'])
    selected={'selection':'validation_only','winner':winner['variant'],'selected_run':chosen['id'],
              'selected_checkpoint':str(out/chosen['id']/'best.pt'),'selected_checkpoint_sha256':sha256(out/chosen['id']/'best.pt'),
              'ranking':ranking,'source_sha256':source,'test_manifest_sha256':audit['test']['manifest_sha256'],
              'global_sota_claim':False,'all_algorithms_and_seeds_retained':True}
    (out/'final_selection.json').write_text(json.dumps(selected,indent=2))
    progress.update(status='training_completed_validation_selection_locked',winner=winner['variant'])
    (out/'suite_progress.json').write_text(json.dumps(progress,indent=2))
    print(json.dumps(selected,indent=2),flush=True)


if __name__=='__main__':main()
