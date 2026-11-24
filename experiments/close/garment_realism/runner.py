"""Persistent, resource-limited, dependency-aware v6 research queue."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
from collections import Counter
from .common import ROOT,STUDY,write,digest,records,source_hash

VARIANTS=[
 ('N03_full','evidence_delta','body_scale',True),
 ('A04_raw','evidence_delta','raw',False),
 ('B03_raw_DDL','ddl','raw',False),
 ('B05_raw_Transolver','transolver_plus','raw',False),
 ('N06_normalized_DDL','ddl','body_scale',True),
 ('N07_normalized_Transolver','transolver_plus','body_scale',True),
 ('N01_coordinate_only','evidence_delta','body_scale',False),
 ('N02_loss_only','evidence_delta','raw',True),
 ('N04_state_target','evidence_state_target','body_scale',True),
 ('N05_residual_image','residual_geo_image','body_scale',True),
 ('A05_raw','residual_geo_image','raw',False),
 ('B00_graph','graph','raw',False),
]

def fixed_subset(source,destination,limit):
    rows=records(source);parent=Path(source).parent
    for row in rows:row['cache']=str((parent/row['cache']).resolve())
    groups={category:sorted([r for r in rows if r['category']==category],key=lambda r:__import__('hashlib').sha256(('v6-subset:'+r['sample_id']).encode()).hexdigest()) for category in sorted({r['category'] for r in rows})}
    chosen=[]
    while len(chosen)<min(limit,len(rows)):
        for category in groups:
            if groups[category] and len(chosen)<limit:chosen.append(groups[category].pop(0))
    text=''.join(json.dumps(row)+'\n' for row in chosen)
    if destination.exists():
        if destination.read_text()!=text:raise ValueError('Frozen subset changed')
    else:destination.write_text(text)
    return chosen

def build_plan():
    data=STUDY/'manifests';data.mkdir(exist_ok=True)
    train=fixed_subset(ROOT/'data/prepared_expanded_v5/train.jsonl',data/'synthetic_train600.jsonl',600)
    val=fixed_subset(ROOT/'data/prepared_expanded_v5/val.jsonl',data/'synthetic_val80.jsonl',80)
    jobs=[]
    for seed in [1104,1105,1106]:
        for variant,architecture,coordinate,dimensionless in VARIANTS:
            pre_id=f'{variant}_s{seed}_synthetic';fine_id=f'{variant}_s{seed}_scan_mixed'
            for stage,jobid,dependency in [('synthetic',pre_id,None),('scan_mixed',fine_id,pre_id)]:
                jobs.append({'id':jobid,'variant':variant,'seed':seed,'stage':stage,'architecture':architecture,'coordinate_mode':coordinate,
                             'dimensionless_loss':dimensionless,'dependency':dependency,'epochs':15 if stage=='synthetic' else 10})
    return {'version':'realism_v6_20261104','source_sha256':source_hash(),'jobs':jobs,'training_runs':72,
            'variants':12,'seeds':[1104,1105,1106],'synthetic_training_samples':len(train),'synthetic_validation_samples':len(val),
            'synthetic_train_sha256':digest(data/'synthetic_train600.jsonl'),'synthetic_val_sha256':digest(data/'synthetic_val80.jsonl'),
            'primary_contrasts':[['N03_full','A04_raw'],['N03_full','N06_normalized_DDL'],['N03_full','N07_normalized_Transolver']],
            'new_real_test_never_used_for_selection':True,'input_domain':'real_scan_rerender_not_captured_photographs',
            'grouping':'filename prefix proxy; true person identity not independently verified','preparation_source_separately_frozen':True,
            'all_variants_common_appearance_and_scale_augmentation':True,
            'complete_garment_vs_partial_patch_metrics_must_remain_distinct':True}

def gpu_free():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    return {int(line.split(',')[0]):int(line.split(',')[1]) for line in raw.splitlines()}

def command_for(job,gpu):
    base=STUDY/'manifests';out=STUDY/'runs'/job['id']
    if job['stage']=='synthetic':train=base/'synthetic_train600.jsonl';val=base/'synthetic_val80.jsonl'
    else:train=ROOT/'data/close_v6/train.jsonl';val=ROOT/'data/close_v6/val.jsonl'
    command=[sys.executable,'-m','garment_realism.train','--train',str(train),'--val',str(val),'--out',str(out),
             '--architecture',job['architecture'],'--coordinate-mode',job['coordinate_mode'],'--seed',str(job['seed']),
             '--epochs',str(job['epochs']),'--device',f'cuda:{gpu}','--scale-augmentation','--appearance-augmentation']
    if job['dimensionless_loss']:command+=['--dimensionless-loss']
    if job['stage']=='scan_mixed':command+=['--init',str(STUDY/'runs'/job['dependency']/'best.pt'),'--synthetic-replay',str(base/'synthetic_train600.jsonl'),'--lr','.0003']
    if (out/'latest.pt').exists():command+=['--resume']
    elif out.exists():
        if (out/'result.json').exists():raise RuntimeError('Completed run should have been skipped')
        out.rename(out.with_name(out.name+f'_interrupted_without_checkpoint_{int(time.time())}'))
    return command

def launch_extra_views():
    out=ROOT/'data/close_v6_extra_views';log=(STUDY/'logs/prepare_extra_views.log').open('a')
    command=[sys.executable,'-m','garment_realism.prepare_close','--out',str(out),'--only-split','test','--extra-views']
    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    write(STUDY/'extra_views_job.json',{'pid':proc.pid,'command':command});return proc,log

def main():
    STUDY.mkdir(exist_ok=True);(STUDY/'runs').mkdir(exist_ok=True);(STUDY/'logs').mkdir(exist_ok=True)
    handle=(STUDY/'pipeline.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=build_plan();path=STUDY/'experiment_plan.json'
    if path.exists():
        if json.loads(path.read_text())!=plan:raise ValueError('Plan/source changed; create another version')
    else:write(path,plan)
    active={};failed=[];completed=set();extra=None;extra_started=(ROOT/'data/close_v6_extra_views/receipt.json').exists()
    while len(completed)+len(failed)<len(plan['jobs']):
        for job in plan['jobs']:
            if (STUDY/'runs'/job['id']/'result.json').exists():completed.add(job['id'])
        for jobid,(proc,gpu,log) in list(active.items()):
            code=proc.poll()
            if code is None:continue
            log.close();active.pop(jobid)
            if code!=0:
                failure={'id':jobid,'exit_code':code,'log':str(STUDY/'logs'/(jobid+'.log'))}
                failed.append(failure);write(STUDY/'failures.json',failed)
        ready=(ROOT/'data/close_v6/receipt.json').exists()
        if ready and not extra_started:
            extra=launch_extra_views();extra_started=True
        extra_done=(ROOT/'data/close_v6_extra_views/receipt.json').exists()
        if extra and extra[0].poll() is not None:
            if extra[0].returncode!=0:raise RuntimeError('Extra-view preparation failed; retain log and do not silently omit track')
            extra[1].close();extra=None
        free=gpu_free();load=Counter(gpu for proc,gpu,log in active.values())
        failed_ids={r['id'] for r in failed}
        for job in plan['jobs']:
            if job['id'] in completed or job['id'] in active or job['id'] in failed_ids:continue
            if job['dependency'] and job['dependency'] not in completed:continue
            if job['stage']=='scan_mixed' and not ready:continue
            options=[gpu for gpu in [0,1] if load[gpu]<2 and free[gpu]>1600 and (gpu==0 or (ready and extra_done))]
            if not options:break
            gpu=max(options,key=lambda g:free[g]-load[g]*1200)
            command=command_for(job,gpu);log=(STUDY/'logs'/(job['id']+'.log')).open('a')
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            active[job['id']]=(proc,gpu,log);load[gpu]+=1;free[gpu]-=1300
            write(STUDY/'launches'/(job['id']+'.json'),{'pid':proc.pid,'gpu':gpu,'command':command,'source_sha256':plan['source_sha256']})
        # An upstream failure blocks dependent jobs. Stop visibly instead of looping forever.
        if failed and not active:
            write(STUDY/'status.json',{'status':'stopped_after_training_failure','failed':failed,'completed':len(completed),'total':len(plan['jobs'])})
            raise RuntimeError('Training failures need review; completed results retained')
        write(STUDY/'status.json',{'status':'training','completed':len(completed),'total':len(plan['jobs']),
                                  'active':[{'id':key,'pid':value[0].pid,'gpu':value[1]} for key,value in active.items()],
                                  'failed':failed,'real_preparation_complete':ready,'extra_views_complete':extra_done,
                                  'new_real_test_metrics_read':False})
        time.sleep(20)
    if failed:raise RuntimeError('Training failures retained')
    while not (ROOT/'data/close_v6_extra_views/receipt.json').exists():time.sleep(20)
    checkpoints=[str(STUDY/'runs'/job['id']/'best.pt') for job in plan['jobs']]
    tests=[str(ROOT/'data/close_v6/test.jsonl'),str(ROOT/'data/close_v6_extra_views/test.jsonl')]
    lock={'selection':'validation_only','checkpoint_sha256':[digest(p) for p in checkpoints],
          'test_manifest_sha256':[digest(p) for p in tests],'checkpoints':checkpoints,'test_manifests':tests,
          'plan_sha256':digest(path),'locked_before_reading_new_test_metrics':True,'time':time.time()}
    write(STUDY/'locked_evaluation.json',lock)
    write(STUDY/'status.json',{'status':'locked_evaluating','completed':len(completed),'total':len(plan['jobs']),'new_real_test_metrics_read':False})
    for job in plan['jobs']:
        for track,manifest in zip(['main','extra_views'],tests):
            dest=STUDY/'evaluation'/(job['id']+'_'+track)
            if (dest/'report.json').exists():continue
            if dest.exists():dest.rename(dest.with_name(dest.name+f'_incomplete_{int(time.time())}'))
            full=job['stage']=='scan_mixed' and job['variant'] in ['N03_full','A04_raw','N06_normalized_DDL','N07_normalized_Transolver'] and track=='main'
            command=[sys.executable,'-m','garment_realism.evaluate','--checkpoint',str(STUDY/'runs'/job['id']/'best.pt'),
                     '--manifest',manifest,'--out',str(dest),'--lock',str(STUDY/'locked_evaluation.json'),'--device','cuda:1']
            if not full:command+=['--conditions','clean']
            with (STUDY/'logs/evaluation.log').open('a') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            write(STUDY/'status.json',{'status':'locked_evaluating','completed_training':72,'completed_reports':len(list((STUDY/'evaluation').glob('*/report.json'))),
                                      'total_reports':144,'test_used_for_selection':False})
    subprocess.run([sys.executable,'-m','garment_realism.summarize'],check=True)
    write(STUDY/'status.json',{'status':'completed_pending_scientific_review','completed_training':72,'completed_reports':144,
                              'source_sha256':plan['source_sha256'],'test_used_for_selection':False})

if __name__=='__main__':main()
