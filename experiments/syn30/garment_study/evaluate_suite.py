"""Evaluate all locked configurations; validation selection is immutable."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from garment_lab.reconstruct import sha256
from .train import source_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True);p.add_argument('--data',required=True)
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:0','cuda:1','cuda:1'])
    p.add_argument('--samples',type=int,default=10000);p.add_argument('--noise',nargs='+',type=float,default=[0.,.04,.08,.12])
    args=p.parse_args();suite=Path(args.suite).resolve();data=Path(args.data).resolve()
    snapshot=json.loads((suite/'protocol_snapshot.json').read_text());selection_path=suite/'final_selection.json'
    selection=json.loads(selection_path.read_text());selection_digest=sha256(selection_path)
    expected=snapshot['final_evaluation']
    if args.samples!=expected['samples_per_surface'] or args.noise!=expected['noise_magnitudes_m']:
        raise ValueError('Final evaluation must follow the protocol locked before training.')
    if source_hash()!=selection['source_sha256'] or sha256(data/'test.jsonl')!=selection['test_manifest_sha256']:
        raise ValueError('Code or locked test manifest changed after validation selection.')
    out=suite/'locked_test';out.mkdir(exist_ok=False)
    jobs=[]
    for job in snapshot['jobs']:
        for noise in args.noise:
            label=f'{job["id"]}_noise{noise:.2f}'
            jobs.append({'label':label,'run':job['id'],'noise':noise,'variant':job['variant'],'seed':job['seed']})
    protocol={'samples_per_surface':args.samples,'noise_magnitudes_m':args.noise,'selection_sha256':selection_digest,
              'test_manifest_sha256':sha256(data/'test.jsonl'),'source_sha256':source_hash(),'jobs':jobs,
              'selection_fixed_before_test':True,'test_used_for_hyperparameter_changes':False}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    active={};pending=jobs.copy();completed=[];failed=[]
    while pending or active:
        if sha256(selection_path)!=selection_digest:raise RuntimeError('Validation selection was modified during test evaluation.')
        for slot,device in enumerate(args.devices):
            if slot in active or not pending:continue
            job=pending.pop(0);log=(out/(job['label']+'.log')).open('w')
            command=[sys.executable,'-m','garment_study.train','evaluate','--manifest',str(data/'test.jsonl'),
                     '--checkpoint',str(suite/job['run']/'best.pt'),'--out',str(out/(job['label']+'.json')),
                     '--device',device,'--samples',str(args.samples),'--noise',str(job['noise'])]
            if job['noise']==0 and job['run']==selection['selected_run']:
                command+=['--save-meshes',str(out/'selected_test_meshes')]
            env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            active[slot]={'process':process,'stream':log,'job':job,'began':time.time()}
        for slot,state in list(active.items()):
            code=state['process'].poll()
            if code is None:continue
            state['stream'].close();row={**state['job'],'exit_code':code,'seconds':time.time()-state['began']}
            (completed if code==0 else failed).append(row);del active[slot]
            print(json.dumps({'completed':len(completed),'failed':len(failed),'job':row['label']}),flush=True)
        (out/'progress.json').write_text(json.dumps({'completed':len(completed),'failed':failed,'remaining':len(pending),'active':len(active)}))
        if pending or active:time.sleep(5)
    report={'status':'completed' if not failed else 'failed','evaluations':completed,'failed':failed,
            'selection_sha256':selection_digest,'test_never_used_for_selection':True}
    (out/'receipt.json').write_text(json.dumps(report,indent=2))
    if failed:raise RuntimeError('Failed locked test evaluations must be retained and investigated.')


if __name__=='__main__':main()
