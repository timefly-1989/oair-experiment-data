from common import *
import sys
sys.path.insert(0,str(SOURCE))
import numpy as np,torch
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from garment_realism.dataset import Dataset,move,sample_surface
from garment_realism.evaluate import load_model,changed
from garment_lab.mesh import boundary_loops
from garment_expansion.common import mesh_quality

def boundary_points(vertices,faces,n=1500):
 loops=boundary_loops(faces);starts=[];ends=[]
 for loop in loops:
  v=vertices[np.asarray(loop)];starts.append(v);ends.append(np.roll(v,-1,axis=0))
 a=np.concatenate(starts);b=np.concatenate(ends);length=np.linalg.norm(b-a,axis=1);cum=np.r_[0,np.cumsum(length)]
 if cum[-1]<=0:raise ValueError('No finite positive boundary length')
 position=(np.arange(n)+.5)*(cum[-1]/n)
 ids=np.minimum(np.searchsorted(cum,position,side='right')-1,len(a)-1)
 alpha=(position-cum[ids])/np.maximum(length[ids],1e-12)
 return a[ids]+alpha[:,None]*(b[ids]-a[ids])
def cd(a,b):return float((cKDTree(a).query(b)[0].mean()+cKDTree(b).query(a)[0].mean())/2)

@torch.inference_mode()
def main():
 os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.use_deterministic_algorithms(True)
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.028,1)
 p=read(ROOT/'protocol.json');manifest=ROOT/'holdout/test.jsonl'
 # The extra official batch is an availability amendment before any test score.
 # Preserve the original 26-case pool and record the exact evaluated manifest.
 for attempt in range(120):
  state=read(ROOT/'holdout_expansion_status.json') if (ROOT/'holdout_expansion_status.json').exists() else {}
  if state.get('phase') not in ['acquiring','extracting','preparing']:break
  time.sleep(15)
 if (ROOT/'holdout_v2/receipt.json').exists():manifest=ROOT/'holdout_v2/test.jsonl'
 if not (ROOT/'holdout/receipt.json').exists():
  write(ROOT/'status.json',{'phase':'training_complete_evaluation_missing_fresh_holdout','updated_at':now()});return
 choices={};validation={}
 for model in ['OAIR','DDL','TransolverPP']:
  scores={}
  for recipe in ['no_scale','scale_aug']:
   scores[recipe]=float(np.mean([read(ROOT/'runs'/j['id']/'result.json')['best_score'] for j in p['jobs'] if j['model']==model and j['recipe']==recipe]))
  choices[model]=min(scores,key=scores.get);validation[model]=scores
 lock={'locked_at':now(),'protocol_sha256':digest(ROOT/'protocol.json'),'manifest_sha256':digest(manifest),'manifest':str(manifest),'evaluator_sha256':digest(__file__),
       'checkpoint_sha256':{j['id']:digest(ROOT/'runs'/j['id']/'best.pt') for j in p['jobs']},'selected_recipes':choices,
       'validation_recipe_scores':validation,'test_used_for_selection':False}
 dest=ROOT/'test_lock.json'
 if dest.exists():lock['locked_at']=read(dest)['locked_at'];assert read(dest)==lock
 else:write(dest,lock)
 data=Dataset(manifest);targets=[]
 for row in data.records:
  with np.load(row['cache'],allow_pickle=False) as z:
   targets.append({'points':sample_surface(z['target_vertices'],z['target_faces'],8192,1234),
                   'boundary':boundary_points(z['target_vertices'],z['target_faces']),
                   'initial':z['initial_vertices'].copy(),'faces':z['faces'].copy()})
 contracts=read(ROOT/'runtime_contracts.json')['checks'] if (ROOT/'runtime_contracts.json').exists() else []
 jobs=[{'id':'Initializer','model':'Initializer','recipe':'none','seed':0}]+p['jobs']
 for job in jobs:
  out=ROOT/'evaluation'/(job['id']+'.json')
  if out.exists():continue
  if job['id']!='Initializer':assert digest(ROOT/'runs'/job['id']/'best.pt')==lock['checkpoint_sha256'][job['id']]
  model=load_model(ROOT/'runs'/job['id']/'best.pt','cuda:1') if job['id']!='Initializer' else None
  result=[];started=time.time()
  for condition in ['clean','body_noise012','blur2']:
   for i,row in enumerate(data.records):
    s=changed(move(data[i],'cuda:1'),row,condition)
    seed=int(hashlib.sha256(row['sample_id'].encode()).hexdigest()[:8],16)
    with torch.random.fork_rng(devices=[1]):
     torch.manual_seed(seed);v=model(s)['vertices'] if model else s['initial_vertices']
    if i==0 and condition=='clean' and model is not None:
     isolated={k:value for k,value in s.items() if k not in ['target_points','target_boundary','partial_target']}
     with torch.random.fork_rng(devices=[1]):torch.manual_seed(seed);check=model(isolated)['vertices']
     error=float((v-check).abs().max());assert error==0
     contracts.append({'job':job['id'],'target_removed_max_abs_difference':error})
    v=v.cpu().numpy();faces=targets[i]['faces'];points=sample_surface(v,faces,4096,61234)
    surface=cd(points,targets[i]['points']);boundary=cd(boundary_points(v,faces),targets[i]['boundary'])
    condition_initial=s['initial_vertices'].cpu().numpy()
    initial_points=sample_surface(condition_initial,faces,4096,61234)
    initial_surface=cd(initial_points,targets[i]['points'])
    distance_a=cKDTree(points).query(targets[i]['points'])[0];distance_b=cKDTree(targets[i]['points']).query(points)[0]
    recall=float((distance_a<.01).mean());precision=float((distance_b<.01).mean())
    result.append({'job':job['id'],'model':job['model'],'recipe':job['recipe'],'seed':job['seed'],
        'sample_id':row['sample_id'],'subject_id':row['subject_id'],'category':row['category'],'condition':condition,
        'surface_cd':surface,'boundary_cd':boundary,'composite':surface+.35*boundary,
        'initial_surface_cd':initial_surface,'harmed_over_1mm':surface>initial_surface+.001,
        'fscore_10mm':2*precision*recall/max(precision+recall,1e-12),**mesh_quality(v,condition_initial,faces)})
    if condition=='clean' and i<8:
     folder=ROOT/'qualitative'/job['id'];folder.mkdir(parents=True,exist_ok=True)
     np.savez_compressed(folder/(row['sample_id']+'.npz'),vertices=v,faces=faces)
   write(ROOT/'evaluation_progress.json',{'job':job['id'],'condition':condition,'rows':len(result),'completed_models':len(list((ROOT/'evaluation').glob('*.json'))),'planned_models':31,'updated_at':now()})
  write(out,{'job':job,'rows':result,'seconds':time.time()-started,'test_lock_sha256':digest(dest),'target_kind':'complete_synthetic_garment','alignment':'none','conditions_scope':'refiner cached-input interventions'})
  write(ROOT/'runtime_contracts.json',{'checks':contracts,'all_passed':len(contracts)==30})
  del model
 assert len(contracts)==30
 write(ROOT/'runtime_contracts.json',{'checks':contracts,'all_passed':True})
 write(ROOT/'status.json',{'phase':'completed','training_completed':30,'evaluation_completed':31,'updated_at':now()})
if __name__=='__main__':main()
