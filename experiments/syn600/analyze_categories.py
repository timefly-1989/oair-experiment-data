"""Descriptive category breakdown; no category is used for selection or primary claims."""
from analyze_plot import *

def main():
 lock=read(DATA/'test_lock.json');choices=lock['selected_recipes'];rows=[]
 for p in (DATA/'evaluation').glob('*.json'):
  for r in read(p)['rows']:
   if r['condition']=='clean' and r['recipe']==choices.get(r['model'],'none'):rows.append(r)
 result=[]
 for category in ['top','dress','skirt','pants']:
  for model in ['Initializer','OAIR','DDL','TransolverPP']:
   rr=[r for r in rows if r['category']==category and r['model']==model];seeds=sorted({r['seed'] for r in rr});bodies=sorted({r['subject_id'] for r in rr})
   r={'category':category,'model':model,'samples':len({r['sample_id'] for r in rr}),'bodies':len(bodies),'seeds':len(seeds)}
   for metric in ['surface_cd','boundary_cd','composite']:
    values=[np.mean([np.mean([x[metric] for x in rr if x['seed']==s and x['subject_id']==body]) for body in bodies]) for s in seeds]
    r[metric+'_mean_mm']=float(np.mean(values))*1000
    r[metric+'_seed_sd_mm']=float(np.std(values,ddof=1))*1000 if len(values)>1 else 0
   result.append(r)
 writecsv(OUT/'descriptive_category_metrics.csv',result)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
