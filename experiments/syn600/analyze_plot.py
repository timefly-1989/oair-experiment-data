"""Analyze measured extension results; generate no placeholder accuracy curves."""
from pathlib import Path
import csv,json,math
from collections import defaultdict,Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results';OUT=ROOT/'analysis';FIG=ROOT/'figures'
COLORS={'OAIR':'#326FA8','DDL':'#D18B36','TransolverPP':'#528C74'}
LABELS={'OAIR':'OAIR','DDL':'DDL','TransolverPP':'Transolver++'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(fig,name):
 fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight');fig.savefig(FIG/(name+'.png'),dpi=220,bbox_inches='tight');plt.close(fig)
def writecsv(path,rows):
 if not rows:return
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def learning():
 fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained');summary=[]
 for ax,recipe in zip(axes,['no_scale','scale_aug']):
  for model in COLORS:
   entries=[]
   for directory in sorted((DATA/'runs').glob(model+'_'+recipe+'_s*')):
    p=directory/'history.jsonl'
    if not p.exists():continue
    records=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    entries.append({r['epoch']:r['validation_score']*1000 for r in records if r['validation_score'] is not None})
   epochs=sorted(set(e for d in entries for e in d))
   if not epochs:continue
   means=[];sd=[]
   for epoch in epochs:
    v=[d[epoch] for d in entries if epoch in d];means.append(np.mean(v));sd.append(np.std(v,ddof=1) if len(v)>1 else 0)
    summary.append({'model':model,'recipe':recipe,'epoch':epoch,'available_seeds':len(v),'mean_mm':float(np.mean(v))})
   ax.plot(epochs,means,'o-',label=LABELS[model],color=COLORS[model],ms=4)
   ax.fill_between(epochs,np.array(means)-sd,np.array(means)+sd,color=COLORS[model],alpha=.13)
  ax.set_title('No metric-scale augmentation' if recipe=='no_scale' else 'Metric-scale augmentation')
  ax.set_xlabel('Training epoch');ax.set_ylabel('Validation objective (mm)');ax.spines[['top','right']].set_visible(False)
  if ax.get_legend_handles_labels()[0]:ax.legend(frameon=False,fontsize=8)
 save(fig,'01_validation_curves');writecsv(OUT/'validation_curves.csv',summary)

def real_audit():
 p=DATA/'real_surface_audit.json'
 if not p.exists():return
 data=read(p);rows=data['rows'];classes=sorted({r['category'] for r in rows})
 fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
 total=[sum(r['category']==c for r in rows) for c in classes]
 valid=[sum(r['category']==c and r.get('welded_topology_compatible',False) for r in rows) for c in classes]
 x=np.arange(len(classes));axes[0].bar(x-.17,total,width=.34,color='#8698A9',label='Observed semantic patches');axes[0].bar(x+.17,valid,width=.34,color='#5E947F',label='Topology-compatible after welding')
 axes[0].set_xticks(x,classes);axes[0].set_ylabel('Distinct garment/frame patches');axes[0].legend(frameon=False,fontsize=7)
 for i,n in enumerate(total):axes[0].text(i-.17,n+1,str(n),ha='center',fontsize=8)
 boundaries=[r.get('welded_1micrometer',r)['boundary_components'] for r in rows]
 cap=40
 axes[1].hist(np.minimum(boundaries,cap),bins=np.arange(-.5,cap+1.5),color='#687F98',edgecolor='white')
 axes[1].set_xticks([0,10,20,30,40],['0','10','20','30','40+'])
 axes[1].set_xlabel('Boundary components after 1 micrometer weld');axes[1].set_ylabel('Patch count')
 for ax in axes:ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Real scan audit: topology alone does not verify physical garment completeness',fontsize=10)
 save(fig,'02_real_surface_audit')
 simple=[{k:r[k] for k in ['sample_id','subject_id','category','components','boundary_components','boundary_noncycle_vertices','nonmanifold_edges','topology_compatible','welded_topology_compatible'] if k in r} for r in rows]
 writecsv(OUT/'real_surface_topology.csv',simple)

def bootstrap_pair(a,b,draws=6000,seed=1110):
 assert a.shape==b.shape and a.ndim==2
 rng=np.random.default_rng(seed);delta=(a-b)*1000;s,n=delta.shape;values=np.empty(draws)
 for i in range(draws):values[i]=delta[rng.integers(s,size=s)[:,None],rng.integers(n,size=n)[None,:]].mean()
 return {'difference_mm':float(delta.mean()),'nominal_95_mm':np.quantile(values,[.025,.975]).tolist(),
         'familywise_97_5_mm':np.quantile(values,[.0125,.9875]).tolist(),'seed_differences_mm':delta.mean(1).tolist()}

def metrics():
 paths=sorted((DATA/'evaluation').glob('*.json'))
 if len(paths)!=31:return {'evaluation_complete':False,'reports':len(paths)}
 lock=read(DATA/'test_lock.json');choices=lock['selected_recipes'];records=[]
 for path in paths:records.extend(read(path)['rows'])
 simple=[{k:r[k] for k in ['model','recipe','seed','sample_id','subject_id','category','condition','surface_cd','boundary_cd','composite','harmed_over_1mm','fscore_10mm']} for r in records]
 writecsv(OUT/'per_case_metrics.csv',simple)
 grouped=defaultdict(list)
 for row in records:grouped[(row['model'],row['recipe'],row['condition'],row['seed'],row['subject_id'])].append(row)
 cells=defaultdict(list)
 for (m,recipe,c,s,subject),rr in grouped.items():
  cells[(m,recipe,c,s)].append({metric:float(np.mean([r[metric] for r in rr])) for metric in ['surface_cd','boundary_cd','composite','harmed_over_1mm','fscore_10mm']})
 perseed=[]
 for (m,recipe,c,s),rr in cells.items():
  perseed.append({'model':m,'recipe':recipe,'condition':c,'seed':s,**{metric:float(np.mean([r[metric] for r in rr])) for metric in rr[0]}})
 writecsv(OUT/'per_seed_metrics.csv',perseed)
 agg=[]
 for key in sorted({(r['model'],r['recipe'],r['condition']) for r in perseed}):
  rr=[r for r in perseed if (r['model'],r['recipe'],r['condition'])==key]
  agg.append({'model':key[0],'recipe':key[1],'condition':key[2],'seeds':len(rr),
     **{metric+'_mean':float(np.mean([r[metric] for r in rr])) for metric in ['surface_cd','boundary_cd','composite','harmed_over_1mm','fscore_10mm']},
     'composite_seed_sd':float(np.std([r['composite'] for r in rr],ddof=1)) if len(rr)>1 else 0})
 writecsv(OUT/'aggregate_metrics.csv',agg)
 bodies=sorted({r['subject_id'] for r in records});seeds=[2110,2111,2112,2113,2114]
 def matrix(model):
  result=np.zeros((5,len(bodies)))
  for i,s in enumerate(seeds):
   for j,subject in enumerate(bodies):
    rr=grouped[(model,choices[model],'clean',s,subject)]
    result[i,j]=np.mean([r['composite'] for r in rr])
  return result
 contrasts=[]
 for name in ['DDL','TransolverPP']:
  contrast=bootstrap_pair(matrix(name),matrix('OAIR'));contrasts.append({'control':name,**contrast})
 stable=all(r['familywise_97_5_mm'][0]>0 and all(v>0 for v in r['seed_differences_mm']) for r in contrasts)
 summary={'evaluation_complete':True,'reports':31,'samples':len({r['sample_id'] for r in records}),'body_groups':len(bodies),
          'seeds':5,'selected_recipes':choices,'aggregate':agg,'contrasts':contrasts,'stable_advantage_established':stable,
          'criterion':'both familywise intervals strictly positive and positive mean difference in each of five seeds',
          'real_complete_validation_established':False,'external_full_system_comparison_completed':False,
          'units':'all stored distances meters; plot distances millimeters'}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

 fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
 for ax,recipe in zip(axes,['no_scale','scale_aug']):
  for x,m in enumerate(COLORS):
   rr=[r for r in perseed if r['model']==m and r['recipe']==recipe and r['condition']=='clean'];values=np.array([r['composite']*1000 for r in rr])
   ax.bar(x,values.mean(),yerr=values.std(ddof=1),color=COLORS[m],alpha=.7,capsize=4)
   ax.scatter(x+np.linspace(-.12,.12,len(values)),values,c='black',s=13,zorder=4)
  ax.set_xticks(range(3),list(LABELS.values()));ax.set_title(recipe.replace('_',' '));ax.set_ylabel('Complete-garment composite (mm)');ax.spines[['top','right']].set_visible(False)
 save(fig,'03_matched_recipe_five_seeds')

 fig,ax=plt.subplots(figsize=(6.6,2.5),layout='constrained')
 for i,row in enumerate(contrasts):
  mean=row['difference_mm'];lo,hi=row['familywise_97_5_mm']
  ax.plot([lo,hi],[i,i],color=COLORS[row['control']],lw=3);ax.plot(mean,i,'o',color=COLORS[row['control']]);ax.text(hi+.25,i,f'{mean:.2f} [{lo:.2f}, {hi:.2f}]',va='center',fontsize=8)
 ax.axvline(0,color='gray',ls='--',lw=1);ax.set_yticks(range(2),[LABELS[r['control']]+' minus OAIR' for r in contrasts]);ax.set_xlabel('Paired composite difference (mm); positive favors OAIR');ax.set_ylim(-.6,1.6);ax.spines[['top','right']].set_visible(False)
 save(fig,'04_primary_contrasts')

 fig,axes=plt.subplots(1,2,figsize=(8,3),layout='constrained')
 for m in COLORS:
  rr=[r for r in agg if r['model']==m and r['recipe']==choices[m]];order=['clean','body_noise012','blur2'];rr=sorted(rr,key=lambda r:order.index(r['condition']))
  axes[0].errorbar(range(3),[r['composite_mean']*1000 for r in rr],yerr=[r['composite_seed_sd']*1000 for r in rr],fmt='o-',capsize=3,color=COLORS[m],label=LABELS[m])
  vv=[]
  for body in bodies:
   rr=[r for r in records if r['model']==m and r['recipe']==choices[m] and r['condition']=='clean' and r['subject_id']==body]
   vv.append(np.mean([(r['surface_cd']-r['initial_surface_cd'])*1000 for r in rr]))
  vv=np.sort(vv);axes[1].plot(vv,(np.arange(len(vv))+1)/len(vv),label=LABELS[m],color=COLORS[m])
 axes[0].set_xticks(range(3),['Clean','Body noise','Blur']);axes[0].set_ylabel('Composite error (mm)');axes[0].legend(frameon=False,fontsize=8)
 axes[1].axvline(0,c='gray',ls='--',lw=1);axes[1].set_xlabel('Body-mean CD change from initializer (mm)');axes[1].set_ylabel('Empirical cumulative fraction');axes[1].legend(frameon=False,fontsize=8)
 for ax in axes:ax.spines[['top','right']].set_visible(False)
 save(fig,'05_robustness_and_harm')
 return summary

def main():
 OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.titlesize':10})
 learning();real_audit();result=metrics();print(json.dumps({k:v for k,v in result.items() if k not in ['aggregate']},indent=2))
if __name__=='__main__':main()
