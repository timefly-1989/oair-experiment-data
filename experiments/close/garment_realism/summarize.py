"""Report all completed locked results, including unfavorable outcomes."""
import csv,json
from collections import defaultdict
import numpy as np
from .common import STUDY,write,digest

def main():
    plan=json.loads((STUDY/'experiment_plan.json').read_text());summaries=[];reports={}
    for job in plan['jobs']:
        for track in ['main','extra_views']:
            report=json.loads((STUDY/'evaluation'/(job['id']+'_'+track)/'report.json').read_text())
            reports[(job['variant'],job['seed'],job['stage'],track)]=report
            for condition,summary in report['summary'].items():
                summaries.append({'variant':job['variant'],'seed':job['seed'],'stage':job['stage'],'track':track,
                                  'condition':condition,**summary,'failed_cases':sum(r['condition']==condition for r in report['failures'])})
    grouped=defaultdict(list)
    for row in summaries:grouped[(row['variant'],row['stage'],row['track'],row['condition'])].append(row)
    aggregate=[]
    for (variant,stage,track,condition),rows in grouped.items():
        values=[r['target_to_prediction'] for r in rows if r['target_to_prediction'] is not None]
        aggregate.append({'variant':variant,'stage':stage,'track':track,'condition':condition,'seeds':len(values),
                          'coverage_distance_source_units_mean':float(np.mean(values)),'coverage_distance_source_units_seed_std':float(np.std(values,ddof=1)) if len(values)>1 else None,
                          'cases_per_seed':[r['cases'] for r in rows],'failed_cases_total':sum(r['failed_cases'] for r in rows)})
    contrasts=[];rng=np.random.default_rng(64011)
    for proposed,baseline in plan['primary_contrasts']:
        differences={}
        for seed in plan['seeds']:
            a={r['sample_id']:r for r in reports[(proposed,seed,'scan_mixed','main')]['rows'] if r['condition']=='clean'}
            b={r['sample_id']:r for r in reports[(baseline,seed,'scan_mixed','main')]['rows'] if r['condition']=='clean'}
            for case in sorted(a.keys()&b.keys()):differences[(seed,a[case]['group_id'],case)]=b[case]['target_to_prediction']-a[case]['target_to_prediction']
        groups=sorted({key[1] for key in differences});seeds=plan['seeds'];estimates=[]
        for _ in range(5000):
            seed_sample=rng.choice(seeds,len(seeds));group_sample=rng.choice(groups,len(groups));values=[]
            for seed in seed_sample:
                for group in group_sample:values.extend(v for (s,g,c),v in differences.items() if s==seed and g==group)
            estimates.append(np.mean(values))
        contrasts.append({'proposed':proposed,'baseline':baseline,'sign':'baseline_minus_proposed_positive_favors_proposed',
                          'mean_difference_source_units':float(np.mean(list(differences.values()))),
                          'nominal_95_interval':np.quantile(estimates,[.025,.975]).tolist(),
                          'three_contrast_bonferroni_interval':np.quantile(estimates,[.05/6,1-.05/6]).tolist(),
                          'prefix_groups':len(groups),'paired_seed_cases':len(differences),'bootstrap_replicates':5000,
                          'physical_person_identity_not_independently_verified':True})
    output=STUDY/'analysis';output.mkdir(exist_ok=True)
    for filename,rows in [('per_seed_results.csv',summaries),('aggregate_results.csv',aggregate)]:
        with (output/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'status':'completed_pending_scientific_review','training_runs':len(plan['jobs']),'evaluation_reports':len(reports),
            'aggregate':aggregate,'contrasts':contrasts,'test_used_for_selection':False,
            'input_domain':'real_scan_rerender_not_captured_photographs','target_kind':'partial_semantic_outer_patches_not_complete_garments',
            'plan_sha256':digest(STUDY/'experiment_plan.json'),'evaluation_lock_sha256':digest(STUDY/'locked_evaluation.json')}
    write(output/'summary.json',result)
    text='Real-scan transfer and scale consistency: v6 results\n'+'='*64+'\n\n'
    text+='All distances refer to author-scale recovered scan coordinates. Inputs are scan rerenderings. Targets are observed semantic outer patches; these are not complete-garment or captured-camera accuracy results.\n\n'
    text+='All candidates and failures are retained. Checkpoints were selected only with validation before locking test hashes.\n\n'
    for item in contrasts:
        text+=f"{item['proposed']} vs {item['baseline']}: baseline-minus-proposed={item['mean_difference_source_units']:.6f}, nominal interval={item['nominal_95_interval']}, familywise interval={item['three_contrast_bonferroni_interval']}.\n\n"
    text+='Three seeds and filename-prefix clusters limit inference. Group names are a proxy and do not independently establish physical person identity. Positive intervals do not establish global SOTA or journal acceptance.\n'
    (output/'results_summary.rst').write_text(text)
    print(json.dumps({'status':result['status'],'reports':len(reports)}))
if __name__=='__main__':main()
