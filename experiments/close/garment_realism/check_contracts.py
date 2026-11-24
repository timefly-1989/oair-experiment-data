"""Nontrivial equivariance, loss-scaling, partial-target, and gradient checks."""
import json
import torch
from .common import ROOT,STUDY,write,source_hash
from .dataset import Dataset,move
from .models import ScaleRefiner
from .train import metric_scale,loss_for

def main():
    torch.set_num_threads(2);data=Dataset(ROOT/'data/close_v6_engineering/train.jsonl')
    if len(data)<1:raise ValueError('Engineering real-scan preparation did not yield a case')
    sample=move(data[0],'cpu');torch.manual_seed(7701)
    model=ScaleRefiner('evidence_delta','body_scale')
    # Zero decoder initialization would make this test vacuous.
    torch.nn.init.normal_(model.core.decoder[-1].weight,std=.015)
    torch.nn.init.normal_(model.core.decoder[-1].bias,std=.008)
    model.eval();baseline=model(sample)['displacement'];errors=[]
    for factor in [.5,.75,1.25,1.6,2.]:
        scaled=metric_scale(sample,factor);prediction=model(scaled)['displacement']
        relative=float((prediction-baseline*factor).abs().max()/baseline.abs().max().clamp_min(1e-7))
        errors.append({'factor':factor,'relative_max_displacement_error':relative})
        assert relative<2e-4
    torch.manual_seed(443);loss=loss_for(model(sample)['vertices'],sample,384,True)
    scaled=metric_scale(sample,1.6);torch.manual_seed(443);scaled_loss=loss_for(model(scaled)['vertices'],scaled,384,True)
    assert abs(float(scaled_loss/loss)-1)<2e-3
    model.train();loss.backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    # Forward must ignore supervision, even if target points are maliciously changed.
    altered=dict(sample,target_points=sample['target_points']*100+1000)
    assert torch.equal(model(sample)['vertices'],model(altered)['vertices'])
    assert sample['partial_target'] and len(sample['target_boundary'])==0
    result={'status':'passed','source_sha256':source_hash(),'scale_tests':errors,
            'dimensionless_loss_ratio_at_scale_1_6':float(scaled_loss/loss),
            'nonzero_decoder':True,'finite_nonzero_gradients':True,'target_not_read_by_forward':True,
            'partial_scan_no_natural_boundary_supervision':True,'not_benchmark_accuracy':True}
    write(STUDY/'engineering_contracts.json',result);print(json.dumps(result))
if __name__=='__main__':main()
