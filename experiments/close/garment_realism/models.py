"""Body-scale normalization and camera-space projection with matched old operators.

Conditional positive metric-scale equivariance holds for fixed image/mask/UV,
and jointly scaled initial cloth and predicted body. This is not image-resize
equivariance, camera invariance, or a newly invented group-equivariance principle.
"""
import torch
from torch import nn
from garment_study.model import Refiner,Config,protect_boundary_projection

class ScaleRefiner(nn.Module):
    def __init__(self,architecture='evidence_delta',coordinate_mode='raw',protect=True,width=64):
        super().__init__()
        if coordinate_mode not in ('raw','body_scale'):raise ValueError(coordinate_mode)
        self.coordinate_mode=coordinate_mode;self.protect=protect
        self.core=Refiner(Config(architecture=architecture,width=width,depth=4,protect_projection=False))

    def model_config(self):
        return {'architecture':self.core.config.architecture,'coordinate_mode':self.coordinate_mode,
                'protect':self.protect,'width':self.core.config.width,'depth':4,'reference_extent_m':1.7}

    @staticmethod
    def extent(sample):
        body=sample['body_vertices']
        return (body.max(0).values-body.min(0).values).max().clamp_min(1e-4)

    def forward(self,sample):
        vertices=sample['initial_vertices']
        if self.coordinate_mode=='body_scale':
            body=sample['body_vertices'];center=body.mean(0,keepdim=True)
            factor=self.extent(sample)/1.7
            normalized=dict(sample,initial_vertices=(vertices-center)/factor,body_vertices=(body-center)/factor)
            # Cached nearest-body features and median-normalized geodesic inputs
            # are already dimensionless; their inputs do not need rescaling.
            displacement=self.core(normalized)['displacement']*factor
        else:displacement=self.core(sample)['displacement']
        if self.protect:
            supported=sample['visible_proxy']*sample['projection_support']
            displacement=protect_boundary_projection(displacement,vertices,sample['boundary'],supported)
        return {'vertices':vertices+displacement,'displacement':displacement}
