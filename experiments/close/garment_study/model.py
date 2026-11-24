"""2025/2026-inspired operators with explicit matched task baselines.

Physics slice attention follows Transolver's slice-attend-deslice computation,
with a point-conditioned temperature inspired by Transolver++ (ICML 2025).
The scalar DDL baseline follows the 2026 paper's normalized read/compare/write
with a (0,2) gate. These are task adaptations, not full upstream reproductions.
"""
from dataclasses import dataclass,asdict
import torch
from torch import nn
import torch.nn.functional as F

ARCHITECTURES=('graph','adaptive_slice','transolver_plus','dense_geodesic','ddl','geodesic_slice','geodesic_ddl','evidence_state_target','evidence_delta',
               'residual_geo_image','residual_geo_state','residual_geo_ddl')


@dataclass
class Config:
    architecture:str='evidence_delta'
    width:int=64
    depth:int=4
    interior_slots:int=16
    boundary_slots:int=8
    max_displacement:float=.20
    protect_projection:bool=True
    projection_policy:str='mask_supported'


class AdaptiveSlice(nn.Module):
    def __init__(self,width,slots=24):
        super().__init__()
        self.heads=4; self.dim=width//4; self.slots=slots
        self.key_features=nn.Linear(width,width); self.values=nn.Linear(width,width)
        self.assignment=nn.Linear(self.dim,slots)
        nn.init.orthogonal_(self.assignment.weight)
        self.temperature=nn.Linear(self.dim,1)
        self.q=nn.Linear(self.dim,self.dim,bias=False); self.k=nn.Linear(self.dim,self.dim,bias=False)
        self.v=nn.Linear(self.dim,self.dim,bias=False); self.output=nn.Linear(width,width)

    def forward(self,h,sample):
        n=len(h)
        keys=self.key_features(h).reshape(n,4,self.dim).transpose(0,1)
        values=self.values(h).reshape(n,4,self.dim).transpose(0,1)
        assignment=(self.assignment(keys)/(F.softplus(self.temperature(keys))+.1)).softmax(-1)
        slots=assignment.transpose(1,2)@values/assignment.sum(1)[:,:,None].clamp_min(1e-6)
        attention=(self.q(slots)@self.k(slots).transpose(1,2)/self.dim**.5).softmax(-1)
        lifted=assignment@(attention@self.v(slots))
        return self.output(lifted.transpose(0,1).reshape(n,-1))


class GeodesicBoundarySlice(nn.Module):
    def __init__(self,width,slots=24):
        super().__init__()
        self.assignment=nn.Linear(width,slots); self.temperature=nn.Linear(width,1)
        self.distance_strength=nn.Parameter(torch.tensor(0.))
        self.q=nn.Linear(width,width,bias=False); self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False); self.output=nn.Linear(width,width)
        self.attention_distance_strength=nn.Parameter(torch.tensor(0.))

    def forward(self,h,sample):
        logits=(self.assignment(h)-F.softplus(self.distance_strength)*sample['geodesic_distance'])/(F.softplus(self.temperature(h))+.1)
        assignment=logits.masked_fill(~sample['slice_support'],-1e4).softmax(-1)
        weighted=assignment*sample['vertex_mass'][:,None]
        slots=weighted.T@h/weighted.sum(0)[:,None].clamp_min(1e-6)
        attention=(self.q(slots)@self.k(slots).T/h.shape[-1]**.5-F.softplus(self.attention_distance_strength)*sample['slot_distance']).softmax(-1)
        return self.output(assignment@(attention@self.v(slots)))


class DenseGeodesic(nn.Module):
    """Generic dense geodesic-bias task control, not a reproduction of ICME GSA."""
    def __init__(self,width):
        super().__init__();self.dim=width//4
        self.q=nn.Linear(width,width);self.k=nn.Linear(width,width);self.v=nn.Linear(width,width)
        self.strength=nn.Parameter(torch.zeros(4,1,1));self.output=nn.Linear(width,width)

    def forward(self,h,sample):
        n=len(h)
        q=self.q(h).reshape(n,4,self.dim).transpose(0,1)
        k=self.k(h).reshape(n,4,self.dim).transpose(0,1)
        v=self.v(h).reshape(n,4,self.dim).transpose(0,1)
        bias=-F.softplus(self.strength)*sample['full_geodesic_distance'][None]
        attention=(q@k.transpose(1,2)/self.dim**.5+bias).softmax(-1)
        return self.output((attention@v).transpose(0,1).reshape(n,-1))


class EideticSlice(nn.Module):
    """Single-GPU Transolver++ attention-core transfer, including Gumbel sampling.

    Upstream: thuml/Transolver_plus, models/Transolver_plus.py. Distributed
    reductions are identity on one GPU. The surrounding garment encoder/graph
    is shared across task controls, not the original PDE architecture.
    """
    def __init__(self,width,slots=24):
        super().__init__();self.dim=width//4
        self.norm=nn.LayerNorm(width);self.project=nn.Linear(width,width)
        self.assignment=nn.Linear(self.dim,slots);nn.init.orthogonal_(self.assignment.weight)
        self.temperature=nn.Sequential(nn.Linear(self.dim,slots),nn.GELU(),nn.Linear(slots,1),nn.GELU())
        self.temperature_bias=nn.Parameter(torch.full((4,1,1),.5))
        self.q=nn.Linear(self.dim,self.dim,bias=False);self.k=nn.Linear(self.dim,self.dim,bias=False)
        self.v=nn.Linear(self.dim,self.dim,bias=False);self.output=nn.Linear(width,width)

    def forward(self,h,sample):
        n=len(h);features=self.project(self.norm(h)).reshape(n,4,self.dim).transpose(0,1)
        temperature=(self.temperature(features)+self.temperature_bias).clamp_min(.01)
        logits=self.assignment(features);u=torch.rand_like(logits)
        gumbel=-torch.log(-torch.log(u+1e-8)+1e-8)
        assignment=((logits+gumbel)/temperature).softmax(-1)
        slots=assignment.transpose(1,2)@features/(assignment.sum(1)[:,:,None]+1e-5)
        mixed=F.scaled_dot_product_attention(self.q(slots)[None],self.k(slots)[None],self.v(slots)[None])[0]
        return self.output((assignment@mixed).transpose(0,1).reshape(n,-1))


class Delta(nn.Module):
    def __init__(self,width,anchored=False,state_target=False):
        super().__init__(); self.anchored=anchored; self.state_target=state_target
        self.value=nn.Linear(width,1)
        self.gate=nn.Linear(width*2+2 if anchored else width,1)
        self.norm=nn.LayerNorm(width)

    def forward(self,h,evidence,image_anchor,boundary,visible):
        direction=F.normalize(evidence,dim=-1,eps=1e-6)
        context=self.norm(h)
        if self.anchored:
            value=self.value(context if self.state_target else image_anchor)
            gate=torch.sigmoid(self.gate(torch.cat([context,(context-image_anchor).abs(),boundary[:,None],visible[:,None]],-1)))
        else:
            value=self.value(context); gate=2*torch.sigmoid(self.gate(context))
        return h+gate*direction*(value-(direction*h).sum(-1,keepdim=True))


def protect_boundary_projection(displacement,vertices,boundary,visible):
    ray=F.normalize(vertices,dim=-1,eps=1e-6)
    along_ray=(displacement*ray).sum(-1,keepdim=True)*ray
    strength=(boundary*visible).clamp(0,1)[:,None]
    return displacement*(1-strength)+along_ray*strength


class Refiner(nn.Module):
    def __init__(self,config=None):
        super().__init__()
        if isinstance(config,dict) and 'projection_policy' not in config:config=dict(config,projection_policy='all_front')
        self.config=config if isinstance(config,Config) else Config(**(config or {}))
        c=self.config; w=c.width
        if c.architecture not in ARCHITECTURES or w<8 or w%4 or c.depth<1: raise ValueError('Invalid garment study architecture.')
        self.encoder=nn.Sequential(nn.Conv2d(4,32,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(32,w,3,stride=2,padding=1),nn.GELU(),nn.Conv2d(w,w,3,padding=1),nn.GELU())
        # This image target path has no body features or 3D initial coordinates.
        self.image_input=nn.Sequential(nn.Linear(w+4,w),nn.LayerNorm(w),nn.GELU())
        self.geometry_input=nn.Linear(11,w)
        from garment_lab.model import LocalGraph
        self.graphs=nn.ModuleList([LocalGraph(w) for _ in range(c.depth)])
        if c.architecture in ('adaptive_slice','ddl'):
            self.slices=nn.ModuleList([AdaptiveSlice(w,c.interior_slots+c.boundary_slots) for _ in range(c.depth)])
        elif c.architecture=='transolver_plus':
            self.slices=nn.ModuleList([EideticSlice(w,c.interior_slots+c.boundary_slots) for _ in range(c.depth)])
        elif c.architecture=='dense_geodesic':
            self.slices=nn.ModuleList([DenseGeodesic(w) for _ in range(c.depth)])
        elif c.architecture in ('geodesic_slice','geodesic_ddl','evidence_state_target','evidence_delta','residual_geo_image','residual_geo_state','residual_geo_ddl'):
            self.slices=nn.ModuleList([GeodesicBoundarySlice(w,c.interior_slots+c.boundary_slots) for _ in range(c.depth)])
        else: self.slices=None
        anchored=c.architecture.startswith('evidence_') or c.architecture in ('residual_geo_image','residual_geo_state')
        state_target=c.architecture in ('evidence_state_target','residual_geo_state')
        self.deltas=nn.ModuleList([Delta(w,anchored,state_target) for _ in range(c.depth)]) if c.architecture in ('ddl','geodesic_ddl','evidence_state_target','evidence_delta','residual_geo_image','residual_geo_state','residual_geo_ddl') else None
        self.decoder=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w),nn.GELU(),nn.Linear(w,3))
        nn.init.zeros_(self.decoder[-1].weight); nn.init.zeros_(self.decoder[-1].bias)

    def image_anchor(self,sample):
        features=self.encoder(torch.cat([sample['image'],sample['mask'][None]],0)[None])
        grid=sample['uv']*2-1
        local=F.grid_sample(features,grid[None,None],align_corners=False,padding_mode='zeros')[0,:,0].T
        valid=(grid.abs()<=1).all(-1).float()
        return self.image_input(torch.cat([local,sample['uv'],sample['boundary'][:,None],valid[:,None]],-1))

    def forward(self,sample):
        vertices=sample['initial_vertices']; anchor=self.image_anchor(sample)
        center=vertices.mean(0,keepdim=True); scale=(vertices-center).norm(dim=-1).max().clamp_min(1e-5)
        camera_context=torch.cat([center.expand(len(vertices),-1),scale.expand(len(vertices),1)],-1)
        h=anchor+self.geometry_input(torch.cat([(vertices-center)/scale,sample['body_features'],camera_context],-1))
        for index,graph in enumerate(self.graphs):
            evidence=graph(h,sample['edges'])
            if self.slices is not None: evidence=evidence+self.slices[index](h,sample)
            base=h+evidence if self.config.architecture.startswith('residual_geo_') else h
            h=self.deltas[index](base,evidence,anchor,sample['boundary'],sample['visible_proxy']) if self.deltas is not None else h+evidence
        displacement=self.decoder(h).tanh()*self.config.max_displacement
        if self.config.protect_projection:
            visible=sample['visible_proxy']
            if self.config.projection_policy=='mask_supported':visible=visible*sample['projection_support']
            elif self.config.projection_policy!='all_front':raise ValueError('Unknown projection evidence policy.')
            displacement=protect_boundary_projection(displacement,vertices,sample['boundary'],visible)
        return {'vertices':vertices+displacement,'displacement':displacement}

    def model_config(self): return asdict(self.config)
