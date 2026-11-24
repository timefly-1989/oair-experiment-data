"""Do not relabel observed scan patches as complete standalone garment truth."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.spatial import cKDTree
from garment_lab.mesh import edge_counts,boundary_edges
from garment_study.topology import projection_support
from .common import records,digest

def sample_surface(v,f,count,seed):
    rng=np.random.default_rng(seed);t=v[f];area=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)
    ids=rng.choice(len(f),count,p=area/area.sum());r=rng.random((count,2));a=np.sqrt(r[:,0]);weights=np.c_[1-a,a*(1-r[:,1]),a*r[:,1]]
    return (t[ids]*weights[:,:,None]).sum(1).astype(np.float32)

class Dataset:
    def __init__(self,manifest,limit=None):
        self.records=records(manifest)
        if limit is not None:self.records=self.records[:limit]
        self.samples=[]
        for index,row in enumerate(self.records):
            row['cache']=str((Path(manifest).resolve().parent/row['cache']).resolve())
            if digest(row['cache'])!=row['cache_sha256']:raise ValueError('Cache SHA mismatch: '+row['sample_id'])
            with np.load(row['cache'],allow_pickle=False) as z:
                vertices=z['initial_vertices'].astype(np.float32);faces=z['faces'].astype(np.int64);body=z['body_vertices'].astype(np.float32)
                edges,_=edge_counts(faces);boundary=np.zeros(len(vertices),np.float32);boundary[np.unique(boundary_edges(faces))]=1
                distance,nearest=cKDTree(body).query(vertices);extent=max(np.ptp(vertices,axis=0).max(),1e-5)
                body_features=np.c_[(vertices-body[nearest])/extent,distance/extent].astype(np.float32)
                partial=row.get('target_kind')=='observed_semantic_garment_patch'
                if partial:
                    target=z['patch_points'].astype(np.float32);target_boundary=np.empty((0,3),np.float32)
                else:
                    target=sample_surface(z['target_vertices'],z['target_faces'],8192,27000+index)
                    target_boundary=z['target_vertices'][np.unique(boundary_edges(z['target_faces']))].astype(np.float32)
                sample={'image':torch.from_numpy(z['image'].copy()).permute(2,0,1).float()/255,
                        'mask':torch.from_numpy(z['mask'].copy()).float(),'initial_vertices':torch.from_numpy(vertices),
                        'faces':torch.from_numpy(faces),'uv':torch.from_numpy(z['uv'].copy()).float(),
                        'edges':torch.from_numpy(edges).long(),'boundary':torch.from_numpy(boundary),
                        'body_vertices':torch.from_numpy(body),'body_features':torch.from_numpy(body_features),
                        'projection_support':torch.from_numpy(projection_support(z['mask'],z['uv'])),
                        'target_points':torch.from_numpy(target),'target_boundary':torch.from_numpy(target_boundary),
                        'partial_target':partial}
                for key in ['geodesic_distance','slice_support','vertex_mass','slot_distance','visible_proxy']:
                    sample[key]=torch.from_numpy(z[key].copy())
                self.samples.append(sample)

    def __len__(self):return len(self.samples)
    def __getitem__(self,i):return self.samples[i]

def move(sample,device):return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in sample.items()}

def check_groups(train,val):
    a,b=records(train),records(val)
    for key in ['sample_id','subject_id','garment_id']:
        if {r[key] for r in a}&{r[key] for r in b}:raise ValueError('Split overlap: '+key)
    if any(r['split']!='train' for r in a) or any(r['split']!='val' for r in b):raise ValueError('Training selection requires train/val')
    return {'train':len(a),'val':len(b),'disjoint_fields':['sample_id','subject_id','garment_id']}
