"""Garment-specific manifests: explicit target identity, scale, grouping and provenance."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.spatial import cKDTree
from .mesh import Mesh, edge_counts, boundary_edges

KINDS = {'garment_benchmark','synthetic_software_check'}


def read_manifest(path):
    path = Path(path).resolve()
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:
        raise ValueError('Empty garment manifest.')
    for row in records:
        required = {'sample_id','subject_id','garment_id','split','cache','data_kind','target_kind','units'}
        if required-row.keys():
            raise ValueError(f'Missing garment metadata: {required-row.keys()}')
        if row['target_kind'] != 'independent_garment_surface' or row['data_kind'] not in KINDS:
            raise ValueError('Garment targets must be explicitly independent surfaces, not body/SMPL targets.')
        if row['units'] not in {'meters','arbitrary_image_scale'}:
            raise ValueError('Unsupported garment coordinate units.')
        if row['data_kind']=='garment_benchmark' and len(row.get('cache_sha256',''))!=64:
            raise ValueError('Real garment caches require a SHA256 receipt.')
        row['cache'] = str((path.parent/row['cache']).resolve())
    if len({row['sample_id'] for row in records}) != len(records):
        raise ValueError('Duplicate garment sample ids.')
    return records


def audit_splits(train, val):
    a,b = read_manifest(train),read_manifest(val)
    if any(row['split'] != 'train' for row in a) or any(row['split'] != 'val' for row in b):
        raise ValueError('Expected train and val manifests; never select checkpoints on test.')
    for field in ('sample_id','subject_id','garment_id'):
        if {row[field] for row in a}&{row[field] for row in b}:
            raise ValueError('Garment train/validation overlap: '+field)
    if len({row['units'] for row in a+b}) != 1 or len({row['data_kind'] for row in a+b}) != 1:
        raise ValueError('Do not silently mix metric/normalized or software/benchmark data.')
    return {'train_samples':len(a),'val_samples':len(b),'units':a[0]['units'],'data_kind':a[0]['data_kind']}


def sample_tensors(arrays):
    v,f = arrays['initial_vertices'],arrays['faces']
    Mesh(v,f).validate()
    image = arrays['image']
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError('Cache image must be uint8 HxWx3.')
    uv = np.asarray(arrays['uv'])
    if uv.shape != (len(v),2) or not np.isfinite(uv).all():
        raise ValueError('Per-garment-vertex image UVs are required.')
    mask = arrays['mask'].astype(np.float32)
    if mask.shape != image.shape[:2] or not np.isfinite(mask).all():
        raise ValueError('Mask/image shape mismatch.')
    if mask.min()<0 or mask.max()>1:
        raise ValueError('Cache mask must have values in [0,1].')
    edges,_ = edge_counts(f)
    boundary = np.zeros(len(v),dtype=np.float32)
    boundary[np.unique(boundary_edges(f))] = 1
    body_features = np.zeros((len(v),4),dtype=np.float32)
    if 'body_vertices' in arrays:
        bv = arrays['body_vertices']
        if not np.isfinite(bv).all() or bv.ndim != 2 or bv.shape[1] != 3:
            raise ValueError('Invalid body context.')
        distance,nn = cKDTree(bv).query(v)
        scale = max(np.ptp(v,axis=0).max(),1e-5)
        body_features = np.c_[(v-bv[nn])/scale,distance/scale].astype(np.float32)
    result = {'image':torch.from_numpy(image.copy()).permute(2,0,1).float()/255,
              'mask':torch.from_numpy(mask.copy()),'initial_vertices':torch.from_numpy(v.copy()).float(),
              'uv':torch.from_numpy(uv.copy()).float(),'faces':torch.from_numpy(f.copy()).long(),
              'edges':torch.from_numpy(edges).long(),'boundary':torch.from_numpy(boundary),
              'body_features':torch.from_numpy(body_features)}
    if 'target_vertices' in arrays:
        Mesh(arrays['target_vertices'],arrays['target_faces']).validate()
        result.update(target_vertices=torch.from_numpy(arrays['target_vertices'].copy()).float(),
                      target_faces=torch.from_numpy(arrays['target_faces'].copy()).long())
        if 'vertex_correspondence' in arrays:
            result['vertex_correspondence']=torch.tensor(bool(arrays['vertex_correspondence'].item()))
    return result


class GarmentDataset(Dataset):
    def __init__(self, manifest, allow_synthetic=False):
        self.records = read_manifest(manifest)
        if any(r['data_kind']=='synthetic_software_check' for r in self.records) and not allow_synthetic:
            raise ValueError('Synthetic software fixtures require --allow-synthetic and are not human benchmarks.')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from .reconstruct import sha256
        row=self.records[index]
        if row.get('cache_sha256') and sha256(row['cache'])!=row['cache_sha256']:
            raise ValueError('Garment cache SHA256 mismatch: '+row['sample_id'])
        with np.load(self.records[index]['cache'],allow_pickle=False) as raw:
            arrays = {key:raw[key].copy() for key in raw.files}
        if not {'initial_vertices','faces','uv','image','mask','target_vertices','target_faces'}.issubset(arrays):
            raise ValueError('Garment cache is missing independent mesh/image targets.')
        return sample_tensors(arrays)
