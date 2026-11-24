"""Area/length-weighted sampled mesh distances; no hidden alignment or unit conversion."""
import numpy as np
from scipy.spatial import cKDTree
from .mesh import boundary_edges


def sample_surface(vertices, faces, count=3000, seed=0):
    rng = np.random.default_rng(seed)
    triangles = vertices[faces]
    normal = np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    area = np.linalg.norm(normal,axis=-1)
    ids = rng.choice(len(faces),count,p=area/area.sum())
    a,b = rng.random((2,count))
    a = np.sqrt(a)
    weights = np.c_[1-a,a*(1-b),a*b]
    points = (triangles[ids]*weights[:,:,None]).sum(1)
    normals = normal[ids]/area[ids,None].clip(1e-12)
    return points,normals


def sample_boundary(vertices, faces, count=1500, seed=0):
    edges = boundary_edges(faces)
    if not len(edges):
        return np.empty((0,3),dtype=np.float32)
    rng = np.random.default_rng(seed)
    lengths = np.linalg.norm(vertices[edges[:,1]]-vertices[edges[:,0]],axis=1)
    ids = rng.choice(len(edges),count,p=lengths/lengths.sum())
    t = rng.random((count,1))
    return vertices[edges[ids,0]]*(1-t)+vertices[edges[ids,1]]*t


def surface_metrics(pred, target, count=3000, threshold=.01):
    if threshold <= 0:
        raise ValueError('F-score threshold must be positive in declared mesh units.')
    p,pn = sample_surface(pred.vertices,pred.faces,count,seed=17)
    q,qn = sample_surface(target.vertices,target.faces,count,seed=17)
    dp,ip = cKDTree(q).query(p)
    dq,iq = cKDTree(p).query(q)
    precision,recall = float((dp <= threshold).mean()),float((dq <= threshold).mean())
    bp,bq = sample_boundary(pred.vertices,pred.faces),sample_boundary(target.vertices,target.faces)
    boundary_cd = None if not len(bp) or not len(bq) else float((cKDTree(bq).query(bp)[0].mean()+cKDTree(bp).query(bq)[0].mean())/2)
    return {'sampled_surface_cd':float((dp.mean()+dq.mean())/2),
            'fscore':2*precision*recall/max(precision+recall,1e-12),
            'normal_consistency':float((np.abs((pn*qn[ip]).sum(1)).mean()+np.abs((qn*pn[iq]).sum(1)).mean())/2),
            'sampled_boundary_cd':boundary_cd,'fscore_threshold':threshold,
            'distance_definition':'mean_bidirectional_unsquared_sampled_distance',
            'alignment':'none','surface_samples_per_mesh':count}


def training_loss(pred, sample, points=384):
    import torch
    import torch.nn.functional as F
    def draw(vertices, faces):
        tri = vertices[faces]
        area = torch.linalg.vector_norm(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),dim=-1)
        ids = torch.multinomial(area.detach().clamp_min(1e-10),points,replacement=True)
        r = torch.rand(points,2,device=vertices.device)
        a = r[:,0].sqrt()
        weights = torch.stack([1-a,a*(1-r[:,1]),a*r[:,1]],-1)
        return (tri[ids]*weights[:,:,None]).sum(1)
    p = draw(pred,sample['faces'])
    q = draw(sample['target_vertices'],sample['target_faces'])
    distances = torch.cdist(p[None],q[None])[0]
    surface = (distances.min(0).values.mean()+distances.min(1).values.mean())/2
    pe,qe = boundary_edges(sample['faces'].detach().cpu().numpy()),boundary_edges(sample['target_faces'].detach().cpu().numpy())
    boundary = surface*0
    if len(pe) and len(qe):
        bp = pred[torch.as_tensor(np.unique(pe),device=pred.device)]
        bq = sample['target_vertices'][torch.as_tensor(np.unique(qe),device=pred.device)]
        bd = torch.cdist(bp[None],bq[None])[0]
        boundary = (bd.min(0).values.mean()+bd.min(1).values.mean())/2
    edges = sample['edges']
    before = torch.linalg.vector_norm(sample['initial_vertices'][edges[:,0]]-sample['initial_vertices'][edges[:,1]],dim=-1)
    after = torch.linalg.vector_norm(pred[edges[:,0]]-pred[edges[:,1]],dim=-1)
    stretch = ((after-before)/(before.clamp_min(1e-5))).square().mean()
    displacement = (pred-sample['initial_vertices']).square().mean()
    loss = surface+.35*boundary+.002*stretch+.01*displacement
    return loss, {'surface_loss':float(surface.detach()),'boundary_loss':float(boundary.detach()),'stretch_loss':float(stretch.detach())}
