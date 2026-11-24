"""Input-only geodesic support; target geometry never determines learned assignments."""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from garment_lab.mesh import edge_counts,boundary_edges,boundary_loops


def projection_support(mask,uv):
    """Support an opening projection only near the supplied 2D silhouette contour.

    Internal neckline/cuff edges can be hidden behind another cloth panel in an
    amodal silhouette, so front-panel membership alone is insufficient evidence.
    """
    mask=np.asarray(mask,dtype=bool)
    edge=mask & ~ndimage.binary_erosion(mask)
    distance=ndimage.distance_transform_edt(~edge)
    h,w=mask.shape
    pixels=uv*np.array([w,h])-.5
    sampled=ndimage.map_coordinates(distance,[pixels[:,1],pixels[:,0]],order=1,mode='nearest')
    radius=1.5*max(h,w)/256
    return (sampled<=radius).astype(np.float32)


def dense_geodesic(vertices,faces):
    edges,_=edge_counts(faces);length=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1).clip(1e-7)
    graph=coo_matrix((np.r_[length,length],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(vertices),len(vertices))).tocsr()
    distance=dijkstra(graph,directed=False)
    scale=np.median(distance[np.isfinite(distance)&(distance>0)])
    return np.nan_to_num(distance/max(scale,1e-6),posinf=100).astype(np.float32)


def topology_features(vertices,faces,interior_slots=16,boundary_slots=8):
    edges,_=edge_counts(faces); n=len(vertices)
    length=np.linalg.norm(vertices[edges[:,0]]-vertices[edges[:,1]],axis=1).clip(1e-7)
    graph=coo_matrix((np.r_[length,length],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n)).tocsr()
    loops=boundary_loops(faces)
    if len(loops)>boundary_slots: raise ValueError('Every opening requires at least one boundary slot.')
    boundary=np.zeros(n,dtype=bool); boundary[np.unique(boundary_edges(faces))]=True
    interior=np.where(~boundary)[0]
    if len(interior)<interior_slots: raise ValueError('Mesh has too few interior vertices for anchors.')
    anchors=[int(interior[0])]
    nearest=dijkstra(graph,directed=False,indices=anchors[0])
    for _ in range(interior_slots-1):
        candidate=int(interior[np.argmax(nearest[interior])]); anchors.append(candidate)
        nearest=np.minimum(nearest,dijkstra(graph,directed=False,indices=candidate))
    # Ensure every natural opening owns a slot before assigning extra capacity.
    allocation=np.ones(len(loops),dtype=int)
    for _ in range(boundary_slots-len(loops)):
        index=int(np.argmax([len(loop)/allocation[i] for i,loop in enumerate(loops)])); allocation[index]+=1
    boundary_anchors=[]; slot_loop=[]
    for index,(loop,count) in enumerate(zip(loops,allocation)):
        for offset in np.linspace(0,len(loop),count,endpoint=False).astype(int):
            boundary_anchors.append(int(loop[offset])); slot_loop.append(index)
    anchors+=boundary_anchors
    distances=dijkstra(graph,directed=False,indices=anchors).T
    scale=np.median(distances[np.isfinite(distances)&(distances>0)])
    distances=np.nan_to_num(distances/max(scale,1e-6),posinf=100.).astype(np.float32)
    support=np.zeros_like(distances,dtype=bool)
    nearest_interior=np.argsort(distances[:,:interior_slots],axis=1)[:,:min(3,interior_slots)]
    support[np.arange(n)[:,None],nearest_interior]=True
    vertex_loop=np.full(n,-1,dtype=int)
    for index,loop in enumerate(loops): vertex_loop[loop]=index
    # Boundary pooling is isolated by opening; a tiny cuff cannot disappear into a hem.
    for vertex in np.where(boundary)[0]:
        support[vertex]=False
        allowed=np.where(np.asarray(slot_loop)==vertex_loop[vertex])[0]+interior_slots
        order=allowed[np.argsort(distances[vertex,allowed])[:2]]
        support[vertex,order]=True
    area=np.linalg.norm(np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]]),axis=1)/2
    mass=np.zeros(n,dtype=np.float32)
    np.add.at(mass,faces.reshape(-1),np.repeat(area/3,3))
    bedges=boundary_edges(faces); boundary_mass=np.zeros(n,dtype=np.float32)
    blength=np.linalg.norm(vertices[bedges[:,0]]-vertices[bedges[:,1]],axis=1)
    np.add.at(boundary_mass,bedges.reshape(-1),np.repeat(blength/2,2))
    mass[boundary]=boundary_mass[boundary]
    mass/=max(float(mass.mean()),1e-8)
    slot_distance=distances[np.asarray(anchors)].astype(np.float32)
    return {'geodesic_distance':distances,'slice_support':support,'vertex_mass':mass,
            'slot_distance':slot_distance,'anchors':np.asarray(anchors,dtype=np.int64),
            'vertex_loop':vertex_loop.astype(np.int64)}
