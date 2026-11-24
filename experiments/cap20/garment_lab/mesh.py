"""Validated independent meshes and open-boundary topology. No SMPL vertex subset."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import trimesh


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    uv: np.ndarray | None = None
    colors: np.ndarray | None = None

    def validate(self):
        v, f = np.asarray(self.vertices), np.asarray(self.faces)
        if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all() or len(v) < 3:
            raise ValueError('Mesh requires finite (N,3) vertices.')
        if f.ndim != 2 or f.shape[1] != 3 or len(f) == 0 or not np.issubdtype(f.dtype, np.integer):
            raise ValueError('Mesh requires nonempty integer triangular faces.')
        if f.min() < 0 or f.max() >= len(v):
            raise ValueError('Faces reference nonexistent vertices.')
        triangles = v[f]
        areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
        if np.any(areas < 1e-12):
            raise ValueError('Degenerate garment triangles.')
        if self.uv is not None and (self.uv.shape != (len(v), 2) or not np.isfinite(self.uv).all()):
            raise ValueError('UVs must be finite (N,2).')
        edges, counts = edge_counts(f)
        if np.any(counts > 2):
            raise ValueError('Nonmanifold garment edges.')
        return self


def edge_counts(faces):
    all_edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(all_edges, axis=1), axis=0, return_counts=True)


def boundary_edges(faces):
    edges, counts = edge_counts(np.asarray(faces))
    return edges[counts == 1]


def boundary_loops(faces):
    edges = boundary_edges(faces)
    adjacency = {}
    for a, b in edges:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    if any(len(values) != 2 for values in adjacency.values()):
        raise ValueError('Garment boundary is not a union of closed manifold loops.')
    unseen, loops = set(adjacency), []
    while unseen:
        start = min(unseen)
        loop, previous, current = [], None, start
        while True:
            loop.append(current)
            unseen.discard(current)
            nxt = next(node for node in adjacency[current] if node != previous)
            previous, current = current, nxt
            if current == start:
                break
            if current not in unseen:
                raise ValueError('Invalid or intersecting boundary loops.')
        loops.append(np.asarray(loop, dtype=np.int64))
    return loops


def topology_report(mesh):
    mesh.validate()
    loops = boundary_loops(mesh.faces)
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    return {'vertices': len(mesh.vertices), 'faces': len(mesh.faces),
            'boundary_edges': len(boundary_edges(mesh.faces)), 'boundary_loops': len(loops),
            'loop_lengths': [len(loop) for loop in loops],
            'connected_components': len(tm.split(only_watertight=False)),
            'watertight': bool(tm.is_watertight), 'winding_consistent': bool(tm.is_winding_consistent)}


def load_mesh(path):
    obj = trimesh.load(str(path), force='mesh', process=False)
    if not isinstance(obj, trimesh.Trimesh):
        raise ValueError('Expected a triangle mesh.')
    uv = getattr(obj.visual, 'uv', None)
    colors = np.asarray(obj.visual.vertex_colors) if obj.visual.kind == 'vertex' else None
    return Mesh(np.asarray(obj.vertices, dtype=np.float32), np.asarray(obj.faces, dtype=np.int64), uv, colors).validate()


def export_mesh(mesh, directory, name='garment'):
    mesh.validate()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    colors = np.asarray(mesh.colors)[:, :3]/255.0 if mesh.colors is not None else None
    with (directory / (name+'.obj')).open('w', encoding='utf-8') as stream:
        stream.write('# Independent open garment surface; see reconstruction.json for scale and provenance.\n')
        for i, vertex in enumerate(mesh.vertices):
            suffix = '' if colors is None else ' '+' '.join(f'{x:.6f}' for x in colors[i])
            stream.write('v '+' '.join(f'{x:.8f}' for x in vertex)+suffix+'\n')
        for face in mesh.faces:
            stream.write('f '+' '.join(str(int(x)+1) for x in face)+'\n')
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=mesh.colors, process=False)
    (directory/(name+'.glb')).write_bytes(tm.export(file_type='glb'))
    arrays = {'vertices': mesh.vertices, 'faces': mesh.faces}
    if mesh.uv is not None:
        arrays['uv'] = mesh.uv
    if mesh.colors is not None:
        arrays['colors'] = mesh.colors
    np.savez_compressed(directory/(name+'.npz'), **arrays)
    return topology_report(mesh)
