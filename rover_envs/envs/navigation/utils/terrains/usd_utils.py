import numpy as np
from typing import Tuple
from functools import lru_cache

from pxr import Usd, UsdGeom


@lru_cache(maxsize=1) # Cache the result to avoid repeated imports.
def isaacsim_available():
    """Check if Isaac Sim is available, with automatic caching."""
    try:
        import isaacsim.core
        return True
    except ImportError:
        return False
    

def get_triangles_and_vertices_from_prim(prim_path):
    """Get triangles and vertices from a mesh prim or mesh-containing prim tree."""
    from isaacsim.core.utils.stage import get_current_stage

    stage: Usd.Stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"Invalid or null prim at path: {prim_path}")

    return _get_triangles_and_vertices_from_mesh_tree(root_prim)


def get_triangles_and_vertices_from_prim_standalone(
    usd_file_path: str,
    prim_path: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standalone USD loader that doesn't require Isaac Sim runtime.

    Args:
        usd_file_path: Path to the USD file.
        prim_path: Specific prim path to load. If None, uses the default prim when available,
            otherwise scans the full stage.

    Returns:
        Tuple of (faces, vertices) as numpy arrays (note order matches Isaac Sim function).
    """
    stage = Usd.Stage.Open(usd_file_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_file_path}")

    if prim_path:
        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim or not root_prim.IsValid():
            raise RuntimeError(f"No valid prim found at path: {prim_path}")
    else:
        root_prim = stage.GetDefaultPrim()
        if not root_prim or not root_prim.IsValid():
            root_prim = stage.GetPseudoRoot()

    return _get_triangles_and_vertices_from_mesh_tree(root_prim)


def _get_triangles_and_vertices_from_mesh_tree(root_prim: Usd.Prim) -> Tuple[np.ndarray, np.ndarray]:
    mesh_prims = [prim for prim in Usd.PrimRange(root_prim) if prim.IsA(UsdGeom.Mesh)]
    if not mesh_prims:
        raise RuntimeError(f"No mesh prims found under path: {root_prim.GetPath()}")

    xform_cache = UsdGeom.XformCache()
    vertices_by_mesh = []
    faces_by_mesh = []
    vertex_offset = 0

    for mesh_prim in mesh_prims:
        faces, vertices = _read_mesh_prim(mesh_prim)
        if vertices.size == 0 or faces.size == 0:
            continue

        vertices = _transform_vertices(vertices, xform_cache.GetLocalToWorldTransform(mesh_prim))
        vertices_by_mesh.append(vertices)
        faces_by_mesh.append(faces + vertex_offset)
        vertex_offset += len(vertices)

    if not vertices_by_mesh:
        raise RuntimeError(f"No triangle mesh data found under path: {root_prim.GetPath()}")

    return np.vstack(faces_by_mesh).astype(np.int32), np.vstack(vertices_by_mesh).astype(np.float32)


def _read_mesh_prim(mesh_prim: Usd.Prim) -> Tuple[np.ndarray, np.ndarray]:
    mesh = UsdGeom.Mesh(mesh_prim)
    points = mesh.GetPointsAttr().Get()
    face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()

    if points is None:
        raise RuntimeError(f"Mesh at {mesh_prim.GetPath()} has no points data")
    if face_vertex_indices is None:
        raise RuntimeError(f"Mesh at {mesh_prim.GetPath()} has no face vertex indices")

    vertices = np.asarray(points, dtype=np.float32)
    if vertices.ndim == 2 and vertices.shape[1] > 3:
        vertices = vertices[:, :3]

    faces = _triangulate_face_indices(
        np.asarray(face_vertex_indices, dtype=np.int32),
        np.asarray(face_vertex_counts, dtype=np.int32) if face_vertex_counts is not None else None,
    )
    return faces, vertices


def _triangulate_face_indices(face_vertex_indices: np.ndarray, face_vertex_counts: np.ndarray | None) -> np.ndarray:
    if face_vertex_counts is None:
        return face_vertex_indices.reshape(-1, 3).astype(np.int32)

    triangles = []
    index = 0
    for count in face_vertex_counts:
        face = face_vertex_indices[index:index + count]
        if count == 3:
            triangles.append(face)
        elif count > 3:
            for i in range(1, count - 1):
                triangles.append([face[0], face[i], face[i + 1]])
        index += count

    if not triangles:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(triangles, dtype=np.int32)


def _transform_vertices(vertices: np.ndarray, transform) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    homogeneous = np.ones((len(vertices), 4), dtype=np.float64)
    homogeneous[:, :3] = vertices
    return (homogeneous @ matrix)[:, :3].astype(np.float32)


def check_prim_exists(prim_path):
    """
    Check if a prim exists and is valid in the current Isaac Sim stage.
    
    Args:
        prim_path: Path to the prim to check
        
    Returns:
        bool: True if prim exists and is valid, False otherwise
    """
    try:
        from isaacsim.core.utils.stage import get_current_stage
        stage: Usd.Stage = get_current_stage()
        mesh_prim = stage.GetPrimAtPath(prim_path)

        return (mesh_prim and 
                mesh_prim.IsValid() and 
                mesh_prim.IsA(UsdGeom.Mesh))
    except ImportError:
        # Isaac Sim not available
        return False
    except Exception:
        return False
