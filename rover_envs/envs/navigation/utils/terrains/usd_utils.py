# import isaacsim.core.utils.prims as prim_utils
import numpy as np
# # import isaaclab.utils.kit as kit_utils
# from isaacsim.core.api.materials import PhysicsMaterial
# from isaacsim.core.prims import XFormPrim
from typing import Tuple
from functools import lru_cache

from pxr import Usd, UsdPhysics, UsdGeom


@lru_cache(maxsize=1) # Cache the result to avoid repeated imports.
def isaacsim_available():
    """Check if Isaac Sim is available, with automatic caching."""
    try:
        import isaacsim.core
        return True
    except ImportError:
        return False
    

def get_triangles_and_vertices_from_prim(prim_path):
    from isaacsim.core.utils.stage import get_current_stage
    """ Get triangles and vertices from prim """
    stage: Usd.Stage = get_current_stage()
    mesh_prim = stage.GetPrimAtPath(prim_path)

    # Validate prim exists and is valid
    if not mesh_prim or not mesh_prim.IsValid():
        raise RuntimeError(f"Invalid or null prim at path: {prim_path}")
    
    # Check if it's a mesh prim
    if not mesh_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"Prim at path {prim_path} is not a mesh")

    # Get mesh attributes with validation
    points_attr = mesh_prim.GetAttribute("points")
    face_vertex_indices_attr = mesh_prim.GetAttribute("faceVertexIndices")
    
    if not points_attr:
        raise RuntimeError(f"Mesh at {prim_path} has no points attribute")
    if not face_vertex_indices_attr:
        raise RuntimeError(f"Mesh at {prim_path} has no face vertex indices attribute")

    points = points_attr.Get()
    face_vertex_indices = face_vertex_indices_attr.Get()
    
    if points is None:
        raise RuntimeError(f"Failed to get points data from mesh at {prim_path}")
    if face_vertex_indices is None:
        raise RuntimeError(f"Failed to get face indices data from mesh at {prim_path}")

    # Convert points to numpy array and extract xyz coordinates efficiently
    vertices_array = np.array(points, dtype=np.float32)
    if vertices_array.ndim == 2 and vertices_array.shape[1] >= 3:
        # Take only x, y, z coordinates if there are more than 3 components
        vertices = vertices_array[:, :3]
    else:
        vertices = vertices_array
    
    # Convert face indices to numpy array and reshape to triangles
    face_indices_array = np.array(face_vertex_indices, dtype=np.int32)
    faces = face_indices_array.reshape(-1, 3)

    return faces, vertices

def get_triangles_and_vertices_from_prim_standalone(usd_file_path: str, prim_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standalone USD loader that doesn't require Isaac Sim runtime
    
    Args:
        usd_file_path: Path to the USD file
        prim_path: Specific prim path to load (if None, finds first mesh)
        
    Returns:
        Tuple of (faces, vertices) as numpy arrays (note order matches Isaac Sim function)
    """
    
    # Open the USD stage
    stage = Usd.Stage.Open(usd_file_path)
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {usd_file_path}")
    
    # Find mesh primitive
    mesh_prim = None
    if prim_path:
        mesh_prim = stage.GetPrimAtPath(prim_path)
        if not mesh_prim or not mesh_prim.IsA(UsdGeom.Mesh):
            raise RuntimeError(f"No valid mesh found at path: {prim_path}")
    else:
        # Find first mesh in the stage
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                mesh_prim = prim
                print(f"Found mesh at path: {prim.GetPath()}")
                break
    
    if not mesh_prim:
        raise RuntimeError("No mesh found in USD file")
    
    # Get mesh data
    mesh = UsdGeom.Mesh(mesh_prim)
    
    # Get points (vertices)
    points_attr = mesh.GetPointsAttr()
    if not points_attr:
        raise RuntimeError("Mesh has no points attribute")
    points = points_attr.Get()
    
    # Get face vertex indices
    face_vertex_indices_attr = mesh.GetFaceVertexIndicesAttr()
    if not face_vertex_indices_attr:
        raise RuntimeError("Mesh has no face vertex indices")
    face_vertex_indices = face_vertex_indices_attr.Get()
    
    # Get face vertex counts (usually 3 for triangles)
    face_vertex_counts_attr = mesh.GetFaceVertexCountsAttr()
    face_vertex_counts = face_vertex_counts_attr.Get() if face_vertex_counts_attr else None
    
    # Convert to numpy arrays
    vertices = np.array(points, dtype=np.float32)
    if vertices.ndim == 2 and vertices.shape[1] >= 3:
        vertices = vertices[:, :3]  # Take only x, y, z coordinates
    
    # Convert faces to triangles
    face_indices = np.array(face_vertex_indices, dtype=np.int32)
    
    if face_vertex_counts is not None:
        # Handle polygons with different vertex counts
        faces = []
        start_idx = 0
        for count in face_vertex_counts:
            if count == 3:
                # Triangle - add directly
                faces.append(face_indices[start_idx:start_idx + 3])
            elif count > 3:
                # Polygon - triangulate by fan triangulation
                for i in range(1, count - 1):
                    faces.append([
                        face_indices[start_idx],
                        face_indices[start_idx + i],
                        face_indices[start_idx + i + 1]
                    ])
            start_idx += count
        faces = np.array(faces, dtype=np.int32)
    else:
        # Assume all triangles
        faces = face_indices.reshape(-1, 3)
    
    return faces, vertices

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