"""
Procedural Terrain Generator for Rover Environments.

This module provides tools to procedurally generate terrain meshes with rocks/obstacles
for use in Isaac Lab rover simulations. It generates three USD files:
- terrain_only.usd: Base terrain mesh (for physics collision)
- rocks_merged.usd: Rock obstacles mesh (for collision detection)
- terrain_merged.usd: Combined terrain + rocks (for raycasting)

Based on terrain-generation6.py from RLRoverLab-extra-features.
"""
from __future__ import annotations

import math
import os
import random
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.stats import qmc

# Optional imports - only needed when generating
try:
    import pymeshlab
    PYMESHLAB_AVAILABLE = True
except ImportError:
    PYMESHLAB_AVAILABLE = False

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdPhysics
    PXR_AVAILABLE = True
except ImportError:
    PXR_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Terrain:
    """A data class to hold terrain grid information."""
    terrain_name: str = "terrain"
    width: int = 256  # meters
    length: int = 256  # meters
    vertical_scale: float = 1.0
    horizontal_scale: float = 1.0
    
    def __post_init__(self):
        self.num_rows = int(self.length / self.horizontal_scale)
        self.num_cols = int(self.width / self.horizontal_scale)
        self.height_field_raw = np.zeros(
            (self.num_rows, self.num_cols), dtype=np.float64
        )


@dataclass
class TerrainLayer:
    """Configuration for a terrain feature layer."""
    name: str
    num_features: int
    radius_m_range: Tuple[float, float]
    height_m_range: Tuple[float, float]
    kernel_type: str = "gaussian"  # "gaussian" or "crater"
    kernel_params: Dict[str, Any] = field(default_factory=lambda: {"sigma": 0.4})
    seed: Optional[int] = None


@dataclass 
class RockConfig:
    """Configuration for rock placement."""
    num_rocks: int = 1500  # Reduced to match mars terrain complexity (~480k verts)
    scale_range: Tuple[float, float] = (0.05, 0.25)
    embed_percentage: float = 0.25


@dataclass
class TerrainGeneratorConfig:
    """Full configuration for terrain generation."""
    name: str = "generated_terrain"
    width: int = 200
    length: int = 200
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.05
    target_vertices: int = 200000
    layers: List[TerrainLayer] = field(default_factory=list)
    rock_config: RockConfig = field(default_factory=RockConfig)
    seed: Optional[int] = None


# =============================================================================
# Kernel Generators
# =============================================================================

def normalize_data(data: np.ndarray) -> np.ndarray:
    """Scales data to the [0, 1] range."""
    min_val, max_val = np.min(data), np.max(data)
    if max_val - min_val == 0:
        return data - min_val
    return (data - min_val) / (max_val - min_val)


def gaussian_distribution_1d(n_samples: int, sigma: float = 0.3, normalized: bool = True) -> np.ndarray:
    """Generate 1D Gaussian distribution."""
    x = np.linspace(-1, 1, n_samples)
    gauss = np.exp(-0.5 * (x / sigma)**2) / (sigma * np.sqrt(2 * np.pi))
    return normalize_data(gauss) if normalized else gauss


def gaussian_kernel_2d(diameter: int, sigma: float = 0.4, normalized: bool = True, **kwargs) -> np.ndarray:
    """Generate 2D Gaussian kernel for hills/bumps."""
    gauss_1d = gaussian_distribution_1d(diameter, sigma=sigma, normalized=normalized)
    return np.outer(gauss_1d, gauss_1d)


def circular_crater_kernel(diameter: int, sigma: float = 0.25, **kwargs) -> np.ndarray:
    """Generate circular crater kernel."""
    center = diameter // 2
    y, x = np.ogrid[:diameter, :diameter]
    dist_from_center = np.sqrt((x - center)**2 + (y - center)**2)
    radius = diameter / 2.0
    crater_kernel = np.exp(-(dist_from_center**2) / (2 * (sigma * radius)**2))
    return normalize_data(crater_kernel)


KERNEL_GENERATORS = {
    "gaussian": gaussian_kernel_2d,
    "crater": circular_crater_kernel,
}


# =============================================================================
# Terrain Generation
# =============================================================================

def apply_feature_layer(terrain: Terrain, layer: TerrainLayer) -> int:
    """Apply a feature layer to the terrain (optimized version)."""
    seed = layer.seed if layer.seed is not None else random.randint(0, 100000)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    
    # Use Halton sequence for quasi-random distribution
    sampler = qmc.Halton(d=2, scramble=False)
    feature_centers = sampler.random(n=layer.num_features)
    feature_centers[:, 0] *= terrain.num_rows
    feature_centers[:, 1] *= terrain.num_cols
    feature_centers = feature_centers.astype(int)
    
    kernel_generator = KERNEL_GENERATORS.get(layer.kernel_type, gaussian_kernel_2d)
    
    # Pre-generate all random values at once (much faster)
    radii = rng.uniform(layer.radius_m_range[0], layer.radius_m_range[1], layer.num_features)
    heights = rng.uniform(layer.height_m_range[0], layer.height_m_range[1], layer.num_features)
    
    # Cache for kernels of same diameter (common optimization)
    kernel_cache = {}
    features_added = 0
    
    for idx, (center_r, center_c) in enumerate(feature_centers):
        radius_m = radii[idx]
        height_m = heights[idx]
        kernel_diameter = int((2 * radius_m) / terrain.horizontal_scale) + 1
        radius_grid = (kernel_diameter - 1) // 2
        
        if radius_grid == 0:
            continue
        
        # Use cached kernel if available
        if kernel_diameter not in kernel_cache:
            kernel_cache[kernel_diameter] = kernel_generator(diameter=kernel_diameter, **layer.kernel_params)
        kernel = kernel_cache[kernel_diameter]
        
        # Calculate terrain and kernel slice boundaries
        r_start_terrain = max(0, center_r - radius_grid)
        r_end_terrain = min(terrain.num_rows, center_r + radius_grid)
        c_start_terrain = max(0, center_c - radius_grid)
        c_end_terrain = min(terrain.num_cols, center_c + radius_grid)
        
        r_start_kernel = abs(min(0, center_r - radius_grid))
        r_end_kernel = abs(min(kernel_diameter - 1, terrain.num_rows - (center_r - radius_grid)))
        c_start_kernel = abs(min(0, center_c - radius_grid))
        c_end_kernel = abs(min(kernel_diameter - 1, terrain.num_cols - (center_c - radius_grid)))
        
        terrain_slice = terrain.height_field_raw[
            r_start_terrain:r_end_terrain,
            c_start_terrain:c_end_terrain
        ]
        kernel_slice = kernel[
            r_start_kernel:r_end_kernel,
            c_start_kernel:c_end_kernel
        ]
        
        if terrain_slice.shape == kernel_slice.shape:
            terrain_slice += kernel_slice * (height_m / terrain.vertical_scale)
            features_added += 1
    
    return features_added


def generate_terrain(config: TerrainGeneratorConfig) -> Terrain:
    """Generate terrain from configuration."""
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
    
    terrain = Terrain(
        terrain_name=config.name,
        width=config.width,
        length=config.length,
        horizontal_scale=config.horizontal_scale,
        vertical_scale=config.vertical_scale,
    )
    
    print(f"Generating terrain '{config.name}' ({config.width}x{config.length}m)...")
    
    for i, layer in enumerate(config.layers):
        print(f"  Applying layer {i+1}/{len(config.layers)}: {layer.name}...")
        num_added = apply_feature_layer(terrain, layer)
        print(f"    -> Added {num_added}/{layer.num_features} features")
    
    return terrain


# =============================================================================
# Mesh Operations
# =============================================================================

def convert_heightfield_to_trimesh(
    height_field: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert heightfield to triangle mesh."""
    num_rows, num_cols = height_field.shape
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)
    
    vertices = np.stack(
        (xx.flatten(), yy.flatten(), height_field.flatten() * vertical_scale),
        axis=1
    ).astype(np.float32)
    
    indices = np.arange(num_rows * num_cols).reshape(num_rows, num_cols)
    v0 = indices[:-1, :-1].flatten()
    v1 = indices[:-1, 1:].flatten()
    v2 = indices[1:, :-1].flatten()
    v3 = indices[1:, 1:].flatten()
    
    faces = np.vstack((
        np.stack((v0, v3, v1), axis=1),
        np.stack((v0, v2, v3), axis=1)
    )).astype(np.uint32)
    
    return vertices, faces


def reduce_mesh_polygons(
    height_field: np.ndarray,
    horizontal_scale: float,
    vertical_scale: float,
    target_vertices: int = 50000
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce mesh polygon count using decimation."""
    if not PYMESHLAB_AVAILABLE:
        raise ImportError("pymeshlab is required for mesh reduction. Install with: pip install pymeshlab")
    
    original_rows, original_cols = height_field.shape
    print(f"  Original heightfield: {original_rows}x{original_cols} ({original_rows * original_cols:,} points)")
    
    # Downsample heightfield first
    intermediate_vertices = target_vertices * 4
    downsample_factor = math.sqrt((original_rows * original_cols) / intermediate_vertices)
    new_rows = int(original_rows / downsample_factor)
    new_cols = int(original_cols / downsample_factor)
    
    print(f"  Stage 1: Downsampling to {new_rows}x{new_cols}...")
    height_field_downsampled = cv2.resize(
        height_field, (new_cols, new_rows), interpolation=cv2.INTER_AREA
    )
    
    vertices, triangles = convert_heightfield_to_trimesh(
        height_field_downsampled,
        horizontal_scale * downsample_factor,
        vertical_scale
    )
    
    print("  Stage 2: Applying mesh decimation...")
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertices, triangles))
    ms.apply_filter(
        'meshing_decimation_quadric_edge_collapse',
        targetfacenum=int(2 * target_vertices),
        preservenormal=True
    )
    
    m = ms.current_mesh()
    print(f"  Final mesh: {m.vertex_number():,} vertices, {m.face_number():,} faces")
    
    return m.vertex_matrix().astype('float32'), m.face_matrix().astype('uint32')


def load_obj(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a simple OBJ file."""
    vertices = []
    faces = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(i) for i in line.strip().split()[1:]])
            elif line.startswith('f '):
                face = [int(i.split('/')[0]) - 1 for i in line.strip().split()[1:]]
                faces.append(face)
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.uint32)


def generate_rock_mesh(
    terrain_verts: np.ndarray,
    rock_models: List[Tuple[np.ndarray, np.ndarray]],
    num_rocks: int,
    scale_range: Tuple[float, float],
    embed_percentage: float = 0.20
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate combined mesh for placed rocks (optimized vectorized version)."""
    print(f"  Generating {num_rocks} rocks...")
    
    if not rock_models:
        print("    WARNING: No rock models provided")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # Pre-compute rock model properties for faster access
    rock_data = []
    for rock_verts, rock_faces in rock_models:
        min_coords = np.min(rock_verts, axis=0)
        max_coords = np.max(rock_verts, axis=0)
        original_height = max_coords[2] - min_coords[2]
        pivot_point = np.array([
            (min_coords[0] + max_coords[0]) / 2.0,
            (min_coords[1] + max_coords[1]) / 2.0,
            min_coords[2]
        ])
        centered_verts = rock_verts - pivot_point
        rock_data.append({
            'verts': centered_verts,
            'faces': rock_faces,
            'height': original_height,
            'num_verts': len(rock_verts)
        })
    
    # Pre-generate all random values at once (much faster than per-rock)
    placement_indices = np.random.choice(len(terrain_verts), num_rocks, replace=True)
    rock_choices = np.random.randint(0, len(rock_models), num_rocks)
    scales = np.random.uniform(scale_range[0], scale_range[1], num_rocks)
    spin_angles = np.random.uniform(0, 2 * np.pi, num_rocks)
    
    # Pre-compute sin/cos for all rotations
    cos_angles = np.cos(spin_angles)
    sin_angles = np.sin(spin_angles)
    
    combined_rock_verts = []
    combined_rock_faces = []
    vertex_offset = 0
    
    for i in range(num_rocks):
        rock = rock_data[rock_choices[i]]
        placement_vertex = terrain_verts[placement_indices[i]]
        scale = scales[i]
        
        # Scale vertices
        transformed_verts = rock['verts'] * scale
        
        # Apply rotation using pre-computed sin/cos
        cos_s, sin_s = cos_angles[i], sin_angles[i]
        x_new = transformed_verts[:, 0] * cos_s - transformed_verts[:, 1] * sin_s
        y_new = transformed_verts[:, 0] * sin_s + transformed_verts[:, 1] * cos_s
        transformed_verts = np.column_stack([x_new, y_new, transformed_verts[:, 2]])
        
        # Embed rocks into terrain
        scaled_height = rock['height'] * scale
        offset_distance = scaled_height * embed_percentage
        
        transformed_verts += placement_vertex
        transformed_verts[:, 2] -= offset_distance
        
        combined_rock_verts.append(transformed_verts)
        combined_rock_faces.append(rock['faces'] + vertex_offset)
        vertex_offset += rock['num_verts']
    
    print(f"    Rock mesh: {vertex_offset:,} vertices")
    return np.vstack(combined_rock_verts), np.vstack(combined_rock_faces)


def combine_meshes(
    verts1: np.ndarray, faces1: np.ndarray,
    verts2: np.ndarray, faces2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine two meshes into one."""
    if len(verts2) == 0:
        return verts1, faces1
    combined_verts = np.vstack([verts1, verts2])
    offset_faces2 = faces2 + len(verts1)
    combined_faces = np.vstack([faces1, offset_faces2])
    return combined_verts, combined_faces


# =============================================================================
# USD Export
# =============================================================================

def save_to_usd(
    file_path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    root_prim_name: str = "ground",
    mesh_prim_name: str = "ground",
    material_path: Optional[str] = None,
    material_name: Optional[str] = None,
    texture_scale: Optional[Tuple[float, float]] = None,
    collision_approximation: Optional[str] = "none",
    enable_collision: bool = True,
):
    """
    Save mesh to USD file with optional material binding.
    
    The USD structure follows IsaacLab's terrain import conventions:
    - terrain_only.usd: root="/ground", mesh="/ground/ground"
    - rocks_merged.usd: root="/obstacles", mesh="/obstacles/obstacles"
    - terrain_merged.usd: root="/hidden_terrain", mesh="/hidden_terrain/terrain"
    
    When imported at prim_path "/World/terrain/terrain", the internal paths become:
    - "/World/terrain/terrain/ground" (accessible as mesh)
    
    Args:
        file_path: Output USD file path
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (M, 3)
        root_prim_name: Name of root Xform prim (default: "ground")
        mesh_prim_name: Name of mesh prim under root (default: "ground")
        material_path: Path to MDL material file
        material_name: Material name (defaults to MDL filename)
        texture_scale: Texture scale for triplanar projection
        collision_approximation: Collision approximation type ("none", "sdf", "convexHull", etc.)
        enable_collision: Whether to add physics collision APIs (False for raycasting-only meshes)
    """
    if not PXR_AVAILABLE:
        raise ImportError("pxr (USD) is required for USD export")
    
    print(f"  Saving: {os.path.basename(file_path)} ({len(vertices):,} verts, {len(faces):,} faces)")
    
    stage = Usd.Stage.CreateNew(file_path)
    
    # Create root Xform prim
    root_path = f"/{root_prim_name}"
    root_xform = UsdGeom.Xform.Define(stage, root_path)
    # Add identity transform (matches mars terrain structure)
    root_xform.AddTransformOp().Set(Gf.Matrix4d(1.0))
    
    # Create mesh under root
    mesh_path = f"{root_path}/{mesh_prim_name}"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    
    # Use Vt arrays directly for better performance with large meshes
    from pxr import Vt
    verts_f32 = vertices.astype(np.float32)
    faces_flat = faces.flatten().astype(np.int32)
    
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(verts_f32))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces_flat))
    mesh.GetFaceVertexCountsAttr().Set([3] * len(faces))
    
    # Set subdivision scheme to 'none' (critical for physics performance)
    mesh.GetSubdivisionSchemeAttr().Set("none")
    
    # Compute and set extent (bounding box)
    min_pt = verts_f32.min(axis=0)
    max_pt = verts_f32.max(axis=0)
    mesh.GetExtentAttr().Set([
        Gf.Vec3f(float(min_pt[0]), float(min_pt[1]), float(min_pt[2])),
        Gf.Vec3f(float(max_pt[0]), float(max_pt[1]), float(max_pt[2]))
    ])
    
    # Compute face normals for each vertex of each face (face-varying)
    # This improves rendering quality
    v0 = verts_f32[faces[:, 0]]
    v1 = verts_f32[faces[:, 1]]
    v2 = verts_f32[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    face_normals = face_normals / norms
    # Repeat normal for each vertex of the face (face-varying interpolation)
    normals_face_varying = np.repeat(face_normals, 3, axis=0).astype(np.float32)
    mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals_face_varying))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
    
    # Apply material if provided
    if material_path and os.path.exists(material_path):
        if material_name is None:
            material_name = os.path.splitext(os.path.basename(material_path))[0]
        
        # Create materials scope under root
        material_prim_path = f"{root_path}/Looks/{material_name}"
        material = UsdShade.Material.Define(stage, material_prim_path)
        
        mdl_shader_path = material_prim_path + "/Shader"
        mdl_shader = UsdShade.Shader.Define(stage, mdl_shader_path)
        mdl_shader.CreateIdAttr("mdlMaterial")
        mdl_shader.SetSourceAsset(material_path, "mdl")
        mdl_shader.SetSourceAssetSubIdentifier(material_name, "mdl")
        
        # Enable triplanar projection
        project_uvw_input = mdl_shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool)
        project_uvw_input.Set(True)
        
        if texture_scale is not None:
            texture_scale_input = mdl_shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2)
            texture_scale_input.Set(Gf.Vec2f(texture_scale[0], texture_scale[1]))
        
        material_output = material.CreateSurfaceOutput("mdl")
        material_output.ConnectToSource(mdl_shader.ConnectableAPI(), "out")
        
        binding_api = UsdShade.MaterialBindingAPI(mesh)
        binding_api.Bind(material)
    
    # Apply physics collision APIs to mesh (only if collision is enabled)
    if enable_collision:
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        collision_api = UsdPhysics.CollisionAPI.Apply(mesh_prim)
        collision_api.GetCollisionEnabledAttr().Set(True)
        
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh_prim)
        mesh_collision_api.GetApproximationAttr().Set(collision_approximation)
        
        # Helper function to add API schema to prim via Sdf layer
        def add_api_schema(prim_path, schema_name):
            prim_spec = stage.GetRootLayer().GetPrimAtPath(prim_path)
            if prim_spec:
                api_schemas = prim_spec.GetInfo("apiSchemas")
                if api_schemas:
                    schema_list = list(api_schemas.GetAddedOrExplicitItems()) if hasattr(api_schemas, 'GetAddedOrExplicitItems') else []
                else:
                    schema_list = []
                if schema_name not in schema_list:
                    schema_list.append(schema_name)
                    prim_spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(schema_list))
        
        # Add PhysX-specific collision properties for terrain (contactOffset/restOffset)
        # This helps with stable physics simulation
        if collision_approximation == "none":
            # Add PhysxCollisionAPI to the apiSchemas list for these attributes to be recognized
            add_api_schema(mesh_path, "PhysxCollisionAPI")
            mesh_prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(0.04)
            mesh_prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.01)
        
        # Create physics material as a proper Material-typed prim
        physics_mat_path = f"{root_path}/PhysicsMaterial"
        physics_mat = UsdShade.Material.Define(stage, physics_mat_path)
        physics_mat_prim = physics_mat.GetPrim()
        
        # Apply PhysicsMaterialAPI
        physics_mat_api = UsdPhysics.MaterialAPI.Apply(physics_mat_prim)
        physics_mat_api.GetStaticFrictionAttr().Set(0.1)
        physics_mat_api.GetDynamicFrictionAttr().Set(2.0)
        physics_mat_api.GetRestitutionAttr().Set(0.0)
        physics_mat_api.GetDensityAttr().Set(0.0)
        
        # Add PhysxMaterialAPI to apiSchemas for Isaac Sim to recognize the PhysX attributes
        add_api_schema(physics_mat_path, "PhysxMaterialAPI")
        
        # Set PhysX material attributes (matching mars terrain for performance)
        physics_mat_prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
        physics_mat_prim.CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
        # Compliant contact settings for stable simulation
        physics_mat_prim.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(1000000.0)
        physics_mat_prim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float).Set(20000.0)
        physics_mat_prim.CreateAttribute("physxMaterial:improvePatchFriction", Sdf.ValueTypeNames.Bool).Set(False)
        
        # Bind physics material to mesh
        binding_api = UsdShade.MaterialBindingAPI.Apply(mesh_prim)
        binding_api.Bind(
            physics_mat, 
            UsdShade.Tokens.weakerThanDescendants,
            "physics"
        )
    
    # Set root as default prim
    stage.SetDefaultPrim(stage.GetPrimAtPath(root_path))
    stage.GetRootLayer().Save()


# =============================================================================
# Main Generation Function
# =============================================================================

def get_generator_assets_path() -> str:
    """Get path to generator assets (materials, textures, obstacles)."""
    return os.path.dirname(os.path.abspath(__file__))


def generate_and_save_terrain(
    config: TerrainGeneratorConfig,
    output_dir: str,
    copy_assets: bool = True,
) -> str:
    """
    Generate terrain and save to USD files.
    
    Args:
        config: Terrain generation configuration
        output_dir: Directory to save terrain files
        copy_assets: Whether to copy materials/textures to output directory
        
    Returns:
        Path to the output directory
    """
    if not PYMESHLAB_AVAILABLE:
        raise ImportError("pymeshlab is required. Install with: pip install pymeshlab")
    if not PXR_AVAILABLE:
        raise ImportError("pxr (USD) is required. This should be available in Isaac Lab environment.")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating terrain to: {output_dir}")
    
    # Copy assets if needed
    assets_path = get_generator_assets_path()
    if copy_assets:
        for folder in ["materials", "textures"]:
            src = os.path.join(assets_path, folder)
            dst = os.path.join(output_dir, folder)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copytree(src, dst)
                print(f"  Copied {folder}/")
    
    # Generate terrain
    terrain = generate_terrain(config)
    
    # Reduce mesh polygons
    print("\nReducing mesh polygons...")
    terrain_verts, terrain_faces = reduce_mesh_polygons(
        terrain.height_field_raw,
        terrain.horizontal_scale,
        terrain.vertical_scale,
        target_vertices=config.target_vertices
    )
    
    # Load rock models and generate rocks
    rock_models = []
    obstacle_assets_dir = os.path.join(assets_path, "obstacle_assets")
    if os.path.exists(obstacle_assets_dir):
        for rock_file in ["rock1.obj", "rock2.obj", "rock3.obj"]:
            rock_path = os.path.join(obstacle_assets_dir, rock_file)
            if os.path.exists(rock_path):
                rock_models.append(load_obj(rock_path))
        print(f"\nLoaded {len(rock_models)} rock models")
    
    print("\nGenerating rock layer...")
    rock_verts, rock_faces = generate_rock_mesh(
        terrain_verts=terrain_verts,
        rock_models=rock_models,
        num_rocks=config.rock_config.num_rocks,
        scale_range=config.rock_config.scale_range,
        embed_percentage=config.rock_config.embed_percentage,
    )
    
    # Combine meshes
    combined_verts, combined_faces = combine_meshes(
        terrain_verts, terrain_faces,
        rock_verts, rock_faces
    )
    
    # Save USD files
    print("\nSaving USD files...")
    
    # Material paths (relative to output dir)
    soil_material = os.path.join(output_dir, "materials", "Soil_Rocky.mdl")
    rock_material = os.path.join(output_dir, "materials", "Fieldstone.mdl")
    
    # terrain_only.usd - base terrain for physics
    # Structure: /ground/ground (matches existing mars terrain)
    # Using triangle mesh collision ("none") like mars terrain for best performance.
    # PhysX will cook the collision mesh at first simulation step if not pre-cooked.
    # Pre-cooked data (physxCookedData:triangleMesh:buffer) would make loading faster,
    # but can only be generated inside Isaac Sim.
    save_to_usd(
        os.path.join(output_dir, "terrain_only.usd"),
        terrain_verts, terrain_faces,
        root_prim_name="ground",
        mesh_prim_name="ground",
        material_path=soil_material,
        material_name="Soil_Rocky",
        texture_scale=(0.3, 0.3),
        collision_approximation="none",  # Triangle mesh like mars terrain
    )
    
    # rocks_merged.usd - obstacles for collision
    # Structure: /obstacles/obstacles (matches existing mars terrain)
    # Always create this file (even if empty) since terrain registry expects it
    if len(rock_verts) > 0:
        save_to_usd(
            os.path.join(output_dir, "rocks_merged.usd"),
            rock_verts, rock_faces,
            root_prim_name="obstacles",
            mesh_prim_name="obstacles",
            material_path=rock_material,
            material_name="Fieldstone",
            collision_approximation="sdf",  # SDF for rocks (faster physics)
        )
    else:
        # Create minimal placeholder with single tiny triangle (invisible)
        placeholder_verts = np.array([[0, 0, -1000], [0.001, 0, -1000], [0, 0.001, -1000]], dtype=np.float32)
        placeholder_faces = np.array([[0, 1, 2]], dtype=np.uint32)
        save_to_usd(
            os.path.join(output_dir, "rocks_merged.usd"),
            placeholder_verts, placeholder_faces,
            root_prim_name="obstacles",
            mesh_prim_name="obstacles",
            collision_approximation="sdf",
        )
    
    # terrain_merged.usd - combined terrain + rocks for raycasting (hidden terrain)
    # Structure: /hidden_terrain/terrain (matches existing mars terrain)
    # NOTE: No collision enabled - this is only used for raycasting, not physics
    save_to_usd(
        os.path.join(output_dir, "terrain_merged.usd"),
        combined_verts, combined_faces,
        root_prim_name="hidden_terrain",
        mesh_prim_name="terrain",
        enable_collision=False,  # Raycasting only, no physics
    )
    
    print(f"\n✅ Terrain generated successfully: {output_dir}")
    return output_dir


# =============================================================================
# Preset Configurations
# =============================================================================

def get_default_layers(seed: Optional[int] = None) -> List[TerrainLayer]:
    """Get default terrain layers for Mars-like terrain."""
    if seed is None:
        seed = random.randint(1, 100000)
    
    return [
        TerrainLayer(
            name="Base Rolling Hills",
            num_features=300,
            radius_m_range=(8, 15),
            height_m_range=(-5, 5),
            kernel_type="gaussian",
            kernel_params={"sigma": 0.4, "normalized": True},
            seed=seed,
        ),
        TerrainLayer(
            name="Small Bumps",
            num_features=3000,
            radius_m_range=(1, 6),
            height_m_range=(0.3, 1.0),
            kernel_type="gaussian",
            kernel_params={"sigma": 0.3, "normalized": True},
            seed=seed,
        ),
        TerrainLayer(
            name="Large Hills",
            num_features=20,
            radius_m_range=(20, 60),
            height_m_range=(-15, 15),
            kernel_type="gaussian",
            kernel_params={"sigma": 0.8, "normalized": True},
            seed=seed,
        ),
    ]


def create_default_config(
    name: str = "generated",
    seed: Optional[int] = None,
    width: int = 200,
    length: int = 200,
    num_rocks: int = 2500,
) -> TerrainGeneratorConfig:
    """Create a default terrain generation configuration."""
    return TerrainGeneratorConfig(
        name=name,
        width=width,
        length=length,
        horizontal_scale=0.05,
        vertical_scale=0.05,
        target_vertices=200000,  # Target for decimation, ~200k verts after processing
        layers=get_default_layers(seed),
        rock_config=RockConfig(
            num_rocks=num_rocks,
            scale_range=(0.05, 0.25),
            embed_percentage=0.25,
        ),
        seed=seed,
    )
