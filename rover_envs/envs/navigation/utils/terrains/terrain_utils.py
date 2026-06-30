"""
Professional Terrain Management Utilities for Robotics Simulation

This module provides comprehensive terrain analysis and management capabilities for 
robotics simulation environments, particularly focused on rover navigation scenarios.

Key Features:
    - High-performance heightmap generation from 3D mesh data
    - Gradient-based terrain difficulty analysis
    - Rock/obstacle detection and safety zone computation
    - Intelligent spawn location generation with safety constraints
    - Multi-device support (CPU/CUDA) for scalable performance
    - Debug visualization tools for terrain analysis
    - Robust USD file loading with Isaac Sim integration

Classes:
    HeightmapManager: Core heightmap generation and height query functionality
    TerrainManager: High-level terrain analysis and spawn generation
    DebugVisualizer: Visualization tools for terrain analysis and debugging

Example:
    >>> # Initialize terrain manager
    >>> terrain = TerrainManager(
    ...     num_envs=100,
    ...     device='cuda',
    ...     debug_mode=True,
    ...     terrain_usd_path="path/to/terrain.usd"
    ... )
    >>> 
    >>> # Get spawn locations
    >>> spawn_positions = terrain.spawn_locations
    >>> 
    >>> # Query height at specific positions
    >>> positions = torch.tensor([[10.0, 15.0], [20.0, 25.0]], device='cuda')
    >>> heights = terrain._heightmap_manager.get_height_at(positions)

Author: RLRoverLab Team
Version: 2.0.0
"""

# from isaaclab.markers.visualization_markers import VisualizationMarkersCfg, VisualizationMarkers
# import isaaclab.sim as sim_utils
import os
from typing import Tuple, Optional, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pymeshlab
import torch
from termcolor import colored

from rover_envs.envs.navigation.utils.terrains.risk_map import (
    ObstacleRiskMap,
    ObstacleRiskMapCfg,
)
from rover_envs.envs.navigation.utils.terrains.usd_utils import get_triangles_and_vertices_from_prim_standalone, isaacsim_available
# Try to import Isaac Sim dependencies for runtime, fallback for debugging

if isaacsim_available():
    from rover_envs.envs.navigation.utils.terrains.usd_utils import get_triangles_and_vertices_from_prim




class HeightmapManager:
    """
    A professional heightmap manager for converting 3D mesh data to 2D heightmaps.
    
    This class handles the conversion of 3D mesh vertices and faces into a structured
    2D heightmap representation, enabling efficient height queries and terrain analysis.
    Supports both CPU and CUDA tensor operations for performance optimization.
    
    Attributes:
        resolution_in_m (float): The resolution of the heightmap in meters per pixel.
        device (str): The device to use for tensor operations ('cpu', 'cuda', or 'cuda:0').
        heightmap (np.ndarray): The 2D heightmap as a numpy array.
        min_x (float): Minimum X coordinate of the heightmap bounds.
        min_y (float): Minimum Y coordinate of the heightmap bounds.
        max_x (float): Maximum X coordinate of the heightmap bounds.
        max_y (float): Maximum Y coordinate of the heightmap bounds.
        heightmap_tensor (torch.Tensor): GPU/CPU tensor version of the heightmap.
        offset_tensor (torch.Tensor): Offset tensor for coordinate transformations.
    """

    def __init__(self, resolution_in_m: float, vertices: np.ndarray, faces: np.ndarray, device: str = 'cpu') -> None:
        """
        Initialize the HeightmapManager with mesh data and configuration.
        
        Args:
            resolution_in_m: The resolution of the heightmap in meters per pixel.
            vertices: 3D vertex array of shape (N, 3) containing [x, y, z] coordinates.
            faces: Face indices array of shape (M, 3) referencing vertices.
            device: Device for tensor operations ('cpu', 'cuda', or 'cuda:0').
            
        Raises:
            ValueError: If vertices or faces arrays have invalid shapes.
            RuntimeError: If CUDA is requested but not available.
        """
        if len(vertices.shape) != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Vertices must have shape (N, 3), got {vertices.shape}")
        if len(faces.shape) != 2 or faces.shape[1] != 3:
            raise ValueError(f"Faces must have shape (M, 3), got {faces.shape}")
        
        self.resolution_in_m = resolution_in_m
        self.device = device
        
        # Generate heightmap from mesh data
        self.heightmap, self.min_x, self.min_y, self.max_x, self.max_y = self._mesh_to_heightmap(vertices, faces)
        
        # Initialize tensors based on device
        self._initialize_tensors()
    
    def _initialize_tensors(self) -> None:
        """Initialize heightmap and offset tensors on the specified device."""
        if self.device in ['cuda', 'cuda:0']:
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA requested but not available. Falling back to CPU.")
            self.heightmap_tensor = torch.from_numpy(self.heightmap).cuda()
            self.offset_tensor = torch.tensor([self.min_x, self.min_y], dtype=torch.float32).cuda()
        else:
            self.heightmap_tensor = torch.from_numpy(self.heightmap)
            self.offset_tensor = torch.tensor([self.min_x, self.min_y], dtype=torch.float32)

    def _mesh_to_heightmap(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
        """
        Convert 3D mesh data to a 2D heightmap representation.
        
        This method projects 3D triangular mesh faces onto a 2D grid, recording the maximum
        height at each grid cell. The resulting heightmap provides an efficient way to
        query terrain heights at arbitrary 2D positions.
        
        Args:
            vertices: 3D vertex coordinates array of shape (N, 3).
            faces: Triangle face indices array of shape (M, 3).
            
        Returns:
            A tuple containing:
                - heightmap (np.ndarray): 2D height grid of shape (height, width).
                - min_x (float): Minimum X coordinate of the heightmap bounds.
                - min_y (float): Minimum Y coordinate of the heightmap bounds.
                - max_x (float): Maximum X coordinate of the heightmap bounds.
                - max_y (float): Maximum Y coordinate of the heightmap bounds.
                
        Note:
            A border margin of 1.0 meter is applied to prevent edge effects.
            Grid cells without mesh coverage are initialized to -99.0 meters.
        """
        # Apply border margin to prevent edge effects
        border_margin = 1.0
        
        # Calculate bounding box with margin
        min_x, min_y, _ = np.min(vertices, axis=0) + border_margin
        max_x, max_y, _ = np.max(vertices, axis=0) - border_margin

        # Calculate grid dimensions
        grid_size_x = (max_x - min_x) / self.resolution_in_m
        grid_size_y = (max_y - min_y) / self.resolution_in_m
        
        # Grid dimensions (add 1 for inclusive bounds)
        grid_width = int(grid_size_x + 1)
        grid_height = int(grid_size_y + 1)

        # Initialize heightmap with sentinel value for uncovered areas
        heightmap = np.full((grid_height, grid_width), -99.0, dtype=np.float32)

        # Calculate cell sizes
        cell_size_x = (max_x - min_x) / grid_size_x
        cell_size_y = (max_y - min_y) / grid_size_y

        if len(faces) > 0:
            # Vectorized processing of all triangles
            face_vertices = vertices[faces]
            
            # Extract coordinate components
            x_coords = face_vertices[:, :, 0]
            y_coords = face_vertices[:, :, 1]
            z_coords = face_vertices[:, :, 2]
            
            # Calculate bounding boxes for all triangles
            min_x_tri = np.min(x_coords, axis=1)
            max_x_tri = np.max(x_coords, axis=1)
            min_y_tri = np.min(y_coords, axis=1)
            max_y_tri = np.max(y_coords, axis=1)
            max_z_tri = np.max(z_coords, axis=1)
            
            # Convert world coordinates to grid coordinates
            min_i = np.maximum(0, ((min_x_tri - min_x) / cell_size_x).astype(int))
            max_i = np.minimum(grid_width - 1, ((max_x_tri - min_x) / cell_size_x).astype(int))
            min_j = np.maximum(0, ((min_y_tri - min_y) / cell_size_y).astype(int))
            max_j = np.minimum(grid_height - 1, ((max_y_tri - min_y) / cell_size_y).astype(int))
            
            # Project triangles onto heightmap
            for idx in range(len(faces)):
                i_range = max_i[idx] - min_i[idx] + 1
                j_range = max_j[idx] - min_j[idx] + 1
                
                if i_range > 0 and j_range > 0:
                    # Update heightmap with maximum height in each cell
                    heightmap[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1] = np.maximum(
                        heightmap[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1],
                        max_z_tri[idx]
                    )

        return heightmap, min_x, min_y, max_x, max_y

    def get_height_at(self, position: torch.Tensor) -> torch.Tensor:
        """
        Query the height at specified 2D positions using bilinear interpolation.
        
        This method efficiently retrieves terrain heights at arbitrary 2D world coordinates
        by mapping them to the heightmap grid and performing lookups. Input positions are
        automatically clamped to the heightmap bounds to prevent out-of-bounds access.
        
        Args:
            position: 2D world coordinates tensor of shape (N, 2) containing [x, y] positions.
                     Must be on the same device as the HeightmapManager.
                     
        Returns:
            Height values tensor of shape (N,) corresponding to each input position.
            
        Raises:
            RuntimeError: If position tensor is not on the same device as the heightmap.
            
        Example:
            >>> positions = torch.tensor([[10.0, 15.0], [20.0, 25.0]], device='cuda')
            >>> heights = heightmap_manager.get_height_at(positions)
            >>> print(heights.shape)  # torch.Size([2])
            
        Note:
            Positions outside the heightmap bounds are clamped to the nearest valid grid cell.
            This prevents extrapolation and ensures consistent behavior at terrain edges.
        """
        if position.device != self.heightmap_tensor.device:
            raise RuntimeError(f"Position tensor device ({position.device}) must match "
                             f"heightmap device ({self.heightmap_tensor.device})")
        
        if position.shape[-1] != 2:
            raise ValueError(f"Position tensor must have shape (..., 2), got {position.shape}")
        
        # Transform world coordinates to grid coordinates
        scaled_position = position / self.resolution_in_m + self.offset_tensor
        
        # Convert to integer grid indices
        grid_cell = scaled_position.long()

        # Clamp indices to valid heightmap bounds
        grid_cell[:, 0] = torch.clamp(grid_cell[:, 0], 0, self.heightmap_tensor.shape[1] - 1)
        grid_cell[:, 1] = torch.clamp(grid_cell[:, 1], 0, self.heightmap_tensor.shape[0] - 1)

        # Retrieve heights at the specified grid positions
        return self.heightmap_tensor[grid_cell[:, 1], grid_cell[:, 0]]
    
    def get_heightmap_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get the world coordinate bounds of the heightmap.
        
        Returns:
            A tuple containing (min_x, min_y, max_x, max_y) in world coordinates.
        """
        return self.min_x, self.min_y, self.max_x, self.max_y
    
    def get_heightmap_shape(self) -> Tuple[int, int]:
        """
        Get the dimensions of the heightmap grid.
        
        Returns:
            A tuple containing (height, width) of the heightmap in grid cells.
        """
        return self.heightmap.shape
    
    def is_position_valid(self, position: torch.Tensor) -> torch.Tensor:
        """
        Check if positions are within the valid heightmap bounds.
        
        Args:
            position: 2D world coordinates tensor of shape (N, 2).
            
        Returns:
            Boolean tensor of shape (N,) indicating valid positions.
        """
        within_x_bounds = (position[:, 0] >= self.min_x) & (position[:, 0] <= self.max_x)
        within_y_bounds = (position[:, 1] >= self.min_y) & (position[:, 1] <= self.max_y)
        return within_x_bounds & within_y_bounds

class TerrainManager:
    """
    Professional terrain management system for robotics simulation environments.
    
    This class provides comprehensive terrain analysis capabilities including heightmap generation,
    gradient analysis, obstacle detection, and safe spawn location generation. It supports both
    Isaac Sim runtime environments and standalone debugging with USD files.
    
    Features:
        - Heightmap generation from 3D mesh data
        - Gradient-based terrain difficulty analysis
        - Rock/obstacle detection and safety zone generation
        - Intelligent spawn location generation
        - Multi-device support (CPU/CUDA)
        - Debug visualization capabilities
    
    Attributes:
        num_envs (int): Number of simulation environments.
        device (str): Computation device ('cpu', 'cuda', or 'cuda:0').
        debug_mode (bool): Whether running in debug mode with USD files.
        resolution_in_m (float): Heightmap resolution in meters per pixel.
        gradient_threshold (float): Threshold for steep terrain detection.
        spawn_locations (torch.Tensor): Pre-generated safe spawn positions.
        rock_mask (np.ndarray): Binary mask indicating rock locations.
        gradient_mask (np.ndarray): Binary mask indicating steep terrain.
        safe_rock_mask (np.ndarray): Dilated rock mask for safety margins.
        safe_gradient_mask (np.ndarray): Dilated gradient mask for safety margins.
    """

    def __init__(self, 
                 num_envs: int, 
                 device: str, 
                 debug_mode: bool = False, 
                 terrain_usd_path: Optional[str] = None, 
                 rock_usd_path: Optional[str] = None,
                 safety_margin: float = 2.0,
                 num_spawn_locations: int = 4096,
                 target_distance_to_boundary: float = 7.0,
                 spawn_distance_to_boundary: float = 10.0,
                 safety_margin_to_obstacles: float = 2.0,
                 resolution_in_m: float = 0.05,
                 gradient_threshold: float = 0.35,
                 obstacle_risk_cfg: Optional[ObstacleRiskMapCfg] = None,
                 ) -> None:
        """
        Initialize the TerrainManager with specified configuration.
        
        Args:
            num_envs: Number of simulation environments requiring spawn locations.
            device: Device for tensor computations ('cpu', 'cuda', or 'cuda:0').
            debug_mode: Enable debug mode for standalone USD file loading.
            terrain_usd_path: Path to terrain USD file (debug mode only).
            rock_usd_path: Path to rock/obstacle USD file (debug mode only).
            safety_margin: Safety margin in meters around obstacles and steep terrain.
            
        Raises:
            FileNotFoundError: If USD files are not found in debug mode.
            RuntimeError: If mesh loading fails and fallback is unsuccessful.
        """
        ## Initialize parameters - general configuration
        self.num_envs = num_envs
        self.device = device
        self.debug_mode = debug_mode

        # Initialize parameters - terrain configuration
        self.terrain_usd_path = terrain_usd_path
        self.rock_usd_path = rock_usd_path
        self.resolution_in_m = resolution_in_m # resolution for heightmap generation
        self.gradient_threshold = gradient_threshold  # threshold for steep terrain detection
        self.obstacle_risk_cfg = obstacle_risk_cfg or ObstacleRiskMapCfg()

        # Initialize parameters - spawn generation configuration
        self.safety_margin = safety_margin
        self.num_spawn_locations = num_spawn_locations
        self.spawn_distance_to_boundary = spawn_distance_to_boundary
        self.safety_margin_to_obstacles = safety_margin_to_obstacles

        # Initialize parameters - target distance
        self.target_distance_to_boundary = target_distance_to_boundary

        # Check if running in Isaac Sim environment or debug mode
        self.debug_mode = debug_mode or not isaacsim_available()
        
        
        if isaacsim_available() and not debug_mode:
            # Isaac Sim runtime mode - use default terrain paths
            terrain_path = "/World/terrain/terrain/ground"
            rock_mesh_path = "/World/terrain/obstacles/obstacles"
        else:
            # Debug mode - use provided USD paths or default assets
            if terrain_usd_path:
                terrain_path = terrain_usd_path
                rock_mesh_path = rock_usd_path
            else:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                base_path = os.path.join(dir_path, "..", "..", "..", "..", "assets", "terrains", "mars", "terrain1")
                terrain_path = os.path.join(base_path, "terrain_only.usd")
                rock_mesh_path = os.path.join(base_path, "rocks_merged.usd")

        self.meshes = {
            "terrain": terrain_path,
            "rock": rock_mesh_path
        }

        # Terrain Parameters
        self.heightmap = None

        # Load Terrain (terrain only, without rocks)
        self.log("Getting triangles and vertices from terrain USD file", level='info', block=False)
        #print("Getting triangles and vertices from terrain USD file")
        terrain_vertices, terrain_faces = self.get_mesh(self.meshes["terrain"])
        
        # Load rocks if available and combine with terrain for spawn height queries
        rock_vertices = None
        rock_faces = None
        
        # First check if rock prim exists (when using Isaac Sim)
        rocks_available = True
        if isaacsim_available():
            from rover_envs.envs.navigation.utils.terrains.usd_utils import check_prim_exists
            if not check_prim_exists(self.meshes["rock"]):
                self.log(f"No rock obstacles found at {self.meshes['rock']} - using terrain-only mode", level='warning', block=False)
                rocks_available = False
        
        if rocks_available:
            try:
                self.log("Loading rock obstacles from USD file...", level='debug', block=False)
                rock_vertices, rock_faces = self.get_mesh(self.meshes["rock"])
                
                # Combine terrain and rocks for complete heightmap
                self.log("Combining terrain and rock meshes...", level='debug', block=False)
                combined_vertices = np.vstack([terrain_vertices, rock_vertices])
                combined_faces = np.vstack([terrain_faces, rock_faces + len(terrain_vertices)])
                
                # Create combined heightmap manager (for spawn height queries)
                self.log("Generating combined heightmap with obstacles", level='debug', block=False)
                self._heightmap_manager = HeightmapManager(self.resolution_in_m, combined_vertices, combined_faces, device)
                
                # Create terrain-only heightmap with SAME BOUNDS as combined heightmap
                self.log("Generating terrain-only heightmap with matched bounds", level='debug', block=False)
                self.terrain_only_heightmap_manager = HeightmapManager(self.resolution_in_m, terrain_vertices, terrain_faces, device)
                
                # Resize terrain-only heightmap to match combined heightmap dimensions
                self.terrain_only_heightmap_manager = self.resize_terrain_heightmap_to_match_combined(
                    terrain_vertices, terrain_faces, self._heightmap_manager
                )
                self.log("✓ Successfully loaded terrain and rock obstacles", level='success', block=False)

            except Exception as e:
                self.log(f"Warning: Could not load rock obstacles ({e}). Continuing with terrain-only mode.", level='warning', block=False)
                rocks_available = False
        
        if not rocks_available:
            # Use terrain-only heightmap for everything
            self.log("Using terrain-only mode (no obstacles)", level='warning', block=False)
            self._heightmap_manager = HeightmapManager(self.resolution_in_m, terrain_vertices, terrain_faces, device)
            self.terrain_only_heightmap_manager = self._heightmap_manager

        # Generate Rock Mask if rocks are available
        if rock_vertices is not None:
            self.log("Generating rock obstacle mask from mesh data", level='info', block=False)
            self.rock_mask, self.safe_rock_mask = self.project_rocks_to_heightmap(rock_vertices, rock_faces)
        else:
            self.log("No rock obstacles present - using clear obstacle masks", level='warning', block=False)
            # Create empty rock masks
            height, width = self._heightmap_manager.heightmap.shape
            self.rock_mask = np.zeros((height, width), dtype=np.int32)
            self.safe_rock_mask = np.zeros((height, width), dtype=np.int32)

        self.obstacle_risk_map = self.create_obstacle_risk_map(rock_vertices, rock_faces)

        # Generate Gradient Mask using terrain-only heightmap
        self.log("Generating gradient mask from terrain-only heightmap", level='info', block=False)
        self.gradient_mask, self.safe_gradient_mask = self.compute_gradient_masks(
            self.terrain_only_heightmap_manager.heightmap, self.gradient_threshold)

        # Combine rock and gradient masks for spawn generation
        self.log("Combining rock and gradient masks for spawn generation", level='debug', block=False)
        combined_safe_mask = np.logical_or(self.safe_rock_mask, self.safe_gradient_mask).astype(np.int32)

        # Generate Spawn Locations
        self.spawn_locations = self.random_rover_spawns(
            safe_mask=combined_safe_mask,
            heightmap=self._heightmap_manager.heightmap,
            n_spawns=num_spawn_locations,#num_envs*2 if num_envs > 100 else 200,
            border_offset=25.0,
            seed=12345)
        
        if str(self.device).startswith("cuda"):
            self.spawn_locations = torch.from_numpy(self.spawn_locations).cuda()
            self.safe_rock_mask_tensor = torch.from_numpy(self.safe_rock_mask).cuda().unsqueeze(-1)
        else:
            self.spawn_locations = torch.from_numpy(self.spawn_locations)
            self.safe_rock_mask_tensor = torch.from_numpy(self.safe_rock_mask).unsqueeze(-1)
        # Summary of terrain initialization
        obstacle_mode = "with obstacles" if rock_vertices is not None else "terrain-only"
        spawn_count = len(self.spawn_locations)
        self.log(
            "✓ Terrain Initialization Complete\n"
            f"→ Obstacle Mode: {obstacle_mode}\n"
            f"→ Spawn Locations Generated: {spawn_count}",
            level='success', block=True
        )

    def log(self, message, level='info', block=False):
        """Simple logger with color-coded levels and optional block formatting."""
        if not self.debug_mode and level not in ['success', 'error', 'warning', 'info']:
            return

        colors = {
            'info': 'white',
            'debug': 'cyan',
            'success': 'green',
            'warning': 'yellow',
            'error': 'red'
        }

        if block:
            border = "=" * 60
            msg = f"{border}\n{message}\n{border}"
        else:
            msg = message
        print(colored(msg, colors.get(level, 'white')))

    def get_mesh(self, prim_path: str = "/") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load mesh data from USD file or Isaac Sim prim.
        
        This method attempts to load mesh data using Isaac Sim runtime capabilities first,
        then falls back to standalone USD loading for debug environments. The loaded mesh
        is processed through PyMeshLab.
        
        Args:
            prim_path: Prim path when using Isaac Sim, or USD file path in standalone(debug) mode.
            
        Returns:
            A tuple containing:
                - vertices (np.ndarray): Vertex coordinates of shape (N, 3).
                - faces (np.ndarray): Triangle face indices of shape (M, 3).
        """
        faces, vertices = get_triangles_and_vertices_from_prim(prim_path) if isaacsim_available() \
            else get_triangles_and_vertices_from_prim_standalone(prim_path)
        # Process mesh through PyMeshLab for consistency and optimization
        mesh = pymeshlab.Mesh(vertices, faces)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(mesh)

        # Extract processed mesh data
        processed_mesh = ms.current_mesh()
        vertices = processed_mesh.vertex_matrix().astype('float32')
        faces = processed_mesh.face_matrix().astype('uint32')

        return vertices, faces

    def check_if_target_is_valid(
            self,
            env_ids: torch.Tensor,
            target_positions: torch.Tensor,
            device: str = "cuda:0"
    ) -> Tuple[torch.Tensor, int]:
        """
        Validate target positions against terrain safety constraints.
        
        This method checks if target positions are located in safe areas (not on rocks
        or obstacles) and returns the environment IDs that require reset due to
        unsafe target placement. Additionally checks if targets are within valid boundary
        distance from the map edges.
        
        Note: Targets are allowed on steep terrain - only rock/obstacle constraints apply.
        Rovers can navigate to targets on steep terrain, but shouldn't spawn on them.
        
        Args:
            env_ids: Environment IDs to check, tensor of shape (N,).
            target_positions: Target positions to validate, tensor of shape (N, 2 or 3).
            device: Device for tensor operations (legacy parameter, uses manager's device).
            
        Returns:
            A tuple containing:
                - env_ids_to_reset (torch.Tensor): Environment IDs requiring reset.
                - num_resets (int): Number of environments requiring reset.
                
        Note:
            Only uses the first 2 dimensions (x, y) of target_positions for validation.
            The safety check is performed against rock safety masks only (not gradient masks)
            and boundary distance constraints using target_distance_to_boundary parameter.
        """
        # Check 1: Boundary distance constraints
        # Ensure targets are not too close to the map boundaries
        x_coords = target_positions[:, 0]
        y_coords = target_positions[:, 1]
        
        # Get heightmap bounds
        min_x, min_y, max_x, max_y = self._heightmap_manager.get_heightmap_bounds()
        
        # Check if targets are within valid boundary distance
        boundary_margin = self.target_distance_to_boundary
        boundary_violations = (
            (x_coords < min_x + boundary_margin) |
            (x_coords > max_x - boundary_margin) |
            (y_coords < min_y + boundary_margin) |
            (y_coords > max_y - boundary_margin)
        )
        
        # Check 2: Rock/obstacle safety constraints (targets allowed on steep terrain)
        # Transform world coordinates to grid coordinates
        scaled_position = target_positions[:, 0:2] / \
            self._heightmap_manager.resolution_in_m + self._heightmap_manager.offset_tensor
        
        # Convert to integer grid indices and clamp to valid bounds
        grid_cell = scaled_position.long()
        grid_cell[:, 0] = torch.clamp(grid_cell[:, 0], 0, self._heightmap_manager.heightmap_tensor.shape[1] - 1)
        grid_cell[:, 1] = torch.clamp(grid_cell[:, 1], 0, self._heightmap_manager.heightmap_tensor.shape[0] - 1)

        # Check rock safety mask only (targets can be on steep terrain, just not on rocks): 1 indicates unsafe areas requiring reset
        safety_violations = torch.where(self.safe_rock_mask_tensor[grid_cell[:, 1], grid_cell[:, 0]] == 1, 1, 0).squeeze(-1)
        
        # Combine both violation checks: reset if either boundary or safety is violated
        combined_violations = boundary_violations | (safety_violations == 1)
        env_ids_to_reset = env_ids[combined_violations]
        
        return env_ids_to_reset, len(env_ids_to_reset)

    def create_obstacle_risk_map(
            self,
            rock_vertices: Optional[np.ndarray],
            rock_faces: Optional[np.ndarray],
    ) -> ObstacleRiskMap:
        """Create the smooth obstacle risk map used by conservative teacher rewards."""
        heightmap_manager = self._heightmap_manager
        bounds = (
            heightmap_manager.min_x,
            heightmap_manager.min_y,
            heightmap_manager.max_x,
            heightmap_manager.max_y,
        )

        if rock_vertices is None or rock_faces is None:
            risk_map = ObstacleRiskMap.empty(
                heightmap_shape=heightmap_manager.heightmap.shape,
                bounds=bounds,
                device=self.device,
                cfg=self.obstacle_risk_cfg,
            )
        else:
            risk_map = ObstacleRiskMap.from_rock_mesh(
                rock_vertices=rock_vertices,
                rock_faces=rock_faces,
                heightmap_shape=heightmap_manager.heightmap.shape,
                bounds=bounds,
                device=self.device,
                cfg=self.obstacle_risk_cfg,
            )

        stats = risk_map.statistics()
        self.log(
            "Obstacle risk map created\n"
            f"Obstacle cells: {stats['obstacle_cells']} ({stats['obstacle_percentage']:.2f}%)\n"
            f"Cost range: [{stats['cost_min']:.4f}, {stats['cost_max']:.4f}]",
            level='debug',
            block=True,
        )
        return risk_map

    def project_rocks_to_heightmap(self, rock_vertices: np.ndarray, rock_faces: np.ndarray, safety_margin: float = 2.0):
        """Project rock mesh triangles onto XY plane to create rock masks"""
        import cv2
        from scipy import ndimage
        
        # Get heightmap dimensions and bounds from the combined heightmap
        height, width = self._heightmap_manager.heightmap.shape
        min_x = self._heightmap_manager.min_x
        min_y = self._heightmap_manager.min_y
        
        # Initialize rock mask
        rock_mask = np.zeros((height, width), dtype=np.uint8)
        
        self.log(f'Projecting {len(rock_faces)} rock triangles onto XY plane', level='debug', block=False)
        
        if len(rock_faces) > 0:
            # Get all triangle vertices and project to 2D
            face_vertices = rock_vertices[rock_faces]
            
            # Extract XY coordinates
            x_coords = face_vertices[:, :, 0]
            y_coords = face_vertices[:, :, 1]
            
            # Find bounding box for each triangle
            min_x_tri = np.min(x_coords, axis=1)
            max_x_tri = np.max(x_coords, axis=1)
            min_y_tri = np.min(y_coords, axis=1)
            max_y_tri = np.max(y_coords, axis=1)
            
            # Convert to grid coordinates
            cell_size_x = (self._heightmap_manager.max_x - min_x) / (width - 1)
            cell_size_y = (self._heightmap_manager.max_y - min_y) / (height - 1)
            
            min_i = np.maximum(0, ((min_x_tri - min_x) / cell_size_x).astype(int))
            max_i = np.minimum(width - 1, ((max_x_tri - min_x) / cell_size_x).astype(int))
            min_j = np.maximum(0, ((min_y_tri - min_y) / cell_size_y).astype(int))
            max_j = np.minimum(height - 1, ((max_y_tri - min_y) / cell_size_y).astype(int))
            
            # Mark triangles on the mask
            for idx in range(len(rock_faces)):
                i_range = max_i[idx] - min_i[idx] + 1
                j_range = max_j[idx] - min_j[idx] + 1
                
                if i_range > 0 and j_range > 0:
                    rock_mask[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1] = 1
        
        self.log(f'Rocks projected onto heightmap: {np.sum(rock_mask)} cells marked as rocks', level='debug', block=True)
        self.log("Rock projection completed. Applying morphological operations...", level='debug', block=False)

        # Apply morphological operations to clean up the mask
        kernel_small = np.ones((3, 3), np.uint8)
        rock_mask = cv2.morphologyEx(rock_mask, cv2.MORPH_CLOSE, kernel_small)
        rock_mask = ndimage.binary_fill_holes(rock_mask).astype(np.uint8)
        
        # Remove very small isolated regions
        kernel_open = np.ones((5, 5), np.uint8)
        rock_mask = cv2.morphologyEx(rock_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Dilate slightly to account for rock boundaries
        kernel_dilate = np.ones((7, 7), np.uint8)
        rock_mask = cv2.dilate(rock_mask, kernel_dilate, iterations=1)
        
        # Create safety margin for spawn locations
        # This creates a larger exclusion zone around rocks for safer navigation
        safety_margin_size = int(safety_margin / self.resolution_in_m)  # 2 meter safety margin
        kernel_safety = np.ones((safety_margin_size, safety_margin_size), np.uint8)
        safe_rock_mask = cv2.dilate(rock_mask, kernel_safety, iterations=1)

        self.log(
            f"Rock mask created: {np.sum(rock_mask)} cells marked as rocks\n"
            f"Rock safety mask created: {np.sum(safe_rock_mask)} cells marked as unsafe",
            level='debug', block=True
        )

        return rock_mask.astype(np.int32), safe_rock_mask.astype(np.int32)

    def compute_gradient_masks(self, heightmap, threshold=0.1, safety_margin=2.0):
        """
        Compute gradient masks for terrain steepness (similar to find_rocks_in_heightmap but only for terrain)
        """
        import cv2
        from scipy import ndimage
        from scipy.signal import convolve2d

        # Sobel operators for gradient in x and y directions
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        # Compute the gradient components
        grad_x = convolve2d(heightmap, sobel_x, mode='same', boundary='wrap')
        grad_y = convolve2d(heightmap, sobel_y, mode='same', boundary='wrap')

        # Compute the overall gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Create mask for steep areas
        gradient_mask = np.zeros_like(heightmap, dtype=np.int32)
        gradient_mask[grad_magnitude > threshold] = 1
        gradient_mask = gradient_mask.astype(np.uint8)
        
        # Apply morphological operations to clean up the mask
        kernel_small = np.ones((3, 3), np.uint8)
        gradient_mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_CLOSE, kernel_small)
        gradient_mask = ndimage.binary_fill_holes(gradient_mask).astype(np.uint8)
        
        # Remove very small isolated regions
        kernel_open = np.ones((5, 5), np.uint8)
        gradient_mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Dilate slightly to account for gradient boundaries
        kernel_dilate = np.ones((7, 7), np.uint8)
        gradient_mask = cv2.dilate(gradient_mask, kernel_dilate, iterations=1)
        
        # Create safety margin for gradient areas
        # This creates a larger exclusion zone around steep areas for safer navigation
        safety_margin_size = int(safety_margin / self.resolution_in_m)  # default 2.0 meter safety margin
        kernel_safety = np.ones((safety_margin_size, safety_margin_size), np.uint8)
        safe_gradient_mask = cv2.dilate(gradient_mask, kernel_safety, iterations=1)

        self.log(
            f"Gradient mask created: {np.sum(gradient_mask)} cells marked as steep\n"
            f"Gradient safety mask created: {np.sum(safe_gradient_mask)} cells marked as unsafe",
            level='debug', block=True
        )

        return gradient_mask.astype(np.int32), safe_gradient_mask.astype(np.int32)

    def random_rover_spawns(
            self,
            safe_mask: np.ndarray,
            heightmap: np.ndarray, 
            n_spawns: int = 100,
            border_offset: float = 20.0,
            seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate safe random spawn locations for rover deployment.
        
        This method creates a specified number of spawn locations by randomly sampling
        positions within the heightmap bounds and validating them against safety constraints.
        Generated positions avoid rocks, steep terrain, and maintain border margins.
        
        Args:
            safe_mask: Binary safety mask where 0=safe, 1=unsafe areas.
            heightmap: Height data for Z-coordinate assignment.
            n_spawns: Number of spawn locations to generate.
            border_offset: Safety margin from heightmap edges in meters.
            seed: Random seed for reproducible generation.
            
        Returns:
            Array of spawn locations with shape (n_spawns, 3) containing [x, y, z] coordinates
            in world space.
            
        Raises:
            AssertionError: If border_offset is too large for the heightmap dimensions.
            
        Note:
            The method attempts up to 1000 iterations per spawn to find valid locations.
            Failed spawns will generate a warning but won't halt the process.
        """
        if seed is not None:
            np.random.seed(seed)

        height, width = safe_mask.shape
        min_xy = int(border_offset / self.resolution_in_m)
        max_xy = int(min(height, width) - min_xy)

        assert max_xy < width, f"Border offset too large: max_xy ({max_xy}) >= width ({width})"
        assert max_xy < height, f"Border offset too large: max_xy ({max_xy}) >= height ({height})"
        assert max_xy > min_xy, f"Invalid range: max_xy ({max_xy}) <= min_xy ({min_xy})"

        spawn_locations = np.zeros((n_spawns, 3), dtype=np.float32)

        for i in range(n_spawns):
            valid_location = False
            attempts = 0
            max_attempts = 1000
            
            while not valid_location and attempts < max_attempts:
                # Generate random grid coordinates within safe bounds
                x = np.random.randint(min_xy, max_xy)
                y = np.random.randint(min_xy, max_xy)
                
                # Validate against safety mask
                if safe_mask[y, x] == 0:  # 0 indicates safe area
                    spawn_locations[i, 0] = x
                    spawn_locations[i, 1] = y
                    spawn_locations[i, 2] = heightmap[y, x]
                    valid_location = True
                
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"Warning: Could not find valid location for spawn {i} after {max_attempts} attempts")

        # Convert grid coordinates to world coordinates
        spawn_locations[:, 0] = spawn_locations[:, 0] * self.resolution_in_m + self._heightmap_manager.min_x
        spawn_locations[:, 1] = spawn_locations[:, 1] * self.resolution_in_m + self._heightmap_manager.min_y

        return spawn_locations

    def resize_terrain_heightmap_to_match_combined(self, terrain_vertices: np.ndarray, terrain_faces: np.ndarray, reference_heightmap_manager):
        """Resize terrain-only heightmap to match combined heightmap dimensions"""
        # Get target dimensions from combined heightmap
        target_height, target_width = reference_heightmap_manager.heightmap.shape
        target_min_x = reference_heightmap_manager.min_x
        target_min_y = reference_heightmap_manager.min_y
        target_max_x = reference_heightmap_manager.max_x
        target_max_y = reference_heightmap_manager.max_y
        
        # Initialize terrain-only heightmap with same dimensions and bounds
        terrain_heightmap = np.full((target_height, target_width), -99.0, dtype=np.float32)
        
        # Calculate cell size
        cell_size_x = (target_max_x - target_min_x) / (target_width - 1)
        cell_size_y = (target_max_y - target_min_y) / (target_height - 1)
        
        if len(terrain_faces) > 0:
            # Get all triangle vertices from terrain only
            face_vertices = terrain_vertices[terrain_faces]
            
            # Extract coordinates
            x_coords = face_vertices[:, :, 0]
            y_coords = face_vertices[:, :, 1]
            z_coords = face_vertices[:, :, 2]
            
            # Find bounding box for each triangle
            min_x_tri = np.min(x_coords, axis=1)
            max_x_tri = np.max(x_coords, axis=1)
            min_y_tri = np.min(y_coords, axis=1)
            max_y_tri = np.max(y_coords, axis=1)
            max_z_tri = np.max(z_coords, axis=1)
            
            # Convert to grid coordinates using target bounds
            min_i = np.maximum(0, ((min_x_tri - target_min_x) / cell_size_x).astype(int))
            max_i = np.minimum(target_width - 1, ((max_x_tri - target_min_x) / cell_size_x).astype(int))
            min_j = np.maximum(0, ((min_y_tri - target_min_y) / cell_size_y).astype(int))
            max_j = np.minimum(target_height - 1, ((max_y_tri - target_min_y) / cell_size_y).astype(int))
            
            # Process triangles
            for idx in range(len(terrain_faces)):
                i_range = max_i[idx] - min_i[idx] + 1
                j_range = max_j[idx] - min_j[idx] + 1
                
                if i_range > 0 and j_range > 0:
                    terrain_heightmap[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1] = np.maximum(
                        terrain_heightmap[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1],
                        max_z_tri[idx]
                    )
        
        self.log(f"Resizing terrain-only heightmap to match combined heightmap dimensions: {target_height}x{target_width}", level='debug', block=False)

        
        # Create a new HeightmapManager with the resized terrain heightmap
        # Use same bounds as the reference heightmap
        resized_heightmap_manager = HeightmapManager.__new__(HeightmapManager)
        resized_heightmap_manager.resolution_in_m = self.resolution_in_m
        resized_heightmap_manager.device = self.device
        resized_heightmap_manager.heightmap = terrain_heightmap
        resized_heightmap_manager.min_x = target_min_x
        resized_heightmap_manager.min_y = target_min_y
        resized_heightmap_manager.max_x = target_max_x
        resized_heightmap_manager.max_y = target_max_y
        
        # Set up tensor attributes
        if self.device == 'cuda' or self.device == 'cuda:0':
            resized_heightmap_manager.heightmap_tensor = torch.from_numpy(terrain_heightmap).cuda()
            resized_heightmap_manager.offset_tensor = torch.tensor([target_min_x, target_min_y]).cuda()
        else:
            resized_heightmap_manager.heightmap_tensor = torch.from_numpy(terrain_heightmap)
            resized_heightmap_manager.offset_tensor = torch.tensor([target_min_x, target_min_y])
        
        return resized_heightmap_manager
    
    # TODO : Remove
    def get_valid_targets(self, target_positions: torch.Tensor, device: str = "cuda:0") -> torch.Tensor:
        """
        Filter target positions to return only those in safe areas.
        
        This method filters target positions to only include those that are safe for
        navigation targets. Targets are allowed on steep terrain - only rock/obstacle
        constraints and boundary distance requirements apply.
        
        Args:
            target_positions: Candidate target positions, tensor of shape (N, 2 or 3).
            device: Device for computations (legacy parameter).
            
        Returns:
            Filtered target positions that are in safe areas (not on rocks) and within 
            boundary distance constraints.
        """
        # Check 1: Boundary distance constraints
        # Ensure targets are not too close to the map boundaries
        x_coords = target_positions[:, 0]
        y_coords = target_positions[:, 1]
        
        # Get heightmap bounds
        min_x, min_y, max_x, max_y = self._heightmap_manager.get_heightmap_bounds()
        
        # Check if targets are within valid boundary distance
        boundary_margin = self.target_distance_to_boundary
        boundary_valid = (
            (x_coords >= min_x + boundary_margin) &
            (x_coords <= max_x - boundary_margin) &
            (y_coords >= min_y + boundary_margin) &
            (y_coords <= max_y - boundary_margin)
        )
        
        # Check 2: Rock/obstacle safety constraints (targets allowed on steep terrain)
        # Transform to grid coordinates
        scaled_position = target_positions[:, 0:2] / \
            self._heightmap_manager.resolution_in_m + self._heightmap_manager.offset_tensor
        grid_cell = scaled_position.long()
        
        # Clamp to valid bounds
        grid_cell[:, 0] = torch.clamp(grid_cell[:, 0], 0, self._heightmap_manager.heightmap_tensor.shape[1] - 1)
        grid_cell[:, 1] = torch.clamp(grid_cell[:, 1], 0, self._heightmap_manager.heightmap_tensor.shape[0] - 1)
        
        # Filter safe positions (rock safe_mask == 0 means safe, targets can be on steep terrain)
        safe_mask_values = self.safe_rock_mask_tensor[grid_cell[:, 1], grid_cell[:, 0]].squeeze(-1)
        safety_valid = (safe_mask_values == 0)
        
        # Combine both constraints: valid if both boundary and safety checks pass
        combined_valid = boundary_valid & safety_valid
        valid_indices = torch.where(combined_valid)[0]
        
        return target_positions[valid_indices]
    
    def get_terrain_statistics(self) -> dict:
        """
        Get comprehensive statistics about the terrain.
        
        Returns:
            Dictionary containing terrain analysis statistics.
        """
        height, width = self._heightmap_manager.heightmap.shape
        total_cells = height * width
        
        stats = {
            "heightmap_shape": (height, width),
            "resolution_m": self.resolution_in_m,
            "world_bounds": {
                "min_x": self._heightmap_manager.min_x,
                "max_x": self._heightmap_manager.max_x,
                "min_y": self._heightmap_manager.min_y,
                "max_y": self._heightmap_manager.max_y,
            },
            "height_stats": {
                "min": float(np.min(self._heightmap_manager.heightmap)),
                "max": float(np.max(self._heightmap_manager.heightmap)),
                "mean": float(np.mean(self._heightmap_manager.heightmap)),
                "std": float(np.std(self._heightmap_manager.heightmap))
            }
        }
        
        if hasattr(self, 'rock_mask'):
            rock_cells = np.sum(self.rock_mask)
            safe_rock_cells = np.sum(self.safe_rock_mask)
            stats["rock_analysis"] = {
                "rock_cells": int(rock_cells),
                "rock_percentage": float(rock_cells / total_cells * 100),
                "safe_rock_cells": int(safe_rock_cells),
                "safe_rock_percentage": float(safe_rock_cells / total_cells * 100)
            }

        if hasattr(self, 'obstacle_risk_map'):
            stats["obstacle_risk"] = self.obstacle_risk_map.statistics()
        
        if hasattr(self, 'gradient_mask'):
            gradient_cells = np.sum(self.gradient_mask)
            safe_gradient_cells = np.sum(self.safe_gradient_mask)
            stats["gradient_analysis"] = {
                "steep_cells": int(gradient_cells),
                "steep_percentage": float(gradient_cells / total_cells * 100),
                "safe_gradient_cells": int(safe_gradient_cells),
                "safe_gradient_percentage": float(safe_gradient_cells / total_cells * 100),
                "gradient_threshold": self.gradient_threshold
            }
        
        if hasattr(self, 'spawn_locations'):
            stats["spawn_analysis"] = {
                "total_spawns": len(self.spawn_locations),
                "spawn_density_per_km2": len(self.spawn_locations) / ((self._heightmap_manager.max_x - self._heightmap_manager.min_x) * 
                                                                     (self._heightmap_manager.max_y - self._heightmap_manager.min_y) / 1_000_000)
            }
        
        return stats
