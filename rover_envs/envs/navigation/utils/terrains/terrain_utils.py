# from isaaclab.markers.visualization_markers import VisualizationMarkersCfg, VisualizationMarkers
# import isaaclab.sim as sim_utils
import os
from typing import Tuple, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pymeshlab
import torch

# Try to import Isaac Sim dependencies for runtime, fallback for debugging
try:
    from rover_envs.envs.navigation.utils.terrains.usd_utils import get_triangles_and_vertices_from_prim, get_triangles_and_vertices_from_prim_standalone
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False
    print("Isaac Sim dependencies not available - running in debug mode")
    from rover_envs.envs.navigation.utils.terrains.usd_utils import get_triangles_and_vertices_from_prim_standalone

# Import standalone USD capabilities for debugging
try:
    from pxr import Usd, UsdGeom
    USD_STANDALONE_AVAILABLE = True
except ImportError:
    USD_STANDALONE_AVAILABLE = False
    print("Warning: USD standalone libraries not available")

class HeightmapManager():

    def __init__(self, resolution_in_m, vertices, faces, device='cpu'):
        self.resolution_in_m = resolution_in_m
        self.device = device
        self.heightmap, self.min_x, self.min_y, self.max_x, self.max_y = self.mesh_to_heightmap(vertices, faces)
        if device == 'cuda' or device == 'cuda:0':
            self.heightmap_tensor = torch.from_numpy(self.heightmap).cuda()
            self.offset_tensor = torch.tensor([self.min_x, self.min_y]).cuda()
        else:
            self.heightmap_tensor = torch.from_numpy(self.heightmap)
            self.offset_tensor = torch.tensor([self.min_x, self.min_y])

    def mesh_to_heightmap(self, vertices, faces):
        # Border Margin
        border_margin = 1.0
        # Define bounding box
        min_x, min_y, _ = np.min(vertices, axis=0) + border_margin
        max_x, max_y, _ = np.max(vertices, axis=0) - border_margin

        # Calculate the grid size
        grid_size_x = (max_x - min_x) / self.resolution_in_m
        grid_size_y = (max_y - min_y) / self.resolution_in_m
        
        # Grid dimensions
        grid_width = int(grid_size_x + 1)
        grid_height = int(grid_size_y + 1)

        # Initialize the heightmap
        heightmap = np.full((grid_height, grid_width), -99.0, dtype=np.float32)

        # Calculate the size of a grid cell
        cell_size_x = (max_x - min_x) / grid_size_x
        cell_size_y = (max_y - min_y) / grid_size_y

        if len(faces) > 0:
            # Get all triangle vertices
            face_vertices = vertices[faces]
            
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
            
            # Convert to grid coordinates
            min_i = np.maximum(0, ((min_x_tri - min_x) / cell_size_x).astype(int))
            max_i = np.minimum(grid_width - 1, ((max_x_tri - min_x) / cell_size_x).astype(int))
            min_j = np.maximum(0, ((min_y_tri - min_y) / cell_size_y).astype(int))
            max_j = np.minimum(grid_height - 1, ((max_y_tri - min_y) / cell_size_y).astype(int))
            
            # Process triangles
            for idx in range(len(faces)):
                i_range = max_i[idx] - min_i[idx] + 1
                j_range = max_j[idx] - min_j[idx] + 1
                
                if i_range > 0 and j_range > 0:
                    heightmap[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1] = np.maximum(
                        heightmap[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1],
                        max_z_tri[idx]
                    )

        return heightmap, min_x, min_y, max_x, max_y

    def get_height_at(self, position: torch.Tensor) -> torch.Tensor:
        """
        Returns the height at the specified position.

        Args:
            position (torch.Tensor): The position at which to get the height. Shape (N, 2).

        Returns:
            torch.Tensor: The height at the specified position. Shape (N,).
        """
        # Find the grid cell in self.heightmap_tensor

        # Scale the position to match the heightmap indices
        scaled_position = position / self.resolution_in_m + self.offset_tensor
        # Convert to long to get the grid cell
        grid_cell = scaled_position.long()

        # Clamp the grid cell to the heightmap dimensions
        grid_cell[:, 0] = torch.clamp(grid_cell[:, 0], 0, self.heightmap_tensor.shape[1]-1)
        grid_cell[:, 1] = torch.clamp(grid_cell[:, 1], 0, self.heightmap_tensor.shape[0]-1)

        # Return the heights at the specified positions
        return self.heightmap_tensor[grid_cell[:, 1], grid_cell[:, 0]]

class TerrainManager():

    def __init__(self, 
                 num_envs: int, 
                 device: str, 
                 debug_mode: bool = False, 
                 terrain_usd_path: str = None, 
                 rock_usd_path: str = None,
                 safety_margin: float = 2.0
                 ):
        """
        """
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.debug_mode = debug_mode or not ISAAC_SIM_AVAILABLE
        
        if self.debug_mode and terrain_usd_path:
            # Debug mode with custom USD files
            terrain_path = terrain_usd_path
            rock_mesh_path = rock_usd_path
        elif self.debug_mode:
            # Debug mode with default Mars terrain paths
            base_path = os.path.join(self.dir_path, "..", "..", "..", "..", "assets", "terrains", "mars", "terrain1")
            terrain_path = os.path.join(base_path, "terrain_only.usd")
            rock_mesh_path = os.path.join(base_path, "rocks_merged.usd")
            if not os.path.exists(terrain_path):
                raise FileNotFoundError(f"Debug mode requires terrain USD file at: {terrain_path}")
        else:
            # Isaac Sim runtime mode
            terrain_path = "/World/terrain/hidden_terrain/terrain"
            rock_mesh_path = "/World/terrain/obstacles/obstacles"

        self.meshes = [terrain_path, rock_mesh_path]

        self.meshes = {
            "terrain": terrain_path,
            "rock": rock_mesh_path
        }

        # Terrain Parameters
        self.heightmap = None
        self.resolution_in_m = 0.05
        self.gradient_threshold = 0.4
        self.device = device

        # Load Terrain (terrain only, without rocks)
        print("Getting triangles and vertices from terrain USD file")
        try:
            terrain_vertices, terrain_faces = self.get_mesh(self.meshes["terrain"])
        except Exception as e:
            print(f"Failed to load terrain: {e}")
            if not self.debug_mode:
                print("Trying to fallback to debug mode...")
                self.debug_mode = True
                terrain_vertices, terrain_faces = self.get_mesh(self.meshes["terrain"])
            else:
                raise
        
        # Load rocks if available and combine with terrain for spawn height queries
        try:
            print("Getting triangles and vertices from rock USD file")
            rock_vertices, rock_faces = self.get_mesh(self.meshes["rock"])
            
            # Combine terrain and rocks for complete heightmap
            print("Combining terrain and rock meshes...")
            combined_vertices = np.vstack([terrain_vertices, rock_vertices])
            combined_faces = np.vstack([terrain_faces, rock_faces + len(terrain_vertices)])
            
            # Create combined heightmap manager (for spawn height queries)
            print("Generating combined heightmap")
            self._heightmap_manager = HeightmapManager(self.resolution_in_m, combined_vertices, combined_faces, device)
            
            # Create terrain-only heightmap with SAME BOUNDS as combined heightmap
            print("Generating terrain-only heightmap with matched bounds")
            self.terrain_only_heightmap_manager = HeightmapManager(self.resolution_in_m, terrain_vertices, terrain_faces, device)
            
            # Resize terrain-only heightmap to match combined heightmap dimensions
            self.terrain_only_heightmap_manager = self.resize_terrain_heightmap_to_match_combined(
                terrain_vertices, terrain_faces, self._heightmap_manager
            )
            
        except Exception as e:
            print(f"Failed to load rocks: {e}. Using terrain-only heightmap.")
            # If rocks fail to load, use terrain-only heightmap for everything
            self._heightmap_manager = HeightmapManager(self.resolution_in_m, terrain_vertices, terrain_faces, device)
            self.terrain_only_heightmap_manager = self._heightmap_manager
            rock_vertices = None
            rock_faces = None

        # Generate Gradient Masks (terrain-only for visualization)
        print("Generating gradient masks")
        self.gradient_mask, self.safe_gradient_mask = self.compute_gradient_masks(
            self._heightmap_manager.heightmap, self.gradient_threshold, safety_margin=safety_margin)

        # Generate Rock Mask if rocks are available
        if rock_vertices is not None:
            print("Generating rock mask from rock mesh")
            self.rock_mask, self.safe_rock_mask = self.project_rocks_to_heightmap(rock_vertices, rock_faces)
        else:
            print("No rocks available, creating empty rock masks")
            # Create empty rock masks
            height, width = self._heightmap_manager.heightmap.shape
            self.rock_mask = np.zeros((height, width), dtype=np.int32)
            self.safe_rock_mask = np.zeros((height, width), dtype=np.int32)

        # Generate Gradient Mask using terrain-only heightmap
        print("Generating gradient mask from terrain-only heightmap")
        self.gradient_mask, self.safe_gradient_mask = self.compute_gradient_masks(
            self.terrain_only_heightmap_manager.heightmap, self.gradient_threshold)

        # Combine rock and gradient masks for spawn generation
        print("Combining rock and gradient masks for spawn generation")
        combined_safe_mask = np.logical_or(self.safe_rock_mask, self.safe_gradient_mask).astype(np.int32)

        # Generate Spawn Locations
        self.spawn_locations = self.random_rover_spawns(
            safe_mask=combined_safe_mask,
            heightmap=self._heightmap_manager.heightmap,
            n_spawns=num_envs*2 if num_envs > 100 else 200,
            border_offset=25.0,
            seed=12345)
        if device == 'cuda:0' or device == 'cuda':
            self.spawn_locations = torch.from_numpy(self.spawn_locations).cuda()
            self.safe_rock_mask_tensor = torch.from_numpy(self.safe_rock_mask).cuda().unsqueeze(-1)
        else:
            self.spawn_locations = torch.from_numpy(self.spawn_locations)
            self.safe_rock_mask_tensor = torch.from_numpy(self.safe_rock_mask).unsqueeze(-1)

    def get_mesh(self, prim_path="/") -> Tuple[np.ndarray, np.ndarray]:
        """ This function reads a USD from the specified prim path and return vertices and faces.
        
        Args:
            prim_path: Prim path for Isaac Sim runtime, or USD file path for debug mode
            
        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """

        try:
            if ISAAC_SIM_AVAILABLE:
                # Try Isaac Sim runtime first
                faces, vertices = get_triangles_and_vertices_from_prim(prim_path)
            else:
                raise ImportError("Isaac Sim not available, falling back to standalone mode")
        except (ImportError, Exception) as e:
            print(f"Isaac Sim method failed ({e}), trying standalone USD loading...")
            if USD_STANDALONE_AVAILABLE and os.path.exists(prim_path):
                # Fallback to standalone USD loading for debug mode
                faces, vertices = get_triangles_and_vertices_from_prim_standalone(prim_path)
            else:
                raise RuntimeError(f"Cannot load mesh: Isaac Sim not available and USD file not found at {prim_path}")

        # Create pymeshlab mesh and meshset
        mesh = pymeshlab.Mesh(vertices, faces)

        ms = pymeshlab.MeshSet()
        ms.add_mesh(mesh)

        # get the mesh
        mesh = ms.current_mesh()  # get the mesh

        # Get vertices as float32 array
        vertices = mesh.vertex_matrix().astype('float32')

        # Get faces as uint32 array
        faces = mesh.face_matrix().astype('uint32')

        return vertices, faces

    def check_if_target_is_valid(
            self,
            env_ids: torch.Tensor,
            target_positions: torch.Tensor,
            device: str = "cuda:0"
    ) -> torch.Tensor:
        # Find the grid cell in self.heightmap_tensor

        # Scale the position to match the heightmap indices
        scaled_position = target_positions[:, 0:2] / \
            self._heightmap_manager.resolution_in_m + self._heightmap_manager.offset_tensor
        # Convert to long to get the grid cell
        grid_cell = scaled_position.long()

        # Clamp the grid cell to the heightmap dimensions
        grid_cell[:, 0] = torch.clamp(grid_cell[:, 0], 0, self._heightmap_manager.heightmap_tensor.shape[1]-1)
        grid_cell[:, 1] = torch.clamp(grid_cell[:, 1], 0, self._heightmap_manager.heightmap_tensor.shape[0]-1)

        reset_buf = torch.where(self.safe_rock_mask_tensor[grid_cell[:, 1], grid_cell[:, 0]] == 1, 1, 0).squeeze(-1)
        env_ids = env_ids[reset_buf == 1]
        reset_buf_len = len(env_ids)
        return env_ids, reset_buf_len

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
        
        print(f"Projecting {len(rock_faces)} rock triangles onto XY plane...")
        
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
        
        print("Rock projection completed. Applying morphological operations...")
        
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
        
        print(f"Rock mask created: {np.sum(rock_mask)} cells marked as rocks")
        print(f"Rock safety mask created: {np.sum(safe_rock_mask)} cells marked as unsafe")
        
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
        
        print(f"Gradient mask created: {np.sum(gradient_mask)} cells marked as steep")
        print(f"Gradient safety mask created: {np.sum(safe_gradient_mask)} cells marked as unsafe")
        
        return gradient_mask.astype(np.int32), safe_gradient_mask.astype(np.int32)

    def random_rover_spawns(
            self,
            safe_mask: np.ndarray,
            heightmap, n_spawns: int = 100,
            border_offset: float = 20.0,
            seed=None
    ) -> np.ndarray:
        """Generate random rover spawn locations. Calculates random x,y checks if it is a rock, if not,
        add to list of spawn locations with corresponding z value from heightmap.

        Args:
            safe_mask (np.ndarray): A binary mask indicating the locations of safe areas. (0 for safe, 1 for unsafe).
            n_spawns (int, optional): The number of spawn locations to generate. Defaults to 1.
            border_offset (float, optional): Border offset in meters. Defaults to 20.0.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            np.ndarray: An array of shape (n_spawns, 3) containing the spawn locations.
        """
        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Get the heightmap dimensions
        height, width = safe_mask.shape
        min_xy = int(border_offset / self.resolution_in_m)
        max_xy = int((min(height, width) - min_xy))

        assert max_xy < width, f"max_xy ({max_xy}) must be less than width ({width})"
        assert max_xy < height, f"max_xy ({max_xy}) must be less than height ({height})"

        # Initialize the spawn locations array
        spawn_locations = np.zeros((n_spawns, 3), dtype=np.float32)

        # Generate spawn locations
        for i in range(n_spawns):
            valid_location = False
            attempts = 0
            max_attempts = 1000
            
            while not valid_location and attempts < max_attempts:
                # Generate random x, y coordinates within bounds
                x = np.random.randint(min_xy, max_xy)
                y = np.random.randint(min_xy, max_xy)
                
                # Check if the location is not marked as rock and gradient is safe
                if safe_mask[y, x] == 0:
                    valid_location = True
                    spawn_locations[i, 0] = x
                    spawn_locations[i, 1] = y
                    spawn_locations[i, 2] = heightmap[y, x]
                
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"Warning: Could not find valid location for spawn {i}")

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
        
        print(f"Terrain-only heightmap resized to match combined dimensions: {terrain_heightmap.shape}")
        
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

class DebugVisualizer:
    """Debug visualization helper for terrain analysis"""
    
    def __init__(self, terrain_manager: TerrainManager):
        """Initialize with terrain manager"""
        self.terrain_manager = terrain_manager
        self.heightmap_manager = terrain_manager._heightmap_manager  # Use the actual attribute name
        self.resolution_in_m = terrain_manager.resolution_in_m

    def visualize_combined_rock_gradient_mask(self, spawn_locations: np.ndarray):
        """Visualize combined rock and gradient masks with spawn points"""
        plt.figure(figsize=(15, 10))
        
        # Convert spawn locations to grid coordinates for plotting
        spawn_grid_x = (spawn_locations[:, 0] - self.heightmap_manager.min_x) / self.resolution_in_m
        spawn_grid_y = (spawn_locations[:, 1] - self.heightmap_manager.min_y) / self.resolution_in_m
        
        # Show combined heightmap as background
        plt.imshow(self.heightmap_manager.heightmap, cmap='terrain', origin='lower', alpha=0.7)
        
        # Overlay gradient mask (steep terrain)
        gradient_mask = self.terrain_manager.gradient_mask
        gradient_mask_overlay = np.ma.masked_where(gradient_mask == 0, gradient_mask)
        plt.imshow(gradient_mask_overlay, cmap='Purples', alpha=0.8, origin='lower', vmin=0, vmax=1)

        # Overlay rock mask if available
        if hasattr(self.terrain_manager, 'rock_mask') and self.terrain_manager.rock_mask is not None:
            rock_mask = self.terrain_manager.rock_mask
            safe_rock_mask = self.terrain_manager.safe_rock_mask
            
            rock_mask_overlay = np.ma.masked_where(rock_mask == 0, rock_mask)
            plt.imshow(rock_mask_overlay, cmap='Reds', alpha=0.8, origin='lower', label='Rock Mask')
            
            # Overlay rock safety mask
            safety_rock_mask_overlay = np.ma.masked_where(safe_rock_mask == 0, safe_rock_mask)
            plt.imshow(safety_rock_mask_overlay, cmap='Oranges', alpha=0.3, origin='lower', label='Rock Safety Zone')
        
        # Overlay gradient safety mask
        safe_gradient_mask = self.terrain_manager.safe_gradient_mask
        safety_gradient_mask_overlay = np.ma.masked_where(safe_gradient_mask == 0, safe_gradient_mask)
        plt.imshow(safety_gradient_mask_overlay, cmap='Blues', alpha=0.3, origin='lower', label='Gradient Safety Zone')
        
        # Plot spawn points
        plt.scatter(spawn_grid_x, spawn_grid_y, c='cyan', marker='o', s=30, 
                   edgecolor='blue', linewidth=1, label='Spawn Points', zorder=5)
        
        plt.colorbar(label='Height (m)')
        plt.title('Combined Rock and Gradient Masks with Spawn Points')
        plt.xlabel('X Grid Coordinate')
        plt.ylabel('Y Grid Coordinate')
        
        # Create custom legend
        legend_elements = [
            Patch(facecolor='purple', alpha=0.8, label='Steep Terrain'),
            Patch(facecolor='blue', alpha=0.3, label='Gradient Safety Zones'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
                      markeredgecolor='blue', markersize=8, label='Spawn Points')
        ]
        
        # Add rock-related legend items if rocks are available
        if hasattr(self.terrain_manager, 'rock_mask') and self.terrain_manager.rock_mask is not None:
            legend_elements.insert(1, Patch(facecolor='red', alpha=0.8, label='Rock Areas'))
            legend_elements.insert(3, Patch(facecolor='orange', alpha=0.3, label='Rock Safety Zones'))
        
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_terrain_analysis_subplots(self, spawn_locations: np.ndarray):
        """Visualize terrain gradients and rocks in two subplots on the same page"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # Convert spawn locations to grid coordinates for plotting
        spawn_grid_x = (spawn_locations[:, 0] - self.heightmap_manager.min_x) / self.resolution_in_m
        spawn_grid_y = (spawn_locations[:, 1] - self.heightmap_manager.min_y) / self.resolution_in_m
        
        # Left subplot: Terrain gradients only
        terrain_only_heightmap = getattr(self.terrain_manager, 'terrain_only_heightmap_manager', self.heightmap_manager)
        ax1.imshow(terrain_only_heightmap.heightmap, cmap='terrain', origin='lower', alpha=0.8)
        
        # Overlay gradient mask (steep terrain only, no rocks)
        gradient_mask = self.terrain_manager.gradient_mask
        gradient_mask_overlay = np.ma.masked_where(gradient_mask == 0, gradient_mask)
        ax1.imshow(gradient_mask_overlay, cmap='Purples', alpha=0.9, origin='lower', vmin=0, vmax=1)
        
        # Overlay gradient safety mask
        safe_gradient_mask = self.terrain_manager.safe_gradient_mask
        safety_gradient_mask_overlay = np.ma.masked_where(safe_gradient_mask == 0, safe_gradient_mask)
        ax1.imshow(safety_gradient_mask_overlay, cmap='Blues', alpha=0.4, origin='lower')
        
        # Plot spawn points
        ax1.scatter(spawn_grid_x, spawn_grid_y, c='cyan', marker='o', s=20, 
                   edgecolor='blue', linewidth=1, zorder=5)
        
        ax1.set_title('Terrain Gradients Only (No Rocks)', fontsize=14)
        ax1.set_xlabel('X Grid Coordinate')
        ax1.set_ylabel('Y Grid Coordinate')
        
        # Right subplot: Rocks only (if available)
        ax2.imshow(self.heightmap_manager.heightmap, cmap='terrain', origin='lower', alpha=0.8)
        
        if hasattr(self.terrain_manager, 'rock_mask') and self.terrain_manager.rock_mask is not None:
            # Overlay rock mask only
            rock_mask = self.terrain_manager.rock_mask
            rock_mask_overlay = np.ma.masked_where(rock_mask == 0, rock_mask)
            ax2.imshow(rock_mask_overlay, cmap='Reds', alpha=0.9, origin='lower', vmin=0, vmax=1)
            
            # Overlay rock safety mask
            safe_rock_mask = self.terrain_manager.safe_rock_mask
            safety_rock_mask_overlay = np.ma.masked_where(safe_rock_mask == 0, safe_rock_mask)
            ax2.imshow(safety_rock_mask_overlay, cmap='Oranges', alpha=0.4, origin='lower')
            
            ax2.set_title('Rock Areas Only (No Terrain Gradients)', fontsize=14)
        else:
            ax2.text(0.5, 0.5, 'No Rock Data Available', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=16, bbox=dict(boxstyle='round', facecolor='wheat'))
            ax2.set_title('No Rock Data Available', fontsize=14)
        
        # Plot spawn points
        ax2.scatter(spawn_grid_x, spawn_grid_y, c='cyan', marker='o', s=20, 
                   edgecolor='blue', linewidth=1, zorder=5)
        
        ax2.set_xlabel('X Grid Coordinate')
        ax2.set_ylabel('Y Grid Coordinate')
        
        # Create custom legend for both subplots
        legend_elements = [
            Patch(facecolor='purple', alpha=0.9, label='Steep Terrain'),
            Patch(facecolor='blue', alpha=0.4, label='Gradient Safety Zones'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
                      markeredgecolor='blue', markersize=8, label='Spawn Points')
        ]
        
        # Add rock-related legend items if rocks are available
        if hasattr(self.terrain_manager, 'rock_mask') and self.terrain_manager.rock_mask is not None:
            legend_elements.insert(1, Patch(facecolor='red', alpha=0.9, label='Rock Areas'))
            legend_elements.insert(3, Patch(facecolor='orange', alpha=0.4, label='Rock Safety Zones'))
        
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(legend_elements))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # Make room for legend
        plt.show()


def main():
    """Main debug function"""
    print("=== Terrain Utils Debug Tool ===")
    
    # Configuration - Mars terrain paths
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "assets", "terrains", "debug", "debug1")
    terrain_usd_path = os.path.join(base_path, "terrain_only.usd")
    rock_usd_path = os.path.join(base_path, "rocks_merged.usd")
    
    # Check if USD files exist
    if not os.path.exists(terrain_usd_path):
        print(f"Terrain USD file not found: {terrain_usd_path}")
        print("Please check that the Mars terrain assets are available")
        return
    
    if not os.path.exists(rock_usd_path):
        print(f"Rock USD file not found: {rock_usd_path}")
        print("Continuing without rock mesh...")
        rock_usd_path = None

    try:
        # Initialize terrain manager in debug mode
        print("Initializing terrain manager in debug mode...")
        terrain_manager = TerrainManager(
            num_envs=1,  # Only need 1 environment for debugging
            device='cuda' if torch.cuda.is_available() else 'cpu',
            debug_mode=True,
            terrain_usd_path=terrain_usd_path,
            rock_usd_path=rock_usd_path
        )
        
        # Initialize visualizer
        visualizer = DebugVisualizer(terrain_manager)
        
        # Generate spawn locations
        print("Generating spawn locations...")
        spawn_locations = terrain_manager.random_rover_spawns(
            safe_mask=np.logical_or(
                terrain_manager.safe_rock_mask,
                terrain_manager.safe_gradient_mask
            ),
            heightmap=terrain_manager._heightmap_manager.heightmap,
            n_spawns=2000, 
            seed=42
        )
        spawn_locations_np = spawn_locations.cpu().numpy() if isinstance(spawn_locations, torch.Tensor) else spawn_locations
        print(f"Generated {len(spawn_locations_np)} spawn locations")
        
        # Visualizations
        print("Creating visualizations...")
        
        # Show terrain analysis in subplots (gradients and rocks side by side)
        visualizer.visualize_terrain_analysis_subplots(spawn_locations_np)
        
        # Show combined rock and gradient masks - Second plot
        visualizer.visualize_combined_rock_gradient_mask(spawn_locations_np)
        
        # Test height queries
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Testing height queries on {device}...")
        test_positions = torch.tensor([[5.0, 5.0], [10.0, 10.0], [15.0, 15.0]], device=device)
        heights = terrain_manager._heightmap_manager.get_height_at(test_positions)
        print(f"Test positions: {test_positions}")
        print(f"Heights: {heights}")
        
        print("Debug session completed successfully!")
        
    except Exception as e:
        print(f"Error during debug session: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
