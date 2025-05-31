#!/usr/bin/env python3
"""
Standalone debug script for terrain utilities that can load USD files 
without requiring Isaac Sim/Isaac Lab to be running.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from pxr import Usd, UsdGeom
import pymeshlab


class StandaloneUSDLoader:
    """Standalone USD loader that doesn't require Isaac Sim runtime"""
    
    @staticmethod
    def load_usd_mesh(usd_file_path: str, prim_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load mesh data from a USD file without Isaac Sim dependencies
        
        Args:
            usd_file_path: Path to the USD file
            prim_path: Specific prim path to load (if None, finds first mesh)
            
        Returns:
            Tuple of (vertices, faces) as numpy arrays
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
        
        print(f"Loaded mesh with {len(vertices)} vertices and {len(faces)} faces")
        return vertices, faces


class StandaloneHeightmapManager:
    """Heightmap manager that works without Isaac Lab dependencies"""
    
    def __init__(self, resolution_in_m: float, vertices: np.ndarray, faces: np.ndarray):
        self.resolution_in_m = resolution_in_m
        self.heightmap, self.min_x, self.min_y, self.max_x, self.max_y = self.mesh_to_heightmap(vertices, faces)
        
        # Convert to torch tensors if CUDA is available
        if torch.cuda.is_available():
            self.heightmap_tensor = torch.from_numpy(self.heightmap).cuda()
            self.offset_tensor = torch.tensor([self.min_x, self.min_y]).cuda()
        else:
            self.heightmap_tensor = torch.from_numpy(self.heightmap)
            self.offset_tensor = torch.tensor([self.min_x, self.min_y])
    
    def mesh_to_heightmap(self, vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, float, float, float, float]:
        """Convert mesh to heightmap"""
        # Border margin
        border_margin = 1.0
        
        # Define bounding box
        min_x, min_y, _ = np.min(vertices, axis=0) + border_margin
        max_x, max_y, _ = np.max(vertices, axis=0) - border_margin
        
        # Calculate grid size
        grid_size_x = (max_x - min_x) / self.resolution_in_m
        grid_size_y = (max_y - min_y) / self.resolution_in_m
        
        # Grid dimensions
        grid_width = int(grid_size_x + 1)
        grid_height = int(grid_size_y + 1)
        
        # Initialize heightmap
        heightmap = np.full((grid_height, grid_width), -99.0, dtype=np.float32)
        
        # Calculate cell size
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
        """Get height at specified positions"""
        # Scale position to match heightmap indices
        scaled_position = position / self.resolution_in_m + self.offset_tensor
        grid_cell = scaled_position.long()
        
        # Clamp to heightmap dimensions
        grid_cell[:, 0] = torch.clamp(grid_cell[:, 0], 0, self.heightmap_tensor.shape[1]-1)
        grid_cell[:, 1] = torch.clamp(grid_cell[:, 1], 0, self.heightmap_tensor.shape[0]-1)
        
        return self.heightmap_tensor[grid_cell[:, 1], grid_cell[:, 0]]
    


class StandaloneTerrainDebugger:
    """Standalone terrain debugger that works without Isaac Lab"""
    
    def __init__(self, terrain_usd_path: str, rock_usd_path: str = None, terrain_prim_path: str = None, rock_prim_path: str = None):
        self.terrain_usd_path = terrain_usd_path
        self.rock_usd_path = rock_usd_path
        self.terrain_prim_path = terrain_prim_path
        self.rock_prim_path = rock_prim_path
        self.resolution_in_m = 0.05
        self.gradient_threshold = 0.3  # Lower threshold to match terrain_utils.py
        
        # Load terrain mesh
        print(f"Loading terrain from USD file: {terrain_usd_path}")
        self.loader = StandaloneUSDLoader()
        self.terrain_vertices, self.terrain_faces = self.loader.load_usd_mesh(terrain_usd_path, terrain_prim_path)
        
        # Load rock mesh if provided
        self.rock_vertices = None
        self.rock_faces = None
        if rock_usd_path and os.path.exists(rock_usd_path):
            print(f"Loading rocks from USD file: {rock_usd_path}")
            self.rock_vertices, self.rock_faces = self.loader.load_usd_mesh(rock_usd_path, rock_prim_path)
        
        # Combine terrain and rocks for heightmap generation
        if self.rock_vertices is not None:
            print("Combining terrain and rock meshes...")
            self.vertices = np.vstack([self.terrain_vertices, self.rock_vertices])
            self.faces = np.vstack([self.terrain_faces, self.rock_faces + len(self.terrain_vertices)])
        else:
            self.vertices = self.terrain_vertices
            self.faces = self.terrain_faces
        
        # Create combined heightmap manager (for spawn height queries)
        print("Generating combined heightmap...")
        self.heightmap_manager = StandaloneHeightmapManager(
            self.resolution_in_m, self.vertices, self.faces
        )
        
        # Create terrain-only heightmap manager (for gradient calculation)
        print("Generating terrain-only heightmap...")
        self.terrain_only_heightmap_manager = StandaloneHeightmapManager(
            self.resolution_in_m, self.terrain_vertices, self.terrain_faces
        )
        
        # Generate rock mask
        print("Generating rock mask...")
        if self.rock_vertices is not None:
            self.rock_mask, self.safe_rock_mask = self.project_rocks_to_xy_plane(
                self.rock_vertices, self.rock_faces
            )
        else:
            # No rocks available, create empty masks
            height, width = self.heightmap_manager.heightmap.shape
            self.rock_mask = np.zeros((height, width), dtype=np.int32)
            self.safe_rock_mask = np.zeros((height, width), dtype=np.int32)
        
        # Generate gradient mask for steep terrain (using terrain-only heightmap with same bounds)
        print("Generating gradient mask for steep terrain...")
        # Resize terrain-only heightmap to match combined heightmap dimensions
        terrain_only_heightmap_resized = self.resize_terrain_heightmap_to_match_combined()
        self.gradient_mask, self.safe_gradient_mask = self.calculate_gradient_mask(
            terrain_only_heightmap_resized, self.gradient_threshold
        )
        
        # Combine rock and gradient masks for final safe spawning areas
        print("Combining rock and gradient masks...")
        self.combined_safe_mask = np.logical_or(self.safe_rock_mask, self.safe_gradient_mask).astype(np.int32)
        
        print("Terrain debugger initialized successfully!")
    
    def project_rocks_to_xy_plane(self, rock_vertices: np.ndarray, rock_faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project rock mesh triangles onto XY plane to create rock masks"""
        import cv2
        from scipy import ndimage
        
        # Get heightmap dimensions and bounds
        height, width = self.heightmap_manager.heightmap.shape
        min_x = self.heightmap_manager.min_x
        min_y = self.heightmap_manager.min_y
        resolution = self.resolution_in_m
        
        # Initialize rock mask
        rock_mask = np.zeros((height, width), dtype=np.uint8)
        
        print(f"Projecting {len(rock_faces)} rock triangles onto XY plane...")
        
        # Vectorized processing of rock triangles
        if len(rock_faces) > 0:
            # Get all triangle vertices and project to 2D
            # rock_vertices shape: (num_total_vertices, 3)
            # rock_faces shape: (num_faces, num_vertices_per_face)
            # all_triangle_vertices shape: (num_faces, num_vertices_per_face, 3)
            all_triangle_vertices = rock_vertices[rock_faces]
            # all_triangle_2d shape: (num_faces, num_vertices_per_face, 2)
            all_triangle_2d = all_triangle_vertices[:, :, :2]

            # Convert world coordinates to grid coordinates
            all_grid_coords_x = (all_triangle_2d[:, :, 0] - min_x) / resolution
            all_grid_coords_y = (all_triangle_2d[:, :, 1] - min_y) / resolution
            
            # all_grid_coords shape: (num_faces, num_vertices_per_face, 2)
            all_grid_coords = np.stack((all_grid_coords_x, all_grid_coords_y), axis=-1)

            # Check which vertices are within heightmap bounds
            # vert_coords_in_bounds_x shape: (num_faces, num_vertices_per_face)
            vert_coords_in_bounds_x = (all_grid_coords[:, :, 0] >= 0) & (all_grid_coords[:, :, 0] < width)
            vert_coords_in_bounds_y = (all_grid_coords[:, :, 1] >= 0) & (all_grid_coords[:, :, 1] < height)

            # Identify triangles where all vertices are within bounds
            # triangle_in_bounds_mask shape: (num_faces,)
            # A triangle is in bounds if all its vertices are in bounds.
            triangle_in_bounds_mask = np.all(vert_coords_in_bounds_x & vert_coords_in_bounds_y, axis=1)

            # Select the grid coordinates of valid triangles and cast to int32 for cv2.fillPoly
            # polygons_to_fill shape: (num_valid_faces, num_vertices_per_face, 2)
            polygons_to_fill = all_grid_coords[triangle_in_bounds_mask].astype(np.int32)

            # Fill these polygons on the mask using OpenCV
            if polygons_to_fill.shape[0] > 0:
                cv2.fillPoly(rock_mask, polygons_to_fill, 1)
        
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
        safety_margin_size = int(2.0 / resolution)  # 2 meter safety margin
        kernel_safety = np.ones((safety_margin_size, safety_margin_size), np.uint8)
        safe_rock_mask = cv2.dilate(rock_mask, kernel_safety, iterations=1)
        
        print(f"Rock mask created: {np.sum(rock_mask)} cells marked as rocks")
        print(f"Safety mask created: {np.sum(safe_rock_mask)} cells marked as unsafe")
        
        return rock_mask.astype(np.int32), safe_rock_mask.astype(np.int32)
    
    def calculate_gradient_mask(self, heightmap: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate gradient mask for terrain steepness"""
        import cv2
        from scipy import ndimage
        from scipy.signal import convolve2d
        
        # Sobel operators for gradient in x and y directions (same as terrain_utils.py)
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
        
        # Debug: Print gradient statistics
        print(f"Gradient magnitude - Min: {np.min(grad_magnitude):.4f}, Max: {np.max(grad_magnitude):.4f}, Mean: {np.mean(grad_magnitude):.4f}")
        print(f"Gradient threshold: {threshold}")
        print(f"Pixels above threshold: {np.sum(grad_magnitude > threshold)}")
        
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
        
        # Create safety margin for spawn locations
        # This creates a larger exclusion zone around steep areas for safer navigation
        safety_margin_size = int(2.0 / self.resolution_in_m)  # 2 meter safety margin
        kernel_safety = np.ones((safety_margin_size, safety_margin_size), np.uint8)
        safe_gradient_mask = cv2.dilate(gradient_mask, kernel_safety, iterations=1)
        
        print(f"Gradient mask created: {np.sum(gradient_mask)} cells marked as steep")
        print(f"Safety mask created: {np.sum(safe_gradient_mask)} cells marked as unsafe")
        
        return gradient_mask.astype(np.int32), safe_gradient_mask.astype(np.int32)
    
    def generate_spawn_locations(self, n_spawns: int = 100, border_offset: float = 20.0, seed: int = None) -> np.ndarray:
        """Generate random spawn locations"""
        if seed is not None:
            np.random.seed(seed)
        
        height, width = self.safe_rock_mask.shape
        min_xy = int(border_offset / self.resolution_in_m)
        max_xy = int(min(height, width) - min_xy)
        
        spawn_locations = np.zeros((n_spawns, 3), dtype=np.float32)
        
        for i in range(n_spawns):
            valid_location = False
            attempts = 0
            max_attempts = 1000
            
            while not valid_location and attempts < max_attempts:
                x = np.random.randint(min_xy, max_xy)
                y = np.random.randint(min_xy, max_xy)
                
                if self.safe_rock_mask[y, x] == 0 and self.safe_gradient_mask[y, x] == 0:
                    valid_location = True
                    spawn_locations[i, 0] = x
                    spawn_locations[i, 1] = y
                    spawn_locations[i, 2] = self.heightmap_manager.heightmap[y, x]
                
                attempts += 1
            
            if attempts >= max_attempts:
                print(f"Warning: Could not find valid location for spawn {i}")
        
        # Scale and offset
        spawn_locations[:, 0] = spawn_locations[:, 0] * self.resolution_in_m + self.heightmap_manager.min_x
        spawn_locations[:, 1] = spawn_locations[:, 1] * self.resolution_in_m + self.heightmap_manager.min_y
        
        return spawn_locations
    
    def visualize_terrain_gradients_only(self, spawn_locations: np.ndarray):
        """Visualize terrain gradients only (excluding rocks) with spawn points"""
        plt.figure(figsize=(15, 10))
        
        # Convert spawn locations to grid coordinates for plotting
        spawn_grid_x = (spawn_locations[:, 0] - self.heightmap_manager.min_x) / self.resolution_in_m
        spawn_grid_y = (spawn_locations[:, 1] - self.heightmap_manager.min_y) / self.resolution_in_m
        
        # Show terrain-only heightmap as background
        plt.imshow(self.terrain_only_heightmap_manager.heightmap, cmap='terrain', origin='lower', alpha=0.8)
        
        # Overlay gradient mask (steep terrain only, no rocks)
        gradient_mask_overlay = np.ma.masked_where(self.gradient_mask == 0, self.gradient_mask)
        plt.imshow(gradient_mask_overlay, cmap='Purples', alpha=0.9, origin='lower', vmin=0, vmax=1)
        
        # Overlay gradient safety mask
        safety_gradient_mask_overlay = np.ma.masked_where(self.safe_gradient_mask == 0, self.safe_gradient_mask)
        plt.imshow(safety_gradient_mask_overlay, cmap='Blues', alpha=0.4, origin='lower', label='Gradient Safety Zone')
        
        # Plot spawn points
        plt.scatter(spawn_grid_x, spawn_grid_y, c='cyan', marker='o', s=30, 
                   edgecolor='blue', linewidth=1, label='Spawn Points', zorder=5)
        
        plt.colorbar(label='Height (m)')
        plt.title('Terrain Gradients Only (No Rocks) with Spawn Points')
        plt.xlabel('X Grid Coordinate')
        plt.ylabel('Y Grid Coordinate')
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='purple', alpha=0.9, label='Steep Terrain'),
            Patch(facecolor='blue', alpha=0.4, label='Gradient Safety Zones'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
                      markeredgecolor='blue', markersize=8, label='Spawn Points')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()

    def visualize_combined_rock_gradient_mask(self, spawn_locations: np.ndarray):
        """Visualize combined rock and gradient masks with spawn points"""
        plt.figure(figsize=(15, 10))
        
        # Convert spawn locations to grid coordinates for plotting
        spawn_grid_x = (spawn_locations[:, 0] - self.heightmap_manager.min_x) / self.resolution_in_m
        spawn_grid_y = (spawn_locations[:, 1] - self.heightmap_manager.min_y) / self.resolution_in_m
        
        # Show combined heightmap as background
        plt.imshow(self.heightmap_manager.heightmap, cmap='terrain', origin='lower', alpha=0.7)
        
        # Overlay gradient mask (steep terrain)
        gradient_mask_overlay = np.ma.masked_where(self.gradient_mask == 0, self.gradient_mask)
        plt.imshow(gradient_mask_overlay, cmap='Purples', alpha=0.8, origin='lower', vmin=0, vmax=1)

        # Overlay rock mask
        rock_mask_overlay = np.ma.masked_where(self.rock_mask == 0, self.rock_mask)
        plt.imshow(rock_mask_overlay, cmap='Reds', alpha=0.8, origin='lower', label='Rock Mask')
        
        # Overlay rock safety mask
        safety_rock_mask_overlay = np.ma.masked_where(self.safe_rock_mask == 0, self.safe_rock_mask)
        plt.imshow(safety_rock_mask_overlay, cmap='Oranges', alpha=0.3, origin='lower', label='Rock Safety Zone')
        
        # Overlay gradient safety mask
        safety_gradient_mask_overlay = np.ma.masked_where(self.safe_gradient_mask == 0, self.safe_gradient_mask)
        plt.imshow(safety_gradient_mask_overlay, cmap='Blues', alpha=0.3, origin='lower', label='Gradient Safety Zone')
        
        # Plot spawn points
        plt.scatter(spawn_grid_x, spawn_grid_y, c='cyan', marker='o', s=30, 
                   edgecolor='blue', linewidth=1, label='Spawn Points', zorder=5)
        
        plt.colorbar(label='Height (m)')
        plt.title('Combined Rock and Gradient Masks with Spawn Points')
        plt.xlabel('X Grid Coordinate')
        plt.ylabel('Y Grid Coordinate')
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.8, label='Rock Areas'),
            Patch(facecolor='purple', alpha=0.8, label='Steep Terrain'),
            Patch(facecolor='orange', alpha=0.3, label='Rock Safety Zones'),
            Patch(facecolor='blue', alpha=0.3, label='Gradient Safety Zones'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
                      markeredgecolor='blue', markersize=8, label='Spawn Points')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_gradient_debug(self):
        """Debug visualization to show gradient mask separately"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Show original heightmap
        im1 = axes[0].imshow(self.heightmap_manager.heightmap, cmap='terrain', origin='lower')
        axes[0].set_title('Original Heightmap')
        plt.colorbar(im1, ax=axes[0])
        
        # Show gradient mask only
        im2 = axes[1].imshow(self.gradient_mask, cmap='viridis', origin='lower')
        axes[1].set_title(f'Gradient Mask (threshold={self.gradient_threshold})')
        plt.colorbar(im2, ax=axes[1])
        
        # Show gradient safety mask
        im3 = axes[2].imshow(self.safe_gradient_mask, cmap='plasma', origin='lower')
        axes[2].set_title('Gradient Safety Mask')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
        
        # Print some statistics
        print(f"Gradient mask stats:")
        print(f"  Total cells: {self.gradient_mask.size}")
        print(f"  Steep cells: {np.sum(self.gradient_mask)}")
        print(f"  Percentage steep: {100 * np.sum(self.gradient_mask) / self.gradient_mask.size:.2f}%")
    
    def resize_terrain_heightmap_to_match_combined(self):
        """Resize terrain-only heightmap to match combined heightmap dimensions"""
        # Get target dimensions from combined heightmap
        target_height, target_width = self.heightmap_manager.heightmap.shape
        target_min_x = self.heightmap_manager.min_x
        target_min_y = self.heightmap_manager.min_y
        target_max_x = self.heightmap_manager.max_x
        target_max_y = self.heightmap_manager.max_y
        
        # Initialize terrain-only heightmap with same dimensions and bounds
        terrain_heightmap = np.full((target_height, target_width), -99.0, dtype=np.float32)
        
        # Calculate cell size
        cell_size_x = (target_max_x - target_min_x) / (target_width - 1)
        cell_size_y = (target_max_y - target_min_y) / (target_height - 1)
        
        if len(self.terrain_faces) > 0:
            # Get all triangle vertices from terrain only
            face_vertices = self.terrain_vertices[self.terrain_faces]
            
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
            for idx in range(len(self.terrain_faces)):
                i_range = max_i[idx] - min_i[idx] + 1
                j_range = max_j[idx] - min_j[idx] + 1
                
                if i_range > 0 and j_range > 0:
                    terrain_heightmap[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1] = np.maximum(
                        terrain_heightmap[min_j[idx]:max_j[idx]+1, min_i[idx]:max_i[idx]+1],
                        max_z_tri[idx]
                    )
        
        print(f"Terrain-only heightmap resized to match combined dimensions: {terrain_heightmap.shape}")
        return terrain_heightmap

    def visualize_terrain_rocks_only(self, spawn_locations: np.ndarray):
        """Visualize rock areas only (excluding terrain gradients) with spawn points"""
        plt.figure(figsize=(15, 10))
        
        # Convert spawn locations to grid coordinates for plotting
        spawn_grid_x = (spawn_locations[:, 0] - self.heightmap_manager.min_x) / self.resolution_in_m
        spawn_grid_y = (spawn_locations[:, 1] - self.heightmap_manager.min_y) / self.resolution_in_m
        
        # Show combined heightmap as background
        plt.imshow(self.heightmap_manager.heightmap, cmap='terrain', origin='lower', alpha=0.8)
        
        # Overlay rock mask only
        rock_mask_overlay = np.ma.masked_where(self.rock_mask == 0, self.rock_mask)
        plt.imshow(rock_mask_overlay, cmap='Reds', alpha=0.9, origin='lower', vmin=0, vmax=1)
        
        # Overlay rock safety mask
        safety_rock_mask_overlay = np.ma.masked_where(self.safe_rock_mask == 0, self.safe_rock_mask)
        plt.imshow(safety_rock_mask_overlay, cmap='Oranges', alpha=0.4, origin='lower', label='Rock Safety Zone')
        
        # Plot spawn points
        plt.scatter(spawn_grid_x, spawn_grid_y, c='cyan', marker='o', s=30, 
                   edgecolor='blue', linewidth=1, label='Spawn Points', zorder=5)
        
        plt.colorbar(label='Height (m)')
        plt.title('Rock Areas Only (No Terrain Gradients) with Spawn Points')
        plt.xlabel('X Grid Coordinate')
        plt.ylabel('Y Grid Coordinate')
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.9, label='Rock Areas'),
            Patch(facecolor='orange', alpha=0.4, label='Rock Safety Zones'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
                      markeredgecolor='blue', markersize=8, label='Spawn Points')
        ]
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
        ax1.imshow(self.terrain_only_heightmap_manager.heightmap, cmap='terrain', origin='lower', alpha=0.8)
        
        # Overlay gradient mask (steep terrain only, no rocks)
        gradient_mask_overlay = np.ma.masked_where(self.gradient_mask == 0, self.gradient_mask)
        ax1.imshow(gradient_mask_overlay, cmap='Purples', alpha=0.9, origin='lower', vmin=0, vmax=1)
        
        # Overlay gradient safety mask
        safety_gradient_mask_overlay = np.ma.masked_where(self.safe_gradient_mask == 0, self.safe_gradient_mask)
        ax1.imshow(safety_gradient_mask_overlay, cmap='Blues', alpha=0.4, origin='lower')
        
        # Plot spawn points
        ax1.scatter(spawn_grid_x, spawn_grid_y, c='cyan', marker='o', s=20, 
                   edgecolor='blue', linewidth=1, zorder=5)
        
        ax1.set_title('Terrain Gradients Only (No Rocks)', fontsize=14)
        ax1.set_xlabel('X Grid Coordinate')
        ax1.set_ylabel('Y Grid Coordinate')
        
        # Right subplot: Rocks only
        ax2.imshow(self.heightmap_manager.heightmap, cmap='terrain', origin='lower', alpha=0.8)
        
        # Overlay rock mask only
        rock_mask_overlay = np.ma.masked_where(self.rock_mask == 0, self.rock_mask)
        ax2.imshow(rock_mask_overlay, cmap='Reds', alpha=0.9, origin='lower', vmin=0, vmax=1)
        
        # Overlay rock safety mask
        safety_rock_mask_overlay = np.ma.masked_where(self.safe_rock_mask == 0, self.safe_rock_mask)
        ax2.imshow(safety_rock_mask_overlay, cmap='Oranges', alpha=0.4, origin='lower')
        
        # Plot spawn points
        ax2.scatter(spawn_grid_x, spawn_grid_y, c='cyan', marker='o', s=20, 
                   edgecolor='blue', linewidth=1, zorder=5)
        
        ax2.set_title('Rock Areas Only (No Terrain Gradients)', fontsize=14)
        ax2.set_xlabel('X Grid Coordinate')
        ax2.set_ylabel('Y Grid Coordinate')
        
        # Create custom legend for both subplots
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='purple', alpha=0.9, label='Steep Terrain'),
            Patch(facecolor='red', alpha=0.9, label='Rock Areas'),
            Patch(facecolor='blue', alpha=0.4, label='Gradient Safety Zones'),
            Patch(facecolor='orange', alpha=0.4, label='Rock Safety Zones'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
                      markeredgecolor='blue', markersize=8, label='Spawn Points')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=5)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)  # Make room for legend
        plt.show()

def main():
    """Main debug function"""
    print("=== Standalone Terrain Debug Tool ===")
    
    # Configuration - Mars terrain paths
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "assets", "terrains", "mars", "terrain1")
    terrain_usd_path = os.path.join(base_path, "terrain_only.usd")
    rock_usd_path = os.path.join(base_path, "rocks_merged.usd")
    
    terrain_prim_path = None  # Will auto-detect first mesh if None
    rock_prim_path = None     # Will auto-detect first mesh if None
    
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
        # Initialize debugger with separate terrain and rock paths
        debugger = StandaloneTerrainDebugger(
            terrain_usd_path=terrain_usd_path,
            rock_usd_path=rock_usd_path,
            terrain_prim_path=terrain_prim_path,
            rock_prim_path=rock_prim_path
        )
        
        # Generate spawn locations
        print("Generating spawn locations...")
        spawn_locations = debugger.generate_spawn_locations(n_spawns=100, seed=42)
        print(f"Generated {len(spawn_locations)} spawn locations")
        
        # Visualizations
        print("Creating visualizations...")
        
        # Show terrain analysis in subplots (gradients and rocks side by side)
        debugger.visualize_terrain_analysis_subplots(spawn_locations)
        
        # Show combined rock and gradient masks - Second plot
        debugger.visualize_combined_rock_gradient_mask(spawn_locations)
        
        # Test height queries
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
            
        print(f"Testing height queries on {device}...")
        test_positions = torch.tensor([[5.0, 5.0], [10.0, 10.0], [15.0, 15.0]], device=device)
        heights = debugger.heightmap_manager.get_height_at(test_positions)
        print(f"Test positions: {test_positions}")
        print(f"Heights: {heights}")
        
        print("Debug session completed successfully!")
        
    except Exception as e:
        print(f"Error during debug session: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
