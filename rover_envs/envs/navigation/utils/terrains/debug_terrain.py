
from terrain_utils import TerrainManager
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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


def debug_func():
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
            debug_mode=False,
            terrain_usd_path=terrain_usd_path,
            rock_usd_path=rock_usd_path
        )
        
        # Initialize visualizer
        visualizer = DebugVisualizer(terrain_manager)
        
        # Generate spawn locations
        #print("Generating spawn locations...")
        spawn_locations = terrain_manager.random_rover_spawns(
            safe_mask=np.logical_or(
                terrain_manager.safe_rock_mask,
                terrain_manager.safe_gradient_mask
            ),
            heightmap=terrain_manager._heightmap_manager.heightmap,
            n_spawns=200, 
            seed=42
        )
        spawn_locations_np = spawn_locations.cpu().numpy() if isinstance(spawn_locations, torch.Tensor) else spawn_locations
        #print(f"Generated {len(spawn_locations_np)} spawn locations")
        
        # Visualizations
        #print("Creating visualizations...")
        
        # Show terrain analysis in subplots (gradients and rocks side by side)
        visualizer.visualize_terrain_analysis_subplots(spawn_locations_np)
        
        # Show combined rock and gradient masks - Second plot
        visualizer.visualize_combined_rock_gradient_mask(spawn_locations_np)
        
        # Test height queries
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #print(f"Testing height queries on {device}...")
        test_positions = torch.tensor([[5.0, 5.0], [10.0, 10.0], [15.0, 15.0]], device=device)
        heights = terrain_manager._heightmap_manager.get_height_at(test_positions)
        #print(f"Test positions: {test_positions}")
        #print(f"Heights: {heights}")
        #print(terrain_manager.get_terrain_statistics())
        #print("Debug session completed successfully!")
        
    except Exception as e:
        print(f"Error during debug session: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_func()