# from omni.isaac.lab.markers.visualization_markers import VisualizationMarkersCfg, VisualizationMarkers
# import omni.isaac.lab.sim as sim_utils
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pymeshlab
import torch

from rover_envs.envs.navigation.utils.terrains.usd_utils import get_triangles_and_vertices_from_prim

directory_terrain_utils = os.path.dirname(os.path.abspath(__file__))
# SPHERE_MARKER_CFG = VisualizationMarkersCfg(
#     markers={
#         "sphere": sim_utils.SphereCfg(
#             radius=0.4,
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
#         ),
#     }
# )


class HeightmapManager():

    def __init__(self, resolution_in_m, vertices, faces):
        self.resolution_in_m = resolution_in_m
        self.heightmap, self.min_x, self.min_y, self.max_x, self.max_y = self.mesh_to_heightmap(vertices, faces)
        if torch.cuda.is_available():
            self.heightmap_tensor = torch.from_numpy(self.heightmap).cuda()
            self.offset_tensor = torch.tensor([self.min_x, self.min_y]).cuda()

    def mesh_to_heightmap(self, vertices, faces):
        # Border Margin
        border_margin = 1.0
        # Define bounding box
        min_x, min_y, _ = np.min(vertices, axis=0) + border_margin
        max_x, max_y, _ = np.max(vertices, axis=0) - border_margin

        # Calculate the grid size
        # grid_size = grid_size_in_m / resolution_in_m
        grid_size_x = (max_x - min_x) / self.resolution_in_m
        grid_size_y = (max_y - min_y) / self.resolution_in_m

        # Initialize the heightmap
        heightmap = np.ones((int(grid_size_x+1), int(grid_size_y+1)), dtype=np.float32) * -99

        # Calculate the size of a grid cell
        cell_size_x = (max_x - min_x) / (grid_size_x)
        cell_size_y = (max_y - min_y) / (grid_size_y)

        # Iterate over each face to update the heightmap
        for face in faces:
            # Get the vertices
            v1, v2, v3 = vertices[face]
            # Project to 2D grid
            min_i = int((min(v1[0], v2[0], v3[0]) - min_x) / cell_size_x)
            max_i = min(int((max(v1[0], v2[0], v3[0]) - min_x) / cell_size_x), heightmap.shape[1] - 1)
            min_j = int((min(v1[1], v2[1], v3[1]) - min_y) / cell_size_y)
            max_j = min(int((max(v1[1], v2[1], v3[1]) - min_y) / cell_size_y), heightmap.shape[0] - 1)

            # Update the heightmap
            for i in range(min_i, max_i + 1):
                for j in range(min_j, max_j + 1):
                    heightmap[j, i] = max(heightmap[j, i], v1[2], v2[2], v3[2])

        return heightmap, min_x, min_y, max_x, max_y

    def get_heightmap(self):
        return self.heightmap

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

    def get_minimums(self):
        return self.min_x, self.min_y


class TerrainManager():

    def __init__(self, num_envs: int, device: str):
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        terrain_path = os.path.join(self.dir_path, "../terrain_data/map.ply")
        rock_mesh_path = os.path.join(self.dir_path, "../terrain_data/big_stones.ply")
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
        self.gradient_threshold = 0.3

        # Load Terrain
        print("Getting triangles and vertices from USD file")
        vertices, faces = self.get_mesh(self.meshes["terrain"])
        print("Generating heightmap")
        self._heightmap_manager = HeightmapManager(self.resolution_in_m, vertices, faces)

        # Generate Rock Mask
        print("Generating rock mask")
        self.rock_mask, self.safe_rock_mask = self.find_rocks_in_heightmap(
            self._heightmap_manager.heightmap, self.gradient_threshold)

        # Generate Spawn Locations
        self.spawn_locations = self.random_rover_spawns(
            rock_mask=self.safe_rock_mask,
            heightmap=self._heightmap_manager.heightmap,
            n_spawns=num_envs*2 if num_envs > 100 else 200,
            border_offset=25.0,
            seed=12345)
        if device == 'cuda:0':
            self.spawn_locations = torch.from_numpy(self.spawn_locations).cuda()
            self.rock_mask_tensor = torch.from_numpy(self.safe_rock_mask).cuda().unsqueeze(-1)

        # if not hasattr(self, "sphere_goal_visualizer"):
        #     sphere_cfg = SPHERE_MARKER_CFG.copy()
        #     sphere_cfg.prim_path = "/Visuals/Command/position_goal"
        #     sphere_cfg.markers["sphere"].radius = 0.2
        #     self.sphere_goal_visualizer = VisualizationMarkers(sphere_cfg)

        # self.sphere_goal_visualizer.set_visibility(True)
        # self.sphere_goal_visualizer.visualize(self.spawn_locations[:, 0:3])

    def get_mesh(self, prim_path="/") -> Tuple[np.ndarray, np.ndarray]:
        """ This function reads a USD from the specified prim path and return vertices and faces.

        """

        # Assert that the specified path exists in the list of meshes.
        # assert path in self.meshes, f"The provided path '{path}' must exist in the 'self.meshes' list."

        # Get faces and vertices from the mesh using the provided prim path
        faces, vertices = get_triangles_and_vertices_from_prim(prim_path)

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

    # TODO Remove this function
    def get_valid_targets(self, target_positions: torch.Tensor, device: str = "cuda:0") -> torch.Tensor:
        """
        Computes the closest valid target positions from a set of potential target positions.
        Valid targets are determined based on a rock mask, which indicates traversable terrain.
        The computations are performed on the specified device (e.g., 'cuda' or 'cpu').

        Args:
            target_positions (torch.Tensor): A tensor of shape (N, 2) containing the target positions.
            device (str, optional): The device on which to perform the computations. Defaults to "cuda:0".

        Returns:
            torch.Tensor: A tensor of the closest valid target positions.
        """

        # Generate a coordinate grid based on the rock mask dimensions
        coordinate_grid_y, coordinate_grid_x = torch.meshgrid(
            torch.arange(self.rock_mask.shape[0], device=device),
            torch.arange(self.rock_mask.shape[1], device=device),
            indexing='ij'
        )
        coordinate_grid = torch.stack((coordinate_grid_x, coordinate_grid_y), dim=-1).view(-1, 2)

        # Adjust grid coordinates based on resolution and offset
        scaled_grid = coordinate_grid.float()
        scaled_grid *= self.resolution_in_m
        scaled_grid[:, 0] += self._heightmap_manager.min_x
        scaled_grid[:, 1] += self._heightmap_manager.min_y

        # Flatten the target positions and transfer to the correct device
        target_positions_flat = target_positions.view(-1, 2).to(device)

        # Compute pairwise distances between grid points and target positions
        distances = torch.cdist(scaled_grid, target_positions_flat, p=2.0).view(*self.rock_mask.shape, -1)

        # Mask out distances on the rock mask to consider only valid positions
        distances_masked = torch.where(self.rock_mask_tensor == 1, torch.inf, distances)

        # Find the closest valid target positions
        min_distances, min_indices = torch.min(distances_masked.view(-1, distances_masked.size(2)), dim=0)

        # Returns closest valid target positions
        return scaled_grid[min_indices]

    # TODO finish this function
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

        reset_buf = torch.where(self.rock_mask_tensor[grid_cell[:, 1], grid_cell[:, 0]] == 1, 1, 0).squeeze(-1)
        env_ids = env_ids[reset_buf == 1]
        reset_buf_len = len(env_ids)
        return env_ids, reset_buf_len

    def mesh_to_heightmap(self, vertices, faces, grid_size_in_m, resolution_in_m=0.1):

        # Border Margin
        border_margin = 1.0
        # Define bounding box
        min_x, min_y, _ = np.min(vertices, axis=0) + border_margin
        max_x, max_y, _ = np.max(vertices, axis=0) - border_margin

        # Calculate the grid size
        # grid_size = grid_size_in_m / resolution_in_m
        grid_size_x = (max_x - min_x) / resolution_in_m
        grid_size_y = (max_y - min_y) / resolution_in_m

        # Initialize the heightmap
        heightmap = np.ones((int(grid_size_x+1), int(grid_size_y+1)), dtype=np.float32) * -99

        # Calculate the size of a grid cell
        cell_size_x = (max_x - min_x) / (grid_size_x)
        cell_size_y = (max_y - min_y) / (grid_size_y)

        # Iterate over each face to update the heightmap
        for face in faces:
            # Get the vertices
            v1, v2, v3 = vertices[face]
            # Project to 2D grid
            min_i = int((min(v1[0], v2[0], v3[0]) - min_x) / cell_size_x)
            max_i = min(int((max(v1[0], v2[0], v3[0]) - min_x) / cell_size_x), heightmap.shape[1] - 1)
            min_j = int((min(v1[1], v2[1], v3[1]) - min_y) / cell_size_y)
            max_j = min(int((max(v1[1], v2[1], v3[1]) - min_y) / cell_size_y), heightmap.shape[0] - 1)

            # Update the heightmap
            try:
                for i in range(min_i, max_i + 1):
                    for j in range(min_j, max_j + 1):
                        heightmap[j, i] = max(heightmap[j, i], v1[2], v2[2], v3[2])
            except Exception as e:
                print(f"Error: {e}, max_i: {max_i}, max_j: {max_j}, min_i: {min_i}, min_j: {min_j}")

        return heightmap

    def find_rocks_in_heightmap(self, heightmap, threshold=0.1):
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

        # Initialize a rock mask with zeros
        rock_mask = np.zeros_like(heightmap, dtype=np.int32)
        # Mark the areas where gradient magnitude is greater than the threshold as rocks
        rock_mask[grad_magnitude > threshold] = 1

        # Perform dilation to add a safety margin around the rocks
        kernel = np.ones((3, 3), np.uint8)
        rock_mask = cv2.morphologyEx(rock_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        # = cv2.morphologyEx(rock_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        # # Perform dilation to add a safety margin around the rocks
        # closed_rock_mask = binary_dilation(rock_mask, iterations=1)
        # filled_rock_mask = ndimage.binary_fill_holes(closed_rock_mask).astype(int)
        rock_mask = ndimage.binary_fill_holes(rock_mask).astype(int)

        kernel = np.ones((7, 7), np.uint8)
        rock_mask = cv2.morphologyEx(rock_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        kernel = np.ones((11, 11), np.uint8)
        rock_mask = cv2.dilate(rock_mask.astype(np.uint8), kernel, iterations=1)
        # # Safety margin around the rocks
        kernel = np.ones((42, 42), np.uint8)
        safe_rock_mask = cv2.dilate(rock_mask.astype(np.uint8), kernel, iterations=1)

        return rock_mask, safe_rock_mask

    def show_heightmap(self, heightmap, name="2D Heightmap", vmax=None):
        plt.figure(figsize=(10, 10))

        if vmax is None:
            vmax = np.max(heightmap)
            vmin = np.min(heightmap)
        else:
            vmin = np.min(heightmap)
        # Display the heightmap
        plt.imshow(heightmap, cmap='terrain', origin='lower', vmin=vmin, vmax=vmax, extent=[0, 200, 0, 200])

        # Add a color bar for reference
        plt.colorbar(label='Height')

        # Add labels and title
        plt.xlabel('X Coordinate (m)')
        plt.ylabel('Y Coordinate (m)')
        plt.title(f"{name}")

        # Show the plot
        plt.show()

    def random_rover_spawns(
            self,
            rock_mask: np.ndarray,
            heightmap, n_spawns: int = 100,
            border_offset: float = 20.0,
            seed=None
    ) -> np.ndarray:
        """Generate random rover spawn locations. Calculates random x,y checks if it is a rock, if not,
        add to list of spawn locations with corresponding z value from heightmap.

        Args:
            rock_mask (np.ndarray): A binary mask indicating the locations of rocks.
            n_spawns (int, optional): The number of spawn locations to generate. Defaults to 1.
            min_dist (float, optional): The minimum distance between two spawn locations. Defaults to 1.0.

        Returns:
            np.ndarray: An array of shape (n_spawns, 3) containing the spawn locations.
        """
        # max_xy = int(max_xy / self.resolution_in_m)
        # min_xy = int(min_xy / self.resolution_in_m)

        # Set the random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Get the heightmap dimensions
        height, width = rock_mask.shape
        min_xy = int(border_offset / self.resolution_in_m)
        max_xy = int((min(height, width) - min_xy))

        assert max_xy < width, f"max_xy ({max_xy}) must be less than width ({width})"
        assert max_xy < height, f"max_xy ({max_xy}) must be less than height ({height})"

        # Initialize the spawn locations array
        spawn_locations = np.zeros((n_spawns, 3), dtype=np.float32)

        # Generate spawn locations
        for i in range(n_spawns):

            valid_location = False
            while not valid_location:
                # Generate a random x and y
                x = np.random.randint(min_xy, max_xy)
                y = np.random.randint(min_xy, max_xy)

                # Check if the location is too close to a previous location
                if rock_mask[y, x] == 0:
                    valid_location = True
                    spawn_locations[i, 0] = x
                    spawn_locations[i, 1] = y
                    spawn_locations[i, 2] = heightmap[y, x]

        # Scale and offset the spawn locations
        spawn_locations[:, 0] = spawn_locations[:, 0] * self.resolution_in_m + self._heightmap_manager.min_x
        spawn_locations[:, 1] = spawn_locations[:, 1] * self.resolution_in_m + self._heightmap_manager.min_y
        return spawn_locations


def show_heightmap(heightmap, cmap="terrain"):
    plt.figure(figsize=(10, 10))
    vmin = np.min(heightmap)
    vmax = np.max(heightmap)
    # Display the heightmap
    plt.imshow(heightmap, cmap=cmap, origin='lower', vmin=vmin, vmax=vmax)

    # Get current ticks
    x_ticks = plt.xticks()[0]
    y_ticks = plt.yticks()[0]

    # Scale ticks by 0.1
    x_ticks_scaled = x_ticks * 0.05
    y_ticks_scaled = y_ticks * 0.05

    # Set new ticks and labels
    plt.xticks(x_ticks, x_ticks_scaled)
    plt.yticks(y_ticks, y_ticks_scaled)

    # Add a color bar for reference
    plt.colorbar(label='Height')

    # Add labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('2D Heightmap')

    # Show the plot
    plt.show()


def visualize_spawn_points(spawn_locations: np.ndarray, heightmap: np.ndarray, scale=1):
    """
    Visualize the spawn locations on the heightmap as separate plots.

    Args:
        spawn_locations (np.ndarray): An array of shape (n_spawns, 3) containing the spawn locations.
        heightmap (np.ndarray): A 2D array representing the heightmap.
        rock_mask (np.ndarray): A binary mask indicating the locations of rocks.

    Returns:
        None
    """
    # Create a 3D plot for heightmap and spawn locations
    fig1 = plt.figure(figsize=(12, 12))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.set_title('3D Visualization of Spawn Locations and Heightmap')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_zlabel('Z Coordinate (Height)')
    # ax1.set_xlim([0, 600])  # Set the limits for the X-axis
    ax1.set_zlim([0, 10])
    x = np.linspace(0, heightmap.shape[1] - 1, heightmap.shape[1])
    y = np.linspace(0, heightmap.shape[0] - 1, heightmap.shape[0])
    X, Y = np.meshgrid(x, y)

    ax1.plot_surface(X, Y, heightmap, alpha=0.5, cmap='viridis')
    ax1.scatter(spawn_locations[:, 0], spawn_locations[:, 1], spawn_locations[:, 2]+0.5, c='r', marker='o', s=50)

    # Create a 2D plot for heightmap and spawn locations
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_title('2D Heightmap with Spawn Locations')
    ax2.set_xlabel('X Coordinate (m)')
    ax2.set_ylabel('Y Coordinate (m)')

    ax2.imshow(heightmap, cmap='terrain', origin='lower', extent=[0, 200, 0, 200])
    # ax2.imshow(np.ma.masked_where(rock_mask == 0, rock_mask), cmap='coolwarm', alpha=0.4)
    ax2.scatter(spawn_locations[:, 0], spawn_locations[:, 1], c='r', marker='o')

    plt.show()


if __name__ == "__main__":
    terrain = TerrainManager(device='cuda:0')

    # vertices, faces = terrain.load_mesh(terrain.meshes[0])
    # heightmap = terrain.mesh_to_heightmap(vertices, faces, grid_size_in_m=60, resolution_in_m=0.1)
    # rock_mask = terrain.find_rocks_in_heightmap(heightmap, threshold=0.7)
    # spawn_locations = terrain.random_rover_spawns(rock_mask=rock_mask, n_spawns=100, min_dist=1.0, seed=41)
    # vertices, faces = load_mesh()
    # heightmap = trimesh_to_heightmap(vertices, faces, grid_size_in_m=60, resolution_in_m=0.1)
    # rock_mask = find_rocks_in_heightmap(heightmap, threshold=0.7)
    # # show_heightmap(rock_mask)
    # # show_heightmap(heightmap)

    # # heightmap = np.random.rand(100, 100)  # Replace with your actual heightmap
    # # rock_mask = np.random.randint(0, 2, (100, 100))  # Replace with your actual rock mask

    # Generate spawn locations using the random_rover_spawns function
    # spawn_locations = random_rover_spawns(rock_mask=rock_mask, n_spawns=1000, min_dist=1.0, seed=41)

    # Visualize the spawn locations using the visualize_spawn_points function
    # show_heightmap(terrain._heightmap_manager.heightmap)
    # generate_random_grid_cell_tensor_100_by_100 = torch.randint(0, 100, (100, 100), device='cuda:0')
    # print(f'generate_random_grid_cell_tensor_100_by_100 shape: {generate_random_grid_cell_tensor_100_by_100.shape}')
    # Create x,y rand indices 2d tensor
    # random_indices = torch.randint(0, 100, (10, 2), device='cuda:0')
    # get_height_at = terrain._heightmap_manager.get_height_at(random_indices)
    # print(generate_random_grid_cell_tensor_100_by_100[random_indices[0], random_indices[1]])
    # print(get_height_at)

    target_positionss = torch.tensor([[5, 5], [10, 10], [15, 15], [11.1, 12.2], [50, 37]], device='cuda:0')
    # set torch seed
    torch.manual_seed(41)

    # Generate 100 random target float positions between 5 and 50
    target_positionss = torch.rand(100, 2, device='cuda:0') * 45 + 5
    # target_positionss = torch.rand(2, 2, device='cuda:0') * 45 + 5
    # print(terrain.get_valid_targets(target_positions=target_positionss,device='cuda:0'))
    target_poss = terrain.get_valid_targets(target_positions=target_positionss, device='cuda:0')
    zero_coloumn = torch.zeros((target_poss.shape[0], 1), device='cuda:0')
    target_poss = torch.cat((target_poss, zero_coloumn), dim=1)
    target_poss[:, 2] = terrain._heightmap_manager.get_height_at(target_poss[:, 0:2])
    target_poss = target_poss.cpu().numpy()
    # print(target_poss)
    target_poss[:, 0:2] = target_poss[:, 0:2] / terrain._heightmap_manager.resolution_in_m

    # visualize_spawn_points(target_poss, terrain._heightmap_manager.heightmap)
    visualize_spawn_points(target_poss, terrain.rock_mask)
    show_heightmap(terrain.safe_rock_mask, cmap='gray')

    # spawns = terrain.spawn_locations
    # spawns[:, 0:2] = spawns[:, 0:2] / terrain._heightmap_manager.resolution_in_m
    # #terrain.spawn_locations[0:2] = terrain.spawn_locations[0:2] * 10
    # visualize_spawn_points(spawns, terrain._heightmap_manager.heightmap)
