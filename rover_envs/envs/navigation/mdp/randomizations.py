from typing import TYPE_CHECKING  # noqa: F401

import torch
import warp as wp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

from ..utils.terrains.terrain_importer import RoverTerrainImporter


def reset_root_state_rover(
    env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg, z_offset: float = 0.5
):
    """
    Generate random root states for the rovers, based on terrain_based_spawn_locations.
    """
    # Get the rover asset
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Get the terrain and sample new spawn locations
    terrain: RoverTerrainImporter = env.scene.terrain
    spawn_locations = terrain.get_spawn_locations()
    spawn_index = torch.randperm(len(spawn_locations), device=env.device)[: len(env_ids)]
    spawn_locations = spawn_locations[spawn_index].clone()

    # Add a small z offset to the spawn locations to avoid spawning the rover inside the terrain.
    positions = spawn_locations
    positions[:, 2] += z_offset

    # Random yaw about world Z-axis. Isaac Lab 3.0 uses quaternion ordering (x, y, z, w).
    default_root_pose = wp.to_torch(asset.data.default_root_pose)[env_ids].clone()
    default_root_vel = wp.to_torch(asset.data.default_root_vel)[env_ids].clone()
    angle = torch.rand(len(env_ids), device=env.device) * 2 * torch.pi
    yaw_quat = math_utils.quat_from_euler_xyz(
        torch.zeros_like(angle), torch.zeros_like(angle), angle
    )
    orientations = math_utils.quat_mul(default_root_pose[:, 3:7], yaw_quat)

    # Update the environment origins, so that the terrain targets are sampled around the new origin.
    env.scene.terrain.env_origins[env_ids] = positions
    # Set the root state.
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=default_root_vel, env_ids=env_ids)
