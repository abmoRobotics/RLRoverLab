from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def obstacle_risk_cost(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return normalized C(x, y) for the robot position from the terrain obstacle risk map."""
    asset = env.scene[asset_cfg.name]
    root_pos_w = wp.to_torch(asset.data.root_link_pos_w)
    xy = root_pos_w[:, :2]

    terrain_manager = getattr(env.scene.terrain, "_terrainManager", None)
    risk_map = getattr(terrain_manager, "obstacle_risk_map", None)
    if risk_map is None:
        return torch.zeros(xy.shape[0], device=xy.device, dtype=torch.float32)

    risk_cost = risk_map.cost_at_world_xy(xy).to(device=xy.device, dtype=torch.float32)
    return risk_cost / env.max_episode_length
