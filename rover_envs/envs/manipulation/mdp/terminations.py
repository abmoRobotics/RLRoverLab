from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omni.isaac.lab.managers import SceneEntityCfg  # noqa: F401
from omni.isaac.lab.sensors import ContactSensor

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def collision_with_table(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """
    Checks for collision with obstacles.
    """
    # Accessing the contact sensor and its data
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Reshape as follows (num_envs, num_bodies, 3)
    print(contact_sensor.data.force_matrix_w.shape)
    force_matrix = contact_sensor.data.force_matrix_w.view(env.num_envs, -1, 3)

    # Calculating the force and returning true if it is above the threshold
    normalized_forces = torch.norm(force_matrix, dim=1)
    forces_active = torch.sum(normalized_forces, dim=-1) > 1

    return torch.where(forces_active, True, False)
