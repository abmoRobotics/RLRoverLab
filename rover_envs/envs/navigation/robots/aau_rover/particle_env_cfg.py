"""AAU rover waypoint navigation on Newton MPM soil.

Kept apart from ``env_cfg.py`` because it imports Isaac Lab's Newton MPM and coupling APIs,
which only the ``docker/Dockerfile.particles`` image provides.
"""

from __future__ import annotations

from isaaclab.utils import configclass

import rover_envs.mdp as mdp
from rover_envs.assets.robots.aau_rover_simple import aau_rover_simple_newton_cfg
from rover_envs.envs.navigation.rover_env_particles_cfg import RoverParticleEnvCfg


@configclass
class AAURoverParticleEnvCfg(RoverParticleEnvCfg):
    """Simple AAU rover driving to waypoints over a 30 x 30 m soil layer on the debug terrain."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = aau_rover_simple_newton_cfg("{ENV_REGEX_NS}/Robot")
        self.actions.actions = mdp.AckermannActionCfg(
            asset_name="robot",
            wheelbase_length=0.849,
            middle_wheel_distance=0.894,
            rear_and_front_wheel_distance=0.77,
            wheel_radius=0.1,
            min_steering_radius=0.8,
            steering_joint_names=[".*Steer_Revolute"],
            drive_joint_names=[".*Drive_Continuous"],
            offset=-0.0135,
        )
