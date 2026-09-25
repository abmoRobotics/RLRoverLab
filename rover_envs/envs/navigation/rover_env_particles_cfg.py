"""Waypoint navigation on a Newton MPM soil layer.

The task, observations and actions match :class:`RoverEnvCfg`, so policies trained on rigid
terrain run unchanged. Newton couples an MJWarp rover solver to an implicit-MPM soil solver
through proxy wheel bodies. This needs the Isaac Lab and Newton versions in
``docker/Dockerfile.particles``.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg

from rover_envs.envs.navigation.rover_env_cfg import RoverEnvCfg, RoverSceneCfg
from rover_envs.envs.navigation.utils.terrains.soil import (
    SOIL_COLLIDER_LABEL,
    SoilLayerCfg,
    SoilMPMSolverCfg,
    SoilTerrainImporterCfg,
)

POLICY_DT = 0.2
"""Policy step [s]. The navigation policies were trained at 5 Hz."""


def soil_physics_cfg(soil: SoilLayerCfg, wheel_bodies: str) -> NewtonCfg:
    """Couple the rovers' MJWarp solver to the MPM soil through proxies of their wheels.

    Args:
        soil: Soil layer that the MPM grid covers.
        wheel_bodies: Newton body-label regex of the wheels that touch the soil.
    """
    return NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rover",
                    solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False, njmax=256, nconmax=512),
                    bodies=[r"/World/envs/env_.*/Robot"],
                    include_static_shapes=True,
                    substeps=3,
                ),
                CouplerEntryCfg(
                    name="soil",
                    solver_cfg=SoilMPMSolverCfg(voxel_size=soil.voxel_size, max_iterations=24),
                    bodies=[SOIL_COLLIDER_LABEL],
                    all_particles=True,
                    in_place=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(source="rover", destination="soil", bodies=[wheel_bodies], collision_pipeline=None)
            ],
        ),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
    )


@configclass
class RoverParticleSceneCfg(RoverSceneCfg):
    """Rover scene whose terrain carries a Newton MPM soil layer. Defaults to the debug terrain."""

    # Newton's coupled solvers do not report contact forces.
    contact_sensor = None

    def __post_init__(self):
        if self.terrain is None:
            self.set_terrain("debug")

    def set_terrain(self, terrain_name: str) -> None:
        """Set the terrain and keep the configured soil layer."""
        soil = self.terrain.soil if isinstance(self.terrain, SoilTerrainImporterCfg) else SoilLayerCfg()
        super().set_terrain(terrain_name)
        terrain = self.terrain
        self.terrain = SoilTerrainImporterCfg(
            prim_path=terrain.prim_path,
            terrain_type=terrain.terrain_type,
            collision_group=terrain.collision_group,
            usd_path=terrain.usd_path,
            spawn_obstacle_mesh_prim_path=terrain.spawn_obstacle_mesh_prim_path,
            soil=soil,
        )


@configclass
class RoverParticleEnvCfg(RoverEnvCfg):
    """Waypoint navigation over a shared MPM soil layer."""

    scene: RoverParticleSceneCfg = RoverParticleSceneCfg(
        num_envs=16, env_spacing=4.0, replicate_physics=True, clone_in_fabric=True
    )

    wheel_bodies: str = r"/World/envs/env_.*/Robot/.*_Drive"
    """Newton body-label regex of the wheels coupled to the soil."""

    def __post_init__(self):
        super().__post_init__()
        # Without contact forces, hitting a rock can neither be penalized nor end an episode.
        self.rewards.collision = None
        self.terminations.collision = None

        # MJWarp takes three substeps per MPM step. Decimation keeps the trained policy rate.
        self.sim.dt = 1 / 120
        self.decimation = round(POLICY_DT / self.sim.dt)
        self.sim.render_interval = 4
        self.scene.height_scanner.update_period = POLICY_DT
        self.update_physics()

    def update_physics(self) -> None:
        """Rebuild the Newton physics config. Call it after changing the soil layer."""
        self.sim.physics = soil_physics_cfg(self.scene.terrain.soil, self.wheel_bodies)
