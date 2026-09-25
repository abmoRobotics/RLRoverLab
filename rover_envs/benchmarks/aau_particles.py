"""Two AAU rover particle scenes with the same initial particle lattice."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from rover_envs.assets.robots.aau_rover_simple import AAU_ROVER_SIMPLE_CFG, aau_rover_simple_newton_cfg


DT = 1.0 / 120.0
BED_LOWER = (-0.75, -0.55, 0.0)
BED_UPPER = (0.75, 0.55, 0.12)
FLOOR_SIZE = (3.0, 2.0, 0.1)
FLOOR_POS = (0.0, 0.0, -0.05)
DENSITY = 1600.0
FRICTION = 0.65


def particle_lattice(voxel_size: float, particles_per_cell: int) -> tuple[list[tuple[float, float, float]], float, float]:
    """Return cell-centred particle positions, mass, and radius."""
    counts = [math.ceil(particles_per_cell * (hi - lo) / voxel_size) for lo, hi in zip(BED_LOWER, BED_UPPER)]
    cell = [(hi - lo) / count for lo, hi, count in zip(BED_LOWER, BED_UPPER, counts)]
    positions = [
        (BED_LOWER[0] + (i + 0.5) * cell[0], BED_LOWER[1] + (j + 0.5) * cell[1], BED_LOWER[2] + (k + 0.5) * cell[2])
        for i in range(counts[0])
        for j in range(counts[1])
        for k in range(counts[2])
    ]
    volume = math.prod(cell)
    return positions, DENSITY * volume, 0.5 * volume ** (1.0 / 3.0)


def make_sim_cfg(backend: str, device: str, voxel_size: float, num_envs: int) -> SimulationCfg:
    if backend == "physx":
        from isaaclab_physx.physics import PhysxCfg

        physics = PhysxCfg(gpu_max_particle_contacts=1 << 20)
    elif backend == "newton":
        from isaaclab_newton.physics import MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
        from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg

        wheel_bodies = r"/World/envs/env_.*/Robot/.*_Drive"
        physics = NewtonCfg(
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
                        solver_cfg=MPMSolverCfg(
                            voxel_size=voxel_size,
                            grid_type="sparse",
                            separate_worlds=True,
                            # Keep headroom for moving soil without paying for an oversized captured grid.
                            max_active_cell_count=int(2048 * num_envs * (0.08 / voxel_size) ** 3),
                            max_iterations=24,
                        ),
                        bodies=[r"/World/envs/env_.*/MPMFloor"],
                        all_particles=True,
                        in_place=True,
                    ),
                ],
                proxies=[
                    CouplerProxyMappingCfg(
                        source="rover", destination="soil", bodies=[wheel_bodies], collision_pipeline=None
                    )
                ],
            ),
            collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    cfg = SimulationCfg(dt=DT, device=device, physics=physics)
    if backend == "newton":
        cfg.use_newton_actuators = True
    return cfg


def make_scene_cfg(
    backend: str, num_envs: int, positions: list[tuple[float, float, float]], mass: float, radius: float
) -> InteractiveSceneCfg:
    if backend == "newton":
        from isaaclab_newton.assets import MPMObjectCfg
        from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg
        from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, MPMPointsCfg
        from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg

        rover_cfg = aau_rover_simple_newton_cfg("{ENV_REGEX_NS}/Robot")
        floor_collision = NewtonCollisionPropertiesCfg(contact_margin=0.004)
    else:
        rover_cfg = AAU_ROVER_SIMPLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        floor_collision = sim_utils.CollisionPropertiesCfg()
    rover_cfg.init_state.pos = (-0.1, 0.0, 0.45)

    @configclass
    class ParticleSceneCfg(InteractiveSceneCfg):
        robot = rover_cfg
        floor = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Floor",
            spawn=sim_utils.CuboidCfg(size=FLOOR_SIZE, collision_props=floor_collision),
            init_state=AssetBaseCfg.InitialStateCfg(pos=FLOOR_POS),
        )

        if backend == "newton":
            mpm_floor = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/MPMFloor",
                spawn=sim_utils.CuboidCfg(
                    size=FLOOR_SIZE,
                    rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
                    collision_props=NewtonCollisionPropertiesCfg(contact_margin=0.5 * radius),
                    visible=False,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=FLOOR_POS),
            )
            soil = MPMObjectCfg(
                prim_path="{ENV_REGEX_NS}/Soil",
                spawn=MPMPointsCfg(
                    positions=positions,
                    mass=mass,
                    radius=radius,
                    material=MPMParticleMaterialCfg(density=DENSITY, friction=FRICTION, yield_pressure=1.0e5),
                ),
            )

    return ParticleSceneCfg(
        num_envs=num_envs,
        env_spacing=3.5,
        replicate_physics=backend == "newton",
        clone_in_fabric=backend == "newton",
    )


def spawn_physx_particles(num_envs: int, positions: list[tuple[float, float, float]], mass: float, radius: float) -> None:
    """Author solid PhysX particles before the first simulation reset."""
    import omni.usd
    from omni.physx.scripts import particleUtils
    from pxr import Gf, Sdf, UsdShade

    stage = omni.usd.get_context().get_stage()
    system_path = Sdf.Path("/World/ParticleSystem")
    system = particleUtils.add_physx_particle_system(
        stage,
        system_path,
        particle_contact_offset=2.1 * radius,
        solid_rest_offset=radius,
        solver_position_iterations=8,
    )
    material_path = Sdf.Path("/World/ParticleMaterial")
    particleUtils.add_pbd_particle_material(stage, material_path, friction=FRICTION, density=DENSITY)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    UsdShade.MaterialBindingAPI.Apply(system.GetPrim()).Bind(material)

    points = [Gf.Vec3f(*p) for p in positions]
    velocities = [Gf.Vec3f(0.0)] * len(points)
    widths = [2.0 * radius] * len(points)
    for env_id in range(num_envs):
        path = Sdf.Path(f"/World/envs/env_{env_id}/Soil")
        particleUtils.add_physx_particleset_points(
            stage, path, points, velocities, widths, system_path, True, False, env_id, mass, DENSITY
        )
