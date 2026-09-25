"""Newton MPM soil layer draped over part of a rover terrain.

All rovers drive on one particle set in Newton's global world, the same way they share the
terrain mesh. While Newton builds its model, the terrain importer adds the particles and
particle-only copies of the terrain and rocks that hold them up. The rover solver keeps the
original terrain and rocks, so wheels that dig through the soil still stop on the ground.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp
from isaaclab.physics import PhysicsEvent
from isaaclab.utils import configclass
from isaaclab_newton.physics import MPMSolverCfg, NewtonManager
from isaaclab_newton.physics.mpm_manager import NewtonMPMManager, _make_solver_config
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg, create_mpm_particle_visualization
from newton.solvers import SolverImplicitMPM
from warp.fem import TemporaryStore

from .terrain_importer import RoverTerrainImporter
from .terrain_importer_cfg import RoverTerrainImporterCfg
from .usd_utils import check_prim_exists, get_triangles_and_vertices_from_prim


SOIL_COLLIDER_LABEL = "/World/terrain/soil_collider"
"""Newton body-label prefix of the kinematic terrain and rock copies that hold the soil."""

SOIL_PARTICLES_PRIM_PATH = "/World/terrain/soil/Particles"
"""USD ``Points`` prim that mirrors the soil particles in the Kit viewport."""


@configclass
class SoilLayerCfg:
    """Granular layer covering a rectangle of the terrain."""

    center: tuple[float, float] | None = None
    """Layer centre in world XY [m]. ``None`` centres the layer on the terrain."""

    size: tuple[float, float] = (30.0, 30.0)
    """Layer extent along X and Y [m]."""

    depth: float = 0.15
    """Layer thickness above the terrain surface [m]."""

    voxel_size: float = 0.1
    """MPM grid voxel size [m]."""

    particles_per_cell: int = 2
    """Particles per voxel edge. The lattice spacing is ``voxel_size / particles_per_cell``."""

    material: MPMParticleMaterialCfg = MPMParticleMaterialCfg(density=1600.0, friction=0.65, yield_pressure=1.0e5)
    """Soil material. Density, friction and yield pressure match the AAU particle benchmark."""

    rock_clearance: float = 0.1
    """Horizontal gap kept between particles and rocks [m]."""

    max_slope: float = 30.0
    """Terrain steeper than this stays bare [deg]. Sand slides off slopes near its friction angle."""

    collider_margin: float = 2.0
    """Distance the soil's terrain and rock colliders extend past the layer edges [m]."""

    spawn_margin: float = 1.5
    """Minimum distance between the layer edge and rover spawns or their targets [m]."""

    color: tuple[float, float, float] = (0.62, 0.5, 0.36)
    """Particle display colour in the Kit viewport."""

    visual_update_frequency: int = 1
    """Copy particle positions to the viewport every N rendered frames."""

    @property
    def spacing(self) -> float:
        """Particle lattice spacing [m]."""
        return self.voxel_size / self.particles_per_cell


class SharedSoilMPMSolver(SolverImplicitMPM):
    """Implicit MPM whose per-world resets keep the shared soil's warm starts."""

    def _validate_reset_warmstart_fields(self, world_mask):
        # Newton cannot clear grid warm starts for one world of a shared grid. They only seed the
        # next rheology solve, so a rover's reset keeps them. Full resets still clear them, and
        # masked resets still refresh the reset rovers' collider poses.
        if world_mask is not None:
            return None, None
        return super()._validate_reset_warmstart_fields(world_mask)


class SoilMPMManager(NewtonMPMManager):
    """Newton MPM manager that steps a :class:`SharedSoilMPMSolver`.

    With a :class:`SoilMPMSolverCfg`, it sizes the sparse grid from the voxels the particles
    occupy. A captured sparse grid runs every kernel over its full capacity, so a loose bound
    costs time and memory in proportion.
    """

    @classmethod
    def _create_solver(cls, model, solver_cfg: MPMSolverCfg) -> SharedSoilMPMSolver:
        if isinstance(solver_cfg, SoilMPMSolverCfg):
            voxels = np.unique(np.floor(model.particle_q.numpy() / solver_cfg.voxel_size).astype(np.int64), axis=0)
            leaves = np.unique(voxels >> 3, axis=0)  # NanoVDB leaves hold 8^3 voxels
            solver_cfg = solver_cfg.replace(
                max_active_cell_count=math.ceil(solver_cfg.grid_headroom * len(voxels)),
                max_leaf_node_count=math.ceil(solver_cfg.grid_headroom * len(leaves)),
            )
            print(
                f"[INFO] Soil MPM grid: {len(voxels):,} active cells in {len(leaves):,} leaves; capacity "
                f"{solver_cfg.max_active_cell_count:,} cells and {solver_cfg.max_leaf_node_count:,} leaves"
            )
        return SharedSoilMPMSolver(model, _make_solver_config(solver_cfg), temporary_store=TemporaryStore())


@configclass
class SoilMPMSolverCfg(MPMSolverCfg):
    """Implicit MPM for a shared soil layer, with a sparse grid sized from its particles."""

    class_type: type = SoilMPMManager

    grid_type: str = "sparse"

    grid_headroom: float = 1.5
    """Grid capacity relative to the cells and leaves the particles occupy at the start.

    Newton raises an error if moving soil outgrows it.
    """


class SoilTerrainImporter(RoverTerrainImporter):
    """Rover terrain with a shared Newton MPM soil layer.

    Rovers spawn far enough inside the layer that their targets, sampled at
    :attr:`target_distance`, also land on it.
    """

    cfg: SoilTerrainImporterCfg

    def __init__(self, cfg: SoilTerrainImporterCfg):
        super().__init__(cfg)
        self.soil_particle_offset = 0
        self.soil_particle_count = 0
        self._soil_bounds: tuple[float, float, float, float] | None = None
        self._soil_points: np.ndarray | None = None
        self._soil_spawn_locations: torch.Tensor | None = None
        self._soil_callbacks = [
            NewtonManager.register_callback(self._add_soil_to_model, PhysicsEvent.MODEL_INIT, name="rover_soil"),
            NewtonManager.register_callback(self._show_soil, PhysicsEvent.PHYSICS_READY, name="rover_soil_visual"),
        ]

    @property
    def soil_bounds(self) -> tuple[float, float, float, float]:
        """Layer bounds ``(min_x, min_y, max_x, max_y)`` in world coordinates [m]."""
        if self._soil_bounds is None:
            raise RuntimeError("The soil layer is created when the simulation starts.")
        return self._soil_bounds

    def get_spawn_locations(self) -> torch.Tensor:
        if self._soil_spawn_locations is None:
            self._soil_spawn_locations = self._spawn_locations_on_soil(super().get_spawn_locations())
        return self._soil_spawn_locations

    def _spawn_locations_on_soil(self, spawn_locations: torch.Tensor) -> torch.Tensor:
        min_x, min_y, max_x, max_y = self.soil_bounds
        margin = self.cfg.soil.spawn_margin + self.target_distance
        if min(max_x - min_x, max_y - min_y) <= 2.0 * margin:
            raise ValueError(
                f"A {max_x - min_x:.1f} x {max_y - min_y:.1f} m soil layer cannot keep {self.target_distance} m "
                f"targets {self.cfg.soil.spawn_margin} m inside its edges. Enlarge SoilLayerCfg.size."
            )
        x, y = spawn_locations[:, 0], spawn_locations[:, 1]
        inside = (x > min_x + margin) & (x < max_x - margin) & (y > min_y + margin) & (y < max_y - margin)
        if not inside.any():
            raise ValueError(
                f"No terrain spawn location lies {margin} m inside the soil layer {self.soil_bounds}. "
                "Move SoilLayerCfg.center over the terrain's spawn area."
            )
        return spawn_locations[inside]

    def _add_soil_to_model(self, _payload) -> None:
        """Add the soil particles and their terrain and rock colliders to Newton's model builder."""
        import newton

        soil = self.cfg.soil
        builder = NewtonManager._builder
        if builder is None:
            raise RuntimeError("Newton has no model builder to receive the soil layer.")

        terrain_faces, terrain_vertices = get_triangles_and_vertices_from_prim(f"{self.cfg.prim_path}/terrain")
        rocks = None
        obstacle_path = self.cfg.spawn_obstacle_mesh_prim_path
        if obstacle_path is not None and check_prim_exists(obstacle_path):
            rocks = get_triangles_and_vertices_from_prim(obstacle_path)

        if soil.center is None:
            center = 0.5 * (terrain_vertices[:, :2].min(axis=0) + terrain_vertices[:, :2].max(axis=0))
        else:
            center = np.asarray(soil.center, dtype=np.float32)
        half = 0.5 * np.asarray(soil.size, dtype=np.float32)
        self._soil_bounds = (*(center - half).tolist(), *(center + half).tolist())

        points, coverage = _drape_particles(
            bounds=self._soil_bounds,
            spacing=soil.spacing,
            depth=soil.depth,
            terrain=_warp_mesh(terrain_vertices, terrain_faces, self.device),
            rocks=None if rocks is None else _warp_mesh(rocks[1], rocks[0], self.device),
            rock_clearance=soil.rock_clearance,
            max_slope=soil.max_slope,
            top=float(terrain_vertices[:, 2].max() + 10.0),
            device=self.device,
        )
        if len(points) == 0:
            raise ValueError(f"The soil layer {self._soil_bounds} does not overlap the terrain.")

        radius = 0.5 * soil.spacing
        # Only the MPM solver sees these copies; rover contacts use the original terrain and rocks.
        collider_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            mu=soil.material.friction,
            margin=0.5 * radius,
            has_shape_collision=False,
            has_particle_collision=True,
            is_visible=False,
        )
        meshes = {"terrain": (terrain_faces, terrain_vertices)}
        if rocks is not None:
            meshes["rocks"] = rocks
        collider_faces = 0
        for name, (faces, vertices) in meshes.items():
            vertices, faces = _crop_mesh(vertices, faces, self._soil_bounds, margin=soil.collider_margin)
            if len(faces) == 0:
                continue
            # One body per mesh: Newton merges a body's shapes into one mesh, and the closest point
            # on a merged mesh can fall on a buried rock face and flip which side is inside.
            label = f"{SOIL_COLLIDER_LABEL}/{name}"
            body = builder.add_link(mass=0.0, is_kinematic=True, lock_inertia=True, label=label)
            builder.add_shape_mesh(
                body,
                mesh=newton.Mesh(vertices, faces.reshape(-1), compute_inertia=False),
                cfg=collider_cfg,
                label=f"{label}/mesh",
            )
            collider_faces += len(faces)

        count = len(points)
        self.soil_particle_offset = builder.particle_count
        builder.add_particles(
            pos=points.tolist(),
            vel=np.zeros_like(points).tolist(),
            mass=[soil.material.density * soil.spacing**3] * count,
            radius=[radius] * count,
            custom_attributes={
                f"mpm:{name}": float(value) for name, value in soil.material.to_dict().items() if name != "density"
            },
        )
        self.soil_particle_count = count
        self._soil_points = points
        print(
            f"[INFO] Soil layer {tuple(round(v, 2) for v in self._soil_bounds)}: {count:,} particles "
            f"({soil.spacing:.3f} m spacing, {soil.depth} m deep) covering {coverage:.0%} of the area; "
            f"colliders with {collider_faces:,} triangles"
        )

    def _show_soil(self, _payload) -> None:
        """Mirror the particles to a USD ``Points`` prim when something renders the stage."""
        from isaaclab.sim import SimulationContext

        sim = SimulationContext.instance()
        if self._soil_points is None or sim is None or not (sim.is_rendering or sim.can_render_rgb_array()):
            return
        create_mpm_particle_visualization(
            prim_paths=[SOIL_PARTICLES_PRIM_PATH],
            positions=self._soil_points[None],
            widths=np.full(self.soil_particle_count, self.cfg.soil.spacing, dtype=np.float32),
            color=self.cfg.soil.color,
        )
        NewtonManager.register_particle_visual_prim(
            SOIL_PARTICLES_PRIM_PATH,
            particle_offset=self.soil_particle_offset,
            particle_count=self.soil_particle_count,
            sync_frequency=self.cfg.soil.visual_update_frequency,
        )
        self._soil_points = None


@configclass
class SoilTerrainImporterCfg(RoverTerrainImporterCfg):
    """Rover terrain importer with a Newton MPM soil layer."""

    class_type: type = SoilTerrainImporter

    soil: SoilLayerCfg = SoilLayerCfg()
    """Soil layer placed on the terrain."""


def _warp_mesh(vertices: np.ndarray, faces: np.ndarray, device: str) -> wp.Mesh:
    return wp.Mesh(
        points=wp.array(vertices, dtype=wp.vec3, device=device),
        indices=wp.array(faces.reshape(-1), dtype=wp.int32, device=device),
    )


@wp.kernel
def _surface_heights(
    terrain: wp.uint64,
    rocks: wp.uint64,
    has_rocks: int,
    xy: wp.array(dtype=wp.vec2),
    top: float,
    rock_clearance: float,
    ground: wp.array(dtype=float),
    blocked: wp.array(dtype=wp.int32),
):
    """Cast down onto the terrain (NaN on a miss) and flag columns within ``rock_clearance`` of a rock."""
    tid = wp.tid()
    down = wp.vec3(0.0, 0.0, -1.0)
    hit = wp.mesh_query_ray(terrain, wp.vec3(xy[tid][0], xy[tid][1], top), down, 1.0e6)
    if not hit.result:
        ground[tid] = wp.nan
        return
    height = top - hit.t
    ground[tid] = height
    if has_rocks == 0:
        return
    for i in range(5):
        offset = wp.vec2(0.0, 0.0)
        if i == 1:
            offset = wp.vec2(rock_clearance, 0.0)
        elif i == 2:
            offset = wp.vec2(-rock_clearance, 0.0)
        elif i == 3:
            offset = wp.vec2(0.0, rock_clearance)
        elif i == 4:
            offset = wp.vec2(0.0, -rock_clearance)
        origin = wp.vec3(xy[tid][0] + offset[0], xy[tid][1] + offset[1], top)
        rock_hit = wp.mesh_query_ray(rocks, origin, down, 1.0e6)
        if rock_hit.result and top - rock_hit.t > height + 0.01:
            blocked[tid] = 1
            return


def _drape_particles(
    bounds: tuple[float, float, float, float],
    spacing: float,
    depth: float,
    terrain: wp.Mesh,
    rocks: wp.Mesh | None,
    rock_clearance: float,
    max_slope: float,
    top: float,
    device: str,
) -> tuple[np.ndarray, float]:
    """Return particle positions stacked ``depth`` high on gentle terrain away from rocks.

    Also returns the fraction of the layer's columns that received particles.
    """
    min_x, min_y, max_x, max_y = bounds
    xs = min_x + spacing * (np.arange(math.floor((max_x - min_x) / spacing)) + 0.5)
    ys = min_y + spacing * (np.arange(math.floor((max_y - min_y) / spacing)) + 0.5)
    xy = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2).astype(np.float32)
    ground = wp.empty(len(xy), dtype=float, device=device)
    blocked = wp.zeros(len(xy), dtype=wp.int32, device=device)
    wp.launch(
        _surface_heights,
        dim=len(xy),
        inputs=[
            terrain.id,
            rocks.id if rocks is not None else wp.uint64(0),
            int(rocks is not None),
            wp.array(xy, dtype=wp.vec2, device=device),
            top,
            rock_clearance,
        ],
        outputs=[ground, blocked],
        device=device,
    )
    heights = ground.numpy()
    # Slope over a 0.5 m baseline, the scale at which a thin layer slides.
    step = max(1, round(0.25 / spacing))
    grid = np.pad(heights.reshape(len(xs), len(ys)), step, mode="edge")
    slope_x = (grid[2 * step :, step:-step] - grid[: -2 * step, step:-step]) / (2 * step * spacing)
    slope_y = (grid[step:-step, 2 * step :] - grid[step:-step, : -2 * step]) / (2 * step * spacing)
    with np.errstate(invalid="ignore"):
        steep = np.degrees(np.arctan(np.hypot(slope_x, slope_y))).reshape(-1) > max_slope
    keep = np.isfinite(heights) & (blocked.numpy() == 0) & ~steep
    coverage = float(keep.mean())
    xy, heights = xy[keep], heights[keep]
    layers = max(1, round(depth / spacing))
    z = heights[:, None] + spacing * (np.arange(layers)[None, :] + 0.5)
    points = np.empty((len(xy), layers, 3), dtype=np.float32)
    points[..., :2] = xy[:, None, :]
    points[..., 2] = z
    return points.reshape(-1, 3), coverage


def _crop_mesh(
    vertices: np.ndarray, faces: np.ndarray, bounds: tuple[float, float, float, float], margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Keep triangles with a vertex inside ``bounds`` grown by ``margin`` and drop unused vertices."""
    min_x, min_y, max_x, max_y = bounds
    xy = vertices[:, :2]
    inside = (
        (xy[:, 0] >= min_x - margin) & (xy[:, 0] <= max_x + margin) & (xy[:, 1] >= min_y - margin) & (xy[:, 1] <= max_y + margin)
    )
    faces = faces[inside[faces].any(axis=1)]
    used, remapped = np.unique(faces, return_inverse=True)
    return vertices[used].astype(np.float32), remapped.reshape(-1, 3).astype(np.int32)
