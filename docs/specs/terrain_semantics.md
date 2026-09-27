# Spec: terrain semantic classes for BEV ground truth

Status: proposed. Owner: unassigned. Consumer: the BEV perception task `AAURoverEnvSimpleBEV-v0`.

## Why

The BEV task trains a semantic head against labels from a grid of downward rays (`bev_scanner`, a
`MultiMeshRayCaster` with `update_mesh_ids=True`). A ray cannot read USD semantic labels, but it reports the
index of the ray-cast target it hit, and that index is the class. Phase 1 therefore hardcodes two classes in
`RoverBevSceneCfg.set_terrain` (`rover_envs/envs/navigation/rover_env_camera_cfg.py`):

| class id | name   | ray-cast target                                                       |
|----------|--------|-----------------------------------------------------------------------|
| 0        | ground | `/World/terrain/terrain` (`terrain_only.usd`)                         |
| 1        | rock   | `TerrainConfig.obstacle_mesh_prim_path`                               |

This is correct for `mars`, `debug` and generated terrains, whose rocks are one merged mesh
(`rocks_merged.usd`, `/World/terrain/obstacles/obstacles`) next to the ground mesh. It is not a terrain
contract, and it is wrong or unusable elsewhere:

- Lunar LTE terrains set `obstacle_mesh_prim_path` to the invisible `lethal_collision` mesh, a subset of the
  rendered rocks. The visible `rocks.usd` is not ray-cast at all, so non-lethal rocks are missing from the
  elevation target and their cells are labelled as the ground beneath them. Lunar terrains without
  `lethal_collision.usd` cannot be used.
- There is no way to declare more classes (bedrock, sand, craters, ...).
- Terrain prims carry no USD semantic labels, so cameras cannot render semantic segmentation to cross-check
  the labels or supervise in image space later.

## Contract to implement

1. **Global class vocabulary.** Define the ordered class names once in
   `rover_envs/assets/terrains/terrain_registry.py`, starting with `SEMANTIC_CLASSES = ("ground", "rock")`.
   The class id is the index in this tuple and must mean the same thing on every terrain. Append new classes;
   never reorder, since trained semantic heads depend on the ids.
2. **Per-terrain class targets.** Add `semantic_prim_paths: dict[str, str]` to `TerrainConfig`: class name to
   one runtime prim whose descendant meshes all belong to that class. `MultiMeshRayCaster` merges the meshes
   under a target prim into one ray-cast mesh, so each class is one target and class id = target index.
   - `register_terrain_from_folder`: `{"ground": "/World/terrain/terrain", "rock": "/World/terrain/obstacles"}`.
   - `discover_lunar_lte_terrains`: rock is the visible `/World/terrain/obstacles` (`rocks.usd`), not the
     lethal-collision mesh, which remains the contact-sensor filter only.
   - `register_terrain` takes the mapping as an argument for custom terrains.
3. **Completeness.** Every rendered, solid terrain mesh that can change the elevation must belong to exactly
   one class target, so the BEV scanner hits what the camera sees. Visual-only or collision-only geometry must
   not be a target.
4. **BEV consumer.** `RoverBevSceneCfg.set_terrain` builds
   `bev_scanner.mesh_prim_paths = [terrain_config.semantic_prim_paths[name] for name in SEMANTIC_CLASSES]`, and
   `BevCfg.classes` defaults to `SEMANTIC_CLASSES`. A terrain missing a class fails with a clear error naming
   the terrain and the class. Remove the phase-1 TODO in `set_terrain`.
5. **USD semantic labels.** Label every class target with `class=<name>`: pass
   `semantic_tags=[("class", name)]` to the `UsdFileCfg` spawners in `create_obstacles_cfg` (and the lighting
   or other asset helpers where relevant), and call `isaaclab.sim.utils.semantics.add_labels(prim, [name])` on
   the imported ground prim, since `TerrainImporterCfg` has no semantic tag option. Labels must not change
   physics, rendering or the ray-cast meshes.
6. **Generator.** `rover_envs/assets/terrains/generator/terrain_generator.py` keeps writing one merged mesh per
   class (`terrain_only.usd`, `rocks_merged.usd`) so generated terrains satisfy the contract by construction.

The `bev_target` observation contract must not change: channel 0 elevation in metres (NaN where no hit),
channel 1 class id (-1 where no hit), channel 2 camera visibility (see `mdp.bev_ground_truth`).

## Out of scope

- Several classes inside one mesh (per-face or texture labels). The ray-cast kernel can return face ids, but
  `MultiMeshRayCaster` does not expose them; split such terrains into one mesh per class instead.
- Dynamic objects, and terrains that only provide some of the classes.

## Open questions to resolve during implementation

- Can `MultiMeshRayCaster` parse the lunar `rocks.usd` hierarchy (instanceable prims or point instancers), and
  how long does merging it take at startup? If it cannot, bake a merged `rocks_merged.usd` for those terrains.
- Should the lunar contact filter keep using `lethal_collision` (current behaviour) while the BEV uses the
  visible rocks? This spec assumes yes.

## Acceptance tests

1. `mars` and `debug`: the `bev_target` observation is identical to phase 1 for the same seed.
2. A lunar LTE terrain: the BEV scanner hits the visible non-lethal rocks, and the depth-lift elevation error
   reported by `examples/02_training/train_bev.py` (`BEV/lift_elevation_mae`) is on par with `mars`.
3. With `semantic_segmentation` added to the camera's data types, the rendered class at the pixel of every
   visible BEV cell matches channel 1 of `bev_target` for at least 95% of the visible cells.
4. A unit test that every built-in terrain declares every class in `SEMANTIC_CLASSES`, and that a missing
   class raises.
