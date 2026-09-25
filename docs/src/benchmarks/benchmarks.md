# AAU rover particle backends

`rover_envs/benchmarks/aau_particles.py` builds the same AAU rover, floor, particle lattice, wheel targets, and 120 Hz step in two scenes. PhysX uses solid PBD particles. Newton uses MPM particles coupled to MJWarp wheel bodies through proxy coupling. `tools/debug_aau_particles.py` checks a single run; `tools/compare_aau_particles.py` runs both backends in separate processes and reports median environment steps per second.

## Newton adaptations

- The repository's Isaac Lab `v3.0.0-beta2.patch1` does not provide the MPM scene and proxy coupling APIs. `docker/Dockerfile.particles` installs Isaac Lab `v3.0.0-EA` at commit `ae37b028ea415c91ea2bc32609efcd759ed2b974` with Newton 1.5.2 on the existing Isaac Sim 6.1 image.
- The original `Mars_Rover.usd` crashes the Newton importer. Both scenes use the repository's `aau_rover_simple/rover_instance.usd` instead. Its 13 rover joints import into Newton and PhysX.
- MJWarp rejects a zero actuator force range on the rover's passive bogie joints. The Newton scene gives those joints a negligible `1e-6` N·m limit. Newton also needs its USD rigid body properties left on the asset, while the six wheel bodies are mapped to MPM proxies.
- Newton's sparse MPM grid needs a bounded active-cell count to capture the simulation in a CUDA graph. The capacity scales with environment count and inverse voxel volume; reserving too many cells slowed the small benchmark.
- PhysX particle sets cannot use Isaac Lab's optimized physics replication. The PhysX scene disables that replication so each environment's particle set is parsed.

The particle positions, count, density, mass, and radius match across scenes. PhysX PBD and Newton MPM have different contact and constitutive models, so the rate ratio measures these configured workloads, not identical soil physics. Startup time is reported separately from the timed steps.

## Run the comparison

Build the base image from `docker/Dockerfile` if `rover-lab-base:latest` is not already available, then build the particle image:

```bash
docker build -f docker/Dockerfile.particles -t rover-particles:ea .
```

From the repository root:

```bash
mkdir -p /tmp/aau_particle_results
docker run --rm --gpus all --network host \
  --entrypoint /isaac-sim/python.sh \
  -e ACCEPT_EULA=Y -e PYTHONPATH=/workspace/rlroverlab \
  -v "$PWD":/workspace/rlroverlab \
  -v /tmp/aau_particle_results:/results \
  rover-particles:ea tools/compare_aau_particles.py \
  --num_envs 2 --voxel_size 0.08 --warmup 50 --steps 500 --repeats 3 \
  --output_dir /results
```

The summary is `/tmp/aau_particle_results/comparison.json`. Per-run JSON and logs are saved beside it. `newton_over_physx` is the median Newton rate divided by the median PhysX rate.

On an RTX 4090 Laptop GPU with two environments and a 0.08 m MPM voxel size, headless runs gave:

| Particles/env | Timed steps | Repeats | PhysX env steps/s | Newton env steps/s |
| ---: | ---: | ---: | ---: | ---: |
| 3,192 | 500 | 3 | 395 | 181 |
| 24,750 | 200 | 2 | 355 | 160 |
| 84,411 | 300 | 3 | 104 | 133 |

Use `--particles_per_cell 4` and `6` for the middle and last rows. Newton overtakes PhysX here as particle density grows while MPM grid resolution stays fixed. The two solvers still model different soil physics, so the crossover is specific to these scenes and settings.

## View either scene

These commands use the host's X11 display. Run them from the repository root after building `rover-particles:ea`. Each runs 600 simulation steps, then closes. Increase `--steps` for more viewing time.

PhysX in Isaac Sim Kit:

```bash
docker run --rm --gpus all --network host \
  --entrypoint /isaac-sim/python.sh \
  -e ACCEPT_EULA=Y -e PYTHONPATH=/workspace/rlroverlab \
  -e DISPLAY="$DISPLAY" -e XAUTHORITY=/tmp/.Xauthority \
  -v "$PWD":/workspace/rlroverlab \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$XAUTHORITY":/tmp/.Xauthority:ro \
  rover-particles:ea tools/debug_aau_particles.py \
  --backend physx --num_envs 1 --viz kit --warmup 0 --steps 600
```

Newton in its OpenGL viewer:

```bash
docker run --rm --gpus all --network host \
  --entrypoint /isaac-sim/python.sh \
  -e ACCEPT_EULA=Y -e PYTHONPATH=/workspace/rlroverlab \
  -e DISPLAY="$DISPLAY" -e XAUTHORITY=/tmp/.Xauthority \
  -v "$PWD":/workspace/rlroverlab \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$XAUTHORITY":/tmp/.Xauthority:ro \
  rover-particles:ea tools/debug_aau_particles.py \
  --backend newton --num_envs 1 --viz newton_gl --warmup 0 --steps 600
```

Use the headless comparison command above for timing. Viewer rendering adds a different cost to each backend.

# Waypoint navigation on MPM soil

`AAURoverEnvParticles-v0` is the AAU rover waypoint-navigation task on the debug terrain with a 30 × 30 m layer of Newton MPM soil. Observations, actions, and the 0.2 s policy step match `AAURoverEnvSimple-v0`, so `examples/03_inference/eval.py` runs the trained height-map policy (`best_agent_heightmap.pt`) on it without retraining.

- The soil is one particle set in Newton's global world. All rovers drive on the same soil and cross each other's tracks, the same way they share the terrain mesh. `SoilTerrainImporter` in `rover_envs/envs/navigation/utils/terrains/soil.py` adds it while Newton builds the model: particles stacked 0.15 m deep on the terrain surface, and particle-only copies of the terrain and rocks that hold them up. Rocks and slopes steeper than 30° stay bare.
- MJWarp steps the rovers against the original terrain and rocks, with three substeps per 120 Hz physics step. Implicit MPM steps the soil on a 0.1 m grid with two particles per voxel edge, 980,364 particles on the debug terrain. Proxy coupling exchanges impulses through the six wheels.
- Rovers spawn at least 10.5 m inside the layer's edges, so their targets, 9 m away, also land on soil.
- `SoilLayerCfg` sets the layer's centre, size, depth, grid resolution, and material. After changing it, call `update_physics()` on the environment config so the MPM grid matches.

Compared with the rigid-terrain task:

- Newton's coupled solvers report no contact forces, so the task has no contact sensor and drops the rock-collision penalty and termination.
- The height scanner measures the terrain under the soil, not the soil surface.
- Resetting a rover does not reset the soil. Tracks accumulate for the whole run.

## Adaptations

- Isaac Lab's `MPMObject` emits a copy of its particles into every environment world. The soil importer adds its particles once, in the global world, from a `MODEL_INIT` callback, and mirrors them to a USD `Points` prim for the Kit viewport.
- A sparse MPM grid captured in a CUDA graph runs its kernels over its full reserved capacity, and by default Newton reserves one 8³-voxel leaf per active cell. The soil solver reserves 1.5 times the cells and leaves that the particles occupy at the start: 263,628 cells and 2,973 leaves on the debug terrain. With two rovers, a capacity estimated from the layer's footprint ran at 8–9 physics steps/s in 15 GB; sizing from the particles runs at 19 steps/s in 8 GB.
- Newton merges all shapes of one collider body into one mesh. With terrain and rocks in one body, closest points on buried rock faces flipped the inside test and let particles fall through the ground. Terrain and rocks are separate collider bodies.
- Newton's implicit MPM cannot clear grid warm starts for one world of a shared grid. When a rover resets, the soil keeps its warm starts, which only seed the next solve. The rover's collider poses are still refreshed.
- Isaac Lab 3.0 EA creates the terrain before the obstacle prims below it. `RoverTerrainImporter` builds its spawn and target maps on first use, which also fixes the rigid tasks on that Isaac Lab version.

## View the task in Isaac Sim

After building `rover-particles:ea` as described above, from the repository root:

```bash
docker run --rm --gpus all --network host \
  --entrypoint /isaac-sim/python.sh \
  -e ACCEPT_EULA=Y -e PYTHONPATH=/workspace/rlroverlab \
  -e DISPLAY="$DISPLAY" -e XAUTHORITY=/tmp/.Xauthority \
  -v "$PWD":/workspace/rlroverlab \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$XAUTHORITY":/tmp/.Xauthority:ro \
  rover-particles:ea examples/03_inference/eval.py \
  --task AAURoverEnvParticles-v0 --num_envs 4 --viz kit
```

The task defaults to the debug terrain. Building the model and capturing the CUDA graph takes under a minute, then the viewport shows the soil and the tracks the rovers leave. With four rovers the viewer runs at about 0.13 times real time. Pass `--steps N` to stop after N policy steps. `eval.py` writes its logs to `logs/` in the working directory.

## Measure scaling

`tools/benchmark_particle_nav.py` builds the task and agent like `eval.py`, runs the policy headless, and reports throughput, rover speed, rover height above the terrain, and how far the particles moved. It also runs rigid tasks, such as `--task AAURoverEnvSimple-v0 --terrain debug`, as a reference. `tools/scale_particle_nav.py` runs it for each rover count in separate processes and reports medians:

```bash
mkdir -p /tmp/particle_nav_results
docker run --rm --gpus all --network host \
  --entrypoint /isaac-sim/python.sh \
  -e ACCEPT_EULA=Y -e PYTHONPATH=/workspace/rlroverlab \
  -v "$PWD":/workspace/rlroverlab \
  -v /tmp/particle_nav_results:/results \
  rover-particles:ea tools/scale_particle_nav.py \
  --num_envs 2 8 32 64 128 --warmup 10 --steps 50 --repeats 2 --output_dir /results
```

The summary is `/tmp/particle_nav_results/scaling.json`, with per-run JSON and logs beside it. Arguments after `--` go to every run, for example `-- --physics_dt 0.0166667` or `-- --particles_per_cell 1`.

On an RTX PRO 6000 Blackwell Max-Q (300 W) with default settings, 10 warm-up and 50 timed policy steps, and the median of two runs:

| Rovers | Env steps/s | Policy steps/s | Physics steps/s | Real-time factor | GPU memory [GB] | Startup [s] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.7 | 0.87 | 20.9 | 0.17 | 7.9 | 13 |
| 8 | 6.7 | 0.83 | 20.0 | 0.17 | 8.9 | 14 |
| 32 | 23.2 | 0.72 | 17.4 | 0.14 | 13.3 | 15 |
| 64 | 39.3 | 0.61 | 14.7 | 0.12 | 19.0 | 18 |
| 128 | 60.8 | 0.47 | 11.4 | 0.09 | 30.5 | 23 |

The two runs at each count differed by less than 0.5%. One soil solve serves every rover, so throughput grows almost linearly up to 32 rovers. The MPM step costs about 40 ms, which caps a single physics step at about 21 per second. Beyond 32 rovers, the rover side grows: coupling only env 0's wheels to the soil at 128 rovers still runs at 12.7 physics steps/s in the same 30.5 GB. Newton's shared-grid collider query loops over every wheel proxy for each grid node, which costs the remaining 9 ms per step at 128 rovers. Keep the GPU free of other work while timing: another Isaac Sim process running on the same GPU halved the rate at 32 and 64 rovers in an earlier run.

Over 100 s with 16 rovers, the policy reached 3 targets on soil and 63 on rigid terrain (`AAURoverEnvSimple-v0`). On soil the rovers averaged 0.11 m/s, against 0.48 m/s on rigid terrain. Their bodies rode 0.27 m above the terrain under the 0.15 m layer, so the wheels sank about as deep as the layer. The soil material values come from the benchmark above and are not calibrated to a real soil.

Coarser settings trade soil response for speed. With two rovers over 25 timed policy steps:

| Settings | Particles | Real-time factor | GPU memory [GB] | Mean speed [m/s] | Body height above terrain [m] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default: 120 Hz, 2 particles per voxel edge | 980,364 | 0.15 | 7.8 | 0.12 | 0.31 |
| `--physics_dt 0.0166667` (60 Hz) | 980,364 | 0.29 | 7.9 | 0.18 | 0.27 |
| `--particles_per_cell 1` | 81,524 | 0.26 | 3.6 | 0.17 | 0.28 |
| Rigid terrain, `AAURoverEnvSimple-v0` (PhysX) | — | 3.76 | — | 0.45 | 0.26 |
