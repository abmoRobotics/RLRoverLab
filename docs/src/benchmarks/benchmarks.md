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
