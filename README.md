
# RLRoverLab

[![Isaac Sim](https://img.shields.io/badge/IsaacSim-6.1.0-green.svg)](https://docs.isaacsim.omniverse.nvidia.com/6.1.0/)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-v3.0.0--beta2.patch1-green)](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2.patch1)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04%20%7C%2024.04-orange.svg)](https://abmorobotics.github.io/RLRoverLab/installation/installation.html)

## Introduction

Welcome to RLRoverLab! This project implements Reinforcement Learning (RL) agents using Isaac Sim and [Isaac Lab](https://isaac-sim.github.io/IsaacLab/). Navigation is the currently supported workflow; manipulation remains development scaffolding and is not part of the validated release surface.

## Features

- **Navigation Tasks**: RL environments for the AAU Rover and ExoMy, including state, height-map, camera, and RGB-D observations.
- **Isaac Sim and Isaac Lab Integration**: Utilizes the advanced simulation environments of Isaac Sim and the Isaac Lab framework for realistic task scenarios.
- **Expandable Framework**: Architecture designed for easy extension with new tasks and functionalities.

## Getting Started

To get started with RLRoverLab, please refer to our [Installation Guide](https://abmorobotics.github.io/RLRoverLab/installation/installation.html). The guide provides comprehensive steps for setting up the suite using Docker as well as instructions for native installation.

### Train with RSL-RL

The height-map tasks (`AAURoverEnv-v0`, `AAURoverEnvSimple-v0`, `Exomy-v0`, and `Sawppy-v0`) have an RSL-RL PPO configuration. From the repository root:

```bash
python examples/02_training/train_rsl_rl.py --task AAURoverEnvSimple-v0 --num_envs 128 --terrain mars --viz none
```

Use `--max_iterations` to set the training length or `--checkpoint` to resume an RSL-RL run. Training exports `policy.onnx` alongside checkpoints under `logs/rsl_rl/rover_heightmap/`. To export an existing checkpoint without training, pass `--checkpoint PATH --export-only`. Additional agent configurations can be registered with a Gym key and selected with `--agent KEY`. The RSL-RL baseline uses an MLP on the flat height-map observation; the existing skrl trainer keeps its CNN and supports the camera tasks.

### Distill into an RGB student

`AAURoverEnvSimpleDistillation-v0` distills a height-map teacher trained on `AAURoverEnvSimple-v0` into a recurrent student that only sees the ZED2i WVGA camera (672×376 RGB) and the three goal observations. The camera frames go through a frozen DINOv3 ViT-S/16, so place its Hugging Face weights in `~/models/dinov3-vits16` (Docker mounts `~/models` at `/root/models`). The teacher defaults to the included RSL-RL height-map policy (`best_agent_rsl_rl_heightmap.pt`):

```bash
python examples/02_training/train_rsl_rl.py --task AAURoverEnvSimpleDistillation-v0 --num_envs 32 --terrain mars --viz none
```

Pass a PPO checkpoint with `--checkpoint` to use a different teacher, or a distillation checkpoint to resume the student. The exported `policy.onnx` runs one policy step: it takes the raw RGB frame (`uint8`, 1×376×672×3), the goal observations, and the GRU hidden state, and returns the actions and the next hidden state.

### Train the BEV perception model

`AAURoverEnvSimpleBEV-v0` trains an instantaneous bird's-eye-view (BEV) model on the ZED2i WVGA RGB-D camera while the included height-map policy drives the rover. The BEV is 120×120 cells at 5 cm, covering x ∈ [-2, 4] m (forward) and y ∈ [-3, 3] m (left) in the yaw-aligned rover frame. Frozen DINOv3 ViT-S/16 features are lifted into the BEV with the depth, together with per-cell mean elevation, elevation range and point density, and a small U-Net predicts the elevation and the ground/rock class of each cell. The targets come from one downward ray per cell (the hit mesh gives the class) and the losses only use cells the camera sees. The DINOv3 weights go in `~/models/dinov3-vits16`, as for distillation:

```bash
python examples/02_training/train_bev.py --task AAURoverEnvSimpleBEV-v0 --num_envs 32 --terrain mars --viz none
```

Checkpoints and BEV visualizations are written to `logs/bev/rover_bev/`. Use `--checkpoint` to resume and `--driver` to drive with another PPO checkpoint. `BEV/lift_elevation_mae` scores the depth lift alone and should stay below a centimetre. Richer terrain semantics are specified in [docs/specs/terrain_semantics.md](docs/specs/terrain_semantics.md).

Three input ablations are registered next to it. Train them with the same command and a different `--task`; each logs to its own experiment in `logs/bev/` and W&B:

| Task | U-Net input | Question |
|---|---|---|
| `AAURoverEnvSimpleBEVGeometryOnly-v0` | Geometry lifted from the perfect depth; DINOv3 is not loaded | How much can perfect depth solve? |
| `AAURoverEnvSimpleBEVDinoOnly-v0` | DINOv3 features only; the depth still places them in the BEV | How much comes from appearance? |
| `AAURoverEnvSimpleBEVNoisyDepth-v0` | Geometry and DINOv3 from ZED-like stereo depth: disparity noise (about 2.5 cm at 2 m, 10 cm at 4 m) and 5% holes | Closest to the real problem |

The ablations only change the model input; the BEV targets and metrics are computed exactly as for `AAURoverEnvSimpleBEV-v0`.

### Temporal BEV memory

`BevMemory` (`rover_envs/envs/navigation/learning/bev/memory.py`) turns the instantaneous BEVs of the frozen model into a persistent ego BEV. For each environment it keeps the latest and the first observation of every 5 cm world cell in a 12 × 12 m world-aligned window around the rover. It stores only cells that received depth points in the current frame, and returns elevation, class probabilities, the provenance of each cell (current, remembered or unknown), the time and distance travelled since it was observed, and the range it was observed from. The BEV tasks expose the pose it needs as the `pose` observation group. Evaluate it while the height-map policy drives, here in the settled configuration: ZED-like noisy depth, the raw depth elevation stored, and the class probabilities of all observations of a cell averaged:

```bash
python examples/03_inference/eval_bev_memory.py --task AAURoverEnvSimpleBEVNoisyDepth-v0 \
    --checkpoint logs/bev/rover_bev_noisy_depth/<run>/model_<iteration>.pt --elevation_source lift --class_fusion mean \
    --num_envs 64 --terrain mars
```

`--driving circle` drives 1.5 m circles that revisit terrain, `--write_rule closest` keeps the elevation observed closest to the rover, and `--class_fusion` chooses the latest class, the mean or log-odds. `--odometry_noise` feeds the memory drifting poses: a severity multiplier, or a range such as `0.25:4` from which every episode draws its own, and `--drift_field_std` makes the drift depend on where the rover drives. The memory is never told the drift level: `DriftMonitor` (`drift.py`) measures it from terrain the rover observes again, and `--keyframe_distance` adds a keyframe registration that measures pose corrections (`registration.py`). Metrics (also by drift severity), an error-vs-distance plot, and real-time MP4 videos with snapshots of the environments in `--video_envs` are written to `logs/bev_memory/`; add `--wandb` to also log them, with per-step curves, to the `rlroverlab` W&B project. `--dump_samples` also saves remembered cells with the trajectories, on which `examples/03_inference/analyze_bev_reliability.py` fits and compares reliability models offline. Plans and results are in [docs/plans](docs/plans) and [docs/results](docs/results).

### Evaluate with RSL-RL

`AAURoverEnv-v0` and `AAURoverEnvSimple-v0` include a trained RSL-RL height-map policy, and `AAURoverEnvSimpleDistillation-v0` includes the distilled RGB student (`best_agent_rsl_rl_rgb.pt`). The included policy is evaluated by default; pass `--checkpoint` to evaluate another one:

```bash
python examples/03_inference/eval_rsl_rl.py --task AAURoverEnvSimple-v0 --num_envs 32 --terrain mars
python examples/03_inference/eval_rsl_rl.py --task AAURoverEnvSimpleDistillation-v0 --num_envs 32 --terrain mars
```

### Quick Links

- [Installation Guide](https://abmorobotics.github.io/RLRoverLab/installation/installation.html)
- [Examples and Tutorials](https://abmorobotics.github.io/RLRoverLab/examples/examples.html)
- [Adding Custom Robots and Tasks](https://abmorobotics.github.io/RLRoverLab/development/adding_new_robots_or_assets.html)

<!--## Contribution

We welcome contributions to RLRoverLab! Whether it's adding new tasks, or fixing bugs. Check out our [Contribution Guidelines](https://github.com/abmoRobotics/rlroverlab/CONTRIBUTING.md) for more information on how to get involved. -->
## Train agents in parallel


https://github.com/user-attachments/assets/aa8e9215-1483-486b-854b-9e6c3af9e4f2

https://github.com/user-attachments/assets/98032f5c-1cdc-42c2-8a94-57d52740d026



## Video of trained RL-agent
<!--https://github.com/user-attachments/assets/44844311-87cd-45cb-a933-f451376f27d8 -->


https://github.com/user-attachments/assets/7ecc4d9e-a4f3-4d4d-b7db-b04c7f35b083




## Support

If you have questions, suggestions, feel free to contact us.

- **Contact Us**: For direct inquiries, reach out to Anton Bjørndahl Mortensen at [antonbm2008@gmail.com](mailto:antonbm2008@gmail.com).


## Citation

Please cite [this paper](https://ieeexplore.ieee.org/abstract/document/10687686) if you use this suite in your work:

```bibtex
@inproceedings{mortensen2024rlroverlab,
  title={RLRoverLAB: An Advanced Reinforcement Learning Suite for Planetary Rover Simulation and Training},
  author={Mortensen, Anton Bj{\o}rndahl and B{\o}gh, Simon},
  booktitle={2024 International Conference on Space Robotics (iSpaRo)},
  pages={273--277},
  year={2024},
  organization={IEEE}
}
```
