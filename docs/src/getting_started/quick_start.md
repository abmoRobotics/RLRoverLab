# Quick Start Guide

This guide will help you get started with RLRoverLab quickly. Follow these steps to set up the environment and run your first training or evaluation.

## Prerequisites

Before starting, ensure you have:
- NVIDIA GPU with at least 8GB VRAM
- Ubuntu 20.04 or 22.04
- Docker and NVIDIA Container Toolkit installed (see [Docker Installation](../installation/docker.md))

## Quick Setup with Docker

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abmoRobotics/RLRoverLab
   cd RLRoverLab
   ```

2. **Download terrain assets**:
   ```bash
   pip3 install gdown
   python3 download_usd.py
   ```

3. **Start the Docker container**:
   ```bash
   cd docker
   ./run.sh
   docker exec -it rover-lab-base bash
   ```

## Running Your First Example

### 1. Train a Simple Agent

Train a PPO agent on the simple AAU rover environment in forced headless mode:

```bash
cd examples/02_training
/workspace/isaac_lab/isaaclab.sh -p train.py --task="AAURoverEnvSimple-v0" --num_envs=128 --viz none
```

### 2. Evaluate a Pre-trained Model

If you have a trained model, evaluate it:

```bash
cd examples/03_inference
/workspace/isaac_lab/isaaclab.sh -p eval.py --task="AAURoverEnvSimple-v0" --num_envs=32 --checkpoint=path/to/your/model.pt --viz none
```

### 3. Demo with Zero Agent

Run a basic demo in the Kit viewer:

```bash
cd examples/01_demos
/workspace/isaac_lab/isaaclab.sh -p 01_zero_agent.py --viz kit
```

## Available Environments

The suite provides several pre-configured environments:

| Environment ID | Robot | Description |
|---|---|---|
| `AAURoverEnvSimple-v0` | AAU Rover (Simple) | Simplified rover with basic sensors |
| `AAURoverEnv-v0` | AAU Rover | Full rover with advanced sensors |
| `Exomy-v0` | Exomy | ESA's ExoMy rover |

## What's Next?

- [Explore more examples](../examples/examples.md)
- [Learn about available environments](../tasks/environment_overview.md)
- [Understand the training process](../training/training.md)
- [Add your own robot](../development/adding_new_robots_or_assets.md)

## Troubleshooting

### Common Issues

1. **GPU Memory Issues**: Reduce `--num_envs` parameter
2. **Docker Permission Issues**: Ensure your user is in the docker group
3. **Display Issues**: Run `xhost +local:docker` before starting the container

For more detailed troubleshooting, see the [Installation Guide](../installation/installation.md).
