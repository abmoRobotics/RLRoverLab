<!--# RLRoverLab
## Introduction
This Project implements a suite of Reinforcement Learning (RL) agents using Isaac Sim and [ORBIT](https://isaac-orbit.github.io/orbit/). So far we've implemented navigation and manipulation-based takss and are working on implementing more so expect to see updates soon. 

# Installation
In order to ease the setup of this suite, we use docker to install Isaac Sim, ORBIT, and our suite. The following documents the process and requirements of doing this.
## Requirements
### Hardware
- GPU: Any RTX GPU with at least 8 GB VRAM (Tested on NVIDIA RTX 3090 and NVIDIA RTX A6000)
- CPU: Intel i5/i7 or equivalent
- RAM: 32GB or more

### Software
- Operating System: Ubuntu 20.04 or 22.04
- Packages: Docker and Nvidia Container Toolkit

## Building the docker image
1. Clone and build docker:
```bash
# Clone Repo
git clone https://github.com/abmoRobotics/isaac_rover_orbit
cd isaac_rover_orbit

# Build and start docker
cd docker
./run.sh
docker exec -it orbit bash

```

2. Train an agent
Once inside the docker container you can train and agent by using the following command
```bash
/workspace/orbit/orbit.sh -p train.py --task="AAURoverEnv-v0" --num_envs=256
```

## Installing natively
1. Install Isaac Sim 2023.1.1 through the [Omniverse Launcher](https://www.nvidia.com/en-us/omniverse/download/).
2. Install ORBIT using the following steps:
```bash
git clone https://github.com/NVIDIA-Omniverse/orbit
cd Orbit

# create aliases
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-2023.1.1"
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

# Create symbolic link
ln -s ${ISAACSIM_PATH} _isaac_sim

# Create Conda Env
./orbit.sh --conda orbit_env

# Activate Env
conda activate orbit_env

# Install dependencies
conda --install

```
3. Clone Repo

```bash
# Clone Repo
git clone https://github.com/abmoRobotics/isaac_rover_orbit
cd isaac_rover_orbit

# Install Repo (make sure conda is activated)
python -m pip install -e .[all]

# Run training script or evaluate pre-trained policy
cd examples/02_train/train.py
python train.py

cd examples/03_inference_pretrained/eval.py
python eval.py
```

# Contact
For other questions feel free to contact:
* Anton Bjørndahl Mortensen: antonbm2008@gmail.com
-->
# RLRoverLab

## Introduction

Welcome to RLRoverLab! This project implements a suite of Reinforcement Learning (RL) agents using Isaac Sim and [Isaac Lab](https://isaac-sim.github.io/IsaacLab/). Our suite currently supports a variety of tasks within navigation and manipulation, with ongoing efforts to expand our offerings.
## Features

- **Navigation and Manipulation Tasks**: Implementations of RL agents designed for navigation and manipulation tasks, we are working on integrating more tasks.
- **Isaac Sim and ORBIT Integration**: Utilizes the advanced simulation environments of Isaac Sim and the ORBIT framework for realistic task scenarios.
- **Expandable Framework**: Architecture designed for easy extension with new tasks and functionalities.

## Getting Started

To get started with RLRoverLab, please refer to our [Installation Guide](https://github.com/abmoRobotics/isaac_rover_orbit/wiki) available in the project's wiki. The guide provides comprehensive steps for setting up the suite using Docker as well as instructions for native installation.

### Quick Links

- [Installation Guide](https://github.com/abmoRobotics/isaac_rover_orbit/wiki/Installing-the-suite)
- [Examples and Tutorials](https://github.com/abmoRobotics/isaac_rover_orbit/wiki/Examples)
- [Adding Custom Robots and Tasks](https://github.com/abmoRobotics/isaac_rover_orbit/wiki/Development)

<!--## Contribution

We welcome contributions to RLRoverLab! Whether it's adding new tasks, or fixing bugs. Check out our [Contribution Guidelines](https://github.com/abmoRobotics/isaac_rover_orbit/CONTRIBUTING.md) for more information on how to get involved. -->
## Train agents in parralel

https://github.com/abmoRobotics/isaac_rover_orbit/assets/56405924/71b69616-0a69-46c6-b3ce-009554bfa3c4


https://github.com/abmoRobotics/isaac_rover_orbit/assets/56405924/c9daf864-bf26-410b-a515-86b122d7a01d




## Video of trained RL-agent
https://github.com/abmoRobotics/isaac_rover_orbit/assets/56405924/84a9f405-f537-4041-9d1f-8a2307f450a7

## Support

If you have questions, suggestions, feel free to contact us.

- **Contact Us**: For direct inquiries, reach out to Anton Bjørndahl Mortensen at [antonbm2008@gmail.com](mailto:antonbm2008@gmail.com).

