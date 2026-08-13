## Native Installation

Docker is the validated installation path. For native development, use a clean Python 3.12 environment and the exact Isaac Sim/Lab versions below.

### Prerequisites

- Ubuntu 22.04 or 24.04
- Python 3.12
- NVIDIA RTX GPU and a compatible production driver
- At least 50 GB of free disk space
- `git`, `git-lfs`, and [uv](https://docs.astral.sh/uv/)

### Install the pinned simulation stack

```bash
uv venv --python 3.12 --seed env_roverlab
source env_roverlab/bin/activate

uv pip install "isaacsim[all,extscache]==6.0.1.0" \
  --extra-index-url https://pypi.nvidia.com \
  --index-strategy unsafe-best-match \
  --prerelease=allow

git clone https://github.com/isaac-sim/IsaacLab.git \
  --branch v3.0.0-beta2.patch1
cd IsaacLab
test "$(git rev-parse HEAD)" = "ffff603eafc6b74264a5261cc0183d6a65390d78"
./isaaclab.sh --install 'rl[skrl],rl[rsl-rl],visualizer[kit]'
cd ..
```

### Install RLRoverLab

```bash
git clone https://github.com/abmoRobotics/RLRoverLab.git
cd RLRoverLab
uv pip install --editable .
python download_usd.py
```

Set the source checkout location when it differs from the Docker default, then verify the stack:

```bash
export ISAAC_LAB_PATH="$(cd ../IsaacLab && pwd)"
python tools/verify_stack.py
```

### Run navigation

Force headless execution with `--viz none`:

```bash
python examples/01_demos/01_zero_agent.py \
  --task AAURoverEnvSimple-v0 --num_envs 1 --viz none
```

Open the local Kit viewer with `--viz kit`:

```bash
python examples/01_demos/01_zero_agent.py \
  --task AAURoverEnvSimple-v0 --num_envs 1 --viz kit
```

Use `--device cpu` when CPU physics is required. Isaac Lab 3.0 deprecates the old `--headless` and `--cpu` flags.

Manipulation configurations are not currently set up or included in the validated workflow.
