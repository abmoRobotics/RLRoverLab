## Installation Using Docker

Docker is the **recommended installation method** as it provides a consistent environment with all dependencies pre-installed.

The image is pinned to Isaac Sim `6.0.1` (including its multi-architecture image digest), Isaac Lab
`v3.0.0-beta2.patch1` (`ffff603`), and Python 3.12.

### Prerequisites
- **.Xauthority for graphical access:** Run the following command to verify or create .Xauthority.
   ```bash
   [ ! -f ~/.Xauthority ] && touch ~/.Xauthority && echo ".Xauthority created" || echo ".Xauthority already exists"
   ```
- **Nvidia Container Toolkit:** see [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

   _After installing the toolkit remember to configure the container runtime for docker using_
   ```bash
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
   _You may need to allow docker to access X server if you want to use the GUI:_
   ```bash
   xhost +local:docker
   ```
-   **Login to NGC**
    1. [Generate NGC API Key ](https://docs.nvidia.com/ai-enterprise/deployment-guide-spark-rapids-accelerator/0.1.0/appendix-ngc.html)
    2. Login with the NGC API as password
    ```docker login nvcr.io
    Username: $oauthtoken
    Password:
    ```


- **Docker Compose:**
   1. Install Docker Compose
   2. Verify using
    ```
    docker compose version
    ```
### Building the Docker Image

1. **Clone the repository and navigate to the docker directory**:
   ```bash
   git clone https://github.com/abmoRobotics/RLRoverLab
   cd RLRoverLab
   ```
2. **Download terrain assets:**
   ```bash
   pip3 install gdown
   python3 download_usd.py
   ```

3. **Build and start the Docker container**:
   ```bash
   cd docker
   ./run.sh
   docker exec -it rover-lab-base bash
   ```

4. **Verify the installed versions** inside the container:
   ```bash
   cd /workspace/rlroverlab
   python tools/verify_stack.py
   ```

### Usage

#### Training an Agent
To train an agent headlessly, use the following command inside the Docker container:
```bash
cd /workspace/rlroverlab
python examples/02_training/train.py --task="AAURoverEnv-v0" --num_envs=256 --viz none
```

#### Evaluating a Pre-trained Policy
To evaluate a pre-trained policy, use the following command inside the Docker container:
```bash
cd /workspace/rlroverlab
python examples/03_inference/eval.py --task="AAURoverEnv-v0" --num_envs=32 --viz none
```

#### Running with the Kit viewer

Pass `--viz kit` from a machine with a working local display and X11 access:

```bash
cd /workspace/rlroverlab
python examples/01_demos/01_zero_agent.py --task="AAURoverEnvSimple-v0" --num_envs=1 --viz kit
```

Isaac Lab 3.0 deprecates `--headless`; use `--viz none` to force headless execution. Camera and RGB-D tasks enable their required camera extensions automatically.

### Development Workflow

The Docker setup bind-mounts the repository to `/workspace/rlroverlab/`, so any changes you make to the code on your host machine are immediately reflected inside the container. This makes it ideal for development:

1. Edit code on your host machine using your preferred editor
2. Run/test inside the container
3. No need to rebuild the container for code changes
