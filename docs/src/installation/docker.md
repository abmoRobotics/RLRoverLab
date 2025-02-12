## Installation Using Docker
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
   cd RLRoverLab/docker
   ```
2. Download terrains from Google Drive:
   1. Download from Google Drive: https://drive.google.com/file/d/1VXFTD2OgHcsQL_ifO81AzD2HDkA98h93/view?usp=sharing
   2. Unzip files in root folder of the git repository
3. **Build and start the Docker container**:
   ```bash
   ./run.sh
   docker exec -it rover-lab-base bash
   ```
4. **Training an Agent Inside the Docker Container**
   To train an agent, use the following command inside the Docker container:
   ```bash
   cd examples/02_train
   /workspace/isaac_lab/isaaclab.sh -p train.py --task="AAURoverEnv-v0" --num_envs=256
   ```
