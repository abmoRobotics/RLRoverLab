## Native Installation
If you prefer to install the suite natively without using Docker, follow these steps:
1. **Install Isaac Sim 5.0** According to the [Official Documentation](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html).

    First [Download the Isaac Sim 5.0](https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-5.0.0-linux-x86_64.zip) to the `~/Downloads` directory.

    Then run the following commands:

    ```bash
    mkdir ~/isaacsim
    cd ~/Downloads
    unzip "isaac-sim-standalone@5.0.0-linux-x86_64.zip" -d ~/isaacsim
    cd ~/isaacsim
    ./post_install.sh
    ./isaac-sim.selector.sh
    ```
2. **Install Isaac Lab**
   ```bash
   git clone https://github.com/isaac-sim/IsaacLab
   cd isaac_lab

   # create aliases
   export ISAACSIM_PATH="${HOME}/isaacsim"
   export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"

   # Create symbolic link
   ln -s ${ISAACSIM_PATH} _isaac_sim

   # Create Conda Env
   ./isaaclab.sh --conda isaaclab_env

   # Activate Env
   conda activate isaaclab_env

   # Install dependencies
   conda --install

   ```

3. **Set up the RL-suite:**

   ```bash
   # Clone Repo
   git clone https://github.com/abmoRobotics/RLRoverLab
   cd RLRoverLab

   # Install Repo (make sure conda is activated)
   python -m pip install -e .[all]
   ```
4. **Download terrain assets:**
   ```bash
   pip3 install gdown
   python3 download_usd.py
   ```

5. **Running The Suite**

   **To train a model**, navigate to the training script and run:
   ```bash
   cd examples/02_train/train.py
   python train.py
   ```

   **To evaluate a pre-trained policy**, navigate to the inference script and run:
   ```bash
   cd examples/03_inference_pretrained/eval.py
   python eval.py
   ```
