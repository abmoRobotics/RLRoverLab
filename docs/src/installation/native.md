## Native Installation

If you prefer to install the suite natively without using Docker, follow these steps.

### Prerequisites

- **Python 3.11** (required)
- **NVIDIA GPU** with CUDA support
- **~50GB+ free disk space** (Isaac Sim packages are very large)

### Installation Steps

1. **Create a Conda environment with Python 3.11:**
   ```bash
   conda create -n roverlab python=3.11
   conda activate roverlab
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/abmoRobotics/RLRoverLab
   cd RLRoverLab
   ```

3. **Install the package:**

   This will automatically install Isaac Sim 5.1.0, Isaac Lab 2.3.0, and all other dependencies:
   ```bash
   pip install -e .[all]
   ```

   > **Note:** The installation downloads large packages (~30GB+). Ensure you have sufficient disk space and a stable internet connection. If you run out of space in `/tmp`, you can use a different temp directory:
   > ```bash
   > TMPDIR=/path/to/larger/disk pip install -e .[all]
   > ```

4. **Download terrain assets:**
   ```bash
   python download_usd.py
   ```

### Running The Suite

**To train a model**, navigate to the training script and run:
```bash
cd examples/02_training
python train.py --task="AAURoverEnv-v0" --num_envs=256
```

**To evaluate a pre-trained policy**, navigate to the inference script and run:
```bash
cd examples/03_inference
python eval.py --task="AAURoverEnv-v0" --num_envs=32
```

### Troubleshooting

#### "No space left on device" error
Isaac Sim packages are very large. Clear pip cache and ensure you have enough space:
```bash
rm -rf ~/.cache/pip
df -h /tmp ~/.cache
```

Or use a different temp directory with more space:
```bash
TMPDIR=/path/to/larger/disk pip install -e .[all]
```

<!-- #### Package conflicts
If you encounter dependency conflicts, try creating a fresh conda environment:
```bash
conda deactivate
conda remove -n roverlab --all
conda create -n roverlab python=3.11
conda activate roverlab
pip install -e .[all]
``` -->
