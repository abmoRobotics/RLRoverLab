# CloneLab Integration

This directory contains the RLRoverLab side of the CloneLab/CloneRL workflow.
The integration is intentionally narrow:

- RLRoverLab owns Isaac Sim, Isaac Lab, task registration, terrains, cameras, and dataset recording.
- CloneLab owns offline datasets, BC/IQL algorithms, trainers, and model checkpoints.
- `rover_envs.integrations.clonelab` only translates RLRoverLab observations and CloneLab checkpoints into a runtime policy interface.

## Start the Container

Use the CloneLab compose overlay only when you want CloneLab visible from the
RLRoverLab container:

```bash
cd docker
./run_clonelab.sh
docker exec -it rover-lab-base bash
```

If CloneLab is not checked out next to RLRoverLab, set `CLONELAB_HOST_PATH`:

```bash
CLONELAB_HOST_PATH=/absolute/path/to/CloneLab ./run_clonelab.sh
```

The overlay only mounts CloneLab at `/workspace/clonelab` and adds it to
`PYTHONPATH`. It intentionally does not install CloneLab or its Python
dependencies. CloneLab owns training dependencies; RLRoverLab owns the Isaac
runtime and evaluation entrypoint. The default RLRoverLab Docker flow is
unchanged.

## Collect Data

Use the existing RLRoverLab inference path and Isaac Lab recorder:

```bash
cd /workspace/rlroverlab
python examples/03_inference/eval.py \
  --task AAURoverEnvRGBDRaw-v0 \
  --num_envs 64 \
  --dataset_dir ./datasets \
  --dataset_name rover_rgbd_expert \
  --dataset_type RL
```

This produces an Isaac Lab HDF5 dataset that CloneLab can read through its
`HDF5DictDatasetRandom` and sequence datasets.

## Train Offline

Offline training lives in CloneLab. From inside the container:

```bash
cd /workspace/clonelab
python Examples/rlroverlab/train_bc.py \
  --dataset /workspace/rlroverlab/datasets/rover_rgbd_expert.hdf5 \
  --epochs 30 \
  --batch_size 64
```

Recurrent BC, IQL, and recurrent IQL use the same boundary:

```bash
python Examples/rlroverlab/train_bc_recurrent.py \
  --dataset /workspace/rlroverlab/datasets/rover_rgbd_expert.hdf5

python Examples/rlroverlab/train_iql.py \
  --dataset /workspace/rlroverlab/datasets/rover_rgbd_expert.hdf5

python Examples/rlroverlab/train_iql_recurrent.py \
  --dataset /workspace/rlroverlab/datasets/rover_rgbd_expert.hdf5
```

Each training script supports `--eval_after_train`, which shells out to this
repo's evaluator and writes an optional metrics JSON.

## Evaluate a CloneLab Checkpoint

Run the trained CloneLab actor inside an RLRoverLab Isaac Lab task:

```bash
cd /workspace/rlroverlab
python examples/04_clonelab/eval_policy.py \
  --task AAURoverEnvRGBDRaw-v0 \
  --num_envs 32 \
  --checkpoint runs/<wandb-project>/<run-id>/checkpoints \
  --checkpoint_name best_model.pt \
  --steps 1000
```

The evaluator keeps the Isaac Lab launch and environment setup in RLRoverLab.
Only the observation adapter and CloneLab actor loading live in the integration
module.
