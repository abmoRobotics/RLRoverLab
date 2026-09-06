# Dataset Recordings

RLRoverLab records datasets from `examples/03_inference/eval.py` when
`--dataset_name` is set. Files are written to
`<dataset_dir>/<dataset_name>.hdf5` on environment close.

```bash
python examples/03_inference/eval.py \
  --task AAURoverEnvRGBDRawWVGA-v0 \
  --num_envs 1 \
  --steps 1000 \
  --enable_cameras \
  --dataset_dir ./datasets \
  --dataset_name rover_wvga_expert_1000 \
  --dataset_type RL_COMPRESSED
```

## Recorder Types

| `--dataset_type` | Format | Use when |
| --- | --- | --- |
| `RL` | Legacy Isaac Lab HDF5, robomimic-style layout | A loader expects `/data/demo_*/obs`, `/data/demo_*/next_obs`, actions, rewards, and dones. |
| `IL` | Legacy Isaac Lab HDF5, robomimic-style layout | Only observations and actions are needed. |
| `RL_COMPRESSED` | Optimized RLRoverLab RGB-D schema | RGB-D storage size and random-access offline loading matter more than direct robomimic layout compatibility. |

## Legacy HDF5

`RL` and `IL` use Isaac Lab's default HDF5 dataset writer. The layout follows
the robomimic convention of storing demonstrations under `/data/demo_N`.
Current robomimic training compatibility still depends on the loader,
environment metadata, and observation keys used by the training config.

For `RL`, each episode contains:

| Path | Contents |
| --- | --- |
| `/data` | Root data group. Attribute `total` is the total transition count; `env_args` stores environment metadata as JSON. |
| `/data/demo_N` | One recorded episode. Attributes include `num_samples` and optional `seed` and `success`. |
| `/data/demo_N/actions` | Action tensor for each transition. |
| `/data/demo_N/rewards` | Reward tensor for each transition. |
| `/data/demo_N/dones` | Done flags for each transition. |
| `/data/demo_N/obs/...` | Observation at time `t`. |
| `/data/demo_N/next_obs/...` | Observation after the action, at time `t + 1`. |

`IL` uses the same episode layout but records only `actions` and `obs`.
Datasets are gzip-compressed by HDF5. This format is simple and compatible with
many robomimic-style loaders, but RGB-D trajectories are large because
`next_obs` physically duplicates the next observation tree.

Use it with:

```bash
python examples/03_inference/eval.py \
  --task AAURoverEnvRGBDRawWVGA-v0 \
  --num_envs 1 \
  --steps 1000 \
  --enable_cameras \
  --dataset_dir ./datasets \
  --dataset_name rover_wvga_legacy_1000 \
  --dataset_type RL
```

## Optimized RGB-D HDF5

`RL_COMPRESSED` writes RLRoverLab's optimized RGB-D HDF5 schema. It stores RGB
and depth observations once in an indexed timeline instead of duplicating a
physical `next_obs` tree.

Use it when RGB-D storage size and random-access offline loading matter more
than direct robomimic layout compatibility. The full dataloader contract is in
[Optimized RGB-D HDF5](./dataset_recordings_optimized_rgbd.md).

Record an optimized RGB-D dataset with:

```bash
python examples/03_inference/eval.py \
  --task AAURoverEnvRGBDRawWVGA-v0 \
  --num_envs 1 \
  --steps 1000 \
  --enable_cameras \
  --dataset_dir ./datasets \
  --dataset_name rover_wvga_compressed_1000 \
  --dataset_type RL_COMPRESSED
```
