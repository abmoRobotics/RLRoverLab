# Optimized RGB-D HDF5

`--dataset_type RL_COMPRESSED` writes RLRoverLab's optimized RGB-D schema:
`rlroverlab.offline_rgbd_v2` with `format_version = 2`. It is designed for
offline RGB-D dataloaders, not Isaac Lab episode replay or direct robomimic
layout consumption.

The key idea is simple: for an episode with `T` actions, the file stores
`T + 1` observations once in a single observation timeline. A transition stores
the action, reward, done flags, and two integer indices: one pointing to
`obs_t`, and one pointing to `obs_{t+1}`. Therefore the file has no physical
`next_obs` group.

## File Attributes

Only a few attributes are needed to load samples. The others are metadata that
make the file easier to inspect and validate.

| Attribute | Expected value or meaning |
| --- | --- |
| `schema_name` | Must be `rlroverlab.offline_rgbd_v2`. |
| `format_version` | Must be `2` for this document. |
| `writer_status` | Must be `complete`; incomplete or failed files should not be used for training. |
| `recommended_rgb_scale` | Scale applied by the loader after RGB decode. Default is `1 / 255`. |
| `recommended_depth_scale_m` | Scale applied by the loader after depth decode. Default is `0.001` m per integer unit. |
| `depth_invalid_sentinel` | Invalid depth value after decode. Default is `0`. |
| `total_transitions` | Convenience count. It should match `len(/transitions/actions)`. |
| `total_observations` | Convenience count. It should match `len(/observations/rgb_jpeg)`. |

Attributes such as `zero_structural_duplication`, `total_episodes`,
`camera_width`, `camera_height`, `camera_channels`, `depth_min_m`,
`depth_max_m`, codec names, source names, and JPEG quality are useful for
inspection, but a loader can derive or ignore them.

## Groups

The optimized file has three data groups that matter to a loader:
`/observations`, `/transitions`, and `/index`.

| Group | Purpose |
| --- | --- |
| `/observations` | Stores every observation once. Visual observations are compressed byte arrays; non-visual observations are numeric timelines. |
| `/transitions` | Stores transition-level values such as actions, rewards, and done flags. |
| `/index` | Connects transition rows to observation rows and, for sequence loaders, to episode boundaries. |

`/episodes` and `/data` are metadata groups. They are useful for inspection and
environment metadata, but they are not needed to reconstruct training samples.

## Observations Group

`/observations` has one row per observation timestep. For each episode with `T`
actions, this group receives `T + 1` rows: the initial observation and one
post-action observation for each transition.

| Path | Contents |
| --- | --- |
| `/observations/rgb_jpeg` | Shape `(num_observations,)`. Each row is a variable-length `uint8` array containing one JPEG frame. Decode to RGB `uint8` with shape `(H, W, 3)`. |
| `/observations/depth_jp2` | Shape `(num_observations,)`. Each row is a variable-length `uint8` array containing one JPEG2000 depth image. Decode to single-channel `uint16` depth in millimeters. |
| `/observations/state/...` | Optional numeric observation values. Every leaf dataset has length `num_observations`. Keys are task-dependent; common rover keys include `angle_diff`, `distance`, and `heading`. |

RGB and depth rows with the same observation index belong to the same timestep.
State rows, when present, use the same indexing.

## Transitions Group

`/transitions` has one row per action step. A minimal offline RL transition uses
`actions`, `rewards`, and `dones`.

| Path | Contents |
| --- | --- |
| `/transitions/actions` | Action tensor. First dimension is `num_transitions`; remaining dimensions are the action shape. |
| `/transitions/rewards` | Reward tensor. First dimension is `num_transitions`. |
| `/transitions/dones` | Boolean episode-boundary flag for each transition. |
| `/transitions/timeouts` | Optional boolean flag for time-limit truncations. |
| `/transitions/terminals` | Optional boolean flag for true terminal states. If `timeouts` is present, this can be derived as `dones & ~timeouts`. |
| `/transitions/extra/...` | Optional numeric transition data. Every leaf dataset has length `num_transitions`. |

For the simplest loader, `dones` is enough. `timeouts` and `terminals` are kept
to support offline RL algorithms that distinguish real terminal states from
time-limit truncations.

## Index Group

`/index` is what replaces a physical `next_obs` group. It maps each transition
row to the observation row for `obs_t` and `obs_{t+1}`.

| Path | Contents |
| --- | --- |
| `/index/obs_index` | `int64`, shape `(num_transitions,)`. Observation row for `obs_t`. |
| `/index/next_obs_index` | `int64`, shape `(num_transitions,)`. Observation row for `obs_{t+1}`. In the current writer this is always `obs_index + 1`. |
| `/index/episode_lengths` | `int64`, shape `(num_episodes,)`. Optional for random transition loading, required for episode-aware sequence sampling. |
| `/index/obs_offsets` | `int64`, shape `(num_episodes,)`. First observation row for each episode. Required for sequence loading. |
| `/index/transition_offsets` | `int64`, shape `(num_episodes,)`. First transition row for each episode. Required for sequence loading. |
| `/index/episode_id` | Optional per-transition episode id. Derivable from `episode_lengths` and `transition_offsets`. |
| `/index/episode_transition_index` | Optional local timestep inside each episode. Derivable from `transition_offsets`. |

For a random transition dataloader, only `obs_index` and `next_obs_index` are
strictly needed. Episode offsets and lengths are needed when sampling contiguous
sequences, splitting by episode, or doing frame stacking without crossing
episode boundaries.

## Index Invariants

For episode `e`:

| Quantity | Meaning |
| --- | --- |
| `T = /index/episode_lengths[e]` | Number of transitions in episode `e`. |
| `o0 = /index/obs_offsets[e]` | Start of the episode's observation timeline. |
| `t0 = /index/transition_offsets[e]` | Start of the episode's transition rows. |
| Observation rows | `o0, o0 + 1, ..., o0 + T`. There are `T + 1` rows. |
| Transition rows | `t0, t0 + 1, ..., t0 + T - 1`. There are `T` rows. |

For global transition row `i`, a loader reconstructs the transition as:

| Field | Read from |
| --- | --- |
| `obs` | Decode observation at `/index/obs_index[i]`. |
| `action` | `/transitions/actions[i]`. |
| `reward` | `/transitions/rewards[i]`. |
| `next_obs` | Decode observation at `/index/next_obs_index[i]`. |
| `done` | `/transitions/dones[i]`. If absent in older files, use `terminals[i] or timeouts[i]`. |
| `terminal` | `/transitions/terminals[i]`. |
| `timeout` | `/transitions/timeouts[i]`. |

The writer validates that `next_obs_index[i] == obs_index[i] + 1` and that the
next index never points past `total_observations`.

## Visual Decode Rules

RGB frames are stored as raw camera RGB values encoded as JPEG.

1. Read `encoded = file["observations/rgb_jpeg"][obs_index]`.
2. Convert the variable-length `uint8` array to bytes.
3. Decode JPEG to `uint8`.
4. Ensure channel order is RGB. Pillow returns RGB after `.convert("RGB")`; OpenCV returns BGR and must be converted with `[..., ::-1]`.
5. Convert to `float32` and multiply by `recommended_rgb_scale` for training.
6. Common tensor layout for PyTorch is channel-first `(3, H, W)`.

Depth frames are metric camera depth values quantized before compression.

1. Source depth is in meters.
2. During recording, finite depths in `[depth_min_m, depth_max_m]` are rounded to millimeters and stored as `uint16`.
3. Invalid, NaN, too-near, or too-far depth values are stored as `depth_invalid_sentinel`, normally `0`.
4. Read `encoded = file["observations/depth_jp2"][obs_index]`.
5. Decode JPEG2000 to a single-channel `uint16` image.
6. Convert to `float32` meters with `depth_m = decoded_uint16 * recommended_depth_scale_m`.
7. Preserve or mask `decoded_uint16 == depth_invalid_sentinel` as invalid. Do not treat `0` as a valid obstacle distance.
8. Common tensor layout for PyTorch is `(1, H, W)`.

JPEG2000 decode requires a library built with JPEG2000 support. Pillow with
OpenJPEG works on CPU. CUDA loaders can use torchvision or nvJPEG for RGB JPEG
and NVIDIA nvImageCodec or nvJPEG2000 for depth JP2.

## Reconstructing Samples

For global transition row `i`, read:

| Sample field | Source |
| --- | --- |
| `obs` | Decode `/observations/...` at row `/index/obs_index[i]`. |
| `action` | `/transitions/actions[i]`. |
| `reward` | `/transitions/rewards[i]`. |
| `next_obs` | Decode `/observations/...` at row `/index/next_obs_index[i]`. |
| `done` | `/transitions/dones[i]`. |
| `timeout` | `/transitions/timeouts[i]` if present. |
| `terminal` | `/transitions/terminals[i]` if present, otherwise `done and not timeout` when timeout is available. |
| `extra` | Any matching row in `/transitions/extra/...`. |

The current writer validates that `next_obs_index[i] == obs_index[i] + 1` and
that the next index never points past the observation timeline.

For sequence loaders, select one episode `e`, choose a local start timestep
`s`, then read transition rows
`transition_offsets[e] + s : transition_offsets[e] + s + sequence_length` and
observation rows `obs_offsets[e] + s : obs_offsets[e] + s + sequence_length`.
The corresponding `next_obs` rows are the same observation rows shifted by one.
