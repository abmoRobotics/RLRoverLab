
We provide a number of examples on how to use the suite, these can be found in the examples directory. Below we show how to run the files.
## Training a new agent
In the example we show how to train a new agent using the suite:
```bash
# Run training script or evaluate pre-trained policy
cd examples/02_training/train.py
python train.py --task="AAURoverEnv-v0" --num_envs=128
python train.py --task="AAURoverEnvSimple-v0" --num_envs=128
```

### Stereo CD-frame training

The `AAURoverCDStereoEnv-v0` task uses two parallel pinhole cameras with a 12 cm baseline centered on the
existing CD camera mount. Each camera runs an independent GPU event simulation at physics rate. The policy
receives a `180×320×2` CD image, with the left camera in channel 0 and the right camera in channel 1, plus the
existing five navigation/action inputs. CD values remain OFF=0.0, no event=0.5, and ON=1.0.

Run from the repository root:

```bash
python examples/02_training/train.py --task AAURoverCDStereoEnv-v0 --agent PPO --enable_cameras --num_envs 8
```

The task selects `rover_cd_stereo_ppo.yaml`, which configures the existing CD CNNs for two image channels.
The live observation viewer shows separate `policy-rgb_image-left` and `policy-rgb_image-right` images,
with OFF events in black, no events in gray, and ON events in white. Training still receives the combined
two-channel floating-point observation.
The single-camera `AAURoverCDEnv-v0` task remains available with its existing training configuration.

## Using pre-trained agent
```bash
# Run training script or evaluate pre-trained policy
cd examples/03_inference
python eval.py --task="AAURoverEnv-v0" --num_envs=32
python eval.py --task="AAURoverEnvSimple-v0" --num_envs=32
```
## Recording data
```bash
# Run training script or evaluate pre-trained policy
cd examples/03_inference
python eval.py --task="AAURoverEnv-v0" --num_envs=32 --dataset_name="dataset_name" --dataset_dir="../../datasets"
python eval.py --task="AAURoverEnvSimple-v0" --num_envs=32 --dataset_name="dataset_name" --dataset_dir="../../datasets"
