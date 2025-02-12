
We provide a number of examples on how to use the suite, these can be found in the examples directory. Below we show how to run the files.
## Training a new agent
In the example we show how to train a new agent using the suite:
```bash
# Run training script or evaluate pre-trained policy
cd examples/02_train/train.py
python train.py --task="AAURoverEnv-v0" --num_envs=128
```
## Using pre-trained agent
```bash
# Run training script or evaluate pre-trained policy
cd examples/03_inference_pretrained
python eval.py --task="AAURoverEnv-v0" --num_envs=128
```
## Recording data
```bash
# Run training script or evaluate pre-trained policy
cd examples/03_inference_pretrained
python record.py --task="AAURoverEnv-v0" --num_envs=16
```
