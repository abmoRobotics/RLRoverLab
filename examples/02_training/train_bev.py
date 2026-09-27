"""Train the rover BEV model: one RGB-D frame -> ego elevation and semantic maps.

The task's height-map PPO policy drives the rover while the model learns from the frames it sees.
"""

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from importlib.metadata import version

# Keep BLAS single-threaded before Isaac Lab starts Kit in headless mode.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the rover BEV model on frames collected by a driving policy.")
parser.add_argument("--task", default="AAURoverEnvSimpleBEV-v0")
parser.add_argument("--terrain", default=None, help="Registered terrain name, e.g. mars or debug.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", default=None, help="Resume from a BEV model checkpoint.")
parser.add_argument("--driver", default=None, help="RSL-RL PPO checkpoint that drives; defaults to the task's.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rover_envs.utils.launcher import configure_camera_launcher_args  # noqa: E402

configure_camera_launcher_args(args)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.utils import resolve_callable  # noqa: E402

import rover_envs.envs.navigation.robots  # noqa: E402, F401
from rover_envs.envs.navigation.learning.bev.losses import bev_loss, bev_metrics, bev_statistics  # noqa: E402
from rover_envs.envs.navigation.learning.bev.models import RoverBevModel  # noqa: E402
from rover_envs.envs.navigation.learning.rsl_rl.ppo_cfg import RoverPPORunnerCfg  # noqa: E402

# Observation groups the model trains on.
BEV_GROUPS = ("rgb", "depth", "camera", "bev_target")


def load_driver(env: RslRlVecEnvWrapper, path: str, device: str) -> torch.nn.Module:
    """Load the actor of a rover height-map PPO checkpoint; it acts deterministically on the `policy` group."""
    actor_cfg = handle_deprecated_rsl_rl_cfg(RoverPPORunnerCfg(), version("rsl-rl-lib")).actor.to_dict()
    actor_class = resolve_callable(actor_cfg.pop("class_name"))
    actor = actor_class(env.get_observations(), {"actor": ["policy"]}, "actor", env.num_actions, **actor_cfg)
    actor.load_state_dict(torch.load(path, map_location=device, weights_only=False)["actor_state_dict"])
    return actor.to(device).eval()


def plot_bev(rgb, elevation, semantics, lifted, target, bev_cfg) -> plt.Figure:
    """Show one frame: the camera image and the BEV lifted from its depth, the targets and the predictions."""
    (x_min, x_max), (y_min, y_max) = bev_cfg.x_range, bev_cfg.y_range
    visible = target[2] > 0.5
    limits = target[0][visible].quantile(torch.tensor([0.01, 0.99], device=target.device)) if visible.any() else [-1, 1]
    num_classes = len(bev_cfg.classes)
    elevation_style = {"cmap": "terrain", "vmin": float(limits[0]), "vmax": float(limits[1])}
    class_style = {"cmap": plt.get_cmap("tab10", num_classes), "vmin": -0.5, "vmax": num_classes - 0.5}
    panels = {
        (0, 0): ("depth lift elevation [m]", lifted[0].where(lifted[2] > 0, torch.nan), elevation_style),
        (0, 1): ("target elevation [m]", target[0].where(visible, torch.nan), elevation_style),
        (0, 2): ("predicted elevation [m]", elevation, elevation_style),
        (1, 1): ("target class", target[1].where(visible, torch.nan), class_style),
        (1, 2): ("predicted class", semantics.argmax(dim=0).float(), class_style),
    }
    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes[1, 0].imshow(rgb.cpu().numpy())
    axes[1, 0].set(title="camera", xticks=[], yticks=[])
    for (row, column), (title, image, style) in panels.items():
        # Forward up and left on the left: flip the [x, y] map and let the horizontal axis run from +y to -y.
        handle = axes[row, column].imshow(image.flip(0, 1).cpu().numpy(), extent=(y_max, y_min, x_min, x_max), **style)
        axes[row, column].plot(0.0, 0.0, "k^")
        axes[row, column].set(title=title, xlabel="y (left) [m]", ylabel="x (forward) [m]")
        colorbar = figure.colorbar(handle, ax=axes[row, column], shrink=0.8)
        if style is class_style:
            colorbar.set_ticks(range(num_classes), labels=bev_cfg.classes)
    figure.tight_layout()
    return figure


def train() -> int:
    env = None
    writer = None
    exit_code = 0
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        cfg = load_cfg_from_registry(args.task, "bev_cfg_entry_point")
        if args.terrain:
            env_cfg.scene.set_terrain(args.terrain)
        if args.seed is not None:
            cfg.seed = args.seed
        if args.max_iterations is not None:
            cfg.max_iterations = args.max_iterations
        env_cfg.seed = cfg.seed
        device = args.device

        log_dir = os.path.abspath(
            os.path.join("logs", "bev", cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        )
        env_cfg.log_dir = log_dir
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
        driver = load_driver(env, args.driver or gym.spec(args.task).kwargs["rsl_rl_driver_model_path"], device)

        bev_cfg = env_cfg.bev
        model_kwargs = {
            "num_classes": len(bev_cfg.classes),
            "x_range": bev_cfg.x_range,
            "y_range": bev_cfg.y_range,
            "resolution": bev_cfg.resolution,
            **cfg.model.to_dict(),
        }
        model = RoverBevModel(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.learning_rate)
        start_iteration = 0
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_iteration = checkpoint["iteration"] + 1

        os.makedirs(log_dir, exist_ok=True)
        if cfg.logger == "wandb":
            import wandb
            from rsl_rl.utils.wandb_utils import WandbSummaryWriter

            writer = WandbSummaryWriter(log_dir=log_dir, flush_secs=10, cfg=cfg.to_dict())
            writer.store_config(env_cfg, cfg.to_dict())
            # The writer names the run after the timestamp only; add the experiment to tell ablations apart.
            wandb.run.name = f"{cfg.experiment_name}_{os.path.basename(log_dir)}"
        else:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

        obs = env.get_observations()
        storage = {
            group: torch.empty((cfg.num_steps_per_env, *obs[group].shape), dtype=obs[group].dtype, device=device)
            for group in BEV_GROUPS
        }
        keep = torch.empty(cfg.num_steps_per_env, env.num_envs, dtype=torch.bool, device=device)
        env.episode_length_buf = torch.randint_like(env.episode_length_buf, high=int(env.max_episode_length))

        for iteration in range(start_iteration, start_iteration + cfg.max_iterations):
            start = time.time()
            with torch.inference_mode():
                for step in range(cfg.num_steps_per_env):
                    obs, _, dones, _ = env.step(driver(obs))
                    for group in BEV_GROUPS:
                        storage[group][step].copy_(obs[group])
                    # A reset env's frame was rendered before the reset teleported the rover, so it is skipped.
                    keep[step] = dones == 0
            collection_time = time.time() - start

            start = time.time()
            data = {group: value.flatten(0, 1) for group, value in storage.items()}
            frames = keep.flatten().nonzero().squeeze(1)
            batch_size = max(1, -(-len(frames) // cfg.num_mini_batches))
            losses, statistics, num_batches = torch.zeros(3, device=device), {}, 0
            model.train()
            for epoch in range(cfg.num_learning_epochs):
                for batch in frames[torch.randperm(len(frames), device=device)].split(batch_size):
                    rgb, depth, camera, target = (data[group][batch] for group in BEV_GROUPS)
                    elevation, semantics, lifted = model(rgb, depth, camera)
                    loss = bev_loss(elevation, semantics, target, cfg.semantic_loss_weight)
                    optimizer.zero_grad()
                    loss[0].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    losses += torch.stack(loss).detach()
                    num_batches += 1
                    # Score each frame on its first use, before the model has trained on it.
                    if epoch == 0:
                        for key, value in bev_statistics(elevation, semantics, lifted, target).items():
                            statistics[key] = statistics.get(key, 0) + value
            learning_time = time.time() - start

            metrics = bev_metrics(statistics, bev_cfg.classes)
            losses = (losses / max(1, num_batches)).tolist()
            scalars = {
                "Loss/total": losses[0],
                "Loss/elevation": losses[1],
                "Loss/semantic": losses[2],
                **{f"BEV/{key}": value for key, value in metrics.items()},
                "Perf/frames": len(frames),
                "Perf/collection_time": collection_time,
                "Perf/learning_time": learning_time,
            }
            for key, value in scalars.items():
                writer.add_scalar(key, value, iteration)
            print(
                f"[{iteration}] loss {losses[0]:.4f} | elevation MAE {100 * metrics['elevation_mae']:.2f} cm"
                f" (depth lift {100 * metrics['lift_elevation_mae']:.2f} cm on {100 * metrics['lift_coverage']:.0f}%"
                f" of visible cells) | accuracy {metrics['semantic_accuracy']:.3f} | "
                + " ".join(f"IoU {name} {metrics[f'iou_{name}']:.3f}" for name in bev_cfg.classes)
                + f" | {len(frames)} frames, collect {collection_time:.1f} s, learn {learning_time:.1f} s"
            )

            last = iteration == start_iteration + cfg.max_iterations - 1
            if len(frames) and (iteration % cfg.image_interval == 0 or last):
                model.eval()
                with torch.no_grad():
                    frame = frames[torch.randint(len(frames), (1,), device=device)]
                    rgb, depth, camera, target = (data[group][frame] for group in BEV_GROUPS)
                    elevation, semantics, lifted = model(rgb, depth, camera)
                figure = plot_bev(rgb[0], elevation[0], semantics[0], lifted[0], target[0], bev_cfg)
                figure.savefig(os.path.join(log_dir, f"bev_{iteration}.png"))
                if cfg.logger == "wandb":
                    wandb.log({"BEV/visualization": wandb.Image(figure)}, step=iteration)
                plt.close(figure)
            if (iteration + 1) % cfg.save_interval == 0 or last:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "iteration": iteration,
                        "model_kwargs": model_kwargs,
                        "cfg": cfg.to_dict(),
                    },
                    os.path.join(log_dir, f"model_{iteration}.pt"),
                )
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        if writer is not None:
            writer.close()
            if cfg.logger == "wandb":
                writer.stop()
        if env is not None:
            env.close()
        simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(train())
