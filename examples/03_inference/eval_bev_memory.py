"""Evaluate the temporal BEV memory (Phase 2) with the frozen Phase-1 BEV model while the rover drives.

Every step, the instantaneous BEV and the persistent BEV are compared with the ground truth on three groups of cells
defined by the ground-truth camera visibility: visible now (current), seen earlier but not now (remembered), and never
seen (unknown). The ground truth only defines these groups; the memory itself stores cells with depth evidence only.
Writes metrics.json, an error-vs-distance plot, and videos and snapshots of chosen environments to
logs/bev_memory/, and with --wandb also to W&B.

The memory can be fed drifting poses whose size is unknown to it and varies: --odometry_noise low:high draws a severity
per run, and --drift_field_std varies it with location. Metrics are then also reported by the run's severity.
--dump_samples saves a random sample of remembered and current cells with the trajectories (samples.npz), to fit and
compare reliability models offline.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import traceback
from datetime import datetime
from importlib.metadata import version

# Keep BLAS single-threaded before Isaac Lab starts Kit in headless mode.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from isaaclab.app import AppLauncher  # noqa: E402


def severity(text: str) -> float | tuple[float, float]:
    """A drift severity multiplier, or a range 'low:high' from which every run draws its own log-uniformly."""
    low, _, high = text.partition(":")
    return (float(low), float(high)) if high else float(low)


parser = argparse.ArgumentParser(description="Evaluate the temporal BEV memory with a frozen Phase-1 BEV model.")
parser.add_argument("--task", default="AAURoverEnvSimpleBEV-v0")
parser.add_argument("--checkpoint", required=True, help="Frozen Phase-1 BEV model checkpoint.")
parser.add_argument("--terrain", default=None, help="Registered terrain name, e.g. mars or debug.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300, help="Policy steps to evaluate (0.2 s each).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--driving", choices=("policy", "circle"), default="policy",
    help="The goal-seeking height-map policy, or 1.5 m radius circles that keep revisiting terrain.",
)
parser.add_argument("--driver", default=None, help="RSL-RL PPO checkpoint that drives; defaults to the task's.")
parser.add_argument(
    "--odometry_noise", type=severity, default=0.0,
    help="Odometry drift severity, or a range low:high from which each run draws its own; 0 uses exact poses.",
)
parser.add_argument(
    "--drift_field_std", type=float, default=0.0,
    help="Log standard deviation of the drift's variation with location on the terrain; 0 for none.",
)
parser.add_argument("--drift_field_length", type=float, default=5.0, help="Correlation length of that variation [m].")
parser.add_argument("--write_rule", choices=("latest", "closest"), default="latest")
parser.add_argument("--class_fusion", choices=("latest", "mean", "log_odds"), default="latest")
parser.add_argument(
    "--fusion_window", type=int, default=0, help="Observations the mean class fusion weights equally; 0 for all."
)
parser.add_argument(
    "--elevation_source", choices=("network", "lift"), default="network",
    help="Store the network's elevation or the raw mean elevation of the depth points.",
)
parser.add_argument("--name", default="memory", help="Suffix of the output folder in logs/bev_memory/.")
parser.add_argument("--video_envs", type=int, nargs="+", default=[0], help="Environments to record videos of.")
parser.add_argument("--video_steps", type=int, default=150, help="Steps to record, from the start.")
parser.add_argument("--wandb", action="store_true", help="Also log metrics, plots and videos to W&B (rlroverlab).")
parser.add_argument(
    "--dump_samples", type=int, default=0,
    help="Remembered cells per environment and step to save for reliability models (and a quarter as many current).",
)
parser.add_argument(
    "--save_frames", type=int, default=0,
    help="Save the depth lift and the network's BEV of this many environments at every step (frames.npz), e.g. to "
    "develop the drift registration offline.",
)
parser.add_argument(
    "--keyframe_distance", type=float, default=0.0,
    help="Measure the drift by registering the depth lift against a keyframe this far back [m]; 0 disables. The "
    "registrations are saved with --dump_samples.",
)
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from PIL import Image  # noqa: E402
from rsl_rl.utils import resolve_callable  # noqa: E402

import rover_envs.envs.navigation.robots  # noqa: E402, F401
from rover_envs.envs.navigation.learning.bev.drift import DriftMonitor, local_relief  # noqa: E402
from rover_envs.envs.navigation.learning.bev.memory import CURRENT, REMEMBERED, UNKNOWN, BevMemory  # noqa: E402
from rover_envs.envs.navigation.learning.bev.models import RoverBevModel  # noqa: E402
from rover_envs.envs.navigation.learning.bev.odometry import OdometryDrift  # noqa: E402
from rover_envs.envs.navigation.learning.bev.registration import KeyframeRegistration  # noqa: E402
from rover_envs.envs.navigation.learning.rsl_rl.ppo_cfg import RoverPPORunnerCfg  # noqa: E402

GROUPS = {"current": CURRENT, "remembered": REMEMBERED, "unknown": UNKNOWN}
METHODS = ("instantaneous", "instantaneous_all", "persistent")
"""The Phase-1 BEV on cells with depth evidence, the Phase-1 BEV everywhere, and the memory's persistent BEV."""
DISTANCE_EDGES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# Bins of the runs' drift severity: octaves around the nominal 1x.
SEVERITY_EDGES = (0.5, 1.0, 2.0)

# Reference data-viz palette (light): chart surface, ink, the first two categorical slots and a no-data gray.
SURFACE, INK, INK_SECONDARY, BLUE, ORANGE, NO_DATA = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# Class maps: muted sand for ground so that rocks (violet) stand out; distinct from the no-data gray.
CLASS_COLORS = {"ground": "#b9ad90", "rock": "#4a3aa7"}


def load_driver(env: RslRlVecEnvWrapper, path: str, device: str) -> torch.nn.Module:
    """Load the actor of a rover height-map PPO checkpoint; it acts deterministically on the `policy` group."""
    actor_cfg = handle_deprecated_rsl_rl_cfg(RoverPPORunnerCfg(), version("rsl-rl-lib")).actor.to_dict()
    actor_class = resolve_callable(actor_cfg.pop("class_name"))
    actor = actor_class(env.get_observations(), {"actor": ["policy"]}, "actor", env.num_actions, **actor_cfg)
    actor.load_state_dict(torch.load(path, map_location=device, weights_only=False)["actor_state_dict"])
    return actor.to(device).eval()


def add(sums: dict, key: str, value: torch.Tensor) -> None:
    sums[key] = sums.get(key, 0) + value.double()


def score(sums, prefix, provided, cells, elevation, classes, target, rock) -> None:
    """Add the counts behind the coverage, elevation MAE and rock IoU of one method on one group of cells."""
    hit = provided & cells
    predicted_rock, true_rock = classes == rock, target[:, 1] == rock
    add(sums, f"{prefix}/cells", cells.sum())
    add(sums, f"{prefix}/provided", hit.sum())
    add(sums, f"{prefix}/error", (elevation - target[:, 0]).abs()[hit].sum())
    add(sums, f"{prefix}/rock_tp", (hit & predicted_rock & true_rock).sum())
    add(sums, f"{prefix}/rock_fp", (hit & predicted_rock & ~true_rock).sum())
    add(sums, f"{prefix}/rock_fn", (hit & ~predicted_rock & true_rock).sum())


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def summarize(sums: dict, frames: int) -> dict:
    s = {key: value.tolist() for key, value in sums.items()}
    total = s["status/total"]
    summary = {
        "frames": frames,
        "provenance_share": {name: ratio(s[f"status/{name}"], total) for name in GROUPS},
        "groups": {},
        "error_vs_distance": [],
    }
    for group in GROUPS:
        summary["groups"][group] = {"cell_share": ratio(s[f"persistent/{group}/cells"], total)}
        for method in METHODS:
            p = f"{method}/{group}"
            summary["groups"][group][method] = {
                "coverage": ratio(s[f"{p}/provided"], s[f"{p}/cells"]),
                "elevation_mae_cm": 100 * ratio(s[f"{p}/error"], s[f"{p}/provided"]),
                "rock_iou": ratio(s[f"{p}/rock_tp"], s[f"{p}/rock_tp"] + s[f"{p}/rock_fp"] + s[f"{p}/rock_fn"]),
            }
    edges = (0.0, *DISTANCE_EDGES, math.inf)
    for i, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        cells = s["distance/cells"][i]
        tp, fp, fn = (s[f"distance/rock_{kind}"][i] for kind in ("tp", "fp", "fn"))
        summary["error_vs_distance"].append({
            "distance_m": [low, high],
            "cells": cells,
            "persistent_mae_cm": 100 * ratio(s["distance/persistent"][i], cells),
            "instantaneous_all_mae_cm": 100 * ratio(s["distance/instantaneous_all"][i], cells),
            "persistent_rock_iou": ratio(tp, tp + fp + fn),
        })
    if "pose/count" in s:
        summary["pose_error"] = {
            "position_mean_cm": 100 * ratio(s["pose/position"], s["pose/count"]),
            "yaw_mean_deg": math.degrees(ratio(s["pose/yaw"], s["pose/count"])),
            "height_mean_cm": 100 * ratio(s["pose/height"], s["pose/count"]),
        }
        # Remembered cells and pose errors by the drift severity of the run they belong to.
        summary["by_severity"] = []
        severity_edges, distances = (0.0, *SEVERITY_EDGES, math.inf), len(DISTANCE_EDGES) + 1
        for i, (low, high) in enumerate(zip(severity_edges[:-1], severity_edges[1:])):
            rows = slice(i * distances, (i + 1) * distances)
            cells, error = s["severity/cells"][rows], s["severity/persistent"][rows]
            tp, fp, fn = (s[f"severity/rock_{kind}"][rows] for kind in ("tp", "fp", "fn"))
            summary["by_severity"].append({
                "severity": [low, high],
                "frames": s["severity/frames"][i],
                "position_error_mean_cm": 100 * ratio(s["severity/position"][i], s["severity/frames"][i]),
                "yaw_error_mean_deg": math.degrees(ratio(s["severity/yaw"][i], s["severity/frames"][i])),
                "measured_drift_level": ratio(s["severity/level"][i], s["severity/measured"][i]),
                "remembered_cells": sum(cells),
                "remembered_mae_cm": 100 * ratio(sum(error), sum(cells)),
                "remembered_rock_iou": ratio(sum(tp), sum(tp) + sum(fp) + sum(fn)),
                "remembered_mae_cm_by_distance": [100 * ratio(e, c) for e, c in zip(error, cells)],
                "remembered_rock_iou_by_distance": [ratio(t, t + p + n) for t, p, n in zip(tp, fp, fn)],
            })
    return summary


def show_bev(ax, image: torch.Tensor, bev_cfg, title: str, **style):
    """Draw an [x, y] BEV map with forward up and left on the left."""
    (x_min, x_max), (y_min, y_max) = bev_cfg.x_range, bev_cfg.y_range
    ax.set_facecolor(NO_DATA)
    extent = (y_max, y_min, x_min, x_max)
    handle = ax.imshow(image.flip(0, 1).numpy(), extent=extent, interpolation="nearest", **style)
    ax.plot(0.0, 0.0, marker="^", color=INK, markersize=6)
    ax.set_title(title, color=INK, fontsize=10)
    ax.set(xticks=[], yticks=[])
    return handle


def animate(frames: list[dict], bev_cfg, dt: float, path: str) -> str:
    """Save a real-time MP4 ``path``.mp4 of one environment, and as ``path``_snapshot.png the frame that remembers the
    most cells: the instantaneous, persistent and ground-truth elevation and classes, the camera, the provenance of each
    persistent cell and the distance travelled since it was observed. Returns the video path."""
    targets = np.concatenate([frame["target"].flatten().numpy() for frame in frames])
    low, high = np.nanpercentile(targets, [2, 98])
    elevation = {"cmap": BLUES, "vmin": low, "vmax": high}
    classes = list(bev_cfg.classes)
    semantics = {
        "cmap": ListedColormap([CLASS_COLORS[name] for name in classes]), "vmin": -0.5, "vmax": len(classes) - 0.5
    }
    figure, axes = plt.subplots(3, 3, figsize=(12, 10.8), dpi=80, facecolor=SURFACE, layout="constrained")
    handles = {
        "instantaneous": show_bev(
            axes[0, 0], frames[0]["instantaneous"], bev_cfg, "instantaneous BEV (depth evidence)", **elevation
        ),
        "persistent": show_bev(axes[0, 1], frames[0]["persistent"], bev_cfg, "persistent BEV", **elevation),
        "target": show_bev(axes[0, 2], frames[0]["target"], bev_cfg, "ground truth", **elevation),
        "instantaneous_class": show_bev(
            axes[1, 0], frames[0]["instantaneous_class"], bev_cfg, "instantaneous class (depth evidence)", **semantics
        ),
        "persistent_class": show_bev(
            axes[1, 1], frames[0]["persistent_class"], bev_cfg, "persistent class", **semantics
        ),
        "target_class": show_bev(axes[1, 2], frames[0]["target_class"], bev_cfg, "ground-truth class", **semantics),
        "status": show_bev(axes[2, 1], frames[0]["status"], bev_cfg, "provenance",
                           cmap=ListedColormap([NO_DATA, BLUE, ORANGE]), vmin=-0.5, vmax=2.5),
        "distance": show_bev(axes[2, 2], frames[0]["distance"], bev_cfg, "distance travelled since observed",
                             cmap="Oranges", vmin=0.0, vmax=5.0),
    }
    camera = axes[2, 0].imshow(frames[0]["rgb"].numpy())
    axes[2, 0].set_title("camera", color=INK, fontsize=10)
    axes[2, 0].set(xticks=[], yticks=[])
    figure.colorbar(handles["persistent"], ax=axes[0, :], shrink=0.8, label="elevation [m]")
    colorbar = figure.colorbar(handles["persistent_class"], ax=axes[1, :], shrink=0.8)
    colorbar.set_ticks(range(len(classes)), labels=classes)
    colorbar = figure.colorbar(handles["status"], ax=axes[2, 1], shrink=0.8)
    colorbar.set_ticks([UNKNOWN, CURRENT, REMEMBERED], labels=["unknown", "current", "remembered"])
    figure.colorbar(handles["distance"], ax=axes[2, 2], shrink=0.8, label="distance [m]")
    title = figure.suptitle("", color=INK)
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    encoder = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", f"{1 / dt:g}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", f"{path}.mp4"],
        stdin=subprocess.PIPE,
    )
    fullest = max(range(len(frames)), key=lambda i: int((frames[i]["status"] == REMEMBERED).sum()))
    for i, frame in enumerate(frames):
        camera.set_data(frame["rgb"].numpy())
        for key, handle in handles.items():
            handle.set_data(frame[key].flip(0, 1).numpy())
        title.set_text(f"environment {frame['env']}, step {frame['step']} ({frame['step'] * dt:.1f} s)")
        figure.canvas.draw()
        image = np.asarray(figure.canvas.buffer_rgba())[..., :3]
        encoder.stdin.write(image.tobytes())
        if i == fullest:
            Image.fromarray(image).save(f"{path}_snapshot.png")
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError(f"ffmpeg failed to encode {path}.mp4")
    plt.close(figure)
    return f"{path}.mp4"


def plot_error_vs_distance(summary: dict, folder: str) -> None:
    """Elevation error of the remembered cells against the distance travelled since they were observed."""
    rows = summary["error_vs_distance"]
    edges = [row["distance_m"] for row in rows]
    labels = [f"{low:g}–{high:g}" if math.isfinite(high) else f"≥{low:g}" for low, high in edges]
    figure, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    series = (
        ("persistent_mae_cm", BLUE, "persistent BEV (memory)"),
        ("instantaneous_all_mae_cm", ORANGE, "instantaneous BEV (network guess)"),
    )
    for key, color, label in series:
        values = [row[key] for row in rows]
        ax.plot(range(len(rows)), values, color=color, linewidth=2, marker="o", markersize=8, label=label)
        measured = [i for i, value in enumerate(values) if not math.isnan(value)]
        if measured:
            ax.annotate(label.split(" (")[0], (measured[-1], values[measured[-1]]), xytext=(8, 0),
                        textcoords="offset points", va="center", color=INK_SECONDARY, fontsize=9)
    ax.set_xticks(range(len(rows)), labels)
    ax.set_xlabel("distance travelled since the cell was observed [m]", color=INK_SECONDARY)
    ax.set_ylabel("elevation MAE [cm]", color=INK_SECONDARY)
    ax.set_title("Remembered cells: elevation error against distance travelled", color=INK, fontsize=11, loc="left",
                 pad=28)
    ax.set_ylim(bottom=0)
    ax.set_xlim(-0.3, len(rows) - 0.3)
    ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
    ax.tick_params(colors=INK_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    figure.tight_layout()
    figure.savefig(os.path.join(folder, "error_vs_distance.png"), dpi=130, facecolor=SURFACE)
    plt.close(figure)


def evaluate() -> int:
    env = run = None
    exit_code = 0
    try:
        device = args.device
        env_cfg = parse_env_cfg(args.task, device=device, num_envs=args.num_envs)
        if args.terrain:
            env_cfg.scene.set_terrain(args.terrain)
        env_cfg.seed = args.seed
        folder = os.path.abspath(
            os.path.join("logs", "bev_memory", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.name}")
        )
        os.makedirs(folder)
        env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
        dt = env.unwrapped.step_dt

        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        bev_cfg, kwargs = env_cfg.bev, checkpoint["model_kwargs"]
        if (tuple(kwargs["x_range"]), tuple(kwargs["y_range"]), kwargs["resolution"]) != (
            tuple(bev_cfg.x_range), tuple(bev_cfg.y_range), bev_cfg.resolution
        ):
            raise ValueError(f"The checkpoint's BEV grid {kwargs} does not match the task's {bev_cfg}.")
        model = RoverBevModel(**kwargs).to(device).eval()
        model.load_state_dict(checkpoint["model_state_dict"])
        rock = list(bev_cfg.classes).index("rock")

        if args.driving == "policy":
            driver = load_driver(env, args.driver or gym.spec(args.task).kwargs["rsl_rl_driver_model_path"], device)
        else:
            # Half of the rovers turn left and half right, at 0.45 m/s on a 1.5 m radius.
            turn = torch.where(torch.arange(env.num_envs, device=device) % 2 == 0, 0.3, -0.3)
            circle = torch.stack((torch.full_like(turn, 0.45), turn), dim=1)

        grid = (bev_cfg.x_range, bev_cfg.y_range, bev_cfg.resolution, dt)
        odometry = None
        if args.odometry_noise:
            odometry = OdometryDrift(
                env.num_envs, args.odometry_noise, dt, args.seed, device,
                field_std=args.drift_field_std, field_length=args.drift_field_length,
            )
        severity_edges = torch.tensor(SEVERITY_EDGES, device=device)
        distance_edges = torch.tensor(DISTANCE_EDGES, device=device)
        # Remembered cells binned by the run's severity and the distance travelled since they were observed.
        severity_bins = (len(SEVERITY_EDGES) + 1) * (len(DISTANCE_EDGES) + 1)
        dump = {"samples": [], "trajectory": [], "registrations": []} if args.dump_samples else None
        frames_to_save = [] if args.save_frames else None
        sampler = torch.Generator(device).manual_seed(args.seed)
        registration = None
        if args.keyframe_distance:
            registration = KeyframeRegistration(
                env.num_envs, bev_cfg.x_range, bev_cfg.y_range, bev_cfg.resolution,
                keyframe_distance=args.keyframe_distance, seed=args.seed, device=device,
            )
            key_step = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        if args.wandb:
            import wandb

            run = wandb.init(
                project="rlroverlab", entity=os.environ.get("WANDB_USERNAME"),
                name=f"bev_memory_{args.name}_{os.path.basename(folder)[:19]}", config=vars(args),
            )
        sums, frames, video = {}, 0, {e: [] for e in args.video_envs}
        with torch.inference_mode():
            memory = BevMemory(
                env.num_envs, len(bev_cfg.classes), *grid, write_rule=args.write_rule, class_fusion=args.class_fusion,
                fusion_window=args.fusion_window, device=device,
            )
            # The drift the rover can measure itself, from terrain it observes again.
            monitor = DriftMonitor(env.num_envs, bev_cfg.resolution, device=device)
            # Evaluation only: where the ground truth says the camera has seen each cell.
            seen = BevMemory(env.num_envs, 1, *grid, device=device)
            obs = env.get_observations()
            for step in range(args.steps):
                obs, _, dones, _ = env.step(driver(obs) if args.driving == "policy" else circle)
                reset = dones.bool()
                # A reset env's frame was rendered before the reset moved the rover: it is neither stored nor scored.
                fresh = ~reset
                memory.reset(reset)
                monitor.reset(reset)
                seen.reset(reset)
                true_pose = obs["pose"]
                pose = odometry(true_pose, reset) if odometry else true_pose

                elevation, logits, lifted = model(obs["rgb"], obs["depth"], obs["camera"])
                probabilities = logits.softmax(dim=1)
                evidence = (lifted[:, 2] > 0) & fresh.view(-1, 1, 1)
                stored = elevation if args.elevation_source == "network" else lifted[:, 0]
                travelled = memory.travelled.clone()
                persistent = memory.step(pose, stored, probabilities, evidence)
                drift_estimate = monitor.step(persistent, memory.travelled - travelled)
                if frames_to_save is not None:
                    saved = slice(0, args.save_frames)
                    frames_to_save.append({
                        "true_pose": true_pose[saved], "pose": pose[saved], "reset": reset[saved],
                        "lift_height": lifted[saved, 0], "lift_count": lifted[saved, 2].expm1().round().to(torch.int16),
                        "elevation": elevation[saved], "target": obs["bev_target"][saved, 0],
                    })
                    frames_to_save[-1] = {key: value.cpu() for key, value in frames_to_save[-1].items()}
                if registration is not None:
                    measured = registration.step(pose, lifted[:, 0], lifted[:, 2].expm1(), reset)
                    due = torch.isfinite(measured["interval"])
                    if dump is not None and due.any():
                        record = {key: measured[key][due] for key in
                                  ("registered", "correction", "interval", "cost", "overlap", "curvature")}
                        record.update(env=due.nonzero().squeeze(1), step=torch.full_like(key_step[due], step),
                                      key_step=key_step[due])
                        dump["registrations"].append({key: value.cpu() for key, value in record.items()})
                    key_step = torch.where(measured["keyframe"], step, key_step)

                target = obs["bev_target"]
                visible = (target[:, 2] > 0.5) & fresh.view(-1, 1, 1)
                nothing = torch.zeros_like(elevation)
                history = seen.step(true_pose, nothing, nothing.unsqueeze(1), visible)
                scored = fresh.view(-1, 1, 1) & torch.isfinite(target[:, 0])
                predicted = {
                    "instantaneous": (evidence, elevation, logits.argmax(dim=1)),
                    "instantaneous_all": (torch.ones_like(evidence), elevation, logits.argmax(dim=1)),
                    "persistent": (
                        persistent["status"] != UNKNOWN,
                        persistent["elevation"],
                        torch.nan_to_num(persistent["semantics"]).argmax(dim=1),
                    ),
                }
                for group, status in GROUPS.items():
                    cells = scored & (history["status"] == status)
                    for method, (provided, values, classes) in predicted.items():
                        score(sums, f"{method}/{group}", provided, cells, values, classes, target, rock)
                    add(sums, f"status/{group}", (scored & (persistent["status"] == status)).sum())
                add(sums, "status/total", scored.sum())

                # Remembered cells by the distance travelled since they were observed.
                remembered = scored & (persistent["status"] == REMEMBERED)
                bins = torch.bucketize(persistent["distance"][remembered], distance_edges, right=True)
                predicted_rock = predicted["persistent"][2][remembered] == rock
                true_rock = target[:, 1][remembered] == rock
                per_bin = {
                    "cells": torch.ones_like(bins, dtype=torch.float64),
                    "persistent": (persistent["elevation"] - target[:, 0])[remembered].abs(),
                    "instantaneous_all": (elevation - target[:, 0])[remembered].abs(),
                    "rock_tp": predicted_rock & true_rock,
                    "rock_fp": predicted_rock & ~true_rock,
                    "rock_fn": ~predicted_rock & true_rock,
                }
                for key, values in per_bin.items():
                    counts = torch.bincount(bins, weights=values.double(), minlength=len(DISTANCE_EDGES) + 1)
                    add(sums, f"distance/{key}", counts)
                if odometry:
                    error = (pose - true_pose)[fresh]
                    add(sums, "pose/count", fresh.sum())
                    add(sums, "pose/position", error[:, :2].norm(dim=1).sum())
                    add(sums, "pose/height", error[:, 2].abs().sum())
                    add(sums, "pose/yaw", (torch.remainder(error[:, 3] + math.pi, 2 * math.pi) - math.pi).abs().sum())
                    run_bin = torch.bucketize(odometry.severity, severity_edges, right=True)
                    measured = torch.isfinite(drift_estimate["level"])
                    pose_errors = {
                        "frames": fresh,
                        "position": (pose - true_pose)[:, :2].norm(dim=1),
                        "yaw": (torch.remainder(pose[:, 3] - true_pose[:, 3] + math.pi, 2 * math.pi) - math.pi).abs(),
                        "measured": fresh & measured,
                        "level": torch.where(measured, drift_estimate["level"], 0.0),
                    }
                    for key, values in pose_errors.items():
                        counts = torch.bincount(
                            run_bin[fresh], weights=values[fresh].double(), minlength=len(SEVERITY_EDGES) + 1
                        )
                        add(sums, f"severity/{key}", counts)
                    run_of_cell = run_bin.view(-1, 1, 1).expand_as(remembered)[remembered]
                    cell_bin = run_of_cell * (len(DISTANCE_EDGES) + 1) + bins
                    for key in ("cells", "persistent", "rock_tp", "rock_fp", "rock_fn"):
                        counts = torch.bincount(cell_bin, weights=per_bin[key].double(), minlength=severity_bins)
                        add(sums, f"severity/{key}", counts)
                frames += int(fresh.sum())

                if dump is not None:
                    unset = torch.zeros(env.num_envs, device=device)
                    trajectory = {
                        "true_pose": true_pose, "pose": pose, "fresh": fresh,
                        "level": odometry.level if odometry else unset,
                        "severity": odometry.severity if odometry else unset,
                        "drift_floor": drift_estimate["floor"], "drift_excess": drift_estimate["excess"],
                        "drift_level": drift_estimate["level"], "drift_weight": drift_estimate["weight"],
                        "revisit_statistics": drift_estimate["statistics"],
                    }
                    dump["trajectory"].append({key: value.cpu() for key, value in trajectory.items()})
                    maps = {
                        "age": persistent["age"],
                        "distance": persistent["distance"],
                        "range": persistent["range"],
                        "elevation": persistent["elevation"],
                        "target": target[:, 0],
                        "guess": elevation,
                        "rock_probability": persistent["semantics"][:, rock],
                        "target_class": target[:, 1],
                        "relief": local_relief(persistent["elevation"]),
                        "history": history["status"],
                        "points": lifted[:, 2],
                        **{key: persistent[key] for key in
                           ("first_elevation", "first_age", "first_distance", "first_range")},
                    }
                    # A random sample of the remembered cells, and a quarter as many current ones, of every env.
                    for status, count in ((REMEMBERED, args.dump_samples), (CURRENT, max(1, args.dump_samples // 4))):
                        eligible = (scored & (persistent["status"] == status)).flatten(1)
                        draw = torch.rand(eligible.shape, generator=sampler, device=device).masked_fill(~eligible, -1.0)
                        top = draw.topk(count, dim=1)
                        keep = top.values >= 0
                        env_index = torch.arange(env.num_envs, device=device).unsqueeze(1).expand_as(keep)[keep]
                        cell_index = top.indices[keep]
                        sample = {name: m.flatten(1)[env_index, cell_index] for name, m in maps.items()}
                        sample.update(
                            env=env_index, cell=cell_index, step=torch.full_like(env_index, step),
                            status=torch.full_like(env_index, status),
                        )
                        dump["samples"].append({key: value.cpu() for key, value in sample.items()})

                if run is not None:
                    # How the memory builds up: the output share it knows, and its coverage and elevation error on
                    # the cells the camera saw earlier but not now, against the network's guess there.
                    seen_before = scored & (history["status"] == REMEMBERED)
                    known = persistent["status"] != UNKNOWN
                    memory_error = (persistent["elevation"] - target[:, 0]).abs()[seen_before & known]
                    guess_error = (elevation - target[:, 0]).abs()[seen_before]
                    run.log({
                        "step/known_share": ((known & scored).sum() / scored.sum()).item(),
                        "step/remembered_coverage": ((known & seen_before).sum() / seen_before.sum()).item(),
                        "step/remembered_mae_cm": 100 * memory_error.mean().item(),
                        "step/remembered_guess_mae_cm": 100 * guess_error.mean().item(),
                    }, step=step)

                for e, frames_of_env in video.items():
                    if step < args.video_steps:
                        frames_of_env.append({
                            "env": e,
                            "step": step,
                            "rgb": obs["rgb"][e, ::2, ::2].cpu(),
                            "instantaneous": elevation[e].where(evidence[e], torch.nan).cpu(),
                            "persistent": persistent["elevation"][e].cpu(),
                            "target": target[e, 0].cpu(),
                            "instantaneous_class": logits[e].argmax(dim=0).float().where(evidence[e], torch.nan).cpu(),
                            "persistent_class": predicted["persistent"][2][e].float().where(
                                persistent["status"][e] != UNKNOWN, torch.nan
                            ).cpu(),
                            "target_class": target[e, 1].where(target[e, 1] >= 0, torch.nan).cpu(),
                            "status": persistent["status"][e].float().cpu(),
                            "distance": persistent["distance"][e].where(remembered[e], torch.nan).cpu(),
                        })

        summary = summarize(sums, frames)
        summary["config"] = {**vars(args), "memory_size": memory.size, "resolution": bev_cfg.resolution, "dt": dt}
        with open(os.path.join(folder, "metrics.json"), "w") as file:
            json.dump(summary, file, indent=2, default=str)
        if frames_to_save:
            # Per step [T, E, ...]: poses, resets, the lift's mean height and point count, the network's elevation and
            # the ground-truth elevation, all on the ego BEV grid.
            np.savez_compressed(
                os.path.join(folder, "frames.npz"),
                **{key: torch.stack([f[key] for f in frames_to_save]).numpy() for key in frames_to_save[0]},
                dt=dt, x_range=bev_cfg.x_range, y_range=bev_cfg.y_range, resolution=bev_cfg.resolution,
            )
        if dump is not None:
            # Per step and environment [T, B, ...]: the true and the memory's poses [x, y, z, yaw], whether the frame
            # was used, and the true drift level and run severity (0 with exact poses). Per sampled cell: its env,
            # step, flat ego cell index and provenance, with the persistent BEV's values there. Per registration
            # (prefixed registration_): its env, step and keyframe step with the measured correction and its quality.
            registrations = {
                f"registration_{key}": torch.cat([r[key] for r in dump["registrations"]]).numpy()
                for key in (dump["registrations"][0] if dump["registrations"] else {})
            }
            np.savez_compressed(
                os.path.join(folder, "samples.npz"),
                **{key: torch.stack([t[key] for t in dump["trajectory"]]).numpy() for key in dump["trajectory"][0]},
                **{key: torch.cat([s[key] for s in dump["samples"]]).numpy() for key in dump["samples"][0]},
                **registrations,
                dt=dt, x_range=bev_cfg.x_range, y_range=bev_cfg.y_range, resolution=bev_cfg.resolution, rock=rock,
            )
        plot_error_vs_distance(summary, folder)
        videos = {
            e: animate(frames_of_env, bev_cfg, dt, os.path.join(folder, f"memory_env{e}"))
            for e, frames_of_env in video.items() if frames_of_env
        }
        if run is not None:
            flat = {"frames": frames, **{f"provenance/{k}": v for k, v in summary["provenance_share"].items()}}
            for group, results in summary["groups"].items():
                for method in METHODS:
                    flat.update({f"{group}/{method}/{k}": v for k, v in results[method].items()})
            flat.update({f"pose_error/{k}": v for k, v in summary.get("pose_error", {}).items()})
            for row in summary.get("by_severity", []):
                label = "severity_{:g}-{:g}".format(*row["severity"])
                flat.update({f"{label}/{k}": v for k, v in row.items() if not isinstance(v, list)})
            run.summary.update(flat)
            columns = ["distance_m", "cells", "persistent_mae_cm", "instantaneous_all_mae_cm", "persistent_rock_iou"]
            table = wandb.Table(columns=columns)
            for row in summary["error_vs_distance"]:
                table.add_data(f"{row['distance_m'][0]:g}-{row['distance_m'][1]:g}", *(row[c] for c in columns[1:]))
            media = {
                "error_vs_distance": table,
                "media/error_vs_distance": wandb.Image(os.path.join(folder, "error_vs_distance.png")),
            }
            for e, path in videos.items():
                media[f"media/video_env{e}"] = wandb.Video(path, format="mp4")
                media[f"media/snapshot_env{e}"] = wandb.Image(path.replace(".mp4", "_snapshot.png"))
            run.log(media, step=args.steps)

        print(f"\n{frames} frames, output cells: " + ", ".join(
            f"{name} {100 * share:.1f}%" for name, share in summary["provenance_share"].items()
        ))
        for group, results in summary["groups"].items():
            print(f"{group} cells ({100 * results['cell_share']:.1f}% of the output):")
            for method in METHODS:
                m = results[method]
                print(f"  {method:18s} coverage {100 * m['coverage']:5.1f}% | elevation MAE "
                      f"{m['elevation_mae_cm']:6.2f} cm | rock IoU {m['rock_iou']:.3f}")
        for row in summary["error_vs_distance"]:
            print(f"  remembered {row['distance_m'][0]:g}-{row['distance_m'][1]:g} m: {row['cells']:.0f} cells, "
                  f"memory {row['persistent_mae_cm']:.2f} cm vs guess {row['instantaneous_all_mae_cm']:.2f} cm, "
                  f"memory rock IoU {row['persistent_rock_iou']:.3f}")
        if "pose_error" in summary:
            print("  pose error:", summary["pose_error"])
        for row in summary.get("by_severity", []):
            print("  severity {:g}-{:g}: ".format(*row["severity"]) + f"{row['frames']:.0f} frames, pose error "
                  f"{row['position_error_mean_cm']:.1f} cm {row['yaw_error_mean_deg']:.2f} deg | remembered "
                  f"{row['remembered_mae_cm']:.2f} cm, rock IoU {row['remembered_rock_iou']:.3f} | measured "
                  f"drift level {row['measured_drift_level']:.2f}")
        print(f"Wrote {folder}")
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        if run is not None:
            run.finish(exit_code=exit_code)
        if env is not None:
            env.close()
        simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(evaluate())
