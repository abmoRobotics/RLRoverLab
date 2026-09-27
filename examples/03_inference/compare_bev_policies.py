"""Compare Phase-4 driving policies from eval_bev_policy.py runs and their training logs.

    python examples/03_inference/compare_bev_policies.py \
        --runs teacher=logs/bev_policy/<run> rgb=logs/bev_policy/<run> full=logs/bev_policy/<run> ... \
        --training full=logs/rsl_rl/rover_bev_distillation/<run> ...

Prints the outcome tables (with Wilson 95% intervals on the success rate) and writes two figures to
logs/bev_policy_comparison/: success by drift octave, and training curves. Needs numpy, matplotlib and tensorboard,
e.g. /isaac-sim/python.sh in the container.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SEVERITY_EDGES = (0.5, 1.0, 2.0)
OUTCOMES = ("is_success", "collision", "far_from_target", "time_limit")
SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# Categorical slots in their fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def wilson(successes: int, count: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval of a rate."""
    if not count:
        return math.nan, math.nan
    rate = successes / count
    centre = (rate + z**2 / (2 * count)) / (1 + z**2 / count)
    half = z * math.sqrt(rate * (1 - rate) / count + z**2 / (4 * count**2)) / (1 + z**2 / count)
    return centre - half, centre + half


def load(folder: str) -> tuple[dict, dict]:
    episodes = dict(np.load(os.path.join(folder, "episodes.npz")))
    with open(os.path.join(folder, "metrics.json")) as file:
        return episodes, json.load(file)


def outcome_row(episodes: dict, rows: np.ndarray) -> dict:
    count = int(rows.sum())
    successes = int((episodes["outcome"][rows] == 0).sum())
    low, high = wilson(successes, count)
    row = {"episodes": count, "success": successes / count if count else math.nan, "low": low, "high": high}
    for k, outcome in enumerate(OUTCOMES[1:], start=1):
        row[outcome] = float((episodes["outcome"][rows] == k).mean()) if count else math.nan
    success = rows & (episodes["outcome"] == 0)
    ratio = episodes["path"][success] / np.maximum(episodes["straight"][success], 1e-6)
    row["path_ratio"] = float(np.median(ratio)) if success.any() else math.nan
    row["time_s"] = float(np.median(episodes["duration"][success])) if success.any() else math.nan
    return row


def octaves(episodes: dict) -> list[tuple[str, np.ndarray]]:
    edges = (0.25, *SEVERITY_EDGES, 4.0)
    bins = np.digitize(episodes["severity"], SEVERITY_EDGES)
    return [(f"{low:g}–{high:g}×", bins == k) for k, (low, high) in enumerate(zip(edges[:-1], edges[1:]))]


def training_curves(folder: str) -> dict[str, np.ndarray]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    curves = {}
    for path in glob.glob(os.path.join(folder, "events.out.tfevents.*")):
        events = EventAccumulator(path, size_guidance={"scalars": 0})
        events.Reload()
        for tag in ("Loss/behavior", "Episode_Termination/is_success", "Episode_Termination/collision"):
            if tag in events.Tags()["scalars"]:
                values = events.Scalars(tag)
                curves[tag] = np.array([(value.step, value.value) for value in values])
    return curves


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=28)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Phase-4 driving policies.")
    parser.add_argument("--runs", nargs="+", required=True, help="label=eval_bev_policy.py folder, e.g. full=logs/...")
    parser.add_argument("--training", nargs="*", default=[], help="label=RSL-RL log folder, for the training curves.")
    parser.add_argument("--name", default="comparison")
    args = parser.parse_args()
    runs = dict(item.split("=", 1) for item in args.runs)
    folder = os.path.join(
        "logs", "bev_policy_comparison", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.name}"
    )
    os.makedirs(folder)

    summary, bev = {}, {}
    print("| Policy | Episodes | Success (95% interval) | Collision | Left the area | Time limit | Path ratio | Time |")
    print("|---|---|---|---|---|---|---|---|")
    for label, run in runs.items():
        episodes, metrics = load(run)
        pooled = outcome_row(episodes, np.ones(len(episodes["outcome"]), dtype=bool))
        summary[label] = {"pooled": pooled, "config": metrics["config"]}
        interval = f"({pooled['low']:.3f}–{pooled['high']:.3f})"
        print(f"| {label} | {pooled['episodes']} | {pooled['success']:.3f} {interval} "
              f"| {pooled['collision']:.3f} | {pooled['far_from_target']:.3f} | {pooled['time_limit']:.3f} "
              f"| {pooled['path_ratio']:.2f} | {pooled['time_s']:.1f} s |")
        if metrics["config"]["policy"] == "bev" and episodes["severity"].max() > 0:
            bev[label] = [(name, outcome_row(episodes, rows)) for name, rows in octaves(episodes)]
            summary[label]["by_severity"] = {name: row for name, row in bev[label]}

    if bev:
        names = [name for name, _ in next(iter(bev.values()))]
        print("\n| BEV policy | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for label, rows in bev.items():
            print(f"| {label} | " + " | ".join(
                f"{row['success']:.3f} ({row['low']:.2f}–{row['high']:.2f}), n={row['episodes']}" for _, row in rows
            ) + " |")
        figure, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
        for k, (label, rows) in enumerate(bev.items()):
            x = np.arange(len(rows)) + (k - (len(bev) - 1) / 2) * 0.08
            y = np.array([row["success"] for _, row in rows])
            error = np.array([[row["success"] - row["low"], row["high"] - row["success"]] for _, row in rows]).T
            ax.errorbar(x, y, yerr=error, color=SERIES[k % len(SERIES)], linewidth=2, marker="o", markersize=8,
                        capsize=3, label=label)
        for label, style_name in (("teacher", "--"), ("rgb", ":")):
            if label in summary:
                ax.axhline(summary[label]["pooled"]["success"], color=INK_SECONDARY, linewidth=1,
                           linestyle=style_name, label=f"{label} (not affected by drift)")
        ax.set_xticks(range(len(names)), names)
        style(ax, "Success rate by the episode's drift level, which the policies never see", "episode drift severity",
              "success rate")
        ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="lower left", fontsize=9)
        figure.tight_layout()
        figure.savefig(os.path.join(folder, "success_by_severity.png"), dpi=130, facecolor=SURFACE)
        plt.close(figure)

    if args.training:
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.0), facecolor=SURFACE)
        for k, item in enumerate(args.training):
            label, run = item.split("=", 1)
            curves = training_curves(run)
            color = SERIES[k % len(SERIES)]
            if "Loss/behavior" in curves:
                axes[0].plot(*curves["Loss/behavior"].T, color=color, linewidth=2, label=label)
            if "Episode_Termination/is_success" in curves:
                steps, values = curves["Episode_Termination/is_success"].T
                smooth = np.convolve(values, np.ones(10) / 10, mode="valid")
                axes[1].plot(steps[9:], smooth, color=color, linewidth=2, label=label)
        axes[0].set_yscale("log")
        style(axes[0], "Behaviour-cloning loss", "iteration", "loss")
        style(axes[1], "Successful share of finished episodes (10-iteration mean)", "iteration", "success")
        for ax in axes:
            ax.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=9)
        figure.tight_layout()
        figure.savefig(os.path.join(folder, "training.png"), dpi=130, facecolor=SURFACE)
        plt.close(figure)

    with open(os.path.join(folder, "comparison.json"), "w") as file:
        json.dump(summary, file, indent=2, default=str)
    print(f"Wrote {folder}")


if __name__ == "__main__":
    main()
