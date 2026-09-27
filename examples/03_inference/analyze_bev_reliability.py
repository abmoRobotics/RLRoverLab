"""Fit and compare reliability models of the persistent BEV offline, on the cells dumped by eval_bev_memory.py.

Every model predicts the expected elevation error of remembered cells, through a monotone map fitted on training runs,
and is scored on held-out runs over the whole range of drift levels (Phase 3.2 and 3.4):

- "distance": the distance travelled since the cell was observed, the Phase-2 provenance;
- "U_pose": the nominal pose uncertainty of the cell, with the lever arm of heading errors along the actual path;
- "pose effect, fixed level": what that uncertainty does to the elevation given the local slope, at an assumed level;
- "pose effect, measured level": the same at the drift level the rover measures itself (DriftMonitor);
- "pose effect, oracle level": the same at the true drift level, which the rover never has;
- "oracle displacement": the actual pose error of the cell, the best any pose model could do.

    python examples/03_inference/analyze_bev_reliability.py \
        --train logs/bev_memory/<run> ... --test logs/bev_memory/<run> ...

Needs numpy, scipy and matplotlib, e.g. /isaac-sim/python.sh in the container. Writes reliability.json and figures to
logs/bev_reliability/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import optimize, stats  # noqa: E402

from rover_envs.envs.navigation.learning.bev.drift import NOMINAL_RATES  # noqa: E402

SEVERITY_EDGES = (0.5, 1.0, 2.0)
DISTANCE_EDGES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# Errors above this (m) count as failures for the AUROC.
LARGE_ERROR = 0.05
# Calibration is also reported on cells remembered over at least this distance (m), where drift matters.
MEMORY_LENGTH = 1.0

SURFACE, INK, INK_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
# Categorical slots in their fixed order, and an ordinal blue ramp for the severity octaves, light to dark.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SEVERITY_COLORS = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
# Models in the charts: the Phase-2 proxy, the best fixed model, the rover's adaptive one and its oracle bound.
PLOTTED = ("distance", "pose effect, fixed level", "pose effect, measured level", "pose effect, oracle level")


def rotate(xy: np.ndarray, angle: np.ndarray) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.stack((cos * xy[:, 0] - sin * xy[:, 1], sin * xy[:, 0] + cos * xy[:, 1]), axis=1)


def path_mean(values: np.ndarray, length: np.ndarray, step: np.ndarray, env: np.ndarray, written: np.ndarray):
    """Distance-weighted mean [N] of per-step ``values`` [T, B] over each cell's path since it was observed; the value
    at ``step`` where the rover has not moved, NaN where no value is known."""
    known = np.isfinite(values)
    moments = np.cumsum(np.stack((length * known, length * np.nan_to_num(values)), axis=2), axis=0)
    path = moments[step, env] - moments[written, env]
    return np.where(path[:, 0] > 1e-6, path[:, 1] / np.maximum(path[:, 0], 1e-9), values[step, env])


def load_run(folder: str) -> dict[str, np.ndarray]:
    """The remembered cells of one run, with what the models may use and the ground truth that scores them."""
    data = np.load(os.path.join(folder, "samples.npz"))
    dt, resolution = float(data["dt"]), float(data["resolution"])
    x_range, y_range = data["x_range"], data["y_range"]
    ny = round((y_range[1] - y_range[0]) / resolution)
    keep = (data["status"] == 2) & np.isfinite(data["target"]) & np.isfinite(data["elevation"])
    env, step = data["env"][keep], data["step"][keep]
    written = step - np.rint(data["age"][keep] / dt).astype(np.int64)
    i, j = np.divmod(data["cell"][keep], ny)
    ego = np.stack((x_range[0] + (i + 0.5) * resolution, y_range[0] + (j + 0.5) * resolution), axis=1)

    pose, true_pose = data["pose"].astype(np.float64), data["true_pose"].astype(np.float64)
    # Path moments of the memory's trajectory, counted like its odometer: nothing across a reset.
    length = np.linalg.norm(np.diff(pose[..., :2], axis=0), axis=2) * data["fresh"][1:]
    length = np.concatenate((np.zeros((1, pose.shape[1])), length))
    xy = pose[..., :2]
    moments = np.cumsum(np.concatenate(
        (length[..., None], length[..., None] * xy, (length * (xy**2).sum(-1))[..., None]), axis=2
    ), axis=0)
    delta = moments[step, env] - moments[written, env]
    travelled = delta[:, 0]
    if np.abs(travelled - data["distance"][keep]).max() > 0.02:
        raise ValueError(f"{folder}: the path since observation does not match the memory's distance.")

    # The cell's position in the memory's world frame now, and the heading lever arm sum(ds |c - p|^2) since then.
    now = pose[step, env]
    cell = now[:, :2] + rotate(ego, now[:, 3])
    lever = (cell**2).sum(1) * travelled - 2 * (cell * delta[:, 1:3]).sum(1) + delta[:, 3]
    # Oracle: where the stored terrain truly is, against where the ego cell truly is now, and the height offset.
    then, true_then, true_now = pose[written, env], true_pose[written, env], true_pose[step, env]
    stored = true_then[:, :2] + rotate(rotate(cell - then[:, :2], -then[:, 3]), true_then[:, 3])
    displacement = np.linalg.norm(stored - (true_now[:, :2] + rotate(ego, true_now[:, 3])), axis=1)
    height_offset = (then[:, 2] - true_then[:, 2]) - (now[:, 2] - true_now[:, 2])

    probability = data["rock_probability"][keep].astype(np.float64)
    true_rock = data["target_class"][keep] == int(data["rock"])
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    measured = data["drift_level"].astype(np.float64)
    return {
        "env": env,
        "distance": travelled,
        "slope": data["relief"][keep].astype(np.float64) / resolution,
        "lever": np.maximum(lever, 0.0),
        "severity": data["severity"][step, env].astype(np.float64),
        # The true level, and the rover's own estimates: now, and as it had them along the path since observation.
        "oracle_level": np.sqrt(path_mean(data["level"].astype(np.float64) ** 2, length, step, env, written)),
        "measured_now": measured[step, env],
        "measured_path": np.sqrt(path_mean(measured**2, length, step, env, written)),
        "displacement": displacement,
        "height_offset": height_offset,
        "error": np.abs(data["elevation"][keep] - data["target"][keep]).astype(np.float64),
        "confused": (probability > 0.5) != true_rock,
        "entropy": -(p * np.log(p) + (1 - p) * np.log(1 - p)),
    }


def concatenate(runs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([run[key] for run in runs]) for key in runs[0]}


def pose_variance(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal (trace) and vertical displacement variance of each cell under the nominal model at 1x."""
    translation, heading, height = NOMINAL_RATES
    return translation * data["distance"] + heading * data["lever"], height * data["distance"]


def pose_effect(data: dict) -> np.ndarray:
    """Elevation error variance the nominal 1x pose error causes: a horizontal displacement changes the elevation by
    the slope times its component along the slope, which has half the variance of the displacement's trace."""
    horizontal, vertical = pose_variance(data)
    return data["slope"] ** 2 * horizontal / 2 + vertical


def true_pose_effect(data: dict) -> np.ndarray:
    """The same with the cell's actual displacement and height offset."""
    return data["slope"] ** 2 * data["displacement"] ** 2 / 2 + data["height_offset"] ** 2


def fit_monotone(score: np.ndarray, error: np.ndarray, bins: int = 200):
    """A non-decreasing map from a score to the expected error: isotonic regression over quantile bins of the score."""
    edges = np.unique(np.quantile(score, np.linspace(0.0, 1.0, bins + 1)))
    index = np.clip(np.searchsorted(edges, score, side="right") - 1, 0, len(edges) - 2)
    counts = np.bincount(index, minlength=len(edges) - 1).astype(np.float64)
    means = np.bincount(index, weights=error, minlength=len(edges) - 1) / np.maximum(counts, 1.0)
    used = counts > 0
    fitted = np.empty_like(means)
    fitted[used] = optimize.isotonic_regression(means[used], weights=counts[used]).x
    fitted[~used] = np.interp(np.flatnonzero(~used), np.flatnonzero(used), fitted[used])
    return lambda s: fitted[np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)]


def build_models(train: dict) -> dict[str, callable]:
    """Models that map the cells of a run to their expected elevation error (m), fitted on ``train``."""
    scores = {
        "distance": lambda data: data["distance"],
        "U_pose": lambda data: pose_variance(data)[0],
        "pose effect, fixed level": pose_effect,
    }
    # Where the rover has no estimate yet, e.g. right after a reset, the training runs' median stands in.
    for name, key in (("pose effect, measured level", "measured_path"), ("measured level, now", "measured_now")):
        prior = float(np.nanmedian(train[key]))
        scores[name] = lambda data, key=key, prior=prior: np.where(np.isfinite(data[key]), data[key], prior) ** 2 * (
            pose_effect(data)
        )
    scores["pose effect, oracle level"] = lambda data: data["oracle_level"] ** 2 * pose_effect(data)
    scores["oracle displacement"] = true_pose_effect
    models = {}
    for name, score in scores.items():
        monotone = fit_monotone(score(train), train["error"])
        models[name] = lambda data, monotone=monotone, score=score: monotone(score(data))
    return models


def risk_coverage(predicted: np.ndarray, error: np.ndarray) -> np.ndarray:
    """Mean error of the kept cells when keeping the most reliable ones first, for each number of kept cells."""
    order = np.argsort(predicted, kind="stable")
    return np.cumsum(error[order]) / np.arange(1, len(error) + 1)


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    if positive.all() or not positive.any():
        return math.nan
    ranks = stats.rankdata(score)
    n_positive = positive.sum()
    return float((ranks[positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * (~positive).sum()))


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.nan


def evaluate(data: dict, predicted: np.ndarray) -> dict:
    error = data["error"]
    oracle, curve = np.sort(error).cumsum() / np.arange(1, len(error) + 1), risk_coverage(predicted, error)
    subsample = np.random.default_rng(0).choice(len(error), size=min(len(error), 200_000), replace=False)
    result = {
        "spearman": float(stats.spearmanr(predicted[subsample], error[subsample]).statistic),
        "aurc_cm": 100 * float(curve.mean()),
        "excess_aurc_cm": 100 * float(curve.mean() - oracle.mean()),
        "auroc_large_error": auroc(predicted, error > LARGE_ERROR),
        "calibration": ratio(error.sum(), predicted.sum()),
        "by_severity": [],
    }
    severity_edges = (0.0, *SEVERITY_EDGES, math.inf)
    for low, high in zip(severity_edges[:-1], severity_edges[1:]):
        cells = (data["severity"] >= low) & (data["severity"] < high)
        long = cells & (data["distance"] >= MEMORY_LENGTH)
        result["by_severity"].append({
            "severity": [low, high],
            "cells": int(cells.sum()),
            "observed_mae_cm": 100 * ratio(error[cells].sum(), cells.sum()),
            "calibration": ratio(error[cells].sum(), predicted[cells].sum()),
            "observed_mae_cm_remembered_1m": 100 * ratio(error[long].sum(), long.sum()),
            "calibration_remembered_1m": ratio(error[long].sum(), predicted[long].sum()),
        })
    return result


def monitor_validation(folders: list[str]) -> dict:
    """How the drift monitor's level follows the true drift level, over the steps of held-out runs."""
    measured, level, severity = [], [], []
    for folder in folders:
        data = np.load(os.path.join(folder, "samples.npz"))
        use = data["fresh"] & np.isfinite(data["drift_level"]) & (data["drift_weight"] > 100)
        measured.append(data["drift_level"][use])
        level.append(data["level"][use])
        severity.append(data["severity"][use])
    measured, level, severity = (np.concatenate(values).astype(np.float64) for values in (measured, level, severity))
    positive = measured > 0
    result = {
        "steps": len(measured),
        "spearman": float(stats.spearmanr(measured, level).statistic),
        "zero_share": float(1 - positive.mean()),
        # Slope of log(measured) on log(true level): 1 if the measure is proportional to the level.
        "log_slope": float(np.polyfit(np.log(level[positive]), np.log(measured[positive]), 1)[0]),
        "by_severity": [],
    }
    edges = (0.0, *SEVERITY_EDGES, math.inf)
    for low, high in zip(edges[:-1], edges[1:]):
        rows = (severity >= low) & (severity < high)
        quartiles = np.quantile(measured[rows], [0.25, 0.5, 0.75]).tolist() if rows.any() else [math.nan] * 3
        result["by_severity"].append({"severity": [low, high], "steps": int(rows.sum()), "quartiles": quartiles})
    result["arrays"] = (measured, level)
    return result


def registration_validation(folders: list[str]) -> dict:
    """Measured against true corrections of the keyframe registrations of held-out runs."""
    import torch

    from rover_envs.envs.navigation.learning.bev.registration import relative_correction

    measured, expected = [], []
    for folder in folders:
        data = np.load(os.path.join(folder, "samples.npz"))
        if "registration_env" not in data.files:
            continue
        ok = data["registration_registered"]
        env, step, key = (data[f"registration_{name}"][ok] for name in ("env", "step", "key_step"))
        pose, true = torch.tensor(data["pose"]), torch.tensor(data["true_pose"])
        expected.append(relative_correction(true[key, env], true[step, env], pose[key, env], pose[step, env]).numpy())
        measured.append(data["registration_correction"][ok])
    if not measured:
        return {}
    measured, expected = np.concatenate(measured), np.concatenate(expected)
    result = {"registrations": len(measured)}
    for k, (name, scale) in enumerate((("x_cm", 100), ("y_cm", 100), ("yaw_deg", 180 / math.pi), ("z_cm", 100))):
        m, t = measured[:, k] * scale, expected[:, k] * scale
        result[name] = {
            "true_rms": float(np.sqrt(np.mean(t**2))),
            "error_rms": float(np.sqrt(np.mean((m - t) ** 2))),
            "correlation": float(np.corrcoef(m, t)[0, 1]),
        }
    return result


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=28)
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_SECONDARY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def save(figure, path: str) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(figure)


def severity_labels() -> list[str]:
    edges = (0.25, *SEVERITY_EDGES, 4.0)
    return [f"{low:g}–{high:g}×" for low, high in zip(edges[:-1], edges[1:])]


def plot_error_by_severity(data: dict, path: str) -> None:
    """Observed error of remembered cells against the distance since observation, one line per severity octave."""
    distance_edges = (0.0, *DISTANCE_EDGES, math.inf)
    bins = zip(distance_edges[:-1], distance_edges[1:])
    labels = [f"{a:g}–{b:g}" if math.isfinite(b) else f"≥{a:g}" for a, b in bins]
    distance_bin = np.digitize(data["distance"], DISTANCE_EDGES)
    severity_bin = np.digitize(data["severity"], SEVERITY_EDGES)
    figure, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    for k, (label, color) in enumerate(zip(severity_labels(), SEVERITY_COLORS)):
        cells = severity_bin == k
        counts = np.bincount(distance_bin[cells], minlength=len(labels))
        sums = np.bincount(distance_bin[cells], weights=data["error"][cells], minlength=len(labels))
        values = np.where(counts >= 200, 100 * sums / np.maximum(counts, 1), np.nan)
        ax.plot(range(len(labels)), values, color=color, linewidth=2, marker="o", markersize=8, label=label)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(bottom=0)
    style(ax, "Held-out runs: remembered-cell error fans out with the unknown drift level",
          "distance travelled since the cell was observed [m]", "elevation MAE [cm]")
    ax.legend(title="run severity", frameon=False, labelcolor=INK_SECONDARY, loc="upper left", fontsize=9,
              title_fontsize=9)
    save(figure, path)


def plot_calibration(results: dict, path: str) -> None:
    """Observed over predicted error of cells remembered over at least MEMORY_LENGTH, by severity octave."""
    figure, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    width = 0.8 / len(PLOTTED)
    for k, (name, color) in enumerate(zip(PLOTTED, SERIES)):
        values = [row["calibration_remembered_1m"] for row in results[name]["by_severity"]]
        positions = np.arange(len(values)) + (k - (len(PLOTTED) - 1) / 2) * width
        ax.bar(positions, values, width=width - 0.02, color=color, label=name)
    ax.axhline(1.0, color=INK_SECONDARY, linewidth=1)
    ax.axhspan(1 / 1.5, 1.5, color=GRID, alpha=0.5, linewidth=0, zorder=0)
    ax.set_xticks(range(len(SEVERITY_COLORS)), severity_labels())
    style(ax, f"Calibration on cells remembered over ≥ {MEMORY_LENGTH:g} m (1 = right; band: 0.67–1.5)",
          "run severity", "observed / predicted error")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper left", fontsize=9)
    save(figure, path)


def plot_risk_coverage(data: dict, predictions: dict, path: str) -> None:
    """Risk-coverage curves from 1% coverage on; below it, the mean of a handful of cells is noise."""
    error = data["error"]
    coverage = np.arange(1, len(error) + 1) / len(error)
    shown = slice(len(error) // 100, None)
    figure, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    for name, color in zip(PLOTTED, SERIES):
        risk = 100 * risk_coverage(predictions[name], error)
        ax.plot(coverage[shown], risk[shown], color=color, linewidth=2, label=name)
    perfect = 100 * np.sort(error).cumsum() / np.arange(1, len(error) + 1)
    ax.plot(coverage[shown], perfect[shown], color=INK_SECONDARY, linewidth=1, linestyle="--", label="perfect ranking")
    ax.set_ylim(0, 1.5 * 100 * error.mean())
    style(ax, "Held-out runs: error of the kept cells when keeping the most reliable first",
          "share of remembered cells kept", "elevation MAE of kept cells [cm]")
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper left", fontsize=9)
    save(figure, path)


def plot_monitor(validation: dict, path: str) -> None:
    """The measured level against the true drift level: medians and interquartile ranges in log-spaced bins."""
    measured, level = validation["arrays"]
    edges = np.geomspace(max(level.min(), 0.05), level.max(), 13)
    centres, median, low, high = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        rows = (level >= a) & (level < b)
        if rows.sum() < 50:
            continue
        centres.append(math.sqrt(a * b))
        median.append(np.median(measured[rows]))
        low.append(np.quantile(measured[rows], 0.25))
        high.append(np.quantile(measured[rows], 0.75))
    figure, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    ax.fill_between(centres, low, high, color=SERIES[0], alpha=0.2, linewidth=0)
    ax.plot(centres, median, color=SERIES[0], linewidth=2, marker="o", markersize=8, label="median, interquartile band")
    ax.set_xscale("log")
    ax.set_xlim(centres[0] / 1.2, centres[-1] * 1.2)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    ax.set_ylim(bottom=0)
    style(ax, "Held-out runs: the drift level the rover measures follows the true level it never sees",
          "true drift level at the rover (severity × location factor)", "measured level")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper left", fontsize=9)
    save(figure, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare reliability models of the persistent BEV offline.")
    parser.add_argument("--train", nargs="+", required=True, help="eval_bev_memory.py run folders to fit on.")
    parser.add_argument("--test", nargs="+", required=True, help="Held-out run folders to score on.")
    parser.add_argument("--name", default="reliability")
    args = parser.parse_args()

    train = concatenate([load_run(folder) for folder in args.train])
    tests = {folder: load_run(folder) for folder in args.test}
    test = concatenate(list(tests.values()))
    models = build_models(train)
    predictions = {name: model(test) for name, model in models.items()}
    results = {name: evaluate(test, predicted) for name, predicted in predictions.items()}
    per_run = {
        os.path.basename(folder.rstrip("/")): {name: evaluate(data, model(data)) for name, model in models.items()}
        for folder, data in tests.items()
    }
    rock = {signal: auroc(test[signal], test["confused"]) for signal in ("entropy", "distance", "slope", "lever")}
    monitor = monitor_validation(args.test)
    folder = os.path.join("logs", "bev_reliability", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.name}")
    os.makedirs(folder)
    summary = {
        "train": args.train,
        "test": args.test,
        "cells": {"train": len(train["error"]), "test": len(test["error"])},
        "pooled": results,
        "per_run": per_run,
        "rock_confusion_auroc": rock,
        "drift_monitor": {key: value for key, value in monitor.items() if key != "arrays"},
        "registration": registration_validation(args.test),
    }
    with open(os.path.join(folder, "reliability.json"), "w") as file:
        json.dump(summary, file, indent=2)
    plot_error_by_severity(test, os.path.join(folder, "error_by_severity.png"))
    plot_calibration(results, os.path.join(folder, "calibration_by_severity.png"))
    plot_risk_coverage(test, predictions, os.path.join(folder, "risk_coverage.png"))
    plot_monitor(monitor, os.path.join(folder, "drift_monitor.png"))

    print(f"{len(train['error'])} training and {len(test['error'])} test cells.")
    print(f"Calibration (observed / predicted) by run severity is on cells remembered over >= {MEMORY_LENGTH:g} m.")
    header = "".join(f"{label:>11s}" for label in severity_labels())
    print(f"{'model':28s} {'spearman':>8s} {'AURC':>6s} {'excess':>6s} {'AUROC':>6s} {'calib':>6s} |{header}")
    for name, result in results.items():
        by_severity = "".join(f"{row['calibration_remembered_1m']:11.2f}" for row in result["by_severity"])
        print(f"{name:28s} {result['spearman']:8.3f} {result['aurc_cm']:6.2f} {result['excess_aurc_cm']:6.2f}"
              f" {result['auroc_large_error']:6.3f} {result['calibration']:6.2f} |{by_severity}")
    observed = "".join(f"{row['observed_mae_cm_remembered_1m']:8.2f} cm" for row in results["distance"]["by_severity"])
    print(f"{'observed MAE':28s} {' ' * 42}|{observed}")
    print("rock confusion AUROC:", {k: round(v, 3) for k, v in rock.items()})
    quartiles = [[round(q, 2) for q in row["quartiles"]] for row in monitor["by_severity"]]
    print(f"drift monitor: Spearman with the true level {monitor['spearman']:.3f}, log-log slope "
          f"{monitor['log_slope']:.2f}, zero share {monitor['zero_share']:.2f}; quartiles by severity {quartiles}")
    print("registration:", json.dumps(summary["registration"]))
    print(f"Wrote {folder}")


if __name__ == "__main__":
    main()
