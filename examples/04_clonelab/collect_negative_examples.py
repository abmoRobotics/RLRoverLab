#!/usr/bin/env python3
"""Collect isolated near-rock negative examples for RC-IQL."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser("Collect near-rock negative examples for offline RC-IQL.")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--teacher_name", required=True)
parser.add_argument("--task", default="AAURoverEnvRGBDRawHD720-v0")
parser.add_argument("--agent", default="PPO")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--terrain", default=None)
parser.add_argument("--terrain-seed", type=int, default=None)
parser.add_argument("--keep-terrain", action="store_true", default=False)
parser.add_argument("--max_steps", type=int, default=150000)
parser.add_argument("--torch_device", default=None)
parser.add_argument("--stochastic", action="store_true", default=False)
parser.add_argument("--no_clip_actions", action="store_true", default=False)
parser.add_argument("--dataset_dir", default="logs/datasets_speedcap_negatives")
parser.add_argument("--candidate_dataset_name", default="speedcap_negative_candidates_hd720")
parser.add_argument("--final_dataset_name", default="speedcap_near_rock_negatives_hd720_50k")
parser.add_argument("--target_negative_transitions", type=int, default=50000)
parser.add_argument("--negative_profile", default="mixed_action")
parser.add_argument(
    "--mode_weights_json",
    default=None,
    help="Optional JSON object overriding profile mode weights, e.g. '{\"fast_near_rock\": 1.0}'.",
)
parser.add_argument(
    "--speed_filter",
    choices=["auto", "none", "moderate", "conservative"],
    default="auto",
    help="Teacher speed filter to apply before negative intervention.",
)
parser.add_argument("--include_timeouts", action="store_true", default=False)
parser.add_argument("--allow_successes", action="store_true", default=False)
parser.add_argument("--max_failure_min_clearance_m", type=float, default=2.5)
parser.add_argument("--manifest_out", default=None)
parser.add_argument("--positive_dataset", action="append", default=[])
parser.add_argument("--negative_fraction", type=float, default=0.143)
parser.add_argument("--d_ref", type=float, default=5.0)
parser.add_argument("--clearance_exponent", type=float, default=2.5)
parser.add_argument("--summary_out", default=None)
parser.add_argument("--log_interval", type=int, default=100)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
checkpoint_path = Path(args_cli.checkpoint)
if not checkpoint_path.exists():
    print(f"[ERROR] Teacher checkpoint does not exist: {checkpoint_path}", file=sys.stderr, flush=True)
    raise SystemExit(2)
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.managers import DatasetExportMode  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from skrl.utils import set_seed  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

import rover_envs  # noqa: E402, F401
import rover_envs.envs.navigation.robots  # noqa: E402, F401
from rover_envs.integrations.clonelab.negative_dataset_tools import (  # noqa: E402
    filter_negative_dataset,
    validate_negative_dataset,
    write_multishard_manifest,
)
from rover_envs.integrations.clonelab.negative_interventions import (  # noqa: E402
    NegativeInterventionController,
    blind_height_scan_observations,
    make_negative_intervention_profile,
    set_negative_intervention_state,
)
from rover_envs.integrations.clonelab.negative_recorders import (  # noqa: E402
    NegativeCompressedRGBDRecorderManagerCfg,
)
from rover_envs.integrations.clonelab.risk_evaluation_metrics import (  # noqa: E402
    current_clearance,
    termination_masks,
)
from rover_envs.integrations.clonelab.speed_risk_filter import (  # noqa: E402
    SPEED_RISK_FILTER_PROFILES,
    SpeedRiskFilterStats,
    apply_speed_risk_filter,
    infer_speed_risk_filter_profile,
)
from rover_envs.learning.agents import create_agent  # noqa: E402
from rover_envs.utils.config import parse_skrl_cfg  # noqa: E402
from rover_envs.utils.terrain_utils import cleanup_terrain, handle_terrain_config  # noqa: E402


simulation_app = app_launcher.app


def _done_tensor(terminated, truncated, device: str) -> torch.Tensor:
    terminated = torch.as_tensor(terminated, device=device).bool()
    truncated = torch.as_tensor(truncated, device=device).bool()
    return torch.logical_or(terminated, truncated).reshape(-1)


def _teacher_actions(agent, observations, *, stochastic: bool, max_steps: int) -> torch.Tensor:
    if stochastic:
        return agent.act(observations, timestep=0, timesteps=max_steps)[0]

    policy = agent.models["policy"]
    means, _ = policy.compute({"observations": observations}, role="policy")
    return means


def _speed_filter_profile():
    if args_cli.speed_filter == "none":
        return None
    if args_cli.speed_filter in SPEED_RISK_FILTER_PROFILES:
        return SPEED_RISK_FILTER_PROFILES[args_cli.speed_filter]
    return infer_speed_risk_filter_profile(args_cli.teacher_name, args_cli.checkpoint, args_cli.task)


def _write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    if args_cli.max_steps <= 0:
        raise ValueError("--max_steps must be positive.")
    if args_cli.target_negative_transitions <= 0:
        raise ValueError("--target_negative_transitions must be positive.")

    seed = args_cli.seed if args_cli.seed is not None else random.randint(0, 100000000)
    device = args_cli.torch_device or ("cpu" if args_cli.cpu else "cuda:0")
    set_seed(seed)
    mode_weights = json.loads(args_cli.mode_weights_json) if args_cli.mode_weights_json else None
    profile = make_negative_intervention_profile(args_cli.negative_profile, mode_weights=mode_weights)
    speed_filter_profile = _speed_filter_profile()
    speed_filter_stats = SpeedRiskFilterStats(speed_filter_profile)

    dataset_dir = Path(args_cli.dataset_dir)
    candidate_path = dataset_dir / f"{args_cli.candidate_dataset_name}.hdf5"
    final_path = dataset_dir / f"{args_cli.final_dataset_name}.hdf5"
    if candidate_path == final_path:
        raise ValueError("--candidate_dataset_name and --final_dataset_name must be different.")

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.recorders = NegativeCompressedRGBDRecorderManagerCfg()
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    env_cfg.recorders.dataset_export_dir_path = str(dataset_dir)
    env_cfg.recorders.dataset_filename = f"{args_cli.candidate_dataset_name}.hdf5"
    env_cfg.recorders.export_in_close = True

    terrain_name, terrain_cleanup_path = handle_terrain_config(
        terrain_arg=args_cli.terrain,
        terrain_seed=args_cli.terrain_seed,
        keep_terrain=args_cli.keep_terrain,
    )
    if terrain_name is not None:
        env_cfg.scene.set_terrain(terrain_name)

    raw_env = None
    env = None
    summary: dict[str, object] = {
        "seed": seed,
        "task": args_cli.task,
        "teacher_name": args_cli.teacher_name,
        "checkpoint": args_cli.checkpoint,
        "candidate_path": str(candidate_path),
        "final_path": str(final_path),
        "negative_profile": profile.name,
    }
    try:
        raw_env = gym.make(args_cli.task, cfg=env_cfg)
        env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")

        experiment_cfg_file = gym.spec(args_cli.task).kwargs["skrl_cfgs"][args_cli.agent.upper()]
        experiment_cfg = parse_skrl_cfg(experiment_cfg_file)
        experiment_cfg["agent"]["rollouts"] = 1
        experiment_cfg["agent"]["experiment"]["wandb"] = False
        print("[INFO] Creating teacher agent.", flush=True)
        agent = create_agent(args_cli.agent, env, experiment_cfg)
        print(f"[INFO] Loading teacher checkpoint: {args_cli.checkpoint}", flush=True)
        try:
            agent.load(args_cli.checkpoint)
        except BaseException as exc:
            print(f"[ERROR] Failed to load teacher checkpoint: {exc!r}", flush=True)
            raise
        print("[INFO] Loaded teacher checkpoint.", flush=True)
        for model in agent.models.values():
            model.eval()

        observations, _ = env.reset()
        num_envs = int(env.num_envs)
        action_dim = int(env.action_space.shape[-1])
        intervention = NegativeInterventionController(
            profile,
            num_envs=num_envs,
            action_dim=action_dim,
            device=device,
            seed=seed,
        )

        termination_counts: dict[str, int] = {}
        completed_episodes = 0
        total_reward = 0.0
        total_rows = 0
        simulation_start = time.perf_counter()

        if speed_filter_profile is None:
            print("[INFO] Collecting negatives without teacher speed filter.", flush=True)
        else:
            print(
                "[INFO] Applying teacher speed filter before negative intervention: "
                f"{speed_filter_profile.name} (v_near={speed_filter_profile.v_near})",
                flush=True,
            )
        print(f"[INFO] Negative profile: {profile.name}", flush=True)
        print(f"[INFO] Candidate dataset: {candidate_path}", flush=True)

        progress = tqdm(
            range(args_cli.max_steps),
            total=args_cli.max_steps,
            desc="Collecting negative candidates",
            unit="step",
            dynamic_ncols=True,
            mininterval=1.0,
            disable=args_cli.log_interval == 0,
        )
        with torch.inference_mode():
            for step in progress:
                decision_clearance = current_clearance(raw_env)
                policy_observations = observations
                if intervention.has_heightmap_blind_mode:
                    gate = intervention.intervention_gate(decision_clearance)
                    policy_observations = blind_height_scan_observations(
                        observations,
                        mode_ids=intervention.mode_ids,
                        gate=gate,
                    )

                actions = _teacher_actions(
                    agent,
                    policy_observations,
                    stochastic=args_cli.stochastic,
                    max_steps=args_cli.max_steps,
                )
                if not torch.isfinite(actions).all():
                    raise RuntimeError("Teacher produced non-finite actions.")
                if not args_cli.no_clip_actions:
                    actions = actions.clamp(-1.0, 1.0)

                if speed_filter_profile is not None:
                    raw_actions = actions
                    actions = apply_speed_risk_filter(actions, decision_clearance, speed_filter_profile)
                    speed_filter_stats.update(raw_actions, actions, decision_clearance)

                actions, metadata = intervention.intervene(actions, decision_clearance)
                set_negative_intervention_state(raw_env, metadata)

                observations, rewards, terminated, truncated, _ = env.step(actions)
                done = _done_tensor(terminated, truncated, device)
                completed_episodes += int(done.sum().item())
                intervention.reset_done(done)

                rewards = torch.as_tensor(rewards, device=device, dtype=torch.float32).reshape(-1)
                total_reward += float(rewards.sum().item())
                total_rows += int(rewards.numel())
                for name, mask in termination_masks(raw_env, device).items():
                    termination_counts[name] = termination_counts.get(name, 0) + int(mask.sum().item())

                should_log = args_cli.log_interval and (
                    (step + 1) % args_cli.log_interval == 0 or step + 1 == args_cli.max_steps
                )
                if should_log:
                    progress.set_postfix(
                        episodes=completed_episodes,
                        collisions=termination_counts.get("collision", 0),
                        timeouts=termination_counts.get("time_limit", 0),
                        successes=termination_counts.get("is_success", 0),
                        active=f"{intervention.summary()['active_intervention_fraction'] or 0.0:.3f}",
                        refresh=True,
                    )

        summary.update(
            {
                "simulation_steps": args_cli.max_steps,
                "num_envs": num_envs,
                "candidate_transition_rows_requested": args_cli.max_steps * num_envs,
                "completed_episodes": completed_episodes,
                "termination_counts": termination_counts,
                "mean_step_reward": total_reward / total_rows if total_rows else None,
                "elapsed_seconds": time.perf_counter() - simulation_start,
                "speed_filter": speed_filter_stats.summary(),
                "negative_intervention": intervention.summary(),
            }
        )
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        elif raw_env is not None:
            with contextlib.suppress(Exception):
                raw_env.close()
        cleanup_terrain(terrain_cleanup_path if "terrain_cleanup_path" in locals() else None)

    candidate_validation = validate_negative_dataset(candidate_path, require_timeout_penalty=False)
    summary["candidate_validation"] = candidate_validation
    if not candidate_validation["valid"]:
        raise RuntimeError(f"Candidate dataset failed validation: {candidate_validation['errors']}")

    filter_summary = filter_negative_dataset(
        candidate_path,
        final_path,
        target_transitions=args_cli.target_negative_transitions,
        failures_only=not args_cli.allow_successes,
        include_timeouts=args_cli.include_timeouts,
        max_min_clearance_m=args_cli.max_failure_min_clearance_m,
        seed=seed,
        metadata={
            "collector": "examples/04_clonelab/collect_negative_examples.py",
            "teacher_name": args_cli.teacher_name,
            "negative_profile": profile.name,
            "speed_filter": None if speed_filter_profile is None else speed_filter_profile.name,
        },
    )
    summary["filter"] = filter_summary

    if args_cli.manifest_out:
        manifest = write_multishard_manifest(
            args_cli.manifest_out,
            positive_paths=args_cli.positive_dataset,
            negative_path=str(final_path),
            negative_transition_fraction=args_cli.negative_fraction,
            d_ref=args_cli.d_ref,
            clearance_exponent=args_cli.clearance_exponent,
        )
        summary["manifest"] = manifest

    summary_path = (
        Path(args_cli.summary_out)
        if args_cli.summary_out
        else dataset_dir / f"{args_cli.final_dataset_name}_summary.json"
    )
    _write_json(summary_path, summary)
    print(f"[INFO] Wrote negative collection summary: {summary_path}", flush=True)
    print(f"[INFO] Final negative shard: {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
