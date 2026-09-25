#!/usr/bin/env python3
"""Time a trained navigation policy on the Newton MPM soil task.

Builds the task and skrl agent like ``examples/03_inference/eval.py``, runs the policy
deterministically, and reports throughput. It also checks that the rovers drive on the soil:
how far the particles moved, how high the rovers ride above the terrain, and how many targets
were reached. Rigid-terrain tasks such as ``AAURoverEnvSimple-v0`` also run, as a reference.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

# Match eval.py: keep BLAS single-threaded before Isaac Lab loads NumPy's OpenBLAS runtime.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AAURoverEnvParticles-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--terrain", help="Registered terrain name; the soil task defaults to 'debug'")
parser.add_argument("--warmup", type=int, default=10, help="Untimed policy steps before timing")
parser.add_argument("--steps", type=int, default=50, help="Timed policy steps")
parser.add_argument("--agent", default="PPO")
parser.add_argument("--checkpoint", help="Policy checkpoint; defaults to the task's best model")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--voxel_size", type=float, help="Override the soil's MPM voxel size [m]")
parser.add_argument("--particles_per_cell", type=int, help="Override the soil particles per voxel edge")
parser.add_argument("--depth", type=float, help="Override the soil depth [m]")
parser.add_argument("--size", type=float, nargs=2, metavar=("X", "Y"), help="Override the soil extent [m]")
parser.add_argument("--physics_dt", type=float, help="Override the physics step [s]; decimation keeps the 0.2 s policy step")
parser.add_argument("--rover_substeps", type=int, help="Override the MJWarp substeps per physics step")
parser.add_argument("--mpm_iterations", type=int, help="Override the MPM rheology solver iteration limit")
parser.add_argument("--output", help="Write one JSON result to this path")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_newton.physics import NewtonManager  # noqa: E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from skrl.trainers.torch import SequentialTrainer  # noqa: E402
from skrl.utils import set_seed  # noqa: E402

import rover_envs.envs.navigation.robots  # noqa: E402, F401
from rover_envs.envs.navigation.rover_env_particles_cfg import POLICY_DT  # noqa: E402
from rover_envs.learning.agents import create_agent  # noqa: E402
from rover_envs.utils.config import parse_skrl_cfg  # noqa: E402


def gpu_used_gb() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1e9


def gpu_utilization(samples: int = 5) -> float | None:
    """Mean device-wide GPU utilization [%]. Before the run starts, it reveals other GPU workloads."""
    readings = []
    for _ in range(samples):
        try:
            query = ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "--id=0"]
            readings.append(float(subprocess.check_output(query, text=True).split()[0]))
        except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
            return None
        time.sleep(0.2)
    return sum(readings) / len(readings)


def main() -> None:
    gpu_before = gpu_used_gb()
    gpu_busy_before = gpu_utilization()
    started = time.perf_counter()

    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if args.terrain is not None:
        env_cfg.scene.set_terrain(args.terrain)
    soil = getattr(env_cfg.scene.terrain, "soil", None)
    soil_options = ("voxel_size", "particles_per_cell", "depth", "size", "physics_dt", "rover_substeps", "mpm_iterations")
    if soil is None and any(getattr(args, name) is not None for name in soil_options):
        raise ValueError(f"{args.task} has no soil layer to configure.")
    if soil is not None:
        for name in ("voxel_size", "particles_per_cell", "depth", "size"):
            if getattr(args, name) is not None:
                setattr(soil, name, tuple(getattr(args, name)) if name == "size" else getattr(args, name))
        env_cfg.update_physics()
        if args.physics_dt is not None:
            env_cfg.sim.dt = args.physics_dt
            env_cfg.decimation = round(POLICY_DT / args.physics_dt)
        rover, soil_entry = env_cfg.sim.physics.solver_cfg.entries
        if args.rover_substeps is not None:
            rover.substeps = args.rover_substeps
        if args.mpm_iterations is not None:
            soil_entry.solver_cfg.max_iterations = args.mpm_iterations
    env_cfg.seed = args.seed

    task_spec = gym.spec(args.task)
    experiment_cfg = parse_skrl_cfg(task_spec.kwargs["skrl_cfgs"][args.agent.upper()])
    experiment_cfg["agent"]["rollouts"] = 1
    experiment = experiment_cfg["agent"]["experiment"]
    experiment.update(directory=os.path.join("/tmp", "skrl_benchmark"), write_interval=0, checkpoint_interval=0)
    experiment["wandb"] = False

    env = SkrlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), ml_framework="torch")
    set_seed(args.seed)
    agent = create_agent(args.agent, env, experiment_cfg)
    agent.load(args.checkpoint or task_spec.kwargs["best_model_path"])
    trainer_cfg = experiment_cfg["trainer"]
    trainer_cfg["timesteps"] = args.warmup + args.steps
    trainer_cfg["disable_progressbar"] = True
    SequentialTrainer(cfg=trainer_cfg, agents=agent, env=env)  # initializes the agent as eval.py does
    agent.enable_training_mode(False)

    base = env.unwrapped
    terrain = base.scene.terrain
    robot = base.scene["robot"]
    terminations = base.termination_manager
    observations, _ = env.reset()
    if soil is not None:
        particles = slice(terrain.soil_particle_offset, terrain.soil_particle_offset + terrain.soil_particle_count)
        initial_particles = particle_positions()[particles].clone()
    heightmap = terrain._terrainManager.terrain_only_heightmap_manager
    created = ready = time.perf_counter()
    counts = dict.fromkeys(terminations.active_terms, 0)
    speed_sum = torch.zeros((), device=base.device)
    clearance_sum = torch.zeros((), device=base.device)

    for step in range(args.warmup + args.steps):
        if step == args.warmup:
            torch.cuda.synchronize()
            ready = time.perf_counter()
            counts = dict.fromkeys(counts, 0)
        with torch.no_grad():
            actions, outputs = agent.act(observations, None, timestep=step, timesteps=args.warmup + args.steps)
            observations, _, terminated, truncated, _ = env.step(outputs.get("mean_actions", actions))
        for name in counts:
            counts[name] += int(terminations.get_term(name).sum())
        if step >= args.warmup:
            speed_sum += robot.data.root_lin_vel_b.torch[:, 0].abs().mean()
            root = robot.data.root_pos_w.torch
            clearance_sum += (root[:, 2] - heightmap.get_height_at(root[:, :2].contiguous())).mean()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - ready

    root = robot.data.root_pos_w.torch
    physics_steps = args.steps * env_cfg.decimation
    result = {"task": args.task, "num_envs": args.num_envs}
    if soil is not None:
        moved = torch.linalg.norm(particle_positions()[particles] - initial_particles, dim=1)
        min_x, min_y, max_x, max_y = terrain.soil_bounds
        on_soil = (root[:, 0] > min_x) & (root[:, 0] < max_x) & (root[:, 1] > min_y) & (root[:, 1] < max_y)
        result |= {
            "soil_particles": terrain.soil_particle_count,
            "soil_bounds_m": terrain.soil_bounds,
            "voxel_size_m": soil.voxel_size,
            "particle_spacing_m": soil.spacing,
            "soil_depth_m": soil.depth,
            "rover_substeps": rover.substeps,
            "mpm_iterations": soil_entry.solver_cfg.max_iterations,
            "particles_moved_over_1cm": int((moved > 0.01).sum()),
            "max_particle_displacement_m": float(moved.max()),
            "rovers_on_soil": int(on_soil.sum()),
        }
    result |= {
        "physics_dt_s": env_cfg.sim.dt,
        "decimation": env_cfg.decimation,
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "startup_s": created - started,
        "warmup_s": ready - created,
        "elapsed_s": elapsed,
        "env_steps_per_s": args.num_envs * args.steps / elapsed,
        "policy_steps_per_s": args.steps / elapsed,
        "physics_steps_per_s": physics_steps / elapsed,
        "real_time_factor": physics_steps * env_cfg.sim.dt / elapsed,
        "gpu_mem_used_gb": gpu_used_gb() - gpu_before,
        "gpu_utilization_before_start_pct": gpu_busy_before,
        "terminations": counts,
        "targets_reached": counts["is_success"],
        "mean_forward_speed_m_s": float(speed_sum) / args.steps,
        "mean_body_height_above_terrain_m": float(clearance_sum) / args.steps,
        "finite_state": bool(torch.isfinite(root).all()) and (soil is None or bool(torch.isfinite(moved).all())),
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2)
    env.close()


def particle_positions() -> torch.Tensor:
    import warp as wp

    return wp.to_torch(NewtonManager.get_state_0().particle_q)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
