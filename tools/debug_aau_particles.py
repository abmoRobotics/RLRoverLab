#!/usr/bin/env python3
"""Smoke-check and time an AAU rover driving on PhysX or Newton particles."""

from __future__ import annotations

import argparse
import json
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--backend", choices=("physx", "newton"), required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--voxel_size", type=float, default=0.08)
parser.add_argument("--particles_per_cell", type=int, default=2)
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--output", help="Write one JSON result to this path")
add_launcher_args(parser)
args = parser.parse_args()

def main() -> None:
    if args.backend == "physx":
        from isaaclab_physx.physics import PhysxCfg

        launcher_cfg = PhysxCfg()
    else:
        from rover_envs.benchmarks.aau_particles import make_sim_cfg

        launcher_cfg = make_sim_cfg(args.backend, args.device, args.voxel_size, args.num_envs)

    started = time.perf_counter()
    with launch_simulation(launcher_cfg, args):
        import isaaclab.sim as sim_utils
        import torch
        from isaaclab.scene import InteractiveScene

        from rover_envs.benchmarks.aau_particles import (
            DT,
            make_scene_cfg,
            make_sim_cfg,
            particle_lattice,
            spawn_physx_particles,
        )

        positions, mass, radius = particle_lattice(args.voxel_size, args.particles_per_cell)
        sim_cfg = make_sim_cfg(args.backend, args.device, args.voxel_size, args.num_envs)
        if args.backend == "newton" and args.visualizer and "newton_gl" in args.visualizer:
            from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

            sim_cfg.visualizer_cfgs = [NewtonGLVisualizerCfg(show_particles=True)]
        scene_cfg = make_scene_cfg(args.backend, args.num_envs, positions, mass, radius)
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(scene_cfg)
        if args.backend == "physx":
            spawn_physx_particles(args.num_envs, positions, mass, radius)
        sim.reset()
        if sim.is_rendering:
            sim.set_camera_view(eye=(3.0, -4.0, 2.5), target=(0.0, 0.0, 0.0))
        robot = scene["robot"]
        drive_ids, drive_names = robot.find_joints([".*Drive_Continuous"])
        steer_ids, steer_names = robot.find_joints([".*Steer_Revolute"])
        if len(drive_ids) != 6 or len(steer_ids) != 4:
            raise RuntimeError(f"Unexpected rover joints: {drive_names}, {steer_names}")
        drive = torch.full((args.num_envs, 6), 2.0, device=sim.device)
        steer = torch.zeros((args.num_envs, 4), device=sim.device)

        def step() -> None:
            robot.set_joint_velocity_target_index(target=drive, joint_ids=drive_ids)
            robot.set_joint_position_target_index(target=steer, joint_ids=steer_ids)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(DT)
            if sim.is_rendering:
                sim.render()

        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()
        ready = time.perf_counter()
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - ready

        pose = robot.data.root_pos_w.torch
        drive_speed = robot.data.joint_vel.torch[:, drive_ids].abs().mean(dim=1)
        if not torch.isfinite(pose).all():
            raise RuntimeError("Rover pose became non-finite")
        if args.backend == "newton" and not torch.isfinite(scene["soil"].data.particle_pos_w.torch).all():
            raise RuntimeError("Newton particle positions became non-finite")

        result = {
            "backend": args.backend,
            "num_envs": args.num_envs,
            "particles_per_env": len(positions),
            "voxel_size": args.voxel_size,
            "steps": args.steps,
            "warmup": args.warmup,
            "startup_s": ready - started,
            "elapsed_s": elapsed,
            "env_steps_per_s": args.num_envs * args.steps / elapsed,
            "sim_seconds_per_wall_second": args.steps * DT / elapsed,
            "rover_x_local_m": (pose[:, 0] - scene.env_origins[:, 0]).tolist(),
            "rover_z_m": pose[:, 2].tolist(),
            "mean_drive_rad_s": drive_speed.tolist(),
        }
        print(json.dumps(result, indent=2))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as output:
                json.dump(result, output, indent=2)


if __name__ == "__main__":
    main()
