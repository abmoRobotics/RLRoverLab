#!/usr/bin/env python3
"""View the AAU rover driving on Newton MPM soil in the Isaac Sim (Kit) viewport.

Uses the same scene and physics as ``tools/debug_aau_particles.py --backend newton``:
MJWarp rover wheels coupled to implicit-MPM soil through proxy coupling. Newton
writes the particles to USD ``Points`` prims every render frame, so Kit shows them
without extra code. The rover alternates driving forward and backward so it stays
on the soil bed; close the window or press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--voxel_size", type=float, default=0.08, help="MPM grid voxel size [m]")
parser.add_argument("--particles_per_cell", type=int, default=2, help="Particles per voxel edge")
parser.add_argument("--collider_basis", default="S2", help='MPM collider basis, "S2" (default) or "Q1"')
parser.add_argument("--max_iterations", type=int, default=None, help="Override the MPM rheology iteration count")
parser.add_argument("--drive_speed", type=float, default=2.0, help="Wheel speed [rad/s]")
parser.add_argument("--reverse_every", type=float, default=4.0, help="Flip driving direction every N seconds")
parser.add_argument("--steps", type=int, default=0, help="Stop after N steps; 0 runs until the window closes")
add_launcher_args(parser)
if not any(arg in ("--viz", "--visualizer") or arg.startswith(("--viz=", "--visualizer=")) for arg in sys.argv):
    sys.argv += ["--viz", "kit"]
args = parser.parse_args()


def make_cfg():
    from rover_envs.benchmarks.aau_particles import make_sim_cfg

    cfg = make_sim_cfg("newton", args.device, args.voxel_size, args.num_envs)
    for entry in cfg.physics.solver_cfg.entries:
        if entry.name == "soil":
            entry.solver_cfg.collider_basis = args.collider_basis
            if args.max_iterations is not None:
                entry.solver_cfg.max_iterations = args.max_iterations
    return cfg


def main() -> None:
    with launch_simulation(make_cfg(), args):
        import isaaclab.sim as sim_utils
        import omni.kit.app
        import torch
        from isaaclab.scene import InteractiveScene

        from rover_envs.benchmarks.aau_particles import DT, make_scene_cfg, particle_lattice

        positions, mass, radius = particle_lattice(args.voxel_size, args.particles_per_cell)
        print(f"[view_aau_particles] {len(positions):,} particles per env, voxel {args.voxel_size} m")
        sim = sim_utils.SimulationContext(make_cfg())
        scene = InteractiveScene(make_scene_cfg("newton", args.num_envs, positions, mass, radius))
        sim.reset()
        sim.set_camera_view(eye=(2.5, -3.0, 1.8), target=(0.0, 0.0, 0.1))

        robot = scene["robot"]
        drive_ids, _ = robot.find_joints([".*Drive_Continuous"])
        steer_ids, _ = robot.find_joints([".*Steer_Revolute"])
        drive = torch.full((args.num_envs, len(drive_ids)), args.drive_speed, device=sim.device)
        steer = torch.zeros((args.num_envs, len(steer_ids)), device=sim.device)
        reverse_steps = max(1, round(args.reverse_every / DT))

        app = omni.kit.app.get_app()
        step = 0
        while app.is_running() and (args.steps == 0 or step < args.steps):
            if step > 0 and step % reverse_steps == 0:
                drive.neg_()
            robot.set_joint_velocity_target_index(target=drive, joint_ids=drive_ids)
            robot.set_joint_position_target_index(target=steer, joint_ids=steer_ids)
            scene.write_data_to_sim()
            sim.step(render=True)
            scene.update(DT)
            step += 1


if __name__ == "__main__":
    main()
