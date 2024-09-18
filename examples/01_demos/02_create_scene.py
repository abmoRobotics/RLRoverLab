from __future__ import annotations

import argparse
import traceback
from typing import TYPE_CHECKING

import carb
import torch
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Empty Scene")

parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create")

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.isaac.lab.sim as sim_utils  # noqa: F401, E402
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: F401, E402
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: F401, E402
from omni.isaac.lab.sim import SimulationContext  # noqa: F401, E402
from omni.isaac.lab.utils import configclass  # noqa: F401, E402

# Avoid Circular Import
from rover_envs.assets.robots.aau_rover_simple import AAU_ROVER_SIMPLE_CFG  # noqa: F401, E402
from rover_envs.assets.robots.exomy import EXOMY_CFG  # noqa: F401, E402

if TYPE_CHECKING:
    from rover_envs.envs.navigation.utils.articulation.articulation import RoverArticulation
# Here we configure the environment


@configclass
class RoverEmptySceneCfg(InteractiveSceneCfg):
    """ Configuration for the empty scene """

    # Add ground plane
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Add lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000, color_temperature=4500.0)
    )

    # Add the robot
    robot: ArticulationCfg = EXOMY_CFG.replace(
        prim_path="/World/Robot")


def setup_scene():
    """ Setup the scene """
    sim_cfg = sim_utils.SimulationCfg(
        device="cpu",
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = SimulationContext(sim_cfg)
    # Set Default Camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_cfg = RoverEmptySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    return sim, scene


def run_simulation(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """ Run the simulation """
    # Get the robot
    robot: RoverArticulation = scene["robot"]

    sim_dt = sim.get_physics_dt()
    count = 0

    def reset_scene(robot: RoverArticulation, scene: InteractiveScene):
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += scene.env_origins
        robot.write_root_state_to_sim(root_state)

        joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        joint_pos += torch.randn_like(joint_pos) * 0.1
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        scene.reset()
        print("Reset")

    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0

            # Reset the scene
            reset_scene(robot, scene)

        # Step the simulation

        velocities = torch.ones_like(robot.data.default_joint_pos) * 0.1
        # robot.set_joint_effort_target(efforts)
        robot.set_joint_velocity_target(velocities)

        scene.write_data_to_sim()

        sim.step()

        count += 1

        scene.update(sim_dt)


def main():
    # First we setup the scene
    sim, scene = setup_scene()
    # Then we run the simulation
    run_simulation(sim, scene)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Error in main: {e}")
        carb.log_error(traceback.format_exc())
    finally:
        simulation_app.close()
