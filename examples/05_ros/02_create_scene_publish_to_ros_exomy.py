from __future__ import annotations

import argparse
import threading
import traceback
from typing import TYPE_CHECKING

import carb
import numpy as np
import rclpy
import torch
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Empty Scene")

parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create")

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
carb.settings.get_settings().set("persistent/app/omniverse/gamepadCameraControl", False)
from omni.isaac.core.utils.extensions import enable_extension  # noqa: F401, E402

enable_extension("omni.isaac.ros2_bridge")

import omni.isaac.lab.sim as sim_utils  # noqa: F401, E402
from omni.isaac.core.utils import prims  # noqa: F401, E402
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: F401, E402
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: F401, E402
from omni.isaac.lab.sim import SimulationContext  # noqa: F401, E402
from omni.isaac.lab.utils import configclass  # noqa: F401, E402
from omni.isaac.sensor import Camera  # noqa: F401, E402

import rover_envs.mdp as mdp  # noqa: F401, E402
# Avoid Circular Import
from rover_envs.assets.robots.exomy import EXOMY_CFG  # noqa: F401, E402
from rover_envs.mdp.actions.ackermann_actions import AckermannActionNonVec  # noqa: F401, E402

if TYPE_CHECKING:
    from rover_envs.envs.navigation.utils.articulation.articulation import RoverArticulation
# Here we configure the environment

DEVICE = "cpu"


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
        prim_path="{ENV_REGEX_NS}/Robot")


def setup_scene():
    """ Setup the scene """
    sim_cfg = sim_utils.SimulationCfg(
        device=DEVICE,
        use_gpu_pipeline=False,
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = SimulationContext(sim_cfg)
    # Set Default Camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_cfg = RoverEmptySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Create Camera
    import omni.isaac.core.utils.numpy.rotations as rot_utils  # noqa: F401
    camera = Camera(
        prim_path="/World/envs/env_0/Robot/Body/Camera",
        resolution=(1280, 720),
        translation=([-0.151, 0, 0.73428]),
        # orientation=([0.64086, 0.29884, -0.29884, -0.64086]),
        # orientation=([-0.64086, 0.64086, 0.29884, -0.64086]),
        orientation=(rot_utils.euler_angles_to_quats(np.array([0, 30, 0]), degrees=True))
        # orientation=([0.64086, 0.29884, -0.29884, -0.64086]),
    )
    camera.initialize()
    camera.set_horizontal_aperture(6.055)
    camera.set_focal_length(2.12)
    camera.set_vertical_aperture(2.968879962)
    camera.set_clipping_range(near_distance=0.01, far_distance=1000000)

    sim.reset()
    return sim, scene, camera


def run_simulation(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """ Run the simulation """
    # Get the robot
    rclpy.init()
    cmd_vel_subscriber = TwistSubscriber()

    thread = threading.Thread(target=rclpy.spin, args=(cmd_vel_subscriber,), daemon=True)
    thread.start()

    robot: RoverArticulation = scene["robot"]

    action_cfg = mdp.AckermannActionCfg(
        asset_name="robot",
        wheelbase_length=0.29778,
        middle_wheel_distance=0.1548,
        rear_and_front_wheel_distance=0.1548,
        wheel_radius=0.1,
        min_steering_radius=0.4,
        steering_joint_names=["FL_Steer_Joint", "RL_Steer_Joint", "RR_Steer_Joint", "FR_Steer_Joint"],
        drive_joint_names=[".*Drive_Joint"],
    )
    rover_articulation_manager = AckermannActionNonVec(action_cfg, robot, num_envs=args_cli.num_envs, device=DEVICE)

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
        if count % 54444400 == 0:
            # reset counter
            count = 0

            # Reset the scene
            reset_scene(robot, scene)
        import copy

        # Step the simulation
        scene.write_data_to_sim()
        lin_vel = copy.deepcopy(cmd_vel_subscriber.velocity)
        ang_vel = copy.deepcopy(cmd_vel_subscriber.angular)
        # print(f'lin vel: {lin_vel}, ang vel {ang_vel}')
        actions = torch.tensor([[lin_vel, ang_vel]], dtype=torch.float32)
        rover_articulation_manager.process_actions(actions)
        rover_articulation_manager.apply_actions()
        sim.step()

        count += 1

        scene.update(sim_dt)


from rover_envs.utils.ros2.publishers import publish_camera_info, publish_depth, publish_rgb  # noqa: F401, E402
from rover_envs.utils.ros2.subscribers import TwistSubscriber  # noqa: F401, E402


def main():
    # First we setup the scene
    sim, scene, camera = setup_scene()
    publish_camera_info(camera, 30)
    publish_rgb(camera, 30)
    publish_depth(camera, 30)

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
