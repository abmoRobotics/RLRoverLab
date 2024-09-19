from __future__ import annotations

import argparse
import threading
import time
import traceback
from typing import TYPE_CHECKING

import carb
import numpy as np
import rclpy
import torch
from omni.isaac.lab.app import AppLauncher
from rclpy.executors import MultiThreadedExecutor

parser = argparse.ArgumentParser(description="Empty Scene")

parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create")

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
carb.settings.get_settings().set("persistent/app/omniverse/gamepadCameraControl", False)
from omni.isaac.core.utils.extensions import enable_extension  # noqa: F401, E402

enable_extension("omni.isaac.ros2_bridge")

import omni.isaac.core.utils.numpy.rotations as rot_utils  # noqa: F401, E402
import omni.isaac.lab.sim as sim_utils  # noqa: F401, E402
from omni.isaac.core.utils import prims  # noqa: F401, E402
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg  # noqa: F401, E402
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: F401, E402
from omni.isaac.lab.sim import SimulationContext  # noqa: F401, E402
from omni.isaac.lab.utils import configclass  # noqa: F401, E402
from omni.isaac.sensor import Camera  # noqa: F401, E402
from pxr import Gf, UsdGeom  # noqa: F401, E402

import rover_envs.mdp as mdp  # noqa: F401, E402
from rover_envs.assets.robots.aau_rover import AAU_ROVER_CFG  # noqa: F401, E402
# Avoid Circular Import
from rover_envs.assets.robots.aau_rover_simple import AAU_ROVER_SIMPLE_CFG  # noqa: F401, E402
from rover_envs.mdp.actions.ackermann_actions import AckermannActionNonVec  # noqa: F401, E402
from rover_envs.utils.ros2.publishers import (RoverPose, goal_position, publish_camera_info,  # noqa: F401, E402
                                              publish_depth, publish_rgb)
from rover_envs.utils.ros2.subscribers import TwistSubscriber  # noqa: F401, E402

if TYPE_CHECKING:
    from rover_envs.envs.navigation.utils.articulation.articulation import RoverArticulation
# Here we configure the environment

DEVICE = "cpu"

# from rover_envs.envs.navigation.utils.terrains.terrain_importer import RoverTerrainImporter  # noqa: F401
from omni.isaac.lab.terrains import TerrainImporter, TerrainImporterCfg  # noqa: F401, E402

from rover_envs.assets.terrains.mars import MarsTerrainSceneCfg  # noqa: F401, E402


@configclass
class RoverEmptySceneCfg(InteractiveSceneCfg):
    """ Configuration for the empty scene """
    # Add ground plane
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    # Ground Terrain
    # terrain = TerrainImporterCfg(
    #     class_type=TerrainImporter,
    #     prim_path="/World/terrain",
    #     terrain_type="usd",
    #     collision_group=-1,
    #     usd_path=os.path.join(
    #         os.path.dirname(os.path.abspath(__file__)),
    #         "..",
    #         "..",
    #         "rover_envs",
    #         "assets",
    #         "terrains",
    #         "mars",
    #         "terrain1",
    #         "terrain_only.usd",
    #     ),
    # )

    # obstacles = AssetBaseCfg(
    #     prim_path="/World/terrain/obstacles",
    #     spawn=sim_utils.UsdFileCfg(
    #         visible=True,
    #         usd_path=os.path.join(
    #             os.path.dirname(os.path.abspath(__file__)),
    #             "..",
    #             "..",
    #             "rover_envs",
    #             "assets",
    #             "terrains",
    #             "mars",
    #             "terrain1",
    #             "rocks_merged.usd",
    #         ),
    #     ),
    # )

    # Add lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000, color_temperature=4500.0)
    )

    # Add the robot
    robot: ArticulationCfg = AAU_ROVER_SIMPLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot")


def setup_scene():
    """ Setup the scene """
    sim_cfg = sim_utils.SimulationCfg(
        device=DEVICE,
        use_gpu_pipeline=False,
        dt=1.0 / 100.0,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = SimulationContext(sim_cfg)
    # Set Default Camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_cfg = RoverEmptySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Create Camera

    sim.reset()
    return sim, scene


def setup_camera():
    camera = Camera(
        prim_path="/World/envs/env_0/Robot/Body/Camera",
        resolution=(1280, 720),
        translation=([-0.151, 0, 0.73428]),
        orientation=(rot_utils.euler_angles_to_quats(np.array([0, 30, 0]), degrees=True))
    )

    camera.initialize()
    camera.set_horizontal_aperture(6.055)
    camera.set_focal_length(2.12)
    camera.set_vertical_aperture(2.968879962)
    camera.set_clipping_range(near_distance=0.01, far_distance=1000000)
    return camera


def spin_executor(executor):
    """
    Target function for the thread. This function will keep running
    the executor until it is explicitly shut down.
    """
    executor.spin()


def run_simulation(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """ Run the simulation """
    # Get the robot
    goal_position_list = torch.tensor([7.0, 7.0], dtype=torch.float32)
    sphere_prim = sim.stage.DefinePrim("/World/target", "Sphere")
    sphere_geom = UsdGeom.Sphere(sphere_prim)

    # Set the sphere's attributes
    sphere_geom.GetRadiusAttr().Set(0.3)
    sphere_geom.AddTranslateOp().Set(value=(goal_position_list[0], goal_position_list[1], 0.0))

    # transform = UsdGeom.TransformAP

    # sphere_geom.GetPrim().SetTranslate(Gf.Vec3d(goal_position_list[0], goal_position_list[1], 0.0))
    rclpy.init()
    cmd_vel_subscriber = TwistSubscriber(topic_name='cmd_vel')
    position_publisher = goal_position(topic_name='goal_position', goal_position=goal_position_list.numpy())
    robot_position_publisher = RoverPose(topic_name='robot_pose')

    # thread = threading.Thread(target=rclpy.spin, args=(cmd_vel_subscriber, ), daemon=True)
    # thread.start()

    executor = MultiThreadedExecutor()
    executor.add_node(cmd_vel_subscriber)
    executor.add_node(position_publisher)
    thread = threading.Thread(target=spin_executor, args=(executor,), daemon=True)
    thread.start()

    robot: RoverArticulation = scene["robot"]

    def reset_scene(robot: RoverArticulation, scene: InteractiveScene):
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] += torch.tensor([12.0, 5.0, 0.1])
        # scene.env_origins
        robot.write_root_state_to_sim(root_state)

        joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
        joint_pos += torch.randn_like(joint_pos) * 0.1
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        scene.reset()
        print("Reset")

    def goal_distance():
        return np.linalg.norm(robot.data.root_pos_w[0, :2] - goal_position_list)

    action_cfg = mdp.AckermannActionCfg(
        asset_name="robot",
        wheelbase_length=0.849,
        middle_wheel_distance=0.894,
        rear_and_front_wheel_distance=0.77,
        wheel_radius=0.1,
        min_steering_radius=0.8,
        steering_joint_names=[".*Steer_Revolute"],
        drive_joint_names=[".*Drive_Continuous"],
        offset=-0.0135
    )
    reset_scene(robot, scene)
    rover_articulation_manager = AckermannActionNonVec(action_cfg, robot, num_envs=args_cli.num_envs, device=DEVICE)

    sim_dt = sim.get_physics_dt()

    timer = time.time()
    while simulation_app.is_running():
        # Step the simulation
        scene.write_data_to_sim()
        lin_vel = cmd_vel_subscriber.velocity
        ang_vel = cmd_vel_subscriber.angular
        actions = torch.tensor([[lin_vel, ang_vel]], dtype=torch.float32)
        rover_articulation_manager.process_actions(actions)
        rover_articulation_manager.apply_actions()
        # print(robot.data.root_pos_w)
        # print(goal_distance())
        sim.step()
        scene.update(sim_dt)
        if goal_distance() < 0.18:
            reset_scene(robot, scene)

        if time.time() - timer > 0.5:
            robot_position_publisher.publish_robot_position(robot.data.root_state_w[0, :7])
            timer = time.time()


def main():
    # First we setup the scene
    sim, scene = setup_scene()
    # Add Camera
    camera = setup_camera()
    # Publish Camera
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
