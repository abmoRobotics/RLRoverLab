from __future__ import annotations

## Debugginh ##
import time

import argparse
import os
import traceback
from typing import TYPE_CHECKING

import carb
import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Manual Rover Drive with Camera")

parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of environments to create")
parser.add_argument("--robot", type=str, default="aau_rover_simple",
                    choices=["aau_rover_simple", "aau_rover", "exomy"],
                    help="Robot to use")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer="kit")

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

# Import robot configurations
from rover_envs.assets.robots.aau_rover_simple import AAU_ROVER_SIMPLE_CFG
from rover_envs.assets.robots.aau_rover import AAU_ROVER_CFG
from rover_envs.assets.robots.exomy import EXOMY_CFG

# Import camera scene configurations
from rover_envs.envs.navigation.rover_env_camera_cfg import RoverCameraSceneCfg, RoverZed2iWVGAEnvCfg, RoverZed2iWVGAEnvCfgTEMP

from rover_envs.mdp.actions.actions_cfg import AckermannActionCfg
from rover_envs.mdp.actions.ackermann_actions import AckermannActionNonVec

if TYPE_CHECKING:
    from rover_envs.envs.navigation.utils.articulation.articulation import RoverArticulation

### CAMERA SCENE CONFIGURATION ###
if args_cli.enable_cameras:
    # from RGBCameraInterface import RGBCameraInterface
    from EventCameraInterfaceGPU import EventCameraInterfaceGPU


def _get_app_window():
    try:
        from omni.appwindow import get_default_app_window
        return get_default_app_window()
    except ModuleNotFoundError:
        pass

    try:
        import omni.kit.app
        kit_app = omni.kit.app.get_app()
        if hasattr(kit_app, "get_window"):
            return kit_app.get_window()
    except (AttributeError, ModuleNotFoundError):
        pass

    try:
        from omni.kit.window.app import get_default_app_window
        return get_default_app_window()
    except (ImportError, ModuleNotFoundError):
        pass

    raise RuntimeError("Could not find a Kit app window for keyboard input.")


def _unsubscribe_keyboard(input_interface, keyboard, keyboard_sub):
    if keyboard_sub is None:
        return
    if hasattr(input_interface, "unsubscribe_to_keyboard_events"):
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
    elif hasattr(keyboard_sub, "unsubscribe"):
        keyboard_sub.unsubscribe()


def _clone_as_torch(tensor_like):
    if isinstance(tensor_like, torch.Tensor):
        return tensor_like.clone()

    try:
        import warp as wp
        return wp.to_torch(tensor_like).clone()
    except (ModuleNotFoundError, TypeError, AttributeError):
        return torch.as_tensor(tensor_like).clone()


@configclass
class RoverSceneCfg(InteractiveSceneCfg):
    """Configuration for the rover scene with Mars terrain"""

    # Add Mars terrain
    terrain = TerrainImporterCfg(
        class_type=TerrainImporter,
        prim_path="/World/terrain",
        terrain_type="usd",
        collision_group=-1,
        usd_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "rover_envs",
            "assets",
            "terrains",
            "mars",
            "terrain1",
            "terrain_only.usd",
        ),
    )

    # Add Mars obstacles (rocks)
    obstacles = AssetBaseCfg(
        prim_path="/World/terrain/obstacles",
        spawn=sim_utils.UsdFileCfg(
            visible=True,
            usd_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "..",
                "rover_envs",
                "assets",
                "terrains",
                "mars",
                "terrain1",
                "rocks_merged.usd",
            ),
        ),
    )

    # Add lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000, color_temperature=4500.0)
    )

    # Robot with camera (will be set dynamically)
    robot: ArticulationCfg = AAU_ROVER_SIMPLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def setup_scene():
    """Setup the scene"""

    # Select robot configuration
    robot_configs = {
        "aau_rover_simple": AAU_ROVER_SIMPLE_CFG,
        "aau_rover": AAU_ROVER_CFG,
        "exomy": EXOMY_CFG,
    }
    selected_robot = robot_configs[args_cli.robot].replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim_cfg = sim_utils.SimulationCfg(
        device="cpu",
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 2.5], [0.0, 0.0, 0.5])

    # Choose scene config based on camera flag
    if args_cli.enable_cameras:
        # Instantiate the camera scene config and override its settings
        #scene_cfg = RoverCameraSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
        # Or use the ZED2i camera configuration:
        # Or use the smaller RoverCameraSceneCfg
        scene_cfg = RoverZed2iWVGAEnvCfgTEMP(num_envs=args_cli.num_envs, env_spacing=2.0)
    else:
        scene_cfg = RoverSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)

    scene_cfg.robot = selected_robot
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    # Print camera info if enabled
    if args_cli.enable_cameras and "tiled_camera" in scene.sensors:
        print(f"[INFO] Camera enabled: {scene['tiled_camera'].cfg.prim_path}")
        print(f"[INFO] Camera resolution: {scene['tiled_camera'].cfg.width}x{scene['tiled_camera'].cfg.height}")

    return sim, scene


def run_simulation(sim: SimulationContext, scene: InteractiveScene):
    """Run the simulation with WASD teleop"""

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    count = 0

    # Setup Ackermann action
    action_cfg = AckermannActionCfg(
        asset_name="robot",
        wheelbase_length=0.849,
        middle_wheel_distance=0.894,
        rear_and_front_wheel_distance=0.77,
        wheel_radius=0.1,
        min_steering_radius=0.8,
        steering_joint_names=[".*Steer_Revolute"],
        drive_joint_names=[".*Drive_Continuous"],
        offset=-0.0135,
    )
    ackermann_action = AckermannActionNonVec(
        action_cfg, robot, num_envs=args_cli.num_envs, device="cpu"
    )

    # Keyboard setup
    import carb.input

    app_window = _get_app_window()
    input_interface = carb.input.acquire_input_interface()
    keyboard = app_window.get_keyboard()

    key_state = {
        carb.input.KeyboardInput.W: True,
        carb.input.KeyboardInput.S: False,
        carb.input.KeyboardInput.A: False,
        carb.input.KeyboardInput.D: False,
        carb.input.KeyboardInput.X: False,
        carb.input.KeyboardInput.R: False,
    }

    def on_keyboard_event(event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input in key_state:
                key_state[event.input] = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in key_state:
                key_state[event.input] = False
        return True

    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    actions = torch.zeros((args_cli.num_envs, 2), device="cpu")

    def reset_scene(robot, scene):
        root_state = _clone_as_torch(robot.data.default_root_state)
        custom_position = torch.tensor([10.0, 10.0, 0.5], device=root_state.device)
        env_origins = _clone_as_torch(scene.env_origins).to(root_state.device)
        root_state[:, :3] = env_origins + custom_position
        robot.write_root_state_to_sim(root_state)

        joint_pos = _clone_as_torch(robot.data.default_joint_pos)
        joint_vel = _clone_as_torch(robot.data.default_joint_vel)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        scene.reset()
        print(f"[INFO] Reset - Robot at: {root_state[0, :3].cpu().numpy()}")

    reset_scene(robot, scene)
    camera_interface = None
    if args_cli.enable_cameras:
        #camera_interface = RGBCameraInterface(
        camera_interface = EventCameraInterfaceGPU(
            scene=scene,
            sim=sim,
            enable_ui=True,  # Enable UI visualization
            device="cuda",
            view_mode="tiled",  # "single" or "tiled"
            env_index=0  # View the first environment
        )

    camera_status = "enabled" if args_cli.enable_cameras else "disabled"
    print(f"[INFO] Simulation started with {args_cli.robot} (camera {camera_status})")
    print("\n=== WASD Keyboard Teleop ===")
    print("W/S: Forward/Backward")
    print("A/D: Turn Left/Right")
    print("X: Stop | R: Reset")
    print("============================\n")

    try:
        while simulation_app.is_running():
            time_start_loop = time.time()
            if count % 50000 == 0 and count > 0:
                reset_scene(robot, scene)
                count = 0

            lin_vel = 0.0
            ang_vel = 0.0

            if key_state[carb.input.KeyboardInput.W]:
                lin_vel = 1.0
            elif key_state[carb.input.KeyboardInput.S]:
                lin_vel = -1.0
            if key_state[carb.input.KeyboardInput.A]:
                ang_vel = 0.5
            elif key_state[carb.input.KeyboardInput.D]:
                ang_vel = -0.5
            if key_state[carb.input.KeyboardInput.X]:
                lin_vel = ang_vel = 0.0
            if key_state[carb.input.KeyboardInput.R]:
                reset_scene(robot, scene)
                key_state[carb.input.KeyboardInput.R] = False
                lin_vel = ang_vel = 0.0

            actions[:, 0] = lin_vel
            actions[:, 1] = ang_vel

            ackermann_action.process_actions(actions)
            ackermann_action.apply_actions()

            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            count += 1

            ### EVENT CAMERA INTERFACE USAGE ###
            if camera_interface is not None:
                events = camera_interface.get_events()
                if events is not None and count % 100 == 0:
                    print(f"[EVENTS] tensor shape={tuple(events.shape)} count={int(events[..., 2].sum().item())}")

            ### RGB CAMERA INTERFACE USAGE ###
            # if camera_interface is not None:
            #     camera_data = camera_interface.get_latest_frame()  # RGB in Omni UI
            #     if camera_data is not None and count % 100 == 0:
            #         print(f"[RGB] Frame at {camera_data.timestamp:.2f}s")

            #print("Elapsed loop time:", time.time() - time_start_loop)
    finally:
        _unsubscribe_keyboard(input_interface, keyboard, keyboard_sub)
        if camera_interface is not None:
            camera_interface.close()


def main():
    sim, scene = setup_scene()
    try:
        run_simulation(sim, scene)
    finally:
        # Proper cleanup sequence
        print("[INFO] Cleaning up simulation...")
        if sim is not None:
            sim.stop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Error: {e}")
        carb.log_error(traceback.format_exc())
    finally:
        simulation_app.close()
        # Give it time to cleanup
        import time
        time.sleep(2)
