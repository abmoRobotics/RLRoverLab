from __future__ import annotations

import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Create and run a simple rover scene.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to create")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer="kit")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

try:
    import warp as wp  # noqa: E402
except ModuleNotFoundError:
    wp = None

# Avoid circular import.
from rover_envs.assets.robots.exomy import EXOMY_CFG  # noqa: E402


def _to_torch(tensor_like: torch.Tensor):
    if isinstance(tensor_like, torch.Tensor):
        return tensor_like
    if wp is None:
        raise TypeError("Expected a torch tensor but got a non-torch value without warp available.")
    return wp.to_torch(tensor_like)


@configclass
class RoverEmptySceneCfg(InteractiveSceneCfg):
    """Configuration for the empty rover scene."""

    # Add ground plane.
    ground = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # Add lights.
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000, color_temperature=4500.0)
    )

    # Add the robot.
    robot: ArticulationCfg = EXOMY_CFG.replace(prim_path="/World/Robot")


def setup_scene() -> tuple[SimulationContext, InteractiveScene]:
    """Set up the simulation and scene."""
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device if not args_cli.cpu else "cpu",
        dt=1.0 / 60.0,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = SimulationContext(sim_cfg)
    # Set default camera.
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    scene_cfg = RoverEmptySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO]: Setup complete...")
    return sim, scene


def run_simulation(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulation loop."""
    robot = scene["robot"]

    sim_dt = sim.get_physics_dt()
    count = 0

    def reset_scene():
        root_pose = _to_torch(robot.data.default_root_pose).clone()
        root_pose[:, :3] += scene.env_origins
        robot.write_root_link_pose_to_sim_index(root_pose=root_pose)

        root_velocity = _to_torch(robot.data.default_root_vel).clone()
        robot.write_root_com_velocity_to_sim_index(root_velocity=root_velocity)

        joint_pos = _to_torch(robot.data.default_joint_pos).clone()
        joint_vel = _to_torch(robot.data.default_joint_vel).clone()
        joint_pos += torch.randn_like(joint_pos) * 0.1
        robot.write_joint_position_to_sim_index(position=joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=joint_vel)

        scene.reset()
        print("[INFO]: Reset scene state")

    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            reset_scene()

        velocities = torch.ones_like(_to_torch(robot.data.default_joint_pos)) * 0.1
        robot.set_joint_velocity_target_index(target=velocities)

        scene.write_data_to_sim()
        sim.step()

        count += 1
        scene.update(sim_dt)


def main():
    sim, scene = setup_scene()
    run_simulation(sim, scene)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Error in main: {e}")
        carb.log_error(traceback.format_exc())
    finally:
        simulation_app.close()
