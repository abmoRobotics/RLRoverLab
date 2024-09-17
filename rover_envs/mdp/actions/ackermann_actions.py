from __future__ import annotations

from typing import TYPE_CHECKING

import carb
import torch
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.envs import ManagerBasedEnv
from omni.isaac.lab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv  # noqa: F811

    from . import actions_cfg


class AckermannAction(ActionTerm):

    cfg: actions_cfg.AckermannActionCfg

    _asset: Articulation

    _wheelbase_length: float

    _middle_wheel_distance: float

    _rear_and_front_wheel_distance: float

    _wheel_radius: float

    _min_steering_radius: float

    _steering_joint_names: list[str]

    _drive_joint_names: list[str]

    _scale: torch.Tensor

    _offset: torch.Tensor

    def __init__(self, cfg: actions_cfg.AckermannActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._drive_joint_ids, self._drive_joint_names = self._asset.find_joints(self.cfg.drive_joint_names)

        self._steering_joint_ids, self._steering_joint_names = self._asset.find_joints(self.cfg.steering_joint_names)

        # remap joints to the order specified in the config.
        steering_order = cfg.steering_order
        drive_order = cfg.drive_order
        sorted_steering_joint_names = sorted(self._steering_joint_names, key=lambda x: steering_order.index(x[:2]))
        sorted_drive_joint_names = sorted(self._drive_joint_names, key=lambda x: drive_order.index(x[:2]))
        original_steering_id_positions = {name: i for i, name in enumerate(self._steering_joint_names)}
        original_drive_id_positions = {name: i for i, name in enumerate(self._drive_joint_names)}
        self._sorted_steering_ids = [self._steering_joint_ids[original_steering_id_positions[name]]
                                     for name in sorted_steering_joint_names]
        self._sorted_drive_ids = [self._drive_joint_ids[original_drive_id_positions[name]]
                                  for name in sorted_drive_joint_names]

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel = torch.zeros(self.num_envs, len(self._drive_joint_ids), device=self.device)
        self._joint_pos = torch.zeros(self.num_envs, len(self._steering_joint_ids), device=self.device)

        # Save the scale and offset for the actions
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 2  # Assuming a 2D action vector (linear velocity, angular velocity)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset

    def apply_actions(self):

        self._joint_pos, self._joint_vel = ackermann(
            self._processed_actions[:, 0], self._processed_actions[:, 1], self.cfg, self.device)

        self._asset.set_joint_velocity_target(self._joint_vel, joint_ids=self._drive_joint_ids)
        self._asset.set_joint_position_target(self._joint_pos, joint_ids=self._steering_joint_ids)


class AckermannActionNonVec():
    def __init__(self,
                 cfg: actions_cfg.AckermannActionCfg,
                 robot: Articulation,
                 num_envs: int,
                 device: torch.device):
        """ Initialize the AckermannActionNonVec

        Args:
            cfg (actions_cfg.AckermannActionCfg): configuration for the ackermann action
            robot (Articulation): robot asset
            num_envs (int): number of environments
            device (torch.device): device to run the operation on
        """
        # Initialize Parameters
        self.cfg = cfg
        self.device = device
        self.num_envs = num_envs
        self._asset = robot

        # Find the joint ids and names for the drive and steering joints
        self._drive_joint_ids, self._drive_joint_names = self._asset.find_joints(self.cfg.drive_joint_names)

        self._steering_joint_ids, self._steering_joint_names = self._asset.find_joints(self.cfg.steering_joint_names)

        # Remap joints to the order specified in the config.
        steering_order = cfg.steering_order
        drive_order = cfg.drive_order
        sorted_steering_joint_names = sorted(self._steering_joint_names, key=lambda x: steering_order.index(x[:2]))
        sorted_drive_joint_names = sorted(self._drive_joint_names, key=lambda x: drive_order.index(x[:2]))
        original_steering_id_positions = {name: i for i, name in enumerate(self._steering_joint_names)}
        original_drive_id_positions = {name: i for i, name in enumerate(self._drive_joint_names)}
        self._sorted_steering_ids = [self._steering_joint_ids[original_steering_id_positions[name]]
                                     for name in sorted_steering_joint_names]
        self._sorted_drive_ids = [self._drive_joint_ids[original_drive_id_positions[name]]
                                  for name in sorted_drive_joint_names]

        carb.log_info(
            f" {self._drive_joint_ids} [{self._drive_joint_names}]"
            f" {self._steering_joint_ids} [{self._steering_joint_names}]"
        )

        # Create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._joint_vel = torch.zeros(self.num_envs, len(self._drive_joint_ids), device=self.device)
        self._joint_pos = torch.zeros(self.num_envs, len(self._steering_joint_ids), device=self.device)

        # Save the scale and offset for the actions
        self._scale = torch.tensor(self.cfg.scale, device=self.device).unsqueeze(0)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).unsqueeze(0)

    @property
    def action_dim(self) -> int:
        return 2  # Assuming a 2D action vector (linear velocity, angular velocity)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions):
        # Store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions = self.raw_actions * self._scale + self._offset

    def apply_actions(self):
        # Apply the actions to the rover
        self._joint_pos, self._joint_vel = ackermann(
            self._processed_actions[:, 0], self._processed_actions[:, 1], self.cfg, self.device)

        self._asset.set_joint_velocity_target(self._joint_vel, joint_ids=self._sorted_drive_ids)
        self._asset.set_joint_position_target(self._joint_pos, joint_ids=self._sorted_steering_ids)


def ackermann(lin_vel, ang_vel, cfg, device):
    """ Ackermann steering model for the rover
    Args:
        lin_vel (torch.Tensor): linear velocity of the rover
        ang_vel (torch.Tensor): angular velocity of the rover
        cfg (actions_cfg.AckermannActionCfg): configuration for the ackermann action
        device (torch.device): device to run the operation on

    Returns:
        torch.Tensor: steering angles for the rover
        torch.Tensor: wheel velocities for the rover
    """

    wheel_radius = cfg.wheel_radius  # wheel radius
    d_fr = cfg.rear_and_front_wheel_distance  # distance between front and rear wheels
    d_mw = cfg.middle_wheel_distance  # distance between middle wheels
    wl = cfg.wheelbase_length  # wheelbase length
    offset = cfg.offset
    # Checking the direction of the linear and angular velocities
    direction: torch.Tensor = torch.sign(lin_vel)
    turn_direction: torch.Tensor = torch.sign(ang_vel)

    direction = torch.where(direction == 0, direction+1, direction)

    # Taking the absolute values of the velocities
    lin_vel = torch.abs(lin_vel)
    ang_vel = torch.abs(ang_vel)

    # Calculates the turning radius of the rover, returns inf if ang_vel is 0
    not_zero_condition = torch.logical_not(ang_vel == 0) | torch.logical_not(lin_vel == 0)

    minimum_radius = (d_mw * 0.8)  # should be 0.5 but 0.8 makes operation more smooth
    turning_radius: torch.Tensor = torch.where(
        not_zero_condition, lin_vel / ang_vel, torch.tensor(float('inf'), device=device))
    # turning_radius = torch.where(turning_radius < minimum_radius, minimum_radius, turning_radius)

    # if turn_radius is shorter than half of wheelbase: point turn else ackermann
    # Calculating the turning radius of the front wheels
    r_ML = turning_radius - (d_mw / 2) * turn_direction
    r_MR = turning_radius + (d_mw / 2) * turn_direction
    r_FL = turning_radius - ((d_fr / 2)) * turn_direction
    r_FR = turning_radius + ((d_fr / 2)) * turn_direction
    r_RL = turning_radius - ((d_fr / 2)) * turn_direction
    r_RR = turning_radius + ((d_fr / 2)) * turn_direction

    # Point turn or ackermann
    # Wheel velocities (m/s)
    # If turning radius is less than distance between middle wheels
    # Set velocities for point turn, else
    # if ang_vel is 0, wheel velocity is equal to linear velocity
    vel_FL = torch.where(turning_radius < minimum_radius,
                         -(ang_vel)*turn_direction,
                         torch.where(ang_vel == 0, lin_vel, (r_FL * ang_vel)) * direction)
    vel_FR = torch.where(turning_radius < minimum_radius,
                         (ang_vel)*turn_direction,
                         torch.where(ang_vel == 0, lin_vel, (r_FR * ang_vel)) * direction)
    vel_RL = torch.where(turning_radius < minimum_radius,
                         -(ang_vel)*turn_direction,
                         torch.where(ang_vel == 0, lin_vel, (r_RL * ang_vel)) * direction)
    vel_RR = torch.where(turning_radius < minimum_radius,
                         (ang_vel)*turn_direction,
                         torch.where(ang_vel == 0, lin_vel, (r_RR * ang_vel)) * direction)
    vel_ML = torch.where(turning_radius < minimum_radius,
                         -(ang_vel)*turn_direction,
                         torch.where(ang_vel == 0, lin_vel, (r_ML * ang_vel)) * direction)
    vel_MR = torch.where(turning_radius < minimum_radius,
                         (ang_vel)*turn_direction,
                         torch.where(ang_vel == 0, lin_vel, (r_MR * ang_vel)) * direction)

    wl = torch.ones_like(r_FL) * wl  # Repeat wl as tensor

    # Steering angles for specifically point turns
    # If turning radius is less than the distance between middle wheels
    # set steering angles for point turn, else
    # ackermann
    theta_FL = torch.where(turning_radius < minimum_radius,
                           torch.tensor(-(torch.pi/4), device=device),
                           torch.atan2((wl/2) - offset, r_FL) * turn_direction)
    theta_RR = torch.where(turning_radius < minimum_radius,
                           torch.tensor(-(torch.pi/4), device=device),
                           torch.atan2((wl/2) + offset, r_RR) * -turn_direction)
    theta_FR = torch.where(turning_radius < minimum_radius,
                           torch.tensor((torch.pi/4), device=device),
                           torch.atan2((wl/2) - offset, r_FR) * turn_direction)
    theta_RL = torch.where(turning_radius < minimum_radius,
                           torch.tensor((torch.pi/4), device=device),
                           torch.atan2((wl/2) + offset, r_RL) * -turn_direction)

    wheel_velocities = torch.stack([vel_FL, vel_FR, vel_ML, vel_MR, vel_RL, vel_RR], dim=1)
    # steering_angles = torch.stack([theta_FL, theta_RL, theta_RR, theta_FR], dim=1)
    steering_angles = torch.stack([theta_FL, theta_FR, theta_RL, theta_RR], dim=1)
    # Convert wheel velocities from m/s to rad/s
    wheel_velocities = wheel_velocities / (wheel_radius*2)

    return steering_angles, wheel_velocities

