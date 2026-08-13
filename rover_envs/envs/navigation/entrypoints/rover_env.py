import torch
from isaaclab.envs.common import VecEnvObs
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

from rover_envs.envs.navigation.rover_env_cfg import RoverEnvCfg
from rover_envs.envs.navigation.utils.articulation import prepare_rover_contact_sensors

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor,
                         torch.Tensor, torch.Tensor, dict]


class RoverEnv(ManagerBasedRLEnv):
    """ Rover environment.

    Note:
        This is a placeholder class for the rover environment. That is, this class is not yet implemented."""

    def __init__(self, cfg: RoverEnvCfg, **kwargs):

        super().__init__(cfg, **kwargs)
        # Populate PhysX report pairs so they are explicit in stage/UI and available for contact filtering.
        prepare_rover_contact_sensors(cfg.scene.contact_sensor.filter_prim_paths_expr)


        # Reset all environments
        # self._reset_idx(env_ids) 


        # Get the terrain and change the origin

        self.global_step_counter = 0

    def _reset_idx(self, idx: torch.Tensor):
        """Reset the environment at the given indices.

        Note:
            This function inherits from :meth:`isaaclab.envs.manager_based_rl_env.ManagerBasedRLEnv._reset_idx`.
            This is done because SKRL requires the "episode" key in the extras dict to be present in order to log.
        Args:
            idx (torch.Tensor): Indices of the environments to reset.
        """
        super()._reset_idx(idx)

        # Done this way because SKRL requires the "episode" key in the extras dict to be present in order to log.
        self.extras["episode"] = self.extras["log"]

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Use upstream Isaac Lab stepping/rendering logic."""
        return super().step(action)
