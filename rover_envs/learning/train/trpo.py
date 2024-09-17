from gymnasium.spaces.box import Box
from omni.isaac.lab.envs import ManagerBasedRLEnv
from skrl.agents.torch.trpo import TRPO, TRPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

from rover_envs.learning.train.get_models import get_models
from rover_envs.utils.config import convert_skrl_cfg


def TRPO_agent(experiment_cfg, observation_space: Box, action_space: Box, env: ManagerBasedRLEnv):

    # Define memory size
    memory_size = experiment_cfg["agent"]["rollouts"]
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Get the models
    models = get_models("TRPO", env, observation_space, action_space)

    # Agent cfg
    agent_cfg = TRPO_DEFAULT_CONFIG.copy()
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"]))

    # Create the agent
    agent = TRPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
    )
    return agent  # noqa R504
