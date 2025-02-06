from gymnasium.spaces.box import Box
from isaaclab.envs import ManagerBasedRLEnv
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

from rover_envs.learning.train.get_models import get_models
from rover_envs.utils.config import convert_skrl_cfg


def PPO_agent(experiment_cfg, observation_space: Box, action_space: Box, env: ManagerBasedRLEnv, conv):

    # Define memory size
    memory_size = experiment_cfg["agent"]["rollouts"]
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)
    # Get the models
    models = get_models("PPO", env, observation_space, action_space, conv)

    # Agent cfg
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"]))

    # Create the agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    return agent  # noqa R504
