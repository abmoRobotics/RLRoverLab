from gym.spaces import Box
from omni.isaac.lab.envs import ManagerBasedRLEnv

from rover_envs.learning.train.ppo import PPO_agent
from rover_envs.learning.train.rpo import RPO_agent
from rover_envs.learning.train.sac import SAC_agent
from rover_envs.learning.train.td3 import TD3_agent
from rover_envs.learning.train.trpo import TRPO_agent


def get_agent(
        agent: str,
        env: ManagerBasedRLEnv,
        observation_space: Box,
        action_space: Box,
        experiment_cfg,
        conv: bool = False):
    """
    Function to get the agent.

    Args:
        agent (str): The agent.

    Returns:
        Agent: The agent.
    """
    if agent == "PPO":
        return PPO_agent(experiment_cfg, observation_space, action_space, env, conv)
    if agent == "TRPO":
        return TRPO_agent(experiment_cfg, observation_space, action_space, env)
    if agent == "RPO":
        return RPO_agent(experiment_cfg, observation_space, action_space, env)
    if agent == "TD3":
        return TD3_agent(experiment_cfg, observation_space, action_space, env)
    if agent == "SAC":
        return SAC_agent(experiment_cfg, observation_space, action_space, env)
    raise ValueError(f"Agent {agent} not supported.")
