from gym.spaces import Box
from omni.isaac.lab.envs import ManagerBasedRLEnv

from rover_envs.envs.navigation.learning.skrl.agents import PPO_agent, RPO_agent, SAC_agent, TD3_agent, TRPO_agent


def get_agent(agent: str, env: ManagerBasedRLEnv, observation_space: Box, action_space: Box, experiment_cfg):
    """
    Function to get the agent.

    Args:
        agent (str): The agent.

    Returns:
        Agent: The agent.
    """
    if agent == "PPO":
        return PPO_agent(experiment_cfg, observation_space, action_space, env)
    if agent == "TRPO":
        return TRPO_agent(experiment_cfg, observation_space, action_space, env)
    if agent == "RPO":
        return RPO_agent(experiment_cfg, observation_space, action_space, env)
    if agent == "TD3":
        return TD3_agent(experiment_cfg, observation_space, action_space, env)
    if agent == "SAC":
        return SAC_agent(experiment_cfg, observation_space, action_space, env)
    raise ValueError(f"Agent {agent} not supported.")
