"""SKRL agent implementations for the general agent factory."""

from typing import Any, Dict, Optional

from gymnasium.spaces.box import Box
from isaaclab.envs import ManagerBasedRLEnv
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.rpo import RPO, RPO_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.agents.torch.trpo import TRPO, TRPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

from ...utils.config import convert_skrl_cfg
from ..models import ModelFactory
from . import AgentFactory


@AgentFactory.register_agent("PPO")
def PPO_agent(experiment_cfg, observation_space: Box, action_space: Box, env: ManagerBasedRLEnv, models: Optional[Dict] = None):
    """Create a PPO agent with the specified configuration."""
    # Define memory size
    memory_size = experiment_cfg["agent"]["rollouts"]
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Get the models - use provided models or create them
    if models is None:
        models = ModelFactory.create_models(env, observation_space, action_space, experiment_cfg["models"])

    # Agent cfg
    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"]))

    # Create the agent
    agent = PPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
    )
    return agent


@AgentFactory.register_agent("TRPO")
def TRPO_agent(experiment_cfg, observation_space: Box, action_space: Box, env: ManagerBasedRLEnv, models: Optional[Dict] = None):
    """Create a TRPO agent with the specified configuration."""
    # Define memory size
    memory_size = experiment_cfg["agent"]["rollouts"]
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Get the models - use provided models or create them
    if models is None:
        models = ModelFactory.create_models(env, observation_space, action_space, experiment_cfg["models"])

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
    return agent


@AgentFactory.register_agent("RPO")
def RPO_agent(experiment_cfg, observation_space: Box, action_space: Box, env: ManagerBasedRLEnv, models: Optional[Dict] = None):
    """Create a RPO agent with the specified configuration."""
    # Define memory size
    memory_size = experiment_cfg["agent"]["rollouts"]
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Get the models - use provided models or create them
    if models is None:
        models = ModelFactory.create_models(env, observation_space, action_space, experiment_cfg["models"])

    # Agent cfg
    agent_cfg = RPO_DEFAULT_CONFIG.copy()
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"]))

    # Create the agent
    agent = RPO(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
    )
    return agent


@AgentFactory.register_agent("SAC")
def SAC_agent(experiment_cfg, observation_space: Box, action_space: Box, env: ManagerBasedRLEnv, models: Optional[Dict] = None):
    """Create a SAC agent with the specified configuration."""
    # Define memory size
    memory_size = experiment_cfg["agent"]["memory_size"]
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Get the models - use provided models or create them
    if models is None:
        models = ModelFactory.create_models(env, observation_space, action_space, experiment_cfg["models"])

    # Agent cfg
    agent_cfg = SAC_DEFAULT_CONFIG.copy()
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"]))

    # Create the agent
    agent = SAC(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
    )
    return agent


@AgentFactory.register_agent("TD3")
def TD3_agent(experiment_cfg, observation_space: Box, action_space: Box, env: ManagerBasedRLEnv, models: Optional[Dict] = None):
    """Create a TD3 agent with the specified configuration."""
    # Define memory size
    memory_size = experiment_cfg["agent"]["memory_size"]
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Get the models - use provided models or create them
    if models is None:
        models = ModelFactory.create_models(env, observation_space, action_space, experiment_cfg["models"])

    # Agent cfg
    agent_cfg = TD3_DEFAULT_CONFIG.copy()
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"]))

    # Create the agent
    agent = TD3(
        models=models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
    )
    return agent
