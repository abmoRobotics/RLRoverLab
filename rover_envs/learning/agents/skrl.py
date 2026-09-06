"""SKRL agent implementations for the general agent factory."""

import copy
import dataclasses
from typing import Any, Dict, Optional

from gymnasium.spaces.box import Box
from isaaclab.envs import ManagerBasedRLEnv
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.agents.torch.rpo import RPO, RPO_CFG
from skrl.agents.torch.sac import SAC, SAC_CFG
from skrl.agents.torch.td3 import TD3, TD3_CFG
from skrl.agents.torch.trpo import TRPO, TRPO_CFG
from skrl.memories.torch import RandomMemory

from ...utils.config import convert_skrl_cfg
from ..models import ModelFactory
from . import AgentFactory


def _build_agent_cfg(default_cfg_cls: type, experiment_agent_cfg: Dict[str, Any]) -> Any:
    """Build and validate a skrl 2.x dataclass config object from defaults plus user overrides."""
    if not dataclasses.is_dataclass(default_cfg_cls):
        raise TypeError(
            f"Expected a skrl 2.x dataclass config class, got: {default_cfg_cls!r}"
        )

    agent_cfg = dataclasses.asdict(default_cfg_cls())
    user_cfg = convert_skrl_cfg(copy.deepcopy(experiment_agent_cfg))
    agent_cfg.update(user_cfg)
    # strict skrl 2.x validation: invalid/legacy keys raise immediately
    return default_cfg_cls(**agent_cfg)


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
    agent_cfg = _build_agent_cfg(PPO_CFG, experiment_cfg["agent"])

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
    agent_cfg = _build_agent_cfg(TRPO_CFG, experiment_cfg["agent"])

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
    agent_cfg = _build_agent_cfg(RPO_CFG, experiment_cfg["agent"])

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
    memory_size = experiment_cfg.get("memory_size", 100000)
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Get the models - use provided models or create them
    if models is None:
        models = ModelFactory.create_models(env, observation_space, action_space, experiment_cfg["models"])

    # Agent cfg
    agent_cfg = _build_agent_cfg(SAC_CFG, experiment_cfg["agent"])

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
    memory_size = experiment_cfg.get("memory_size", 100000)
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Get the models - use provided models or create them
    if models is None:
        models = ModelFactory.create_models(env, observation_space, action_space, experiment_cfg["models"])

    # Agent cfg
    agent_cfg = _build_agent_cfg(TD3_CFG, experiment_cfg["agent"])

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
