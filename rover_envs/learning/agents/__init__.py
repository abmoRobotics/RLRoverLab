"""General agent factory for creating RL agents across different domains."""

from typing import Any, Dict, Optional

from gymnasium.spaces import Box
from isaaclab.envs import ManagerBasedRLEnv

from ..models import ModelFactory


class AgentFactory:
    """Factory class for creating RL agents with automatic model configuration."""

    # Registry for agent creation functions
    _agent_registry = {}

    @classmethod
    def register_agent(cls, agent_name: str):
        """Decorator to register an agent creation function."""
        def decorator(func):
            cls._agent_registry[agent_name.upper()] = func
            return func
        return decorator

    @classmethod
    def create_agent(
        cls,
        agent_name: str,
        env: ManagerBasedRLEnv,
        observation_space: Box,
        action_space: Box,
        experiment_cfg: Dict[str, Any],
        model_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Create an agent with the specified configuration.

        Args:
            agent_name (str): Name of the agent (e.g., "PPO", "SAC", etc.)
            env (ManagerBasedRLEnv): The environment instance
            observation_space (Box): The observation space
            action_space (Box): The action space
            experiment_cfg (Dict): Experiment configuration
            model_config (Optional[Dict]): Model configuration override
            **kwargs: Additional arguments passed to the agent constructor

        Returns:
            Agent: The configured agent instance
        """
        agent_name = agent_name.upper()

        if agent_name not in cls._agent_registry:
            available_agents = list(cls._agent_registry.keys())
            raise ValueError(f"Agent '{agent_name}' not registered. Available agents: {available_agents}")

        # Use model factory to create models if model_config is provided
        models = None
        if model_config:
            models = ModelFactory.create_models(env, observation_space, action_space, model_config)

        # Get the agent creation function and call it
        agent_creator = cls._agent_registry[agent_name]
        return agent_creator(
            experiment_cfg=experiment_cfg,
            observation_space=observation_space,
            action_space=action_space,
            env=env,
            models=models,
            **kwargs
        )

    @classmethod
    def get_available_agents(cls):
        """Get list of available agent types."""
        return list(cls._agent_registry.keys())


# Import SKRL agents to register them
from . import skrl


# Convenience function for backward compatibility
def create_agent(agent_name: str, env: ManagerBasedRLEnv, experiment_cfg: Dict[str, Any], **kwargs):
    """
    Convenience function to create an agent using the AgentFactory.

    Args:
        agent_name (str): Name of the agent (e.g., "PPO", "SAC", etc.)
        env (ManagerBasedRLEnv): The environment instance
        experiment_cfg (Dict): Experiment configuration
        **kwargs: Additional arguments passed to the agent constructor

    Returns:
        Agent: The configured agent instance
    """
    return AgentFactory.create_agent(
        agent_name=agent_name,
        env=env,
        observation_space=env.observation_space,
        action_space=env.action_space,
        experiment_cfg=experiment_cfg,
        **kwargs
    )
