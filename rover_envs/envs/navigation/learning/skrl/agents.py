from gymnasium.spaces.box import Box
from isaaclab.envs import ManagerBasedRLEnv
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.agents.torch.rpo import RPO, RPO_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.agents.torch.trpo import TRPO, TRPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

from rover_envs.envs.navigation.learning.skrl.configure_models import get_models
from rover_envs.utils.config import convert_skrl_cfg


def PPO_agent2(experiment_cfg, observation_space: Box, action_space: Box, env: ManagerBasedRLEnv, conv: bool):
    # ...
    memory_size = experiment_cfg["agent"]["rollouts"]
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Construct agent_model_config based on experiment_cfg and conv flag
    # This part requires you to define how your experiment_cfg stores these model configs.
    # Example: experiment_cfg might have a "model_configs" section:
    # "model_configs": {
    #   "PPO": {
    #     "dense": {
    #       "policy": {"type": "GaussianNeuralNetwork", "params": {"mlp_input_size": 5, ...}},
    #       "value": {"type": "DeterministicNeuralNetwork", "params": {"mlp_input_size": 5, ...}}
    #     },
    #     "conv": {
    #       "policy": {"type": "GaussianNeuralNetworkConv", "params": {"mlp_input_size": 5, ...}},
    #       "value": {"type": "DeterministicNeuralNetworkConv", "params": {"mlp_input_size": 5, ...}}
    #     }
    #   },
    #   "SAC": { ... }
    # }
    agent_name = "PPO"  # Or derive dynamically if this function handles multiple agents
    all_model_configs = experiment_cfg.get("model_configs", {})
    agent_specific_configs = all_model_configs.get(agent_name, {})

    config_key = "conv" if conv else "dense"
    current_agent_model_config = agent_specific_configs.get(config_key)

    if not current_agent_model_config:
        raise ValueError(
            f"Model configuration for agent '{agent_name}' with conv={conv} ('{config_key}')\
            not found in experiment_cfg."
        )

    # Get the models using the new factory method
    skrl_models = get_models(
        env=env,
        observation_space=observation_space,
        action_space=action_space,
        agent_model_config=current_agent_model_config,
    )

    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg.update(convert_skrl_cfg(experiment_cfg["agent"]))

    agent = PPO(
        models=skrl_models,
        memory=memory,
        cfg=agent_cfg,
        observation_space=env.observation_space,  # skrl PPO expects env.observation_space
        action_space=env.action_space,           # skrl PPO expects env.action_space
        device=env.device,
    )

    return agent # noqa R504


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


def RPO_agent(experiment_cfg, observation_space: Box, action_space: Box, env: ManagerBasedRLEnv):

    # Define memory size
    memory_size = experiment_cfg["agent"]["rollouts"]
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Get the models
    models = get_models("RPO", env, observation_space, action_space)

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
    return agent  # noqa R504


def SAC_agent(experiment_cfg, observation_space: Box, action_space: Box, env: ManagerBasedRLEnv):

    # Define memory size
    memory_size = experiment_cfg["agent"]["batch_size"] * 2
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Get the models
    models = get_models("SAC", env, observation_space, action_space)

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
    return agent  # noqa R504


def TD3_agent(experiment_cfg, observation_space: Box, action_space: Box, env: ManagerBasedRLEnv):

    # Define memory size
    memory_size = experiment_cfg["agent"]["batch_size"] * 2
    memory = RandomMemory(memory_size=memory_size, num_envs=env.num_envs, device=env.device)

    # Get the models
    models = get_models("TD3", env, observation_space, action_space)

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
    return agent  # noqa R504


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
