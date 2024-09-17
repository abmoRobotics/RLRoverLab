
from gymnasium.spaces.box import Box
from omni.isaac.lab.envs import ManagerBasedRLEnv

from rover_envs.envs.navigation.learning.skrl.models import (Critic, DeterministicActor, DeterministicNeuralNetwork,
                                                             GaussianNeuralNetwork)


def get_models(agent: str, env: ManagerBasedRLEnv, observation_space: Box, action_space: Box):
    """
    Placeholder function for getting the models.

    Note:
        This function will be further improved in the future, by reading the model config from the experiment config.

    Args:
        agent (str): The agent.

    Returns:
        dict: A dictionary containing the models.
    """

    if agent == "PPO":
        return gaussian_model_skrl(env, observation_space, action_space)
    if agent == "TRPO":
        return gaussian_model_skrl(env, observation_space, action_space)
    if agent == "RPO":
        return gaussian_model_skrl(env, observation_space, action_space)
    if agent == "SAC":
        return double_critic_deterministic_model_skrl(env, observation_space, action_space)
    if agent == "TD3":
        return double_critic_deterministic_model_skrl(env, observation_space, action_space)

    raise ValueError(f"Agent {agent} not supported.")


def gaussian_model_skrl(env: ManagerBasedRLEnv, observation_space: Box, action_space: Box):
    models = {}
    encoder_input_size = env.observation_manager.group_obs_term_dim["policy"][-1][0]

    mlp_input_size = 5

    models["policy"] = GaussianNeuralNetwork(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    models["value"] = DeterministicNeuralNetwork(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    return models


def double_critic_deterministic_model_skrl(env: ManagerBasedRLEnv, observation_space: Box, action_space: Box):
    models = {}
    encoder_input_size = env.observation_manager.group_obs_term_dim["policy"][-1][0]

    mlp_input_size = 4

    models["policy"] = DeterministicActor(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    models["target_policy"] = DeterministicActor(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )

    models["critic_1"] = Critic(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    models["critic_2"] = Critic(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    models["target_critic_1"] = Critic(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    models["target_critic_2"] = Critic(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )

    return models
