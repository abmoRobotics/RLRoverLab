from gymnasium.spaces.box import Box
from isaaclab.envs import ManagerBasedRLEnv

from rover_envs.envs.navigation.learning.skrl.models import (MODEL_REGISTRY, Critic, DeterministicActor,
                                                             DeterministicNeuralNetwork, DeterministicNeuralNetworkConv,
                                                             GaussianNeuralNetwork, GaussianNeuralNetworkConv)


def get_models2(
    env: ManagerBasedRLEnv,
    observation_space: Box,
    action_space: Box,
    agent_model_config: dict,
):
    """
    Creates and returns neural network models based on the provided configuration.

    Args:
        env (ManagerBasedRLEnv): The environment instance.
        observation_space (Box): The observation space.
        action_space (Box): The action space.
        agent_model_config (dict): Configuration for the agent's models.
            Example:
            {
                "policy": {"type": "GaussianNeuralNetwork", "params": {"mlp_input_size": 5, ...}},
                "value": {"type": "DeterministicNeuralNetwork", "params": {"mlp_input_size": 5, ...}}
            }

    Returns:
        dict: A dictionary containing the instantiated models.
    """
    models = {}

    for model_key, config in agent_model_config.items():
        model_type_name = config.get("type")
        model_params = config.get("params", {}).copy()

        if not model_type_name:
            raise ValueError(f"Model type not specified for '{model_key}' in agent_model_config.")

        if model_type_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Model type '{model_type_name}' not found in registry. "
                f"Available models: {list(MODEL_REGISTRY.keys())}"
            )

        ModelClass = MODEL_REGISTRY[model_type_name]

        # Automatically determine encoder_input_size if not provided and encoder_layers are specified
        # This assumes the last observation term in the 'policy' group is for the encoder.
        if "encoder_input_size" not in model_params and model_params.get("encoder_layers"):
            try:
                if hasattr(env.unwrapped, "observation_manager"):
                    policy_obs_terms = env.unwrapped.observation_manager.group_obs_term_dim.get("policy")
                    if policy_obs_terms and len(policy_obs_terms) > 0:
                        model_params["encoder_input_size"] = policy_obs_terms[-1][0]
                    else:
                        model_params["encoder_input_size"] = None  # No encoder if obs structure doesn't provide it
                else:
                    model_params["encoder_input_size"] = None
            except (AttributeError, KeyError, IndexError):
                model_params["encoder_input_size"] = None
        elif "encoder_layers" not in model_params:  # If no encoder_layers, ensure encoder_input_size is None or not passed
            model_params["encoder_input_size"] = None


        # Instantiate the model
        try:
            models[model_key] = ModelClass(
                observation_space=observation_space,
                action_space=action_space,
                device=env.device,
                **model_params,
            )
        except TypeError as e:
            raise TypeError(f"Error instantiating model '{model_type_name}' for '{model_key}' with params {model_params}: {e}")


    return models

def get_models(agent: str, env: ManagerBasedRLEnv, observation_space: Box, action_space: Box, conv: bool = False):
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
        if conv:
            return get_model_gaussian_conv(env, observation_space, action_space)
        return get_model_gaussian(env, observation_space, action_space)
    if agent == "TRPO":
        return get_model_gaussian(env, observation_space, action_space)
    if agent == "RPO":
        return get_model_gaussian(env, observation_space, action_space)
    if agent == "SAC":
        return get_model_double_critic_deterministic(env, observation_space, action_space)
    if agent == "TD3":
        return get_model_double_critic_deterministic(env, observation_space, action_space)

    raise ValueError(f"Agent {agent} not supported.")


def get_model_gaussian(env: ManagerBasedRLEnv, observation_space: Box, action_space: Box):
    models = {}
    encoder_input_size = env.unwrapped.observation_manager.group_obs_term_dim["policy"][-1][0]

    mlp_input_size = 5

    models["policy"] = GaussianNeuralNetwork(
        observation_space=observation_space,
        action_space=action_space,
        device=env.unwrapped.device,
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
        device=env.unwrapped.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
    )
    return models


def get_model_gaussian_conv(env: ManagerBasedRLEnv, observation_space: Box, action_space: Box):
    models = {}
    encoder_input_size = env.unwrapped.observation_manager.group_obs_term_dim["policy"][-1][0]

    mlp_input_size = 5

    models["policy"] = GaussianNeuralNetworkConv(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[8, 16, 32, 64],
        encoder_activation="leaky_relu",
    )
    models["value"] = DeterministicNeuralNetworkConv(
        observation_space=observation_space,
        action_space=action_space,
        device=env.device,
        mlp_input_size=mlp_input_size,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=encoder_input_size,
        encoder_layers=[8, 16, 32, 64],
        encoder_activation="leaky_relu",
    )
    return models


def get_model_double_critic_deterministic(env: ManagerBasedRLEnv, observation_space: Box, action_space: Box):
    models = {}
    encoder_input_size = env.unwrapped.observation_manager.group_obs_term_dim["policy"][-1][0]

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
