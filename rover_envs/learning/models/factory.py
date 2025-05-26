"""Model factory for creating neural network models from configuration."""

from typing import Any, Dict

from gymnasium.spaces import Box
from isaaclab.envs import ManagerBasedRLEnv

from .registry import MODEL_REGISTRY


class ModelFactory:
    """Factory class for creating neural network models based on configuration."""

    @staticmethod
    def create_models(
        env: ManagerBasedRLEnv,
        observation_space: Box,
        action_space: Box,
        agent_model_config: Dict[str, Any],
    ) -> Dict[str, Any]:
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
            # Skip non-model configuration items (like 'separate')
            if not isinstance(config, dict):
                continue

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
