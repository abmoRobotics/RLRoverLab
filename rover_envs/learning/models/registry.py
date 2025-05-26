"""Model registry for neural network models."""

MODEL_REGISTRY = {}


def register_model(name):
    """Decorator to register a model class in the MODEL_REGISTRY.

    Args:
        name (str): The name to register the model under.

    Returns:
        decorator: A decorator function that registers the class.
    """
    def decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model with name '{name}' already registered.")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator
