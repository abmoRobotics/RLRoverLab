"""Base model utilities and activation functions."""

import torch.nn as nn


def get_activation(activation_name):
    """Get the activation function by name.

    Args:
        activation_name (str): Name of the activation function.

    Returns:
        nn.Module: The activation function module.
    """
    activation_fns = {
        "leaky_relu": nn.LeakyReLU(inplace=True),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "relu6": nn.ReLU6(),
        "selu": nn.SELU(),
    }
    if activation_name not in activation_fns:
        raise ValueError(f"Activation function {activation_name} not supported.")
    return activation_fns[activation_name]
