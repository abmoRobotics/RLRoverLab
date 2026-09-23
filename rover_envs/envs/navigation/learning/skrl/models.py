import torch
import torch.nn as nn
from skrl.models.torch.base import Model as BaseModel
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.gaussian import GaussianMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space

from rover_envs.envs.navigation.learning.encoders import ConvHeightmapEncoder, HeightmapEncoder
from rover_envs.learning.models import MODEL_REGISTRY, get_activation, register_model


@register_model("GaussianPolicyConv")
class GaussianPolicyConv(GaussianMixin, BaseModel):
    """Gaussian policy network with convolutional encoder (outputs action distributions)."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=5,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self, clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
        )

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.encoder = ConvHeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += self.encoder.out_features

        self.mlp = nn.ModuleList()

        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        action_space = action_space.shape[0]
        self.mlp.append(nn.Linear(in_channels, action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def compute(self, states, role="actor"):
        # Split the states into proprioception and heightmap if the heightmap is used.

        if self.encoder_input_size is None:
            x = states["observations"]
        else:
            encoder_output = self.encoder(states["observations"][:, self.mlp_input_size - 1:-1])
            x = states["observations"][:, 0:self.mlp_input_size]
            x = torch.cat([x, encoder_output], dim=1)

        # states = self.tensor_to_space(states["observations"], self.observation_space)
        # encoder_output = self.encoder(states["height_scan"])
        # x = torch.cat([states["actions"], states["distance"], states["heading"], states["angle_diff"], encoder_output], dim=1)

        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)

        return x, {"log_std": self.log_std_parameter}

@register_model("ValueNetworkConv")
class ValueNetworkConv(DeterministicMixin, BaseModel):
    """Value network model with convolutional encoder for state value estimation."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.encoder = ConvHeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += self.encoder.out_features

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        if self.encoder_input_size is None:
            x = states["observations"]
        else:
            x = states["observations"][:, :self.mlp_input_size]
            encoder_output = self.encoder(states["observations"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}

@register_model("DeterministicPolicyConv")
class DeterministicPolicyConv(DeterministicMixin, BaseModel):
    """Deterministic policy network with convolutional encoder (outputs specific actions)."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[8, 16, 32, 64],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the deterministic actor model with convolutional encoder.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_layers (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.encoder = ConvHeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += self.encoder.out_features

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, action_space))

    def compute(self, states, role="actor"):
        if self.encoder_input_size is None:
            x = states["observations"]
        else:
            x = states["observations"][:, :self.mlp_input_size]
            encoder_output = self.encoder(states["observations"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}


@register_model("CriticConv")
class CriticConv(DeterministicMixin, BaseModel):
    """Critic model with convolutional encoder."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[8, 16, 32, 64],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the critic model with convolutional encoder.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_layers (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.encoder = ConvHeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += self.encoder.out_features

        self.mlp = nn.ModuleList()

        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        if self.encoder_input_size is None:
            x = torch.cat([states["observations"], states["taken_actions"]], dim=1)
        else:
            x = states["observations"][:, :self.mlp_input_size]
            encoder_output = self.encoder(states["observations"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output, states["taken_actions"]], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}


# create agent that has has to layered fully connected encoder for the 1000 resnet features, the other five features will skip the encoder and go directly to the 3 layered mlp
@register_model("GaussianPolicyResnet")
class GaussianPolicyResnet(GaussianMixin, BaseModel):
    """Gaussian policy network with ResNet encoder (outputs action distributions)."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=5,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self, clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
        )

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = observation_space.shape[0] - mlp_input_size
        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.encoder = HeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels = self.mlp_input_size + encoder_layers[-1]

        self.mlp = nn.ModuleList()

        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        action_space = action_space.shape[0]
        self.mlp.append(nn.Linear(in_channels, action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def compute(self, states, role="actor"):
        if self.encoder_input_size is None:
            x = states["observations"]
        else:
            x = states["observations"][:, :self.mlp_input_size]
            encoder_output = self.encoder(states["observations"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)
        return x, {"log_std": self.log_std_parameter}
    

@register_model("ValueNetworkResnet")
class ValueNetworkResnet(DeterministicMixin, BaseModel):
    """Value network model with ResNet encoder for state value estimation."""
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = observation_space.shape[0] - mlp_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.encoder = HeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels = self.mlp_input_size + encoder_layers[-1]

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        if self.encoder_input_size is None:
            x = states["observations"]
        else:
            x = states["observations"][:, :self.mlp_input_size]
            encoder_output = self.encoder(states["observations"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}
    

@register_model("GaussianPolicyConvDict")
class GaussianPolicyConvDict(GaussianMixin, BaseModel):
    """Gaussian policy network with convolutional encoder (outputs action distributions)."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=5,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        GaussianMixin.__init__(
            self, clip_actions=False, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
        )

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.encoder = ConvHeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += self.encoder.out_features

        self.mlp = nn.ModuleList()

        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        action_space = action_space.shape[0]
        self.mlp.append(nn.Linear(in_channels, action_space))
        self.mlp.append(nn.Tanh())
        self.log_std_parameter = nn.Parameter(torch.zeros(action_space))

    def compute(self, states, role="actor"):

        states = unflatten_tensorized_space(self.observation_space, states["observations"])
        encoder_output = self.encoder(states["height_scan"])
        x = torch.cat([states["actions"], states["distance"], states["heading"], states["angle_diff"], encoder_output], dim=1)

        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)

        return x, {"log_std": self.log_std_parameter}

@register_model("ValueNetworkConvDict")
class ValueNetworkConvDict(DeterministicMixin, BaseModel):
    """Value network model with convolutional encoder for state value estimation."""

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        mlp_input_size=4,
        mlp_layers=[256, 160, 128],
        mlp_activation="leaky_relu",
        encoder_input_size=None,
        encoder_layers=[80, 60],
        encoder_activation="leaky_relu",
        **kwargs,
    ):
        """Initialize the Gaussian neural network model.

        Args:
            observation_space (gym.spaces.Space): The observation space of the environment.
            action_space (gym.spaces.Space): The action space of the environment.
            device (torch.device): The device to use for computation.
            encoder_features (list): The number of features for each encoder layer.
            encoder_activation (str): The activation function to use for each encoder layer.
        """
        BaseModel.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.mlp_input_size = mlp_input_size
        self.encoder_input_size = encoder_input_size

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.encoder = ConvHeightmapEncoder(self.encoder_input_size, encoder_layers, encoder_activation)
            in_channels += self.encoder.out_features

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        
        states = unflatten_tensorized_space(self.observation_space, states["observations"])
        encoder_output = self.encoder(states["height_scan"])
        x = torch.cat([states["actions"], states["distance"], states["heading"], states["angle_diff"], encoder_output], dim=1)


        for layer in self.mlp:
            x = layer(x)

        return x, {}
