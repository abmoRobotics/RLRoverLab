import torch
import torch.nn as nn
from skrl.models.torch.base import Model as BaseModel
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.gaussian import GaussianMixin

# Import general model utilities
from rover_envs.learning.models import MODEL_REGISTRY, get_activation, register_model


class HeightmapEncoder(nn.Module):
    def __init__(self, in_channels, encoder_features=[80, 60], encoder_activation="leaky_relu"):
        super().__init__()
        self.encoder_layers = nn.ModuleList()
        for feature in encoder_features:
            self.encoder_layers.append(nn.Linear(in_channels, feature))
            self.encoder_layers.append(get_activation(encoder_activation))
            in_channels = feature

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x


class ConvHeightmapEncoder(nn.Module):
    def __init__(self, in_channels, encoder_features=[16, 32], encoder_activation="leaky_relu"):
        super().__init__()
        self.heightmap_size = torch.sqrt(torch.tensor(in_channels)).int()
        kernel_size = 3
        stride = 1
        padding = 1
        self.encoder_layers = nn.ModuleList()
        in_channels = 1  # 1 channel for heightmap
        for feature in encoder_features:
            self.encoder_layers.append(
                nn.Conv2d(in_channels, feature, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            self.encoder_layers.append(nn.BatchNorm2d(feature))
            # self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder_layers.append(get_activation(encoder_activation))
            self.encoder_layers.append(nn.Conv2d(feature, feature, kernel_size=kernel_size,
                                       stride=stride, padding=padding, bias=False))
            self.encoder_layers.append(nn.BatchNorm2d(feature))
            self.encoder_layers.append(get_activation(encoder_activation))
            self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature
        out_channels = in_channels
        flatten_size = [self.heightmap_size, self.heightmap_size]
        for _ in encoder_features:
            w = (flatten_size[0] - kernel_size + 2 * padding) // stride + 1
            h = (flatten_size[1] - kernel_size + 2 * padding) // stride + 1
            w = (w - 2) // 2 + 1
            h = (h - 2) // 2 + 1
            flatten_size = [w, h]

        self.conv_out_features = out_channels * flatten_size[0] * flatten_size[1]
        features = [80, 60]

        self.mlps = nn.ModuleList()
        in_channels = self.conv_out_features
        for feature in features:
            self.mlps.append(nn.Linear(in_channels, feature))
            self.mlps.append(get_activation(encoder_activation))
            in_channels = feature

        self.out_features = features[-1]

    def forward(self, x):
        # x is a flattened heightmap, reshape it to 2D
        x = x.view(-1, 1, self.heightmap_size, self.heightmap_size)
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.view(-1, self.conv_out_features)
        for layer in self.mlps:
            x = layer(x)
        return x













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
        BaseModel.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
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
            x = states["states"]
        else:
            encoder_output = self.encoder(states["states"][:, self.mlp_input_size - 1:-1])
            x = states["states"][:, 0:self.mlp_input_size]
            x = torch.cat([x, encoder_output], dim=1)

        # states = self.tensor_to_space(states["states"], self.observation_space)
        # encoder_output = self.encoder(states["height_scan"])
        # x = torch.cat([states["actions"], states["distance"], states["heading"], states["angle_diff"], encoder_output], dim=1)

        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)

        return x, self.log_std_parameter, {}

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
        BaseModel.__init__(self, observation_space, action_space, device)
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
            x = states["states"]
        else:
            x = states["states"][:, :self.mlp_input_size]
            encoder_output = self.encoder(states["states"][:, self.mlp_input_size - 1:-1])
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
        BaseModel.__init__(self, observation_space, action_space, device)
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
            x = states["states"]
        else:
            x = states["states"][:, :self.mlp_input_size]
            encoder_output = self.encoder(states["states"][:, self.mlp_input_size - 1:-1])
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
        BaseModel.__init__(self, observation_space, action_space, device)
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
            x = torch.cat([states["states"], states["taken_actions"]], dim=1)
        else:
            x = states["states"][:, :self.mlp_input_size]
            encoder_output = self.encoder(states["states"][:, self.mlp_input_size - 1:-1])
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
        BaseModel.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
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
            x = states["states"]
        else:
            x = states["states"][:, :self.mlp_input_size]
            encoder_output = self.encoder(states["states"][:, self.mlp_input_size - 1:-1])
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)
        return x, self.log_std_parameter, {}
    

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
        BaseModel.__init__(self, observation_space, action_space, device)
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
            x = states["states"]
        else:
            x = states["states"][:, :self.mlp_input_size]
            encoder_output = self.encoder(states["states"][:, self.mlp_input_size - 1:-1])
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
        BaseModel.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions=True, clip_log_std=True, min_log_std=-20.0, max_log_std=2.0, reduction="sum"
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

        states = self.tensor_to_space(states["states"], self.observation_space)
        encoder_output = self.encoder(states["height_scan"])
        x = torch.cat([states["actions"], states["distance"], states["heading"], states["angle_diff"], encoder_output], dim=1)

        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)

        return x, self.log_std_parameter, {}

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
        BaseModel.__init__(self, observation_space, action_space, device)
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
        
        states = self.tensor_to_space(states["states"], self.observation_space)
        encoder_output = self.encoder(states["height_scan"])
        x = torch.cat([states["actions"], states["distance"], states["heading"], states["angle_diff"], encoder_output], dim=1)


        for layer in self.mlp:
            x = layer(x)

        return x, {}