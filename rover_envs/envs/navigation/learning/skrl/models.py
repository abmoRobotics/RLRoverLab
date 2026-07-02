import math

import torch
import torch.nn as nn

from skrl.models.torch.base import Model as BaseModel
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.gaussian import GaussianMixin
from skrl.utils.spaces.torch import unflatten_tensorized_space

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
        x = x.reshape(-1, 1, self.heightmap_size, self.heightmap_size)
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.reshape(-1, self.conv_out_features)
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





### CD ###

_CD_FRAME_HEIGHT = 180
_CD_FRAME_WIDTH = 320
_CD_FRAME_CHANNELS = 1
_CD_FRAME_FEATURES = _CD_FRAME_HEIGHT * _CD_FRAME_WIDTH * _CD_FRAME_CHANNELS


class ConvCDmapEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        encoder_features=[16, 32],
        encoder_activation="leaky_relu",
        input_shape=None,
    ):
        super().__init__()
        self.input_height, self.input_width, self.input_channels = self._resolve_input_shape(in_channels, input_shape)
        kernel_size = 3
        stride = 1
        padding = 1
        self.encoder_layers = nn.ModuleList()
        in_channels = self.input_channels
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
        flatten_size = [self.input_height, self.input_width]
        for _ in encoder_features:
            w = (flatten_size[0] - kernel_size + 2 * padding) // stride + 1
            h = (flatten_size[1] - kernel_size + 2 * padding) // stride + 1
            w = (w - 2) // 2 + 1
            h = (h - 2) // 2 + 1
            flatten_size = [w, h]
            if w <= 0 or h <= 0:
                raise ValueError(
                    f"CD encoder input shape {(self.input_height, self.input_width, self.input_channels)} "
                    f"is too small for {len(encoder_features)} pooling stages"
                )

        self.conv_out_features = out_channels * flatten_size[0] * flatten_size[1]
        features = [80, 60]

        self.mlps = nn.ModuleList()
        in_channels = self.conv_out_features
        for feature in features:
            self.mlps.append(nn.Linear(in_channels, feature))
            self.mlps.append(get_activation(encoder_activation))
            in_channels = feature

        self.out_features = features[-1]

    @staticmethod
    def _resolve_input_shape(in_channels, input_shape):
        input_size = int(in_channels)
        if input_shape is not None:
            dims = tuple(int(dim) for dim in input_shape)
            if len(dims) == 3:
                height, width, channels = dims
            elif len(dims) == 2:
                height, width = dims
                channels = 1
            elif len(dims) == 1:
                side = math.isqrt(dims[0])
                if side * side != dims[0]:
                    raise ValueError(f"Cannot infer a 2D CD frame shape from flat input shape {dims}")
                height = width = side
                channels = 1
            else:
                raise ValueError(f"Unsupported CD encoder input shape: {dims}")

            if height * width * channels != input_size:
                raise ValueError(
                    f"CD encoder input shape {dims} has {height * width * channels} values, "
                    f"but encoder_input_size is {input_size}"
                )
            return height, width, channels

        side = math.isqrt(input_size)
        if input_size == _CD_FRAME_FEATURES:
            return _CD_FRAME_HEIGHT, _CD_FRAME_WIDTH, _CD_FRAME_CHANNELS
        if side * side != input_size:
            raise ValueError(
                f"Cannot infer a square CD frame from encoder_input_size={input_size}. "
                "Pass encoder_input_shape, for example (180, 320, 1)."
            )
        return side, side, 1

    def forward(self, x):
        expected_features = self.input_height * self.input_width * self.input_channels
        if x.shape[-1] != expected_features:
            raise ValueError(f"Expected {expected_features} CD frame features, got {x.shape[-1]}")

        x = x.reshape(-1, self.input_height, self.input_width, self.input_channels)
        x = x.permute(0, 3, 1, 2).contiguous()
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.reshape(-1, self.conv_out_features)
        for layer in self.mlps:
            x = layer(x)
        return x

#### value and policy

def _infer_cd_encoder_input_size(observation_space, mlp_input_size, encoder_input_size):
    observation_shape = getattr(observation_space, "shape", None)
    if observation_shape is not None:
        inferred_size = math.prod(observation_shape) - mlp_input_size
        if inferred_size > 0:
            return inferred_size
    if encoder_input_size == _CD_FRAME_HEIGHT:
        return _CD_FRAME_FEATURES
    return encoder_input_size


@register_model("GaussianPolicyCDConv")
class GaussianPolicyCDConv(GaussianMixin, BaseModel):
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
        encoder_input_shape=None,
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
        self.encoder_input_size = _infer_cd_encoder_input_size(
            observation_space,
            self.mlp_input_size,
            encoder_input_size,
        )

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.encoder = ConvCDmapEncoder(
                self.encoder_input_size,
                encoder_layers,
                encoder_activation,
                input_shape=encoder_input_shape,
            )
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
        observations = states["observations"]

        if self.encoder_input_size is None:
            x = observations
        else:
            x = observations[:, :self.mlp_input_size]
            encoder_input = observations[:, self.mlp_input_size:self.mlp_input_size + self.encoder_input_size]
            encoder_output = self.encoder(encoder_input)
            x = torch.cat([x, encoder_output], dim=1)

        # states = self.tensor_to_space(states["observations"], self.observation_space)
        # encoder_output = self.encoder(states["height_scan"])
        # x = torch.cat([states["actions"], states["distance"], states["heading"], states["angle_diff"], encoder_output], dim=1)

        # Compute the output of the MLP.
        for layer in self.mlp:
            x = layer(x)

        return x, {"log_std": self.log_std_parameter}

@register_model("ValueNetworkCDConv")
class ValueNetworkCDConv(DeterministicMixin, BaseModel):
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
        encoder_input_shape=None,
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
        self.encoder_input_size = _infer_cd_encoder_input_size(
            observation_space,
            self.mlp_input_size,
            encoder_input_size,
        )

        in_channels = self.mlp_input_size
        if self.encoder_input_size is not None:
            self.encoder = ConvCDmapEncoder(
                self.encoder_input_size,
                encoder_layers,
                encoder_activation,
                input_shape=encoder_input_shape,
            )
            in_channels += self.encoder.out_features

        self.mlp = nn.ModuleList()

        action_space = action_space.shape[0]
        for feature in mlp_layers:
            self.mlp.append(nn.Linear(in_channels, feature))
            self.mlp.append(get_activation(mlp_activation))
            in_channels = feature

        self.mlp.append(nn.Linear(in_channels, 1))

    def compute(self, states, role="actor"):
        observations = states["observations"]
        if self.encoder_input_size is None:
            x = observations
        else:
            x = observations[:, :self.mlp_input_size]
            encoder_input = observations[:, self.mlp_input_size:self.mlp_input_size + self.encoder_input_size]
            encoder_output = self.encoder(encoder_input)
            x = torch.cat([x, encoder_output], dim=1)

        for layer in self.mlp:
            x = layer(x)

        return x, {}
