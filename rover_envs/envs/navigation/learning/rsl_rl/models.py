"""RSL-RL model matching the rover's skrl height-map policy and value networks."""

from __future__ import annotations

import copy
from math import isqrt

import torch
from torch import nn
from torch.distributions import Normal

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import GaussianDistribution

from rover_envs.envs.navigation.learning.encoders import ConvHeightmapEncoder


class ClippedGaussianDistribution(GaussianDistribution):
    """Use the same log-standard-deviation limits as the skrl rover policy."""

    def update(self, mlp_output: torch.Tensor) -> None:
        log_std = self.log_std_param.clamp(-20.0, 2.0)
        self._distribution = Normal(mlp_output, log_std.exp().expand_as(mlp_output))


class RoverConvModel(MLPModel):
    """Encode the flat height map before the policy/value MLP."""

    _scalar_dim = 5

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims,
        activation,
        obs_normalization=False,
        distribution_cfg=None,
    ):
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
            obs_normalization, distribution_cfg,
        )
        heightmap_dim = self.obs_dim - self._scalar_dim
        if heightmap_dim <= 0 or isqrt(heightmap_dim) ** 2 != heightmap_dim:
            raise ValueError(f"Expected 5 scalar observations and a square height map, got {self.obs_dim} values")

        self.encoder = ConvHeightmapEncoder(heightmap_dim, [8, 16, 32, 64], "leaky_relu")
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = MLP(
            self._scalar_dim + self.encoder.out_features,
            mlp_output_dim,
            hidden_dims,
            activation,
            last_activation="tanh" if self.distribution is not None else None,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def get_latent(self, obs, masks=None, hidden_state=None):
        flat = super().get_latent(obs, masks, hidden_state)
        # Match the current skrl policy/value models' height-map slice exactly.
        heightmap_input = flat[:, self._scalar_dim - 1:-1]
        # skrl evaluates BatchNorm during collection and trains it during PPO updates.
        # RSL-RL leaves the model in train mode while collecting under inference_mode.
        if torch.is_inference_mode_enabled() and self.encoder.training:
            self.encoder.eval()
            try:
                heightmap = self.encoder(heightmap_input)
            finally:
                self.encoder.train()
        else:
            heightmap = self.encoder(heightmap_input)
        return torch.cat((flat[:, :self._scalar_dim], heightmap), dim=-1)

    def as_jit(self) -> nn.Module:
        return _ExportRoverConvModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _ExportRoverConvModel(self)


class _ExportRoverConvModel(nn.Module):
    """Export the encoder and MLP together as a deterministic policy."""

    def __init__(self, model: RoverConvModel):
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder = copy.deepcopy(model.encoder)
        # The shared encoder stores these dimensions as scalar tensors. TorchScript
        # requires Python ints for module constants such as Linear.in_features.
        self.encoder.heightmap_size = int(self.encoder.heightmap_size)
        self.encoder.conv_out_features = int(self.encoder.conv_out_features)
        for layer in self.encoder.mlps:
            if isinstance(layer, nn.Linear):
                layer.in_features = int(layer.in_features)
                layer.out_features = int(layer.out_features)
        self.mlp = copy.deepcopy(model.mlp)
        self.deterministic_output = (
            model.distribution.as_deterministic_output_module()
            if model.distribution is not None else nn.Identity()
        )
        self.input_size = model.obs_dim
        self.scalar_dim = model._scalar_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        flat = self.obs_normalizer(obs)
        heightmap = self.encoder(flat[:, self.scalar_dim - 1:-1])
        latent = torch.cat((flat[:, :self.scalar_dim], heightmap), dim=-1)
        return self.deterministic_output(self.mlp(latent))

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]

    @torch.jit.export
    def reset(self) -> None:
        pass
