"""Train the instantaneous BEV model on RGB-D frames collected while the height-map policy drives the rover."""

from isaaclab.utils import configclass


@configclass
class RoverBevModelCfg:
    """Frozen DINOv3 ViT-S/16 features lifted by the depth into a BEV U-Net (see RoverBevModel)."""

    dinov3_path: str = "~/models/dinov3-vits16"
    image_size: tuple[int, int] = (368, 672)
    """DINOv3 input (height, width), multiples of 16. The 376x672 camera frames are resized to it."""
    feature_dim: int = 32
    """Image feature channels splatted into the BEV by the 1x1 adapter."""
    unet_channels: tuple[int, ...] = (64, 128, 256, 256)
    """U-Net widths per level; every level after the first halves the BEV resolution."""
    use_geometry: bool = True
    """Feed the lifted geometry channels (mean elevation, elevation range, point count) to the U-Net."""
    use_dino: bool = True
    """Feed the lifted DINOv3 features to the U-Net. Without them the backbone is not loaded."""


@configclass
class RoverBevTrainerCfg:
    seed: int = 42
    max_iterations: int = 1000
    num_steps_per_env: int = 16
    """Frames collected per environment in each iteration."""
    num_learning_epochs: int = 4
    num_mini_batches: int = 16
    learning_rate: float = 3.0e-4
    max_grad_norm: float = 1.0
    semantic_loss_weight: float = 1.0
    save_interval: int = 50
    image_interval: int = 10
    """Iterations between saved BEV visualizations."""
    experiment_name: str = "rover_bev"
    logger: str = "wandb"
    """"wandb" or "tensorboard"."""
    wandb_project: str = "rlroverlab"
    model: RoverBevModelCfg = RoverBevModelCfg()


# Input ablations; each is registered as its own task and otherwise trains exactly like RoverBevTrainerCfg.


@configclass
class RoverBevGeometryOnlyTrainerCfg(RoverBevTrainerCfg):
    """Only the geometry lifted from the perfect depth: how much can depth alone solve?"""

    experiment_name = "rover_bev_geometry_only"
    model = RoverBevModelCfg(use_dino=False)


@configclass
class RoverBevDinoOnlyTrainerCfg(RoverBevTrainerCfg):
    """Only the DINOv3 features: how much comes from appearance? The depth still places the features."""

    experiment_name = "rover_bev_dino_only"
    model = RoverBevModelCfg(use_geometry=False)


@configclass
class RoverBevNoisyDepthTrainerCfg(RoverBevTrainerCfg):
    """The full model on ZED-like noisy depth (see StereoDepthCfg): the closest to the real problem."""

    experiment_name = "rover_bev_noisy_depth"
