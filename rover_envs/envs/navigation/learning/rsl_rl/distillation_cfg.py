"""Distill the height-map PPO teacher into recurrent students: RGB with frozen DINOv3, or the persistent BEV."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
)

from rover_envs.envs.navigation.learning.rsl_rl.ppo_cfg import RoverPPORunnerCfg


@configclass
class RoverDinoStudentCfg(RslRlRNNModelCfg):
    """Recurrent student that encodes the camera with a frozen DINOv3 ViT-S/16."""

    class_name: str = "rover_envs.envs.navigation.learning.rsl_rl.models:RoverDinoStudentModel"
    dinov3_path: str = "~/models/dinov3-vits16"
    image_size: tuple[int, int] = (288, 512)
    """DINOv3 input (height, width). Camera frames are resized to it."""


@configclass
class RoverDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 60
    max_iterations = 2000
    save_interval = 50
    experiment_name = "rover_distillation"
    logger = "wandb"
    wandb_project = "rlroverlab"
    obs_groups = {"student": ["proprio", "rgb"], "teacher": ["policy"]}

    student = RoverDinoStudentCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=2,
    )
    # The PPO height-map actor; its weights come from the task's registered teacher or --checkpoint.
    teacher = RoverPPORunnerCfg().actor
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=12,
        max_grad_norm=1.0,
    )


@configclass
class RoverBevStudentCfg(RslRlRNNModelCfg):
    """Recurrent student that encodes the persistent BEV with a small CNN."""

    class_name: str = "rover_envs.envs.navigation.learning.rsl_rl.models:RoverBevStudentModel"
    encoder_channels: tuple[int, ...] = (32, 64, 64, 64)
    """Channels of the BEV convolutions; each halves the resolution."""
    encoder_dim: int = 256


@configclass
class RoverBevDistillationRunnerCfg(RoverDistillationRunnerCfg):
    """The student sees the goal and the persistent BEV (the `bev` group added by PersistentBevVecEnv)."""

    max_iterations = 400
    experiment_name = "rover_bev_distillation"
    obs_groups = {"student": ["proprio", "bev"], "teacher": ["policy"]}
    student = RoverBevStudentCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=2,
    )


@configclass
class RoverBevPersistentDistillationRunnerCfg(RoverBevDistillationRunnerCfg):
    experiment_name = "rover_bev_distillation_persistent"


@configclass
class RoverBevInstantaneousDistillationRunnerCfg(RoverBevDistillationRunnerCfg):
    experiment_name = "rover_bev_distillation_instantaneous"
