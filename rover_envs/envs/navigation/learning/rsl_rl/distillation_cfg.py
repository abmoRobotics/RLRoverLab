"""Distill the height-map PPO teacher into a recurrent RGB student with a frozen DINOv3 encoder."""

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
    # The PPO height-map actor; its weights are loaded from the checkpoint passed to the trainer.
    teacher = RoverPPORunnerCfg().actor
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=3.0e-4,
        gradient_length=12,
        max_grad_norm=1.0,
    )
