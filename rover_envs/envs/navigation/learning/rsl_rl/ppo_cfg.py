"""PPO baseline for the rover height-map observation."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 60
    max_iterations = 1000
    save_interval = 50
    experiment_name = "rover_heightmap"
    logger = "wandb"
    wandb_project = "rlroverlab"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        class_name="rover_envs.envs.navigation.learning.rsl_rl.models:RoverConvModel",
        hidden_dims=[256, 160, 128],
        activation="lrelu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            class_name="rover_envs.envs.navigation.learning.rsl_rl.models:ClippedGaussianDistribution",
            init_std=1.0,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        class_name="rover_envs.envs.navigation.learning.rsl_rl.models:RoverConvModel",
        hidden_dims=[256, 160, 128],
        activation="lrelu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=4,
        num_mini_batches=60,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
    )
