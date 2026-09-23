"""Add training progress metrics to the RSL-RL writer."""

from functools import partial

from rsl_rl.utils.logger import Logger


class TrainingProgressLogger(Logger):
    def __init__(self, *args, step_dt: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.step_dt = step_dt

    def log(self, *, it: int, **kwargs) -> None:
        super().log(it=it, **kwargs)
        if self.writer is not None:
            self.writer.add_scalar("train/env_transitions", self.tot_timesteps, it)
            parallel_envs = self.num_envs * self.gpu_world_size
            self.writer.add_scalar("train/sim_time_s", self.tot_timesteps / parallel_envs * self.step_dt, it)


def patch_rsl_rl_logger_for_training_progress(*, step_dt: float) -> None:
    """Use the repo's logger extension when OnPolicyRunner is constructed."""
    import rsl_rl.runners.on_policy_runner as runner_module

    runner_module.Logger = partial(TrainingProgressLogger, step_dt=step_dt)
