"""Compatibility helpers for skrl and Weights & Biases logging."""

from __future__ import annotations

import importlib
from functools import partial


class TorchSummaryWriterAdapter:
    """Expose skrl 2.x's writer API through PyTorch's TensorBoard writer.

    skrl 2.x uses a low-level TensorBoard event writer that W&B's
    ``sync_tensorboard`` path does not detect. PyTorch's SummaryWriter is what
    skrl 1.x used, and W&B still hooks it correctly.
    """

    def __init__(
        self,
        log_dir: str,
        *,
        queue_size: int = 10,
        flush_seconds: int = 120,
        num_envs: int | None = None,
        step_dt: float | None = None,
    ):
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=log_dir, max_queue=queue_size, flush_secs=flush_seconds)
        self._num_envs = num_envs
        self._step_dt = step_dt
        self._last_progress_timestep = None

    def add_scalar(self, *, tag: str, value: float, timestep: int) -> None:
        if self._num_envs is not None and timestep != self._last_progress_timestep:
            # skrl's timestep starts at zero and advances once per vectorized environment step.
            env_steps = timestep + 1
            self._writer.add_scalar("train/env_transitions", env_steps * self._num_envs, timestep)
            if self._step_dt is not None:
                self._writer.add_scalar("train/sim_time_s", env_steps * self._step_dt, timestep)
            self._last_progress_timestep = timestep
        self._writer.add_scalar(tag, value, timestep)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


def patch_skrl_summary_writer_for_wandb(*, num_envs: int | None = None, step_dt: float | None = None) -> None:
    """Patch skrl's imported SummaryWriter symbols to a W&B-compatible writer."""

    writer = partial(TorchSummaryWriterAdapter, num_envs=num_envs, step_dt=step_dt)
    for module_name in ("skrl.agents.torch.base", "skrl.multi_agents.torch.base"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        module.SummaryWriter = writer
