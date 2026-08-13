"""Helpers for configuring Isaac Lab before the simulation app is launched."""

from __future__ import annotations

from argparse import Namespace


_CAMERA_TASK_PREFIXES = (
    "AAURoverEnvCamera",
    "AAURoverEnvCosmos",
    "AAURoverEnvRGBD",
)


def task_uses_cameras(task_name: str | None) -> bool:
    """Return whether a registered navigation task contains RTX camera sensors."""
    return bool(task_name) and task_name.startswith(_CAMERA_TASK_PREFIXES)


def configure_camera_launcher_args(args: Namespace) -> None:
    """Enable camera extensions for camera tasks and video capture before AppLauncher starts."""
    if getattr(args, "video", False) or task_uses_cameras(getattr(args, "task", None)):
        args.enable_cameras = True


__all__ = ["configure_camera_launcher_args", "task_uses_cameras"]
