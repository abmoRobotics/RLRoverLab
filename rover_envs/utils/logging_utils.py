

import os
import logging
from datetime import datetime
import gymnasium as gym
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode # noqa: E402
from rover_envs.mdp.recorders.recorders_cfg import (  # noqa: E402
    CompressedRGBDReinforcementLearningRecorderManagerCfg,
    ImitationLearningRecorderManagerCfg,
    ReinforcementLearningRecorderManagerCfg,
)
import pickle
from typing import Any

logger = logging.getLogger(__name__)


def video_record(
        env: ManagerBasedRLEnv, log_dir: str, video: bool, video_length: int, video_interval: int
) -> ManagerBasedRLEnv:
    """
    Function to check and setup video recording.


    Note:
        Copied from the Isaac Lab framework.


    Args:
        env (ManagerBasedRLEnv): The environment.
        log_dir (str): The log directory.
        video (bool): Whether or not to record videos.
        video_length (int): The length of the video (in steps).
        video_interval (int): The interval between video recordings (in steps).


    Returns:
        ManagerBasedRLEnv: The environment.
    """


    if video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % video_interval == 0,
            "video_length": video_length,
        }
        logger.info("Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        return gym.wrappers.RecordVideo(env, **video_kwargs)


    return env




def log_setup(experiment_cfg, env_cfg, agent):
    """
    Setup the logging for the experiment.


    Note:
        Copied from the Isaac Lab framework.
    """
    # specify directory for logging experiments
    log_root_path = os.path.join(
        "logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    logger.info(f"Logging experiment in directory: {log_root_path}")


    # specify directory for logging runs
