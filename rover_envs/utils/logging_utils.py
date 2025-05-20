
import os
from datetime import datetime
import gymnasium as gym
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode # noqa: E402
from rover_envs.mdp.recorders.recorders_cfg import ReinforcementLearningRecorderManagerCfg, ImitationLearningRecorderManagerCfg # noqa: E402

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
        print("[INFO] Recording videos during training.")
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
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs
    log_dir = datetime.now().strftime("%b%d_%H-%M-%S")
    if experiment_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir = f'_{experiment_cfg["agent"]["experiment"]["experiment_name"]}'

    log_dir += f"_{agent}"

    # set directory into agent config
    experiment_cfg["agent"]["experiment"]["directory"] = log_root_path
    experiment_cfg["agent"]["experiment"]["experiment_name"] = log_dir

    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), experiment_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), experiment_cfg)
    return log_dir


def configure_datarecorder(env_cfg, dataset_dir, dataset_name, dataset_type):
    """
    Configure the data recorder for the environment.

    Args:
        env (ManagerBasedRLEnv): The environment.
        dataset_dir (str): The directory to save the dataset.
        dataset_name (str): The name of the dataset.

    Returns:
        ManagerBasedRLEnv: The environment with the configured data recorder.
    """
    if dataset_name is not None:
        if dataset_type == "IL":
            env_cfg.recorders = ImitationLearningRecorderManagerCfg()
        elif dataset_type == "RL":
            env_cfg.recorders = ReinforcementLearningRecorderManagerCfg()
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        env_cfg.recorders.dataset_export_dir_path = dataset_dir
        env_cfg.recorders.dataset_filename = dataset_name + ".hdf5"
    return env_cfg