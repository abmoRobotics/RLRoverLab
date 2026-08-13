from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from .compressed_rgbd_hdf5 import CompressedRGBDHDF5DatasetFileHandler
from . import recorders


@configclass
class ActionRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = recorders.ActionRecorder

@configclass
class ObservationRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = recorders.ObservationRecorder

@configclass
class RewardRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = recorders.RewardRecorder

@configclass
class DoneRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = recorders.DoneRecorder

@configclass
class TerminalRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = recorders.TerminalRecorder

@configclass
class TimeoutRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = recorders.TimeoutRecorder

@configclass
class RockDistanceRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = recorders.RockDistanceRecorder
    asset_name: str = "robot"

@configclass
class RoverAttitudeRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = recorders.RoverAttitudeRecorder
    asset_name: str = "robot"

@configclass
class TimelineObservationRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = recorders.TimelineObservationRecorder

@configclass
class NextObservationRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = recorders.NextObservationRecorder

@configclass
class ReinforcementLearningRecorderManagerCfg(RecorderManagerBaseCfg):
    record_actions: ActionRecorderCfg = ActionRecorderCfg()
    record_observations: ObservationRecorderCfg = ObservationRecorderCfg()
    record_rewards: RewardRecorderCfg = RewardRecorderCfg()
    record_dones: DoneRecorderCfg = DoneRecorderCfg()
    record_next_observations: NextObservationRecorderCfg = NextObservationRecorderCfg()

@configclass
class ImitationLearningRecorderManagerCfg(RecorderManagerBaseCfg):
    record_actions: ActionRecorderCfg = ActionRecorderCfg()
    record_observations: ObservationRecorderCfg = ObservationRecorderCfg()

@configclass
class CompressedRGBDReinforcementLearningRecorderManagerCfg(RecorderManagerBaseCfg):
    dataset_file_handler_class_type: type = CompressedRGBDHDF5DatasetFileHandler

    record_actions: ActionRecorderCfg = ActionRecorderCfg()
    record_rock_distance: RockDistanceRecorderCfg = RockDistanceRecorderCfg()
    record_rover_attitude: RoverAttitudeRecorderCfg = RoverAttitudeRecorderCfg()
    record_observation_timeline: TimelineObservationRecorderCfg = TimelineObservationRecorderCfg()
    record_rewards: RewardRecorderCfg = RewardRecorderCfg()
    record_dones: DoneRecorderCfg = DoneRecorderCfg()
    record_terminals: TerminalRecorderCfg = TerminalRecorderCfg()
    record_timeouts: TimeoutRecorderCfg = TimeoutRecorderCfg()
