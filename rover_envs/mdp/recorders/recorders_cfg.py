from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

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