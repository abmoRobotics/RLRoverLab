import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


_SAWPPY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sawppy", "SAWPPY.usd")

SAWPPY_CFG = ArticulationCfg(
    articulation_root_prim_path="/SAWPPY/SAWPPY",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_SAWPPY_PATH,
        activate_contact_sensors=True,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.04, rest_offset=0.01),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=1.5,
            max_angular_velocity=4.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*_Steer": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "base_steering": ImplicitActuatorCfg(
            joint_names_expr=[".*_Steer"],
            velocity_limit_sim=6,
            effort_limit_sim=12,
            stiffness=8000.0,
            damping=1000.0,
        ),
        "base_drive": ImplicitActuatorCfg(
            joint_names_expr=[".*_Drive"],
            velocity_limit_sim=6,
            effort_limit_sim=12,
            stiffness=100.0,
            damping=4000.0,
        ),
        "passive_joints": ImplicitActuatorCfg(
            joint_names_expr=["[LR]_(Bogie|Rocker)"],
            velocity_limit_sim=15,
            effort_limit_sim=0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
