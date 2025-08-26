import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from rover_envs.envs.navigation.utils.articulation.articulation import RoverArticulation

# _AAU_ROVER_PATH = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "assets", "rover", "rover_instance.usd"
# )
# _AAU_ROVER_PATH = "http://localhost:8080/omniverse://127.0.0.1/Projects/simple_instanceable_new2/rover_instance.usd"
# _AAU_ROVER_PATH = "http://localhost:8080/omniverse://127.0.0.1/Projects/simplified9.usd"
# _AAU_ROVER_SIMPLE_PATH =
# "http://localhost:8080/omniverse://127.0.0.1/Projects/simple_instanceable_new/rover_instance.usd"
_AAU_ROVER_SIMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "aau_rover_simple", "rover_instance.usd")
AAU_ROVER_SIMPLE_CFG = ArticulationCfg(
    class_type=RoverArticulation,
    spawn=sim_utils.UsdFileCfg(
        usd_path=_AAU_ROVER_SIMPLE_PATH,
        activate_contact_sensors=True,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.04, rest_offset=0.01),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=1.5,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4)
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={".*Steer_Revolute": 0.0},
        joint_vel={".*Steer_Revolute": 0.0, ".*Drive_Continuous": 0.0},
    ),
    actuators={
        "base_steering": ImplicitActuatorCfg(
            joint_names_expr=[".*Steer_Revolute"],
            velocity_limit_sim=6,
            effort_limit_sim=12,
            stiffness=8000.0,
            damping=1000.0,
        ),
        "base_drive": ImplicitActuatorCfg(
            joint_names_expr=[".*Drive_Continuous"],
            velocity_limit_sim=6,
            effort_limit_sim=12,
            stiffness=100.0,
            damping=4000.0,
        ),
        "passive_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*Boogie_Revolute"],
            velocity_limit_sim=15,
            effort_limit_sim=0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
