from omni.isaac.lab.actuators import ActuatorNetMLPCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

import omni.isaac.lab.sim as sim_utils

##
# Configuration - Go1 Model
##
available_usd = {
    # Unitree Go1 without our backpack or cameras
    'Unitree': f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go1/go1.usd",
    # Our Go1 model without the backpack or cameras
    'Unpack': f"./source/assets/go1/go1.usd",
    # Backpacked Go1 with a 2D lydar and 2 cameras
    'BP2D': f"./source/assets/go1/go1_bp2d.usd",
}

GO1_USD = available_usd['BP2D']

##
# Configuration - Actuators.
##
"""Configuration of our Backpacked Go1 actuators using MLP model.

Actuator specifications: https://shop.unitree.com/products/go1-motor
This model is taken from: https://github.com/Improbable-AI/walk-these-ways
"""
GO1_ACTUATOR_CFG = ActuatorNetMLPCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/Unitree/unitree_go1.pt",
    pos_scale=-1.0,
    vel_scale=1.0,
    torque_scale=1.0,
    input_order="pos_vel",
    input_idx=[0, 1, 2],
    effort_limit=23.7,  # taken from spec sheet
    velocity_limit=30.0,  # taken from spec sheet
    saturation_effort=23.7,  # same as effort limit
)

"""Configuration of CROB Backpacked Go1 using MLP-based actuator model."""
GO1_CFG = ArticulationCfg(
    actuators={
        "base_legs": GO1_ACTUATOR_CFG,
    },
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=GO1_USD, #f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go1/go1.usd"
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
)