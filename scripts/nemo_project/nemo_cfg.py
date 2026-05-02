import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# =========================================================================
# Nemo Robot Configuration (ArticulationCfg)
# =========================================================================
NEMO_CFG = ArticulationCfg(
    # ---------------------------------------------------------------------
    # USD file path and basic physics properties
    # ---------------------------------------------------------------------
    spawn=sim_utils.UsdFileCfg(
            usd_path="C:/Users/mmgzz/Desktop/IsaacLab/source/isaaclab_assets/data/Robots/Nemo/nemo.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, # Whether robot links can collide with each other
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    # ---------------------------------------------------------------------
    # Initial state (position, orientation, joint angles)
    # ---------------------------------------------------------------------
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6), # Initial spawn position (x, y, z) - elevated to drop safely
        rot=(1.0, 0.0, 0.0, 0.0), # Initial rotation (w, x, y, z) quaternion
        joint_pos={
            # Default standing pose (slightly bent knees for stability)
            ".*_hip_pitch": 0.2,
            ".*_knee": -0.2,
            ".*_foot_pitch": 0.2,
            ".*_hip_roll": 0.0,
            ".*_hip_yaw": 0.0,
            ".*_foot_roll": 0.0,
        },
        joint_vel={
            ".*": 0.0, # Set initial velocity of all joints to 0.0
        },
    ),
    # ---------------------------------------------------------------------
    # Actuators configuration - PD Controller
    # ---------------------------------------------------------------------
    actuators={
        "nemo_joints": ImplicitActuatorCfg(
            joint_names_expr=[".*"], # Joint names to control (regex: all joints)
            stiffness=800.0,         # P gain (increased to hold the robot's weight)
            damping=80.0,            # D gain (increased to prevent oscillation)
        ),
    },
)
