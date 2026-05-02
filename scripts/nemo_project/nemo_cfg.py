import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

# ----------------------------------------------------------------------------
# Align with ``source/isaaclab_assets/isaaclab_assets/robots/nemo.py``.
# Earlier local edits (uniform actuators without ``effort_limit_sim``,
# self-collisions ON, elevated spawn drop) tended to explode into NaNs in
# thousand-env training.
#
# modest +12% torque headroom vs asset defaults — ablation for dynamic pushes;
# revert if NaNs return. (Keep stiffness unchanged for one-axis-ish physics tweaks.)
# ----------------------------------------------------------------------------
_KP_HIGH = 50.0
_KP_LOW = 40.0
_JOINT_DAMPING = 1.0
_TAU_HIGH = 53.0
_TAU_LOW = 26.5

_HOME_LEG = (
    -0.698132,
    0.0,
    0.0,
    1.22173,
    -0.523599,
    0.0,
)

NEMO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/mmgzz/Desktop/IsaacLab/source/isaaclab_assets/data/Robots/Nemo/nemo.usd",
        activate_contact_sensors=True,
        copy_from_source=False,
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
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5793),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "l_hip_pitch": _HOME_LEG[0],
            "l_hip_roll": _HOME_LEG[1],
            "l_hip_yaw": _HOME_LEG[2],
            "l_knee": _HOME_LEG[3],
            "l_foot_pitch": _HOME_LEG[4],
            "l_foot_roll": _HOME_LEG[5],
            "r_hip_pitch": _HOME_LEG[0],
            "r_hip_roll": _HOME_LEG[1],
            "r_hip_yaw": _HOME_LEG[2],
            "r_knee": _HOME_LEG[3],
            "r_foot_pitch": _HOME_LEG[4],
            "r_foot_roll": _HOME_LEG[5],
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs_high": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch", ".*_knee", ".*_foot_pitch"],
            stiffness=_KP_HIGH,
            damping=_JOINT_DAMPING,
            effort_limit_sim=_TAU_HIGH,
            armature=0.005,
        ),
        "legs_low": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll", ".*_hip_yaw", ".*_foot_roll"],
            stiffness=_KP_LOW,
            damping=_JOINT_DAMPING,
            effort_limit_sim=_TAU_LOW,
            armature=0.005,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
