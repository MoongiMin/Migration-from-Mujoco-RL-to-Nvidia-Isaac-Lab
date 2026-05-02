import math
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.assets.asset_base_cfg import AssetBaseCfg
from isaaclab.sim.spawners.lights.lights_cfg import DomeLightCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.noise import UniformNoiseCfg

# Import the previously created robot configuration
from nemo_cfg import NEMO_CFG
import nemo_custom_mdp as custom_mdp

# MuJoCo ``legged_rl .../models/nemo/nemo5.xml`` <actuator> block order (matches ``data.ctrl`` layout).
NEMO_ACTUATED_JOINT_ORDER: tuple[str, ...] = (
    "l_hip_pitch",
    "l_hip_roll",
    "l_hip_yaw",
    "l_knee",
    "l_foot_pitch",
    "l_foot_roll",
    "r_hip_pitch",
    "r_hip_roll",
    "r_hip_yaw",
    "r_knee",
    "r_foot_pitch",
    "r_foot_roll",
)

_NEMO_ROBOT_JOINTS_ASSET_CFG = SceneEntityCfg(
    "robot",
    joint_names=list(NEMO_ACTUATED_JOINT_ORDER),
    preserve_order=True,
)


@configclass
class CommandsCfg:
    """Joystick-style velocity commands (``joystick.default_config`` + ``sample_command``).

    On each resample, IsaacLab sets ``‖v‖=0`` for a fraction ``rel_standing_envs`` of envs—the same Bernoulli
    idea as JAX ``bernoulli(p=0.1) -> zero(cmd)``. Ranges match ``lin_vel_*`` / ``ang_vel_yaw`` there.
    """

    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.1,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )



# Reward layout: Joystick-derived terms plus gait emergence additions (documented under ``RewardsCfg``).


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(NEMO_ACTUATED_JOINT_ORDER),
        preserve_order=True,
        # Maps policy output [-1,1] to Δq; modestly high for exploratory leg swings during training (stability ↔ coverage).
        scale=0.72,
        use_default_offset=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""
    @configclass
    class PolicyCfg(ObsGroup):
        # 1. Base linear and angular velocity (Local frame)
        filtered_linvel = ObsTerm(
            func=custom_mdp.filtered_linvel, 
            params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"},
            noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1)
        )
        filtered_angvel = ObsTerm(
            func=custom_mdp.filtered_angvel, 
            params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"},
            noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2)
        )
        
        # 2. Projected gravity
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
        )
        
        # 3. Velocity commands
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        # 4. Joint positions and velocities (same order as MJCF actuators / policy actions).
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _NEMO_ROBOT_JOINTS_ASSET_CFG},
            noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={"asset_cfg": _NEMO_ROBOT_JOINTS_ASSET_CFG},
            noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
        )
        
        # 5. Previous actions
        actions = ObsTerm(func=custom_mdp.last_action, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"})
        
        # 6. Phase
        phase = ObsTerm(func=custom_mdp.phase, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Clean-state suffix for the critic only (Joystick ``privileged_state`` after noisy ``state``)."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        root_lin_acc_b = ObsTerm(func=custom_mdp.privileged_root_lin_acc_b)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _NEMO_ROBOT_JOINTS_ASSET_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": _NEMO_ROBOT_JOINTS_ASSET_CFG})
        base_pos_z = ObsTerm(func=mdp.base_pos_z)
        joint_effort = ObsTerm(func=mdp.joint_effort, params={"asset_cfg": _NEMO_ROBOT_JOINTS_ASSET_CFG})
        foot_contact = ObsTerm(
            func=custom_mdp.privileged_foot_contact,
            params={"contact_sensor_name": "contact_forces"},
        )
        feet_lin_vel_w = ObsTerm(
            func=custom_mdp.privileged_feet_lin_vel_w,
            params={"contact_sensor_name": "contact_forces"},
        )
        feet_air_time = ObsTerm(
            func=custom_mdp.privileged_feet_air_time,
            params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()

@configclass
class RewardsCfg:
    """Joystick-based core plus gait emergence terms (details below).

    Reference: Kelly+, *Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition* (IEEE RAL /
    ``arXiv:2011.01387``) — walking benefits from **periodic stance/swing structure** tied to measurable signals
    (forces/velocity per phase); reference-free setups often need **explicit progress** and calibrated contact-phase
    costs to avoid quasi-static collapse.

    Changes vs pure Joystick: softer ``feet_contact`` (avoid frozen feet), stronger planar **command alignment**
    reward, XOR single-contact bonus during locomotion, kept ``locomotion_planar_idle``.
    Isaac-only ``collision``.
    """

    tracking_lin_vel = RewTerm(
        func=custom_mdp.tracking_lin_vel,
        weight=1.15,
        params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces", "std": 0.25},
    )
    tracking_ang_vel = RewTerm(
        func=custom_mdp.tracking_ang_vel,
        weight=0.5,
        params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces", "std": 0.25},
    )

    ang_vel_xy = RewTerm(
        func=custom_mdp.ang_vel_xy,
        weight=-0.15,
        params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"},
    )
    orientation = RewTerm(func=custom_mdp.orientation, weight=-1.0)

    action_rate = RewTerm(
        func=custom_mdp.action_rate,
        weight=-0.005,
        params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"},
    )
    dof_acc = RewTerm(func=custom_mdp.dof_acc, weight=-1.0e-7)

    feet_air_time = RewTerm(
        func=custom_mdp.feet_air_time,
        weight=2.0,
        params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"},
    )
    feet_slip = RewTerm(
        func=custom_mdp.feet_slip,
        weight=-0.25,
        params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"},
    )
    feet_phase = RewTerm(
        func=custom_mdp.feet_phase,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "contact_sensor_name": "contact_forces",
            "max_foot_height": 0.10,
        },
    )

    locomotion_planar_idle = RewTerm(
        func=custom_mdp.locomotion_planar_idle_cost,
        weight=-0.4,
        params={
            "command_name": "base_velocity",
            "contact_sensor_name": "contact_forces",
            "speed_variance": 0.04,
        },
    )

    locomotion_cmd_velocity_alignment = RewTerm(
        func=custom_mdp.locomotion_velocity_command_alignment,
        weight=2.2,
        params={
            "command_name": "base_velocity",
            "contact_sensor_name": "contact_forces",
            "clamp_max": 1.25,
        },
    )
    gait_single_support = RewTerm(
        func=custom_mdp.locomotion_single_support_bonus,
        weight=1.2,
        params={
            "command_name": "base_velocity",
            "contact_sensor_name": "contact_forces",
        },
    )

    # Slightly softer nominal-pose rails (roadmap “Slack’’): frees hip/knee for dynamic stepping vs frozen home.
    joint_deviation_knee = RewTerm(func=custom_mdp.joint_deviation_knee, weight=-0.08)
    joint_deviation_hip = RewTerm(
        func=custom_mdp.joint_deviation_hip, weight=-0.08, params={"command_name": "base_velocity"}
    )
    dof_pos_limits = RewTerm(func=custom_mdp.joint_pos_limits, weight=-1.0, params={"soft_limit_factor": 0.95})
    pose = RewTerm(func=custom_mdp.pose, weight=-0.85)
    feet_distance = RewTerm(
        func=custom_mdp.feet_distance,
        weight=-1.0,
        params={"contact_sensor_name": "contact_forces"},
    )
    collision = RewTerm(
        func=custom_mdp.collision,
        weight=-1.0,
        params={"contact_sensor_name": "contact_forces", "threshold": 0.12},
    )
    feet_contact = RewTerm(
        func=custom_mdp.feet_contact,
        weight=-0.09,
        params={
            "command_name": "base_velocity",
            "contact_sensor_name": "contact_forces",
            "max_foot_height": 0.10,
        },
    )


@configclass
class EventCfg:
    """Reset: default scene pose and joint PD targets aligned to nominal stance to reduce post-reset jitter."""

    reset_stable = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


@configclass
class TerminationsCfg:
    """Episode cap + Joystick-style fall only (``upvector_world.z < 0``). No tilt-only termination."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    tipped_over = DoneTerm(
        func=custom_mdp.torso_up_tip_over,
        params={"min_world_up_z": 0.0},
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Scene specification for the environment."""
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75)
        )
    )
    robot = NEMO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis/.*foot_roll",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        track_air_time=True,
    )

@configclass
class NemoEnvCfg(ManagerBasedRLEnvCfg):
    """The main environment configuration for Nemo."""
    scene: SceneCfg = SceneCfg(num_envs=8192, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.viewer.eye = (1.5, 1.5, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
