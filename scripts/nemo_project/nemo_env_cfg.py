import math
import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
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

@configclass
class CommandsCfg:
    """Command specification for the environment."""
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

@configclass
class ActionsCfg:
    """Action specifications for the environment."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.5,
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
        
        # 4. Joint positions and velocities
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5)
        )
        
        # 5. Previous actions
        actions = ObsTerm(func=custom_mdp.last_action, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"})
        
        # 6. Phase
        phase = ObsTerm(func=custom_mdp.phase, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"})
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """Reward specifications for the environment."""
    # Tracking related rewards.
    tracking_lin_vel = RewTerm(func=custom_mdp.tracking_lin_vel, weight=1.0, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces", "std": 0.25})
    tracking_ang_vel = RewTerm(func=custom_mdp.tracking_ang_vel, weight=0.5, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces", "std": 0.25})
    
    # Base related rewards.
    lin_vel_z = RewTerm(func=custom_mdp.lin_vel_z, weight=0.0, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"})
    ang_vel_xy = RewTerm(func=custom_mdp.ang_vel_xy, weight=-0.15, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"})
    orientation = RewTerm(func=custom_mdp.orientation, weight=-1.0)
    base_height = RewTerm(func=custom_mdp.base_height, weight=0.0, params={"target_height": 0.5793})
    
    # Energy related rewards.
    torques = RewTerm(func=custom_mdp.torques, weight=0.0)
    action_rate = RewTerm(func=custom_mdp.action_rate, weight=-0.005, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"})
    energy = RewTerm(func=custom_mdp.energy, weight=0.0)
    dof_acc = RewTerm(func=custom_mdp.dof_acc, weight=-1e-7)
    dof_vel = RewTerm(func=custom_mdp.dof_vel, weight=0.0)
    
    # Feet related rewards.
    feet_clearance = RewTerm(func=custom_mdp.feet_clearance, weight=0.0, params={"contact_sensor_name": "contact_forces", "max_foot_height": 0.10})
    feet_air_time = RewTerm(func=custom_mdp.feet_air_time, weight=2.0, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"})
    feet_slip = RewTerm(func=custom_mdp.feet_slip, weight=-0.25, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces"})
    feet_height = RewTerm(func=custom_mdp.feet_height, weight=0.0, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces", "max_foot_height": 0.10})
    feet_phase = RewTerm(func=custom_mdp.feet_phase, weight=1.0, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces", "max_foot_height": 0.10})
    
    # Other rewards.
    stand_still = RewTerm(func=custom_mdp.stand_still, weight=0.0, params={"command_name": "base_velocity"})
    alive = RewTerm(func=mdp.is_alive, weight=0.40)
    termination = RewTerm(func=mdp.is_terminated, weight=0.0)
    
    # Pose related rewards.
    joint_deviation_knee = RewTerm(func=custom_mdp.joint_deviation_knee, weight=-0.1)
    joint_deviation_hip = RewTerm(func=custom_mdp.joint_deviation_hip, weight=-0.1, params={"command_name": "base_velocity"})
    dof_pos_limits = RewTerm(func=custom_mdp.joint_pos_limits, weight=-1.0)
    pose = RewTerm(func=custom_mdp.pose, weight=-1.0)
    feet_distance = RewTerm(func=custom_mdp.feet_distance, weight=-1.0, params={"contact_sensor_name": "contact_forces"})
    collision = RewTerm(func=custom_mdp.collision, weight=-1.0, params={"contact_sensor_name": "contact_forces"})
    feet_contact = RewTerm(func=custom_mdp.feet_contact, weight=-0.25, params={"command_name": "base_velocity", "contact_sensor_name": "contact_forces", "max_foot_height": 0.10})

@configclass
class TerminationsCfg:
    """Termination specifications for the environment."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.2}
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
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.viewer.eye = (1.5, 1.5, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
