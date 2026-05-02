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
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.sim.spawners.lights.lights_cfg import DomeLightCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.noise import UniformNoiseCfg

# Import the previously created robot configuration
from nemo_cfg import NEMO_CFG

@configclass
class CommandsCfg:
    """Command specification for the environment.
    Defines the joystick commands (linear velocity x/y and angular velocity yaw).
    """
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0), # Sample new command every 5 to 10 seconds
        rel_standing_envs=0.1, # 10% chance of setting the target command to exactly (0,0,0) (stand still)
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), 
            lin_vel_y=(-0.3, 0.3), 
            ang_vel_z=(-1.0, 1.0), 
            heading=(-math.pi, math.pi),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the environment.
    Maps neural network outputs to joint position targets.
    """
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"], # Control all joints
        scale=1.5, # Matches legacy config.action_scale = 1.5
        use_default_offset=True, # Network output is an offset relative to the default pose
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the environment.
    Defines what the neural network "sees" at every step.
    """
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the base policy (Actor)."""
        # 1. Base linear and angular velocity (Local frame)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))
        
        # 2. Projected gravity (Provides base orientation information)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05)
        )
        
        # 3. Velocity commands (The target velocities we want the robot to track)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        
        # 4. Joint positions and velocities
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, # Relative to default pose
            noise=UniformNoiseCfg(n_min=-0.03, n_max=0.03)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5)
        )
        
        # 5. Previous actions (Helps network understand its own past behavior)
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True # Enable the uniform noise added to sensors
            self.concatenate_terms = True

    # The RL algorithm expects the observations to be in a group called 'policy' (which is mapped to 'actor' by default)
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """Reward specifications for the environment.
    Defines the objective function and penalties for the agent.
    """
    # 1. Tracking Rewards (Reward the agent for following velocity commands)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    
    # 2. Base Related Penalties (Penalize excessive shaking/tilting)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.15)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    
    # 3. Energy/Action Penalties (Encourage smooth and efficient movements)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1e-7)
    
    # 4. Survival & Termination (Reward for staying alive, penalize falling)
    is_alive = RewTerm(func=mdp.is_alive, weight=0.40)
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-0.0)
    
    # 5. Joint Limits (Penalize pushing joints past their physical limits)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

@configclass
class TerminationsCfg:
    """Termination specifications for the environment.
    Defines when an episode should end early.
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True) # Episode reaches max length
    # base_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot.*"), "threshold": 1.0},
    #     # Note: Currently set up to check if any feet collide, but standard practice
    #     # is to check if the base/torso touches the ground. This can be modified later.
    # )

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Scene specification for the environment.
    Defines what objects are spawned in the world.
    """
    # Spawn a flat ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
    )
    
    # Spawn a dome light
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75)
        )
    )
    
    # Spawn the Nemo robot using the NEMO_CFG from nemo_cfg.py
    robot = NEMO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/.*_foot_.*",
    #     update_period=0.0,
    #     history_length=3,
    #     debug_vis=False,
    #     filter_prim_paths_expr=[],
    #     track_air_time=False,
    # )

@configclass
class NemoEnvCfg(ManagerBasedRLEnvCfg):
    """The main environment configuration for Nemo."""
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5) # Spawn 4096 environments in parallel
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        # Control frequency logic:
        # IsaacLab base physics step is typically 0.005s.
        # decimation = 4 means policy runs every 0.02s (0.005 * 4), matching legacy ctrl_dt=0.02
        self.decimation = 4
        
        # Max episode length: 10.0 seconds (matches legacy episode_length=500 steps * 0.02s)
        self.episode_length_s = 10.0
        
        # Default camera position in the viewer
        self.viewer.eye = (1.5, 1.5, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.5)
