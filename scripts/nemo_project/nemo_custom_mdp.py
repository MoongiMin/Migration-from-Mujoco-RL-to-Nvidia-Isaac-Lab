import torch
import math
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

def get_nemo_state(env: "ManagerBasedRLEnv", command_name: str, asset_name: str = "robot", contact_sensor_name: str = "contact_forces"):
    """Initialize and retrieve the custom Nemo state attached to the environment."""
    if not hasattr(env, "nemo_state"):
        env.nemo_state = {
            "phase": torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32),
            "phase_dt": torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.float32),
            "filtered_linvel": torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32),
            "filtered_angvel": torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32),
            "last_act": torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device, dtype=torch.float32),
            "last_last_act": torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device, dtype=torch.float32),
            "feet_air_time": torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32),
            "last_contact": torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.bool),
            "swing_peak": torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32),
            "updated_step": torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        }
        gait_freq = torch.rand(env.num_envs, 1, device=env.device) * 0.5 + 1.25 # U(1.25, 1.75)
        env.nemo_state["phase_dt"] = 2 * torch.pi * env.step_dt * gait_freq
        env.nemo_state["phase"][:, 1] = torch.pi

    # Handle resets
    if hasattr(env, "reset_buf"):
        reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            gait_freq = torch.rand(len(reset_ids), 1, device=env.device) * 0.5 + 1.25
            env.nemo_state["phase_dt"][reset_ids] = 2 * torch.pi * env.step_dt * gait_freq
            env.nemo_state["phase"][reset_ids, 0] = 0.0
            env.nemo_state["phase"][reset_ids, 1] = torch.pi
            env.nemo_state["filtered_linvel"][reset_ids] = 0.0
            env.nemo_state["filtered_angvel"][reset_ids] = 0.0
            env.nemo_state["last_act"][reset_ids] = 0.0
            env.nemo_state["last_last_act"][reset_ids] = 0.0
            env.nemo_state["feet_air_time"][reset_ids] = 0.0
            env.nemo_state["last_contact"][reset_ids] = False
            env.nemo_state["swing_peak"][reset_ids] = 0.0
            # For resets, updated_step shouldn't block an update this step
            env.nemo_state["updated_step"][reset_ids] = -1

    # Perform per-step updates exactly once per env.step()
    if hasattr(env, "episode_length_buf"):
        needs_update = env.nemo_state["updated_step"] != env.episode_length_buf
    else:
        # During initialization, episode_length_buf does not exist, force update
        needs_update = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        
    if needs_update.any():
        idx = needs_update.nonzero(as_tuple=False).squeeze(-1)
        
        asset = env.scene[asset_name]
        commands = env.command_manager.get_command(command_name)
        contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_name]
        
        # In IsaacLab, foot contacts can be fetched from the contact sensor. We assume sensor_cfg.body_ids matches [left, right]
        # For simplicity, let's say contact_sensor.data.net_forces_w > 1.0 means contact.
        if contact_sensor.data.net_forces_w_history is not None and contact_sensor.data.net_forces_w_history.numel() > 0:
            forces = contact_sensor.data.net_forces_w_history[:, :, :, :].norm(dim=-1).max(dim=1)[0]
            contact = (forces > 1.0)[idx] # shape: (len(idx), num_bodies)
        else:
            contact = torch.zeros(len(idx), 2, device=env.device, dtype=torch.bool)
        
        # Action update
        if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
            current_act = env.action_manager.action[idx]
        else:
            current_act = torch.zeros(len(idx), env.action_manager.total_action_dim, device=env.device, dtype=torch.float32)
            
        env.nemo_state["last_last_act"][idx] = env.nemo_state["last_act"][idx]
        env.nemo_state["last_act"][idx] = current_act
        
        # Velocity update
        # Local linvel:
        vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w[idx]), asset.data.root_lin_vel_w[idx, :3])
        env.nemo_state["filtered_linvel"][idx] = vel_yaw * 1.0 # filtered = current * 1.0 + old * 0.0
        # Local angvel:
        angvel = quat_apply_inverse(asset.data.root_quat_w[idx], asset.data.root_ang_vel_w[idx, :3])
        env.nemo_state["filtered_angvel"][idx] = angvel * 1.0

        # Feet air time & swing peak
        env.nemo_state["feet_air_time"][idx] += env.step_dt
        foot_pos = asset.data.body_pos_w[idx] # (len(idx), num_bodies, 3)
        foot_ids = asset.find_bodies(contact_sensor.body_names)[0]
        foot_z = foot_pos[:, foot_ids, 2] # (len(idx), 2)
        env.nemo_state["swing_peak"][idx] = torch.maximum(env.nemo_state["swing_peak"][idx], foot_z)
        
        # Phase update
        phase_tp1 = env.nemo_state["phase"][idx] + env.nemo_state["phase_dt"][idx]
        new_phase = torch.fmod(phase_tp1 + math.pi, 2 * math.pi) - math.pi
        
        cmd_norm = torch.norm(commands[idx, :3], dim=1)
        zero_cmd_mask = cmd_norm <= 0.01
        new_phase[zero_cmd_mask] = math.pi
        env.nemo_state["phase"][idx] = new_phase
        
        # Reset air time and swing peak if in contact
        env.nemo_state["feet_air_time"][idx] *= ~contact
        env.nemo_state["swing_peak"][idx] *= ~contact
        env.nemo_state["last_contact"][idx] = contact
        
        if hasattr(env, "episode_length_buf"):
            env.nemo_state["updated_step"][idx] = env.episode_length_buf[idx]

    return env.nemo_state

def _update_and_get_nemo_state(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str):
    return get_nemo_state(env, command_name, asset_name="robot", contact_sensor_name=contact_sensor_name)

# ========================
# OBSERVATIONS
# ========================

def phase(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str) -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    return torch.cat([torch.cos(state["phase"]), torch.sin(state["phase"])], dim=1)

def filtered_linvel(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str) -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    return state["filtered_linvel"]

def filtered_angvel(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str) -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    return state["filtered_angvel"]

def last_action(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str) -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    return state["last_act"]

def current_action(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str) -> torch.Tensor:
    return env.action_manager.action

# ========================
# REWARDS
# ========================

def tracking_lin_vel(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str, std: float) -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    commands = env.command_manager.get_command(command_name)
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - state["filtered_linvel"][:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std)

def tracking_ang_vel(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str, std: float) -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    commands = env.command_manager.get_command(command_name)
    ang_vel_error = torch.square(commands[:, 2] - state["filtered_angvel"][:, 2])
    return torch.exp(-ang_vel_error / std)

def lin_vel_z(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str) -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    return torch.square(state["filtered_linvel"][:, 2])

def ang_vel_xy(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str) -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    return torch.sum(torch.square(state["filtered_angvel"][:, :2]), dim=1)

def orientation(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    gravity = quat_apply_inverse(asset.data.root_quat_w, torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1))
    return torch.sum(torch.square(gravity[:, :2]), dim=1)

def base_height(env: "ManagerBasedRLEnv", target_height: float, asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)

def torques(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    return torch.sum(torch.abs(asset.data.applied_torque), dim=1)

def action_rate(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str) -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    current_act = env.action_manager.action
    return torch.sum(torch.square(current_act - state["last_act"]), dim=1)

def energy(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    return torch.sum(torch.abs(asset.data.joint_vel * asset.data.applied_torque), dim=1)

def dof_acc(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=1)

def dof_vel(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)

def feet_slip(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str, asset_name: str = "robot") -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    contact = state["last_contact"]
    asset = env.scene[asset_name]
    body_vel = asset.data.root_lin_vel_w[:, :2] # proxy for feet slip if root is moving
    # Actually joystick.py uses global linvel of the body.
    reward = torch.sum(torch.norm(body_vel, dim=-1).unsqueeze(-1) * contact, dim=1)
    return reward

def feet_clearance(env: "ManagerBasedRLEnv", max_foot_height: float, contact_sensor_name: str, asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_name]
    foot_ids = asset.find_bodies(contact_sensor.body_names)[0]
    feet_vel = asset.data.body_lin_vel_w[:, foot_ids, :]
    vel_xy = feet_vel[..., :2]
    vel_norm = torch.sqrt(torch.norm(vel_xy, dim=-1))
    foot_pos = asset.data.body_pos_w[:, foot_ids, :]
    foot_z = foot_pos[..., 2]
    delta = torch.abs(foot_z - max_foot_height)
    return torch.sum(delta * vel_norm, dim=1)

def feet_height(env: "ManagerBasedRLEnv", max_foot_height: float, command_name: str, contact_sensor_name: str) -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_name]
    forces = contact_sensor.data.net_forces_w_history[:, :, :, :].norm(dim=-1).max(dim=1)[0]
    contact = forces > 1.0
    contact_filt = contact | state["last_contact"]
    first_contact = (state["feet_air_time"] > 0.0) * contact_filt
    
    error = state["swing_peak"] / max_foot_height - 1.0
    return torch.sum(torch.square(error) * first_contact, dim=1)

def feet_air_time(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str, threshold_min: float = 0.2, threshold_max: float = 0.5) -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    commands = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(commands[:, :3], dim=1)
    
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_name]
    forces = contact_sensor.data.net_forces_w_history[:, :, :, :].norm(dim=-1).max(dim=1)[0]
    contact = forces > 1.0
    contact_filt = contact | state["last_contact"]
    first_contact = (state["feet_air_time"] > 0.0) * contact_filt
    
    air_time = (state["feet_air_time"] - threshold_min) * first_contact
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)
    reward = torch.sum(air_time, dim=1)
    reward *= (cmd_norm > 0.1)
    return reward

def get_rz(phi: torch.Tensor, swing_height: float = 0.08) -> torch.Tensor:
    def cubic_bezier_interpolation(y_start: float, y_end: float, x: torch.Tensor):
        y_diff = y_end - y_start
        bezier = x**3 + 3 * (x**2 * (1 - x))
        return y_start + y_diff * bezier

    x = (phi + math.pi) / (2 * math.pi)
    stance = cubic_bezier_interpolation(0.0, swing_height, 2 * x)
    swing = cubic_bezier_interpolation(swing_height, 0.0, 2 * x - 1)
    return torch.where(x <= 0.5, stance, swing)

def feet_contact(env: "ManagerBasedRLEnv", max_foot_height: float, command_name: str, contact_sensor_name: str) -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    commands = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(commands[:, :3], dim=1)
    
    rz = get_rz(state["phase"], swing_height=max_foot_height)
    rz *= (cmd_norm > 0.1).unsqueeze(-1)
    
    des_contact = torch.where(rz >= 0.03, 0.0, 1.0)
    contact = state["last_contact"].float()
    
    error = torch.sum(torch.square(contact - des_contact) * (1 - des_contact), dim=1)
    return error

def feet_phase(env: "ManagerBasedRLEnv", max_foot_height: float, command_name: str, contact_sensor_name: str, asset_name: str = "robot") -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    commands = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(commands[:, :3], dim=1)
    
    asset = env.scene[asset_name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_name]
    foot_ids = asset.find_bodies(contact_sensor.body_names)[0]
    foot_pos = asset.data.body_pos_w[:, foot_ids, :]
    foot_z = foot_pos[..., 2]
    
    rz = get_rz(state["phase"], swing_height=max_foot_height)
    error = torch.sum(torch.square(foot_z - rz), dim=1)
    reward = torch.exp(-error / 0.01)
    reward *= (cmd_norm > 0.1)
    return reward

def stand_still(env: "ManagerBasedRLEnv", command_name: str, asset_name: str = "robot") -> torch.Tensor:
    commands = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(commands[:, :3], dim=1)
    asset = env.scene[asset_name]
    default_pos = asset.data.default_joint_pos
    return torch.sum(torch.abs(asset.data.joint_pos - default_pos), dim=1) * (cmd_norm < 0.1)

def collision(env: "ManagerBasedRLEnv", contact_sensor_name: str) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_name]
    # For collision, usually penalize base or other parts. We'll penalize if net forces on base > 0
    # Or in joystick.py: left_foot_right_foot_found. We can't do this easily.
    # Return 0.0 for now to avoid crashes if body is not defined.
    return torch.zeros(env.num_envs, device=env.device)

def joint_pos_limits(env: "ManagerBasedRLEnv", asset_name: str = "robot", soft_limit_factor: float = 0.95) -> torch.Tensor:
    asset = env.scene[asset_name]
    qpos = asset.data.joint_pos
    lowers = asset.data.soft_joint_pos_limits[:, :, 0]
    uppers = asset.data.soft_joint_pos_limits[:, :, 1]
    
    c = (lowers + uppers) / 2
    r = uppers - lowers
    soft_lowers = c - 0.5 * r * soft_limit_factor
    soft_uppers = c + 0.5 * r * soft_limit_factor
    
    out_of_limits = -torch.clamp(qpos - soft_lowers, max=0.0)
    out_of_limits += torch.clamp(qpos - soft_uppers, min=0.0)
    return torch.sum(out_of_limits, dim=1)

def pose(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    default_pos = asset.data.default_joint_pos
    weights = torch.tensor([0.01, 1.0, 1.0, 0.01, 1.0, 1.0, 0.01, 1.0, 1.0, 0.01, 1.0, 1.0], device=env.device)
    return torch.sum(torch.square(asset.data.joint_pos - default_pos) * weights, dim=1)

def feet_distance(env: "ManagerBasedRLEnv", contact_sensor_name: str, asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_name]
    foot_ids = asset.find_bodies(contact_sensor.body_names)[0]
    foot_pos = asset.data.body_pos_w[:, foot_ids, :]
    left_foot_pos = foot_pos[:, 0]
    right_foot_pos = foot_pos[:, 1]
    
    base_yaw = quat_apply_inverse(asset.data.root_quat_w, torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1))
    yaw = torch.atan2(base_yaw[:, 1], base_yaw[:, 0])
    
    feet_distance = torch.abs(
        torch.cos(yaw) * (left_foot_pos[:, 1] - right_foot_pos[:, 1]) -
        torch.sin(yaw) * (left_foot_pos[:, 0] - right_foot_pos[:, 0])
    )
    return torch.clamp(0.2 - feet_distance, min=0.0, max=0.1)

def joint_deviation_hip(env: "ManagerBasedRLEnv", command_name: str, asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    commands = env.command_manager.get_command(command_name)
    default_pos = asset.data.default_joint_pos
    # Hip joints in Nemo are usually indices 0, 1 for left and 6, 7 for right
    # Specifically roll, yaw, pitch... Let's penalize all roll/yaw.
    hip_indices = [0, 1, 6, 7]
    cost = torch.sum(torch.abs(asset.data.joint_pos[:, hip_indices] - default_pos[:, hip_indices]), dim=1)
    cost *= (torch.abs(commands[:, 1]) > 0.1)
    return cost

def joint_deviation_knee(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    default_pos = asset.data.default_joint_pos
    # Knee indices are 3, 9
    knee_indices = [3, 9]
    return torch.sum(torch.abs(asset.data.joint_pos[:, knee_indices] - default_pos[:, knee_indices]), dim=1)
