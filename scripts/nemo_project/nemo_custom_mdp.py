import torch
import math
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

# region agent log
_NEMO_DBG_LOG = Path(__file__).resolve().parents[2] / "debug-8591d2.log"


def _nemo_maybe_dbg_aggregate(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str) -> None:
    """Rare NDJSON aggregates for gait vs stand hypotheses (once per simulation step max). Session 8591d2."""
    cc = getattr(env, "common_step_counter", None)
    if cc is None:
        return
    if getattr(env, "_nemo_dbg_logged_cc", -1) == cc:
        return
    setattr(env, "_nemo_dbg_logged_cc", cc)
    if cc % 256 != 0:
        return
    commands = env.command_manager.get_command(command_name)
    cn = torch.norm(commands[:, :3], dim=1).detach()
    g = cn > LOCOMOTION_CMD_GATE
    data = {
        "common_step": int(cc),
        "hypothesisId": "H_cmd_gait_contact_aggregate",
        "mean_cmd_norm_xyw": float(cn.mean().cpu()),
        "p50_cmd_norm": float(torch.quantile(cn.float(), 0.5).cpu()),
        "p90_cmd_norm": float(torch.quantile(cn.float(), 0.9).cpu()),
        "frac_cn_gt_gate": float(g.float().mean().cpu()),
        "frac_cn_gt_0p2": float((cn > 0.2).float().mean().cpu()),
        "frac_cn_le_0p01_phase_pin": float((cn <= 0.01).float().mean().cpu()),
        "num_envs": int(env.num_envs),
    }
    try:
        st = getattr(env, "nemo_state", None)
        if st is not None:
            contact = _sensor_foot_contact_b(env, contact_sensor_name)
            data["mean_feet_air_time"] = float(st["feet_air_time"].mean().cpu())
            data["both_feet_contact_frac"] = float(contact.all(dim=1).float().mean().cpu())
            data["single_feet_contact_frac"] = float(
                (contact[:, 0] ^ contact[:, 1]).float().mean().cpu()
            )
            if g.any():
                data["mean_feet_air_gate"] = float(st["feet_air_time"][g].mean().cpu())
    except Exception:
        pass
    line = {
        "sessionId": "8591d2",
        "runId": "train",
        "hypothesisId": data.get("hypothesisId", ""),
        "location": "nemo_custom_mdp.py:_nemo_maybe_dbg_aggregate",
        "message": "nemo_mdp_aggregate",
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_NEMO_DBG_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
    except OSError:
        pass


# endregion agent log
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, quat_apply


# Joystick parity: gait shaping only when ``‖v_cmd‖ > 0.1`` (`feet_air_time`, `feet_phase`, `feet_contact` rz mask).
LOCOMOTION_CMD_GATE: float = 0.1


def _sensor_foot_contact_b(
    env: "ManagerBasedRLEnv",
    contact_sensor_name: str,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Foot contact Booleans `(N, 2)` from net contact forces (aligned with Joystick instantaneous ``contact``)."""
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_name]
    hist = contact_sensor.data.net_forces_w_history
    if hist is not None and hist.numel() > 0:
        forces = hist[:, :, :, :].norm(dim=-1).max(dim=1)[0]
        return forces > force_threshold
    return torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.bool)


def finalize_nemo_foot_air_after_rewards(
    env: "ManagerBasedRLEnv",
    contact_sensor_name: str,
    force_threshold: float = 1.0,
) -> None:
    """MuJoCo end-of-step parity: ``feet_air_time *= ~contact``, ``swing_peak *= ~contact``, ``last_contact = contact``.

    Isaac computes rewards **before** observations. Joystick clears air time **after** ``_get_reward``.
    Calling this once per env step **after rewards** restores landing credit and correct ``contact | last_contact`` hysteresis.
    """
    if not hasattr(env, "nemo_state"):
        return
    if getattr(env, "_nemo_air_finalize_at_common_step", None) == env.common_step_counter:
        return
    contact = _sensor_foot_contact_b(env, contact_sensor_name, force_threshold)
    ft = env.nemo_state["feet_air_time"].dtype
    mask = (~contact).to(dtype=ft)
    env.nemo_state["feet_air_time"] *= mask
    env.nemo_state["swing_peak"] *= mask
    env.nemo_state["last_contact"] = contact
    env._nemo_air_finalize_at_common_step = env.common_step_counter


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
        
        contact = _sensor_foot_contact_b(env, contact_sensor_name)[idx]
        
        # Action update
        if hasattr(env, "action_manager") and hasattr(env.action_manager, "action"):
            current_act = env.action_manager.action[idx]
        else:
            current_act = torch.zeros(len(idx), env.action_manager.total_action_dim, device=env.device, dtype=torch.float32)
            
        env.nemo_state["last_last_act"][idx] = env.nemo_state["last_act"][idx]
        env.nemo_state["last_act"][idx] = current_act
        
        # Velocity update
        # Local linvel:
        linvel = quat_apply_inverse(asset.data.root_quat_w[idx], asset.data.root_lin_vel_w[idx, :3])
        env.nemo_state["filtered_linvel"][idx] = linvel * 1.0 # filtered = current * 1.0 + old * 0.0
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

        # Touchdown clears and last_contact lag (Joystick: after _get_reward). See finalize_nemo_foot_air_after_rewards.

        if hasattr(env, "episode_length_buf"):
            env.nemo_state["updated_step"][idx] = env.episode_length_buf[idx]

    # region agent log
    _nemo_maybe_dbg_aggregate(env, command_name, contact_sensor_name)
    # endregion agent log

    return env.nemo_state

def _update_and_get_nemo_state(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str):
    return get_nemo_state(env, command_name, asset_name="robot", contact_sensor_name=contact_sensor_name)


def torso_up_tip_over(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    min_world_up_z: float = 0.0,
) -> torch.Tensor:
    """Joystick ``_get_termination``: ``fall_termination = upvector_world[-1] < 0`` (past horizontal flip).

    Here ``upvector`` is the torso's local +Z axis expressed in world. Upright humanoid has ``up_world_z > 0``;
    clearly tipped / upside-down crosses to ``< 0`` (threshold optional).
    """
    asset = env.scene[asset_name]
    ez = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=torch.float32).expand(env.num_envs, 3)
    up_world_z = quat_apply(asset.data.root_quat_w, ez)[:, 2]
    return up_world_z < min_world_up_z


# ========================
# PRIVILEGED OBSERVATIONS (critic suffix; asymmetric actor–critic like joystick privileged_state)
# ========================

def privileged_root_lin_acc_b(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    """Root link linear acceleration in the root (base) frame, approx. IMU linear acceleration."""
    asset = env.scene[asset_name]
    lin_acc_w = asset.data.body_lin_acc_w[:, 0, :]
    acc_b = quat_apply_inverse(asset.data.root_quat_w, lin_acc_w)
    return torch.nan_to_num(acc_b, nan=0.0, posinf=0.0, neginf=0.0)


def privileged_foot_contact(env: "ManagerBasedRLEnv", contact_sensor_name: str, asset_name: str = "robot") -> torch.Tensor:
    """Per-foot binary contact flags (aligned with ContactSensor bodies), shape (N, 2)."""
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_name]
    if contact_sensor.data.net_forces_w_history is not None and contact_sensor.data.net_forces_w_history.numel() > 0:
        forces = contact_sensor.data.net_forces_w_history[:, :, :, :].norm(dim=-1).max(dim=1)[0]
        return (forces > 1.0).to(dtype=torch.float32)
    return torch.zeros(env.num_envs, 2, device=env.device, dtype=torch.float32)


def privileged_feet_lin_vel_w(env: "ManagerBasedRLEnv", contact_sensor_name: str, asset_name: str = "robot") -> torch.Tensor:
    """Flattened world-frame linear velocities of foot bodies, shape (N, 6) for two feet."""
    asset = env.scene[asset_name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_name]
    foot_ids = asset.find_bodies(contact_sensor.body_names)[0]
    v = asset.data.body_lin_vel_w[:, foot_ids, :]
    return v.reshape(env.num_envs, -1)


def privileged_feet_air_time(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str) -> torch.Tensor:
    finalize_nemo_foot_air_after_rewards(env, contact_sensor_name)
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    return state["feet_air_time"]


# ========================
# OBSERVATIONS
# ========================

def phase(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str) -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    return torch.cat([torch.cos(state["phase"]), torch.sin(state["phase"])], dim=1)

def filtered_linvel(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str) -> torch.Tensor:
    finalize_nemo_foot_air_after_rewards(env, contact_sensor_name)
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


def locomotion_planar_idle_cost(
    env: "ManagerBasedRLEnv",
    command_name: str,
    contact_sensor_name: str,
    speed_variance: float = 0.04,
) -> torch.Tensor:
    """Cost in ``[0, 1]``: high when locomotion velocity is commanded but base planar speed (body frame) is ~0.

    Suppresses “upright statue” optimum under non-zero velocity commands without rewarding mere survival each step (no ``alive`` term).
    Stand / near-zero-command envs (``‖v_cmd‖`` below gait gate) are unaffected.
    """
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    commands = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(commands[:, :3], dim=1)
    v_xy = state["filtered_linvel"][:, :2]
    s2 = torch.sum(torch.square(v_xy), dim=1)
    idle = torch.exp(-s2 / max(speed_variance, 1e-8))
    gate = (cmd_norm > LOCOMOTION_CMD_GATE).to(dtype=idle.dtype)
    return idle * gate


def locomotion_velocity_command_alignment(
    env: "ManagerBasedRLEnv",
    command_name: str,
    contact_sensor_name: str,
    clamp_max: float = 1.25,
) -> torch.Tensor:
    """Convex ``forward progress'' cue: planar base velocity projected onto planar velocity command.

    Complements exponential ``tracking_lin_vel`` by giving **linear gradient** along the commanded direction when
    under-speed (Kelly et al. note underspecified velocity-only cues have high-variance gait; aligning phase-based
    contact costs with softened penalties still needs explicit progress signal in practice).
    """
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    commands = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(commands[:, :3], dim=1)
    c_xy = commands[:, :2]
    planar = torch.norm(c_xy, dim=1)
    plane_gate = planar > 0.05
    v_xy = state["filtered_linvel"][:, :2]
    align = torch.sum(v_xy * c_xy, dim=1) / torch.clamp(planar, min=1e-4)
    r = torch.clamp(align, min=0.0, max=clamp_max)
    cmd_gate = (cmd_norm > LOCOMOTION_CMD_GATE).float()
    return r * cmd_gate * plane_gate.to(dtype=r.dtype)


def locomotion_single_support_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str,
    contact_sensor_name: str,
) -> torch.Tensor:
    """Bonus when exactly one foot is in contact — breaks frozen double-support under locomotion gate.

    Real walking has stance/swing; XOR contact is a cheap proxy (Kelly-style periodic decomposition is richer but
    needs per-foot force curves). Weight kept moderate so nominal double stance is still allowed occasionally.
    """
    _ = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    contact = _sensor_foot_contact_b(env, contact_sensor_name)
    xor_f = torch.logical_xor(contact[:, 0], contact[:, 1]).to(dtype=torch.float32)
    commands = env.command_manager.get_command(command_name)
    cn = torch.norm(commands[:, :3], dim=1)
    gate = (cn > LOCOMOTION_CMD_GATE).to(dtype=torch.float32)
    return xor_f * gate


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


def stand_double_support(
    env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str, cmd_eps: float = 0.1
) -> torch.Tensor:
    """1 when ‖v_cmd‖ < eps (stand episode) and both feet report contact."""
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    commands = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(commands[:, :3], dim=1)
    both = state["last_contact"].all(dim=1).float()
    return both * (cmd_norm < cmd_eps).float()


def stand_base_motion_cost(
    env: "ManagerBasedRLEnv",
    command_name: str,
    contact_sensor_name: str,
    cmd_eps: float = 0.1,
    ang_scale: float = 0.08,
    asset_name: str = "robot",
) -> torch.Tensor:
    """When ``stand``: cost on root linear/angular world velocity (defines ``still'' stance)."""
    _ = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    commands = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(commands[:, :3], dim=1)
    asset = env.scene[asset_name]
    mask = (cmd_norm < cmd_eps).float()
    lin_c = torch.sum(torch.square(asset.data.root_lin_vel_w[:, :3]), dim=1)
    ang_c = torch.sum(torch.square(asset.data.root_ang_vel_w[:, :3]), dim=1)
    return (lin_c + ang_scale * ang_c) * mask


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
    acc = torch.square(torch.nan_to_num(asset.data.joint_acc, nan=0.0, posinf=0.0, neginf=0.0))
    return torch.sum(acc, dim=1)

def dof_vel(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    return torch.sum(torch.square(asset.data.joint_vel), dim=1)

def feet_slip(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str, asset_name: str = "robot") -> torch.Tensor:
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    contact = _sensor_foot_contact_b(env, contact_sensor_name).to(dtype=torch.float32)
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
    contact = _sensor_foot_contact_b(env, contact_sensor_name)
    contact_filt = contact | state["last_contact"]
    air_post = state["feet_air_time"]
    air_pre = torch.clamp(air_post - env.step_dt, min=0.0)
    fc = air_pre.gt(0.0).to(air_post.dtype) * contact_filt.to(air_post.dtype)

    error = state["swing_peak"] / max_foot_height - 1.0
    return torch.sum(torch.square(error) * fc, dim=1)

def feet_air_time(env: "ManagerBasedRLEnv", command_name: str, contact_sensor_name: str, threshold_min: float = 0.2, threshold_max: float = 0.5) -> torch.Tensor:
    """Joystick ``_reward_feet_air_time`` (uses pre-increment air for ``first_contact``, post-increment for duration)."""
    state = _update_and_get_nemo_state(env, command_name, contact_sensor_name)
    commands = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(commands[:, :3], dim=1)

    contact = _sensor_foot_contact_b(env, contact_sensor_name)
    contact_filt = contact | state["last_contact"]
    air_post = state["feet_air_time"]
    air_pre = torch.clamp(air_post - env.step_dt, min=0.0)
    first_contact = air_pre.gt(0.0).to(air_post.dtype) * contact_filt.to(air_post.dtype)

    air_piece = (air_post - threshold_min) * first_contact
    air_piece = torch.clamp(air_piece, max=threshold_max - threshold_min)
    reward = torch.sum(air_piece, dim=1)
    reward *= (cmd_norm > LOCOMOTION_CMD_GATE).to(reward.dtype)
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
    rz *= (cmd_norm > LOCOMOTION_CMD_GATE).unsqueeze(-1).to(rz.dtype)

    des_contact = torch.where(rz >= 0.03, 0.0, 1.0)
    contact = _sensor_foot_contact_b(env, contact_sensor_name).to(dtype=torch.float32)

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
    reward *= (cmd_norm > LOCOMOTION_CMD_GATE).to(reward.dtype)
    return reward

def stand_still(env: "ManagerBasedRLEnv", command_name: str, asset_name: str = "robot") -> torch.Tensor:
    commands = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(commands[:, :3], dim=1)
    asset = env.scene[asset_name]
    default_pos = asset.data.default_joint_pos
    return torch.sum(torch.abs(asset.data.joint_pos - default_pos), dim=1) * (cmd_norm < 0.1)

def collision(
    env: "ManagerBasedRLEnv",
    contact_sensor_name: str,
    asset_name: str = "robot",
    threshold: float = 0.12,
) -> torch.Tensor:
    """Binary-ish proxy for MuJoCo ``left_foot_right_foot_found`` (Isaac lacks that sensor tuple)."""

    asset = env.scene[asset_name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_name]
    foot_ids = asset.find_bodies(contact_sensor.body_names)[0]
    lf = asset.data.body_pos_w[:, foot_ids[0], :2]
    rf = asset.data.body_pos_w[:, foot_ids[1], :2]
    sep = torch.norm(lf - rf, dim=-1)
    return (sep < threshold).to(dtype=torch.float32)

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
    
    if not hasattr(env, "nemo_pose_weights"):
        weights = torch.ones(asset.num_joints, device=env.device)
        for i, name in enumerate(asset.data.joint_names):
            if "hip_pitch" in name or "knee" in name:
                weights[i] = 0.01
            else:
                weights[i] = 1.0
        env.nemo_pose_weights = weights
        
    return torch.sum(torch.square(asset.data.joint_pos - default_pos) * env.nemo_pose_weights, dim=1)

def _lateral_spacing_loss(
    d: torch.Tensor,
    target_sep: float,
    min_sep: float,
    max_sep: float,
    narrow_scale: float = 1.0,
    wide_scale: float = 0.35,
) -> torch.Tensor:
    narrow = torch.relu(min_sep - d)
    wide = torch.relu(d - max_sep)
    center = torch.square(d - target_sep)
    return narrow_scale * narrow.square() + wide_scale * wide.square() + 0.15 * center


def _lateral_sep_in_base_yaw(root_quat_w: torch.Tensor, p_left_xy: torch.Tensor, p_right_xy: torch.Tensor) -> torch.Tensor:
    """Pelvis-facing lateral spacing of two XY points (same convention as MuJoCo / ``feet_distance``)."""

    fwd = torch.tensor([1.0, 0.0, 0.0], device=root_quat_w.device, dtype=torch.float32).expand(root_quat_w.shape[0], 3)
    base_fwd = quat_apply_inverse(root_quat_w, fwd)
    yaw = torch.atan2(base_fwd[:, 1], base_fwd[:, 0])
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    return torch.abs(cos_y * (p_left_xy[:, 1] - p_right_xy[:, 1]) - sin_y * (p_left_xy[:, 0] - p_right_xy[:, 0]))


def lateral_pair_spacing_cost(
    env: "ManagerBasedRLEnv",
    asset_name: str,
    body_left: str,
    body_right: str,
    target_sep: float,
    min_sep: float,
    max_sep: float,
    narrow_scale: float = 1.0,
    wide_scale: float = 0.35,
) -> torch.Tensor:
    """Keep nominal lateral spacing: heavy cost when narrower than ``min_sep`` (crossed legs), lighter when too wide."""
    asset = env.scene[asset_name]
    body_ids, _ = asset.find_bodies([body_left, body_right], preserve_order=True)
    pos = asset.data.body_pos_w[:, body_ids, :2]
    d = _lateral_sep_in_base_yaw(asset.data.root_quat_w, pos[:, 0], pos[:, 1])
    return _lateral_spacing_loss(d, target_sep, min_sep, max_sep, narrow_scale, wide_scale)


def feet_distance(
    env: "ManagerBasedRLEnv",
    contact_sensor_name: str,
    asset_name: str = "robot",
) -> torch.Tensor:
    """MuJoCo ``joystick._cost_feet_distance``: ``clip(0.2 - lateral_sep, 0, 0.1)``."""

    asset = env.scene[asset_name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_name]
    foot_ids = asset.find_bodies(contact_sensor.body_names)[0]
    foot_pos = asset.data.body_pos_w[:, foot_ids, :2]
    d = _lateral_sep_in_base_yaw(asset.data.root_quat_w, foot_pos[:, 0], foot_pos[:, 1])
    return torch.clamp(0.2 - d, min=0.0, max=0.1)


def knee_lateral_spacing(
    env: "ManagerBasedRLEnv",
    asset_name: str = "robot",
    target_sep: float = 0.17,
    min_sep: float = 0.10,
    max_sep: float = 0.38,
) -> torch.Tensor:
    """Same lateral measure between knee links — reduces scissoring when legs adduct excessively."""

    return lateral_pair_spacing_cost(
        env, asset_name, "l_knee", "r_knee", target_sep, min_sep, max_sep, narrow_scale=1.2, wide_scale=0.35
    )

def joint_deviation_hip(env: "ManagerBasedRLEnv", command_name: str, asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    commands = env.command_manager.get_command(command_name)
    default_pos = asset.data.default_joint_pos
    
    if not hasattr(env, "nemo_hip_indices"):
        env.nemo_hip_indices = [i for i, name in enumerate(asset.data.joint_names) if "hip_roll" in name or "hip_yaw" in name]
        
    cost = torch.sum(torch.abs(asset.data.joint_pos[:, env.nemo_hip_indices] - default_pos[:, env.nemo_hip_indices]), dim=1)
    cost *= (torch.abs(commands[:, 1]) > 0.1)
    return cost

def joint_deviation_knee(env: "ManagerBasedRLEnv", asset_name: str = "robot") -> torch.Tensor:
    asset = env.scene[asset_name]
    default_pos = asset.data.default_joint_pos
    
    if not hasattr(env, "nemo_knee_indices"):
        env.nemo_knee_indices = [i for i, name in enumerate(asset.data.joint_names) if "knee" in name]
        
    return torch.sum(torch.abs(asset.data.joint_pos[:, env.nemo_knee_indices] - default_pos[:, env.nemo_knee_indices]), dim=1)
