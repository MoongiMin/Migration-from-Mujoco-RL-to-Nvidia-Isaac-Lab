# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NEMO biped (MuJoCo MJCF ``nemo5.xml``) — Isaac Lab articulation config.

This project does **not** ship a USD. After you generate ``nemo.usd`` (see below),
place it at:

  ``source/isaaclab_assets/data/Robots/Nemo/nemo.usd``

**How to build ``nemo.usd``**

1. Use the MJCF under ``legged_rl-main/legged_rl-main/models/nemo/`` with mesh
   assets (``assets/*.stl`` next to ``nemo5.xml`` — same layout as MuJoCo expects).
2. In **Isaac Sim**: import MJCF (or convert via URDF if you prefer), fix any
   articulation root / joint names, then **Export** as USD to the path above.
3. Joint names in the USD should match the MJCF actuator joints:

   ``l_hip_pitch``, ``l_hip_roll``, ``l_hip_yaw``, ``l_knee``, ``l_foot_pitch``,
   ``l_foot_roll``, and the same with ``r_*``.

   If the importer renames them, update :attr:`NEMO_CFG` actuators or
   ``joint_pos`` keys to match.

Torque ranges and default pose follow ``nemo5.xml`` / ``home`` keyframe.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab_assets
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


def _nemo_usd_path() -> str:
    """``.../source/isaaclab_assets/data/Robots/Nemo/nemo.usd`` (next to the ``isaaclab_assets`` package)."""
    root = Path(isaaclab_assets.__file__).resolve().parent.parent
    return str(root / "data" / "Robots" / "Nemo" / "nemo.usd")


# MJCF <position kp="50"> for hip_pitch, knee, foot_pitch; <position kp="40"> for roll, yaw, foot_roll.
_KP_HIGH = 50.0
_KP_LOW = 40.0
# MJCF joint damping="1" on all actuated joints.
_JOINT_DAMPING = 1.0
# MJCF actuatorfrcrange magnitude.
_TAU_HIGH = 47.4
_TAU_LOW = 23.7

# Home keyframe from nemo5.xml (12 leg joints: left then right).
_HOME_LEG = (
    -0.698132,
    0.0,
    0.0,
    1.22173,
    -0.523599,
    0.0,
)

NEMO_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_nemo_usd_path(),
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
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5793),
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
"""NEMO biped. Requires ``data/Robots/Nemo/nemo.usd`` next to the ``isaaclab_assets`` package root."""
