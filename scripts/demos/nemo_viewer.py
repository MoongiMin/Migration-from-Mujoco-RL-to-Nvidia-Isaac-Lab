# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawn NEMO (``NEMO_CFG``) in Isaac Sim for visual check.

Requires ``source/isaaclab_assets/data/Robots/Nemo/nemo.usd`` (export from MJCF; see
``isaaclab_assets/robots/nemo.py``).

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/nemo_viewer.py

Headless:

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/nemo_viewer.py --headless

"""

"""Launch Isaac Sim first."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View NEMO biped in Isaac Sim.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots.nemo import NEMO_CFG  # isort: skip


def _usd_path() -> Path:
    return Path(NEMO_CFG.spawn.usd_path)


def design_scene(sim: SimulationContext) -> tuple[Articulation, torch.Tensor]:
    p = _usd_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"NEMO USD not found: {p}\n"
            "Export your MJCF to this path (see docstring in isaaclab_assets/robots/nemo.py)."
        )

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    origins = torch.tensor([[0.0, 0.0, 0.0]], device=sim.device)
    robot = Articulation(NEMO_CFG.replace(prim_path="/World/Nemo"))
    return robot, origins


def run_simulator(sim: SimulationContext, robot: Articulation, origins: torch.Tensor):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    while simulation_app.is_running():
        if count % 200 == 0:
            sim_time = 0.0
            count = 0
            joint_pos = robot.data.default_joint_pos
            joint_vel = robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins[0]
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.reset()
            print("[INFO] Reset NEMO")

        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        robot.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        robot.update(sim_dt)


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 1.2], target=[0.0, 0.0, 0.8])

    robot, origins = design_scene(sim)
    sim.reset()
    print("[INFO] NEMO viewer running. Close the window or Ctrl+C to exit.")
    run_simulator(sim, robot, origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
