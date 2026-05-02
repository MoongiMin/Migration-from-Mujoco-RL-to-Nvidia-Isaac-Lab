import argparse
import math

from isaaclab.app import AppLauncher

# Set up argparse
parser = argparse.ArgumentParser(description="Run Nemo robot and move joints.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

# Import the Nemo configuration class
from nemo_cfg import NEMO_CFG

def main():
    # Set up simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set camera view
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.5])

    # Create ground plane
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # Create dome light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Create Nemo robot
    nemo_cfg = NEMO_CFG.replace(prim_path="/World/Nemo")
    nemo_robot = Articulation(cfg=nemo_cfg)

    # Reset and start simulation
    sim.reset()
    print("[INFO]: Setup complete. Moving joints...")

    # Physics simulation loop
    sim_time = 0.0
    while simulation_app.is_running():
        # Calculate target joint positions (Sine wave)
        # 관절 각도를 -0.5 ~ 0.5 라디안 사이로 움직이게 합니다.
        target_pos = 0.5 * math.sin(sim_time * 2.0)
        
        # Create a tensor with the target position for all joints
        # nemo_robot.num_joints는 로봇의 총 관절 개수입니다.
        joint_targets = torch.full((1, nemo_robot.num_joints), target_pos, device=sim.device)

        # Apply the target positions to the robot's PD controller
        nemo_robot.set_joint_position_target(joint_targets)

        # Step the simulation
        sim.step()
        
        # Update robot state
        nemo_robot.update(sim_cfg.dt)
        sim_time += sim_cfg.dt

if __name__ == "__main__":
    main()
    simulation_app.close()
