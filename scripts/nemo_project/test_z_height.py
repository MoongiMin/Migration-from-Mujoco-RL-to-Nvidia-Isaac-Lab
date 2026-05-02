import argparse
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import h5py

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from nemo_env_cfg import NemoEnvCfg

def main():
    env_cfg = NemoEnvCfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()
    for i in range(20):
        obs, rew, term, trunc, info = env.step(torch.zeros(1, env.action_manager.total_action_dim, device=env.device))
        root_z = env.scene["robot"].data.root_pos_w[:, 2]
        print(f"Step {i}: root_z = {root_z.item():.4f}")
        if term.item():
            print(f"Terminated at step {i}")
            break

if __name__ == "__main__":
    main()
    simulation_app.close()
