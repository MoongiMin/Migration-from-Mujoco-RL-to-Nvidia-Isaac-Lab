import argparse
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import h5py
import gymnasium as gym

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
    env_cfg.scene.num_envs = 2
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("Environment initialized successfully!")
    env.reset()
    for _ in range(5):
        obs, rew, term, trunc, info = env.step(torch.zeros(2, env.action_manager.total_action_dim, device=env.device))
    print("Step passed successfully!")

if __name__ == "__main__":
    main()
    simulation_app.close()
