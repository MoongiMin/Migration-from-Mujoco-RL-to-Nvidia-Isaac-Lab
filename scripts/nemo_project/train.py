"""Script to train RL policies using RSL-RL for the Nemo robot.

This script registers the custom Nemo environment and wraps the standard IsaacLab
RSL-RL training workflow.

Usage (Run in terminal, NOT in Script Editor):
    # Train the policy in headless mode
    .\isaaclab.bat -p scripts\nemo_project\train.py --headless

    # Or with a specific number of environments
    .\isaaclab.bat -p scripts\nemo_project\train.py --num_envs 4096 --headless
"""

import argparse
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import h5py
import gymnasium as gym

from isaaclab.app import AppLauncher

# Set up argparse for the IsaacLab application
parser = argparse.ArgumentParser(description="Train an RL agent for Nemo using RSL-RL.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Nemo-Flat-v0", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports must happen after app_launcher
import torch
import os
from isaaclab.envs import ManagerBasedRLEnv

# Import our custom environment configuration
from nemo_env_cfg import NemoEnvCfg

# Register the environment in OpenAI Gym registry
gym.register(
    id="Nemo-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": NemoEnvCfg,
    },
)

print("[INFO] Environment registered successfully as 'Nemo-Flat-v0'.")

# Try to import RSL-RL wrappers and runner
try:
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
        RslRlMLPModelCfg,
        handle_deprecated_rsl_rl_cfg
    )
    from rsl_rl.runners import OnPolicyRunner
    import importlib.metadata as metadata
except ImportError:
    print("[ERROR] RSL-RL is not installed or isaaclab_rl wrapper is missing.")
    print("Please install it using: .\isaaclab.bat -e rsl_rl")
    simulation_app.close()
    sys.exit(1)

def main():
    # 1. Create the environment
    env_cfg = NemoEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Initialize the IsaacLab ManagerBasedRLEnv
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # 2. Wrap the environment for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    # 3. Configure the PPO Algorithm (matching your legacy ppo_params roughly)
    runner_cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=20,          # unroll_length
        max_iterations=306,            # 50,000,000 / (8192 * 20) ~= 305
        save_interval=50,              # Checkpoint save interval
        experiment_name="nemo_locomotion",
        empirical_normalization=True,  # normalize_observations=True
        obs_groups={
            "actor": ["policy"],
            "critic": ["policy"],
        },
        actor=RslRlMLPModelCfg(
            class_name="MLPModel",
            hidden_dims=[512, 256, 256, 128],
            activation="elu",
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
                init_std=1.0,
            ),
        ),
        critic=RslRlMLPModelCfg(
            class_name="MLPModel",
            hidden_dims=[512, 256, 256, 128],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,             # clipping_epsilon
            entropy_coef=0.005,         # entropy_cost
            num_learning_epochs=4,      # num_updates_per_batch
            num_mini_batches=32,
            learning_rate=3e-4,
            max_grad_norm=1.0,
            gamma=0.97,                 # discounting
            lam=0.95,
            desired_kl=0.01,
            schedule="adaptive",
        ),
    )
    
    # 4. Initialize and run the RSL-RL OnPolicyRunner
    installed_version = metadata.version("rsl-rl-lib")
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, installed_version)
    log_dir = os.path.join("logs", runner_cfg.experiment_name)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=env.unwrapped.device)
    
    print(f"[INFO] Starting training... Logs will be saved to: {log_dir}")
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=True)
    
    # Cleanup
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
