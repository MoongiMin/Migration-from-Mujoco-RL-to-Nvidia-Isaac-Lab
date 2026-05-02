"""Script to train RL policies using RSL-RL for the Nemo robot.

This script registers the custom Nemo environment and wraps the standard IsaacLab
RSL-RL training workflow.

Usage (run in a terminal, not the Script Editor):
    # Headless training (default: wipes ``logs/nemo_locomotion`` unless --keep_prior_logs)
    isaaclab.bat -p scripts/nemo_project/train.py --headless

    # Short run after reward / MDP edits (example: 150 iterations)
    isaaclab.bat -p scripts/nemo_project/train.py --num_envs 4096 --max_iterations 150 --headless

    # Keep old TensorBoard checkpoints
    isaaclab.bat -p scripts/nemo_project/train.py --keep_prior_logs --headless
"""

import argparse
import sys
import os
import shutil

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import h5py
import gymnasium as gym

from isaaclab.app import AppLauncher

# Set up argparse for the IsaacLab application
parser = argparse.ArgumentParser(description="Train an RL agent for Nemo using RSL-RL.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Nemo-Flat-v0", help="Name of the task.")
parser.add_argument(
    "--keep_prior_logs",
    action="store_true",
    default=False,
    help="Do not delete the experiment log folder before training (TensorBoard events, checkpoints, etc.).",
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=None,
    help="RL iterations when --max_iterations is omitted (see LONG_RUN_* in main()).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Imports must happen after app_launcher
import torch
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
    print("Please install it using: isaaclab.bat -e rsl_rl (from the Isaac Lab repo root).")
    simulation_app.close()
    sys.exit(1)

def main():
    # Wall time scales roughly with max_iterations × per-iter latency (~4–6 s headless; depends on GPU and num_envs).
    LONG_RUN_ITERATIONS = 500
    LONG_RUN_SAVE_INTERVAL = 25  # e.g. model_150.pt lands on a save tick when extending past 125

    # 1. Create the environment
    env_cfg = NemoEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    
    # Initialize the IsaacLab ManagerBasedRLEnv
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # 2. Wrap the environment for RSL-RL
    env = RslRlVecEnvWrapper(env)
    
    max_iter = args_cli.max_iterations if args_cli.max_iterations is not None else LONG_RUN_ITERATIONS

    # 3. Configure PPO (Joystick-aligned LR/clip/discount); exploration boosted vs JAX ``entropy_cost=0.005`` via
    #    higher entropy_coef + larger actor init_std so rollouts probe wider joint targets early in training.
    runner_cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=20,          # unroll_length
        max_iterations=max_iter,
        save_interval=LONG_RUN_SAVE_INTERVAL,
        experiment_name="nemo_locomotion",
        empirical_normalization=True,  # normalize_observations=True
        # Asymmetric actor–critic: actor uses noisy on-policy obs; critic also sees privileged clean-state suffix
        # (same pattern as JAX joystick: value_obs_key="privileged_state" = policy_state || extras).
        obs_groups={
            "actor": ["policy"],
            "critic": ["policy", "privileged"],
        },
        actor=RslRlMLPModelCfg(
            class_name="MLPModel",
            hidden_dims=[512, 256, 256, 128],
            activation="elu",
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
                init_std=0.48,
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
            entropy_coef=0.017,
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

    # Fresh run: remove prior TensorBoard events, checkpoints, and other run artifacts.
    if not args_cli.keep_prior_logs and os.path.isdir(log_dir):
        print(f"[INFO] Removing prior training data under: {log_dir}", flush=True)
        shutil.rmtree(log_dir)

    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=log_dir, device=env.unwrapped.device)

    print(
        "[INFO] MDP/reward wiring: scripts/nemo_project (nemo_env_cfg + nemo_custom_mdp). "
        "Stale checkpoints from older reward definitions mix poorly—prefer a clean log_dir for a new run.",
        flush=True,
    )
    print(
        f"[INFO] Training run: max_iterations={max_iter}, save_interval={LONG_RUN_SAVE_INTERVAL}, "
        f"num_envs={args_cli.num_envs} | logs -> {log_dir}",
        flush=True,
    )
    runner.learn(num_learning_iterations=runner_cfg.max_iterations, init_at_random_ep_len=False)
    
    # Cleanup
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
