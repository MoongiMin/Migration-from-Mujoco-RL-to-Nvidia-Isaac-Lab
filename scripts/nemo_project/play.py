"""Script to run a trained RL policy for the Nemo robot in the GUI.

Usage:
    # Run the trained policy and watch the robot walk!
    .\isaaclab.bat -p scripts\nemo_project\play.py
"""

import argparse
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import h5py
import gymnasium as gym

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a trained RL agent for Nemo.")
parser.add_argument("--task", type=str, default="Nemo-Flat-v0", help="Name of the task.")
parser.add_argument("--video", action="store_true", default=False, help="Record video of the playback.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from nemo_env_cfg import NemoEnvCfg

# Register environment
gym.register(
    id="Nemo-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": NemoEnvCfg},
)

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
    print("[ERROR] RSL-RL is not installed.")
    simulation_app.close()
    sys.exit(1)

def main():
    # 1. Load Environment Configuration
    env_cfg = NemoEnvCfg()
    env_cfg.scene.num_envs = 36 # Spawn 36 robots to watch them walk
    
    # 2. Create the environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    
    if args_cli.video:
        video_dir = os.path.join("logs/nemo_locomotion", "videos")
        print(f"[INFO] Recording video to: {video_dir}", flush=True)
        env = gym.wrappers.RecordVideo(
            env, 
            video_dir, 
            step_trigger=lambda step: step == 0, 
            video_length=args_cli.video_length
        )
    
    env = RslRlVecEnvWrapper(env)
    
    # 3. Find the trained model directory
    run_dir = "logs/nemo_locomotion"
    if not os.path.exists(run_dir):
        print(f"[ERROR] No trained models found in {run_dir}. Please run train.py first.")
        return
        
    # 4. Configure runner to load the model
    runner_cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=1,
        experiment_name="nemo_locomotion",
        empirical_normalization=False,
        obs_groups={
            "actor": ["policy"],
            "critic": ["policy"],
        },
        actor=RslRlMLPModelCfg(
            class_name="MLPModel",
            hidden_dims=[512, 256, 128],
            activation="elu",
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
                init_std=1.0,
            ),
        ),
        critic=RslRlMLPModelCfg(
            class_name="MLPModel",
            hidden_dims=[512, 256, 128],
            activation="elu",
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            class_name="PPO",
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,             
            entropy_coef=0.01,          
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            max_grad_norm=1.0,
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            schedule="adaptive",
        ),
    )
    
    # Initialize runner and load the policy
    installed_version = metadata.version("rsl-rl-lib")
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, installed_version)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=run_dir, device=env.unwrapped.device)
    
    # Find the latest model file in the directory
    model_files = [f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not model_files:
        print(f"[ERROR] Could not find any model_*.pt files in {run_dir}")
        return
        
    def get_iteration(filename):
        try:
            return int(filename.split("_")[1].split(".")[0])
        except ValueError:
            return -1
            
    latest_model = sorted(model_files, key=get_iteration)[-1]
    resume_path = os.path.join(run_dir, latest_model)
    
    print(f"[INFO] Loading model from: {resume_path}", flush=True)
    runner.load(resume_path)
        
    # Get the policy function
    print("[INFO] Getting inference policy...", flush=True)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    
    print("[INFO] Starting playback. Watch the robots walk!", flush=True)
    obs = env.get_observations()
    # Depending on IsaacLab version, get_observations might return a dict or tuple
    if isinstance(obs, tuple):
        obs = obs[0]
    
    print("[INFO] Entering playback loop...", flush=True)
    # 5. Playback loop
    step_count = 0
    while simulation_app.is_running():
        # Get actions from the trained policy
        actions = policy(obs)
        print("[DEBUG] Policy generated actions", flush=True)
        # Apply actions to the environment
        step_returns = env.step(actions)
        print("[DEBUG] Env stepped", flush=True)
        obs = step_returns[0]
        step_count += 1
        if step_count % 100 == 0:
            print(f"[INFO] Performed {step_count} steps.", flush=True)

if __name__ == "__main__":
    main()
    simulation_app.close()
