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
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to a saved .pt file. If omitted: use latest logs/nemo_locomotion/model_*.pt, else bundled checkpoints/nemo_locomotion_latest.pt.",
)
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
        video_dir = os.path.abspath("logs/nemo_locomotion/videos").replace("\\", "/")
        print(f"[INFO] Recording video to: {video_dir}", flush=True)
        env = gym.wrappers.RecordVideo(
            env, 
            video_dir, 
            step_trigger=lambda step: step == 0, 
            video_length=args_cli.video_length
        )
    
    env = RslRlVecEnvWrapper(env)
    
    # 3. Resolve checkpoint path (optional explicit path vs logs vs bundled artifact)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    bundled_ckpt = os.path.join(_script_dir, "checkpoints", "nemo_locomotion_latest.pt")

    def _iter_from_model_name(filename: str) -> int:
        try:
            return int(filename.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            return -1

    run_dir = "logs/nemo_locomotion"
    resume_path = None

    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
        if not os.path.isfile(resume_path):
            print(f"[ERROR] Checkpoint not found: {resume_path}")
            return
        run_dir = os.path.dirname(resume_path)
    elif os.path.exists(run_dir):
        candidates = [
            f for f in os.listdir(run_dir) if f.startswith("model_") and f.endswith(".pt")
        ]
        if candidates:
            resume_path = os.path.join(run_dir, max(candidates, key=_iter_from_model_name))

    if resume_path is None and os.path.isfile(bundled_ckpt):
        print(f"[INFO] Using bundled checkpoint: {bundled_ckpt}")
        resume_path = bundled_ckpt
        run_dir = os.path.dirname(resume_path)

    if resume_path is None:
        print(
            f"[ERROR] No checkpoint found under {run_dir} (expect model_*.pt), "
            "no bundled checkpoints/nemo_locomotion_latest.pt, and --checkpoint not set."
        )
        return

    # 4. Configure runner to load the model
    runner_cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=20,
        max_iterations=1,
        experiment_name="nemo_locomotion",
        empirical_normalization=True,
        obs_groups={
            "actor": ["policy"],
            "critic": ["policy", "privileged"],
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
            clip_param=0.2,             
            entropy_coef=0.005,          
            num_learning_epochs=4,
            num_mini_batches=32,
            learning_rate=3e-4,
            max_grad_norm=1.0,
            gamma=0.97,
            lam=0.95,
            desired_kl=0.01,
            schedule="adaptive",
        ),
    )

    print(f"[INFO] Loading model from: {resume_path}", flush=True)
    
    # Initialize runner and load the policy
    installed_version = metadata.version("rsl-rl-lib")
    runner_cfg = handle_deprecated_rsl_rl_cfg(runner_cfg, installed_version)
    runner = OnPolicyRunner(env, runner_cfg.to_dict(), log_dir=run_dir, device=env.unwrapped.device)
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
    # 5. Playback loop — inference_mode avoids autograd teardown issues on Windows at env.close().
    step_count = 0
    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs)
            step_returns = env.step(actions)
            obs = step_returns[0]
            step_count += 1

            if step_count % 100 == 0:
                print(f"[INFO] Performed {step_count} steps.", flush=True)

            if args_cli.video and step_count >= args_cli.video_length:
                print(f"[INFO] Video recording finished ({args_cli.video_length} steps). Exiting...", flush=True)
                break

    # RecordVideo.close() closes Isaac before stop_recording(); flush mp4 here first.
    if args_cli.video:
        recorder = getattr(env, "env", None)
        if recorder is not None and getattr(recorder, "recording", False):
            print("[INFO] Finalizing video (stop_recording) before Isaac env shutdown...", flush=True)
            recorder.stop_recording()

    del runner
    del policy

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
