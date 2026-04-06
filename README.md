# MuJoCo NEMO → NVIDIA Isaac Lab migration (starter)

This repository collects **asset configuration and demo scripts** for using the MuJoCo MJCF–based **NEMO** biped in **Isaac Lab**.  
It does **not** include MuJoCo / Brax / `legged_rl` training code (keep that in a separate project).

- **Isaac Lab**: official docs [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)  
- Copy the paths below **as-is** into your Isaac Lab install tree, or keep the same layout relative to the repo root.

---

## Important Model File Locations

When working with this repository, you must place the model files in specific locations:

1. **Source MJCF (MuJoCo format):**
   - The original `nemo5.xml` and its `assets/*.stl` files (from the `legged_rl` repository) should be used as the source for conversion.
   - For stability during conversion on Windows, it is recommended to copy these to a short path like `C:\temp\nemo_mjcf\`.

2. **Target USD (Isaac Sim format):**
   - The converted USD file **must** be saved exactly here:
     `source/isaaclab_assets/data/Robots/Nemo/nemo.usd`
   - This repository provides a `.gitkeep` placeholder for this directory. The `.usd` file itself is large and ignored by git.

---

## What is in this repository

| Path | Purpose |
|------|---------|
| `source/isaaclab_assets/isaaclab_assets/robots/nemo.py` | `NEMO_CFG` (`ArticulationCfg`) aligned with MuJoCo `nemo5.xml`: 12 leg joint names, stiffness, damping, torque limits, and home pose. |
| `source/isaaclab_assets/data/Robots/Nemo/CONVERSION.md` | Detailed guide on how to convert the MJCF to USD, including troubleshooting for common crashes on Windows. |
| `scripts/demos/nemo_viewer.py` | Demo that spawns NEMO in Isaac Sim when `nemo.usd` is present: holds default pose and resets periodically for a **visual sanity check**. |
| `scripts/tools/` | Helper scripts for headless MJCF-to-USD conversion and fixing duplicate articulation roots in the generated USD. |

### What `nemo.py` does (summary)

- Defines **spawn USD path**, **initial root height and joint defaults**, and **implicit actuator groups** (hip/knee/foot_pitch vs roll/yaw/foot_roll) via `ArticulationCfg`.
- Joint names follow MJCF actuators: 12 DoFs as `l_*` / `r_*`.
- If the importer renames joints in USD, update `joint_pos` keys or actuator `joint_names_expr` in `NEMO_CFG` to match.

### What `nemo_viewer.py` does (summary)

- Adds a ground plane and dome light, spawns the robot with `NEMO_CFG`.
- Runs a simple loop: track default joint targets and reset on a fixed interval (basic standing check).

---

## How to integrate into Isaac Lab

1. From your cloned [Isaac Lab](https://github.com/isaac-sim/IsaacLab) root, copy these paths with the **same relative layout**:  
   - `source/isaaclab_assets/isaaclab_assets/robots/nemo.py`  
   - `source/isaaclab_assets/data/Robots/Nemo/` (including `CONVERSION.md` and `.gitkeep`)  
   - `scripts/demos/nemo_viewer.py`
   - `scripts/tools/` (optional, for conversion helpers)
2. `isaaclab_assets` often does **not** auto-import `nemo`. In the demo, import explicitly:  
   `from isaaclab_assets.robots.nemo import NEMO_CFG`
3. Convert the MJCF to USD and save it to **`source/isaaclab_assets/data/Robots/Nemo/nemo.usd`** (see `CONVERSION.md` for details).
4. Example run (Windows typically uses `isaaclab.bat`):

   ```bat
   isaaclab.bat -p scripts\demos\nemo_viewer.py
   ```

Running Isaac Sim alongside heavy training (e.g. Brax) can use a lot of VRAM — **prefer separating training and the viewer**.

---

## Roadmap / next steps

1. **Build and validate `nemo.usd`** — after MJCF import, verify joint names, axis directions, and contacts.
2. **`ManagerBasedRLEnv` (or equivalent)** — port MuJoCo rewards, observations, and commands (e.g. joystick commands) to Isaac Lab observation / reward / action APIs.
3. **Training pipeline with RSL-RL** (or Isaac Lab tutorial patterns) — checkpoints and evaluation loops.
4. **(Optional)** Decide whether to **transfer weights** from a MuJoCo-trained policy or **retrain from scratch** in Isaac.

---

## License

The added `nemo.py` and `nemo_viewer.py` files follow **BSD-3-Clause**, per the SPDX headers (same style as the Isaac Lab project).

---

## Links

- This repository: [Migration-from-Mujoco-RL-to-Nvidia-Isaac-Lab](https://github.com/MoongiMin/Migration-from-Mujoco-RL-to-Nvidia-Isaac-Lab)