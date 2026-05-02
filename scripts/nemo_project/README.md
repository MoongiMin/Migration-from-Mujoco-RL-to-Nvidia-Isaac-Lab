# Nemo Robot Integration Guide for IsaacLab

This guide explains how to import the Nemo robot (from an MJCF `.xml` file) into NVIDIA Isaac Sim and IsaacLab, configure its physics properties, and test its default standing pose.

---

## Step 1: Convert MJCF to USD
Isaac Sim uses the Universal Scene Description (USD) format for physics simulation. First, convert your raw `.xml` (MJCF) file into `.usd`.

1. Open your terminal (e.g., Anaconda Prompt or WSL) and activate your IsaacLab environment.
2. Run the conversion script provided by IsaacLab:
   ```bash
   .\isaaclab.bat -p scripts\tools\convert_mjcf.py "path/to/your/nemo.xml" "source/isaaclab_assets/data/Robots/Nemo/nemo.usd"
   ```
   *(This will generate a `nemo.usd` file in the specified output directory.)*

---

## Step 2: Fix Collisions in Isaac Sim GUI
By default, the conversion script might only import visual meshes without physical collision boundaries. If the robot falls through the floor, you must add colliders manually.

1. Open the **NVIDIA Omniverse Launcher** and launch **Isaac Sim**.
2. Go to **File > Open** and open your newly created `nemo.usd` file.
3. In the **Stage** panel (top right), expand the robot's hierarchy (e.g., `pelvis` -> `l_hip_pitch` -> etc.).
4. Find and select all the **Mesh** objects (the icons that look like wireframe polygons). **Do not select the Xform folders (axis icons).**
   * *Tip: Use `Ctrl` or `Shift` to select multiple meshes at once.*
5. With the meshes selected, scroll down to the bottom of the **Property** panel (bottom right).
6. Click **[+ Add] > Physics > Collider Preset** (or `Collider`).
7. In the newly added **Collider** section, change the **Approximation** setting to **`Convex Hull`**.
8. Go to **File > Save** to overwrite and save the `nemo.usd` file.

---

## Step 3: Configure the Robot in Python (`nemo_cfg.py`)
To use the robot in reinforcement learning environments, you must define an `ArticulationCfg` class. This tells IsaacLab where the USD file is, what its default pose is, and how strong its motors are.

1. Create a file named `nemo_cfg.py` (e.g., in `scripts/nemo_project/`).
2. Define the `ArticulationCfg`. Key parameters to tune:
   * `usd_path`: The absolute or relative path to your `nemo.usd`.
   * `init_state.pos`: The spawn height (e.g., `z = 0.6`) so it doesn't spawn inside the floor.
   * `init_state.joint_pos`: The default standing pose. For bipeds/quadrupeds, slightly bent knees (`-0.4` rad) and pitched hips/ankles (`0.2` rad) offer better stability than completely straight legs (`0.0`).
   * `actuators.stiffness` & `damping`: The PD controller gains. Increase these (e.g., `800.0` and `80.0`) if the robot collapses under its own weight.

*(See the `nemo_cfg.py` file in this directory for the exact code template.)*

---

## Step 4: Test the Standing Pose in Isaac Sim GUI (CRITICAL)

> **⚠️ IMPORTANT WARNING FOR WINDOWS USERS ⚠️**
> Do **NOT** run GUI testing scripts via the terminal (e.g., `.\isaaclab.bat -p script.py`). Doing so on Windows often causes **fatal RTX/Vulkan crashes** during viewport initialization. 
> 
> **You MUST use the built-in Script Editor inside the Isaac Sim application to test Python scripts that require a GUI.**

Before writing complex Reinforcement Learning code, verify that the robot can hold its weight using a simple Python script inside the Isaac Sim Script Editor.

1. Open the standalone **Isaac Sim** application and create an empty stage (**File > New**).
2. Go to **Window > Script Editor** from the top menu bar.
3. Open the testing script:
   * In the Script Editor, click **File > Open**.
   * Navigate to `scripts/nemo_project/test_nemo_in_gui.py` and open it.
4. Click the **Run (Ctrl + Enter)** button at the bottom of the Script Editor.
   * *The robot and a ground plane will spawn in the viewport.*
5. Click the **Play (▶)** button on the left toolbar.
   * *The robot will drop to the floor. Because the script injects PD stiffness (`800.0`) and damping (`80.0`), the robot should act like a stiff spring and hold its default pose without collapsing like a ragdoll.*

---

## Step 5: Install RSL-RL and Train the Robot
Before running the training script, you need to install the reinforcement learning library `rsl_rl`. IsaacLab provides a built-in command for this:

1. Open your terminal and run:
   ```bash
   .\isaaclab.bat -e rsl_rl
   ```
   *(This installs the required `rsl_rl` python module and its IsaacLab wrapper).*

2. Once installed, start the training process by running:
   ```bash
   .\isaaclab.bat -p scripts\nemo_project\train.py --headless
   ```
   *(Running in `--headless` mode skips the GUI to maximize training speed).*

3. The script will output metrics like `Mean reward` and `Mean episode length`. Model checkpoints will be saved automatically every 50 iterations in `logs/nemo_locomotion/`.

---

## Step 6: Watch the Trained Robot (Play)
After training is complete (or you have generated checkpoints in the `logs` folder), you can load the trained neural network and watch the robot walk in the GUI!

1. Close any running instances of Isaac Sim.
2. Open your terminal.
3. Run the playback script (this script automatically turns on the GUI so you can watch):
   ```bash
   .\isaaclab.bat -p scripts\nemo_project\play.py
   ```
   *(This script spawns 36 robots in a grid and applies the trained policy to all of them simultaneously.)*
