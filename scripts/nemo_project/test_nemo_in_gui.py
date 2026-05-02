import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core import World
import numpy as np
import sys

# 1. Clear previous world instances to avoid "expired Xform" errors
if World.instance() is not None:
    World.instance().clear_instance()

# 2. Create a clean simulation world
world = World(stage_units_in_meters=1.0)

# 3. Create a ground plane
ground_path = "/World/defaultGroundPlane"
if not prim_utils.is_prim_path_valid(ground_path):
    world.scene.add_default_ground_plane(prim_path=ground_path)

# 4. Spawn the Nemo robot
robot_path = "/World/Nemo"
usd_path = "C:/Users/mmgzz/Desktop/IsaacLab/source/isaaclab_assets/data/Robots/Nemo/nemo.usd"

if not prim_utils.is_prim_path_valid(robot_path):
    prim_utils.create_prim(
        prim_path=robot_path,
        usd_path=usd_path,
        translation=np.array([0.0, 0.0, 0.6]) # Spawn 0.6m in the air
    )

# Register the robot to the physics engine (World)
if not world.scene.object_exists("nemo_robot"):
    nemo_robot = world.scene.add(Articulation(prim_path=robot_path, name="nemo_robot"))
else:
    nemo_robot = world.scene.get_object("nemo_robot")

# 5. Reset the world to initialize physics
world.reset()

# 6. Apply PD Control (Stiffness and Damping) to hold the pose
# By default, a raw USD might have 0 stiffness, causing it to collapse.
# We inject strong PD gains here so it acts like a stiff spring.
nemo_robot.get_articulation_controller().set_gains(
    kps=np.full(nemo_robot.num_dof, 800.0), # Proportional gain (Stiffness)
    kds=np.full(nemo_robot.num_dof, 80.0)   # Derivative gain (Damping)
)

# 7. Set the target standing pose (0.0 for all joints)
# You can modify this array to test different poses
target_positions = np.zeros(nemo_robot.num_dof)

# Create an ArticulationAction object to apply the target positions
action = ArticulationAction(joint_positions=target_positions)

# Apply the action to the controller
nemo_robot.get_articulation_controller().apply_action(action)

print("=========================================================")
print("[SUCCESS] Nemo Robot spawned with PD Controllers active!")
print("Press the Play button (▶) on the left toolbar.")
print("The robot should drop to the ground and HOLD its pose.")
print("=========================================================")
