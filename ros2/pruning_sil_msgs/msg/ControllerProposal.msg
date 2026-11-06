# What the pruning controller proposes for the next control cycle. The safety gate
# forwards it, or holds the arm instead.

# stamp: the sensor bundle this proposal answers. frame_id: frame of pose.
std_msgs/Header header
uint32 frame_index

# False when the controller has no command (no image observed yet).
bool has_pose
geometry_msgs/Pose pose

# The controller's own decision: tracking, hold, release, ... A controller hold is
# forwarded as a hold; the gate never turns it into motion.
string state
string reason
string motion_phase

# Cutter state after this cycle's observation. The gate uses the previous cycle's
# values as the state "before" the next observation, as SimulatedCutController does.
string cut_phase
bool cut_detached
float64 closure_progress
