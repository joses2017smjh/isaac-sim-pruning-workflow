# The vision measurement the cut check is evaluated on, in one control cycle.
# Mirrors isaaclab_pruning.task.simulated_cut.CutObservation's vision fields.

# stamp: when the measured image was captured (the vision timestamp).
# frame_id: frame of position and axis; must match the tool pose's frame.
std_msgs/Header header

# The control cycle (sensor bundle) this estimate is offered for. The safety gate
# pairs on this stamp and checks the age header.stamp has against it.
builtin_interfaces/Time observation_stamp

string target_id
# branch or spur; anything else fails the cut check.
string target_semantic
bool vision_valid
# live, or closure_hold for the labelled closure-start target (see simulated_cut.VISION_SOURCES).
string vision_source
bool has_held_target_stamp
builtin_interfaces/Time held_target_stamp

# NaN components when the tracker has no estimate.
geometry_msgs/Point position
geometry_msgs/Vector3 axis
float64 radius_m

# The tracker's own state label (tracking, tracking_lost, ...); informational.
string tracker_state
