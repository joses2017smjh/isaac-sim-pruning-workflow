# The safety gate's call for one control cycle: pass the controller's command, or hold.

# stamp: the control cycle (sensor bundle) this decision gates.
std_msgs/Header header
uint32 cycle

# True: the controller's proposal was forwarded unchanged.
bool passed
# Why this cycle holds; empty when passed.
string reason
# The run's first hold; empty until one happens. Holds latch, as in the Isaac recorder.
string latched_reason

# Time-of-flight check (isaaclab_pruning.task.safety_gate.sensor_guard).
string tof_reason
int32 tof_missing_frames
uint32 tof_usable_zones
# Nearest usable return; NaN when there is none.
float64 tof_min_range_m
# That return deprojected into its sensor's frame (tof_nearest_frame); NaN when none.
geometry_msgs/Point tof_nearest_point
string tof_nearest_frame

# Cut check (isaaclab_pruning.task.simulated_cut.evaluate_cut_gate).
bool cut_safe_to_approach
bool cut_ready_to_close
string[] cut_reasons
# NaN when the geometry was invalid.
float64 mouth_distance_m
float64 perpendicularity_error_deg
float64 vision_age_s
string vision_source
string[] waived_checks

# Context the rules used.
string prior_cut_phase
bool prior_detached
bool operator_hold
# Jaw closure may advance only on a passing cycle whose certificate is ready to close.
bool closure_authorized

# Wall time from this cycle's last input arriving to this decision being published.
float64 evaluation_latency_s
