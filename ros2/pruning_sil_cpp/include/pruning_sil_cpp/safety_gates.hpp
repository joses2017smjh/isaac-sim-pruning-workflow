// The two stop-or-go checks of the pruning demo, and the rules that combine them.
//
// Ports of:
//   sensor_guard         isaaclab_pruning.task.safety_gate.sensor_guard
//   evaluate_cut_gate    isaaclab_pruning.task.simulated_cut.evaluate_cut_gate
//   tool_mouth_geometry  isaaclab_pruning.task.simulated_cut.tool_mouth_geometry
//   SafetyGate           isaaclab_pruning.task.safety_gate.SafetyGateReference
//
// The port is checked against the Python on recorded Isaac cycles
// (test/test_safety_gates.cpp and tools/validate_safety_gate.py), not against itself.
// No ROS here: safety_gate_node.cpp wraps this in a node. Geometry limits are the
// demo's (demo_cut_config), not hardware ratings.

#ifndef PRUNING_SIL_CPP__SAFETY_GATES_HPP_
#define PRUNING_SIL_CPP__SAFETY_GATES_HPP_

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace pruning_sil_cpp
{

using Vec3 = std::array<double, 3>;

inline constexpr double kMinimumClearanceM = 0.06;
inline constexpr int kMissingFramesLimit = 4;
inline constexpr double kHazardContactForceN = 5.0;

inline const char * const kOperatorHold = "operator_hold";
inline const char * const kGateLostDuringClosure = "gate_lost_during_closure";

/// Result of one time-of-flight check. `reason` is empty when the check passes.
struct SensorGuardResult
{
  std::string reason;
  int consecutive_missing{0};
  std::size_t usable_zones{0};
  double min_range_m{std::numeric_limits<double>::quiet_NaN()};
  /// Index of the nearest usable zone in the pooled ranges; npos when none.
  std::size_t min_index{static_cast<std::size_t>(-1)};
};

/// A zone counts only if flagged valid, finite and positive. Both sensors are pooled.
SensorGuardResult sensor_guard(
  const std::vector<double> & ranges,
  const std::vector<bool> & valid,
  int consecutive_missing,
  double minimum_clearance_m = kMinimumClearanceM);

/// Cut-gate limits. Defaults are demo_cut_config's, which the recorded runs used.
struct CutConfig
{
  std::string selected_target_id;
  double max_branch_radius_m{0.012};
  double mouth_position_tolerance_m{0.008};
  double alignment_tolerance_deg{15.0};
  double vision_max_age_s{0.25};
  std::vector<std::string> allowed_semantics{"branch", "spur"};

  /// Throws std::invalid_argument on the same inputs CutConfig.__post_init__ rejects.
  void validate() const;
};

struct CutObservation
{
  double time_s{0.0};
  double vision_timestamp_s{0.0};
  std::string target_id;
  std::string target_semantic;
  bool vision_valid{false};
  Vec3 target_position_w{};
  Vec3 target_axis_w{};
  double target_radius_m{0.0};
  Vec3 mouth_position_w{};
  Vec3 cutter_closing_axis_w{};
  bool hazard_contact{false};
  std::string vision_source{"live"};
  std::optional<double> held_target_timestamp_s;
};

struct CutCertificate
{
  bool safe_to_approach{false};
  bool ready_to_close{false};
  std::vector<std::string> reasons;
  std::optional<double> mouth_distance_m;
  std::optional<double> perpendicularity_error_deg;
  std::optional<double> vision_age_s;
  std::string vision_source{"live"};
  std::optional<double> held_target_age_s;
  std::vector<std::string> waived_checks;
};

/// Fails closed on identity, age, contact, radius, or malformed geometry.
CutCertificate evaluate_cut_gate(const CutObservation & observation, const CutConfig & config);

/// The jaw mouth position and closing axis in the pose's frame. Pose is x y z w qx qy qz.
/// Throws std::invalid_argument on a nonfinite pose, zero quaternion or zero axis.
std::pair<Vec3, Vec3> tool_mouth_geometry(
  const std::array<double, 7> & tool_pose_wxyz,
  const Vec3 & mouth_offset_tool,
  const Vec3 & closing_axis_tool);

/// Everything the gate sees in one control cycle.
struct GateInputs
{
  std::vector<double> tof_ranges;
  std::vector<bool> tof_valid;
  CutObservation observation;
  /// Cutter state before this cycle's observation, from the controller's previous report.
  std::string prior_cut_phase{"approach"};
  bool prior_detached{false};
  bool operator_hold{false};
};

struct GateVerdict
{
  bool passed{false};
  std::string reason;
  std::string latched_reason;
  SensorGuardResult tof;
  CutCertificate certificate;
  bool closure_authorized{false};
};

/// Combines the two checks as the Isaac recorder and SimulatedCutController do, and
/// latches the first hold. Precedence: operator, then ToF, then the cut check.
class SafetyGate
{
public:
  explicit SafetyGate(
    CutConfig config,
    double minimum_clearance_m = kMinimumClearanceM,
    int initial_missing = 0,
    std::string initial_latched_reason = {});

  GateVerdict step(const GateInputs & inputs);

  /// Latch a hold the checks did not produce (an input fault, an operator request).
  void latch(const std::string & reason);

  const std::string & latched_reason() const {return latched_reason_;}
  bool latched() const {return !latched_reason_.empty();}
  int consecutive_missing() const {return missing_;}
  const CutConfig & config() const {return config_;}

private:
  CutConfig config_;
  double minimum_clearance_m_;
  int missing_;
  std::string latched_reason_;
};

}  // namespace pruning_sil_cpp

#endif  // PRUNING_SIL_CPP__SAFETY_GATES_HPP_
