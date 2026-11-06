#include "pruning_sil_cpp/safety_gates.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace pruning_sil_cpp
{
namespace
{

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

// The checks a closure-hold observation passes only because it is labelled valid and
// fresh; listed in every closure-hold certificate (simulated_cut.CLOSURE_HOLD_WAIVED_CHECKS).
const std::vector<std::string> kClosureHoldWaivedChecks{
  "vision_invalid",
  "vision_stale_or_future",
  "vision_timestamp_regressed",
  "vision_frame_reused_during_closure",
};

bool finite3(const Vec3 & v)
{
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

double dot(const Vec3 & a, const Vec3 & b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double norm(const Vec3 & v)
{
  return std::sqrt(dot(v, v));
}

// simulated_cut._vector: a finite 3-vector, normalized when `unit`, else nothing.
std::optional<Vec3> as_vector(const Vec3 & value, bool unit)
{
  if (!finite3(value)) {
    return std::nullopt;
  }
  if (!unit) {
    return value;
  }
  const double length = norm(value);
  if (length <= 1e-12) {
    return std::nullopt;
  }
  return Vec3{value[0] / length, value[1] / length, value[2] / length};
}

bool contains(const std::vector<std::string> & values, const std::string & value)
{
  return std::find(values.begin(), values.end(), value) != values.end();
}

}  // namespace

SensorGuardResult sensor_guard(
  const std::vector<double> & ranges,
  const std::vector<bool> & valid,
  int consecutive_missing,
  double minimum_clearance_m)
{
  if (ranges.size() != valid.size()) {
    throw std::invalid_argument("ToF ranges and validity must have the same length");
  }
  SensorGuardResult result;
  for (std::size_t index = 0; index < ranges.size(); ++index) {
    const double range = ranges[index];
    if (!valid[index] || !std::isfinite(range) || !(range > 0.0)) {
      continue;
    }
    ++result.usable_zones;
    if (result.min_index == static_cast<std::size_t>(-1) || range < result.min_range_m) {
      result.min_range_m = range;
      result.min_index = index;
    }
  }
  result.consecutive_missing = result.usable_zones == 0 ? consecutive_missing + 1 : 0;
  if (result.usable_zones > 0 && result.min_range_m < minimum_clearance_m) {
    result.reason = "tof_minimum_clearance";
  } else if (result.consecutive_missing >= kMissingFramesLimit) {
    result.reason = "both_tof_missing_4_frames";
  }
  return result;
}

void CutConfig::validate() const
{
  bool blank = true;
  for (const char ch : selected_target_id) {
    if (!std::isspace(static_cast<unsigned char>(ch))) {
      blank = false;
      break;
    }
  }
  if (blank) {
    throw std::invalid_argument("selected_target_id must name the selected branch.");
  }
  for (const double value : {max_branch_radius_m, mouth_position_tolerance_m, vision_max_age_s}) {
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::invalid_argument("Geometry and time limits must be finite and positive.");
    }
  }
  if (!std::isfinite(alignment_tolerance_deg) || alignment_tolerance_deg < 0.0 ||
    alignment_tolerance_deg >= 90.0)
  {
    throw std::invalid_argument("alignment_tolerance_deg must be in [0, 90).");
  }
  if (allowed_semantics.empty()) {
    throw std::invalid_argument("Only explicitly identified branches/spurs can be cut.");
  }
  for (const auto & semantic : allowed_semantics) {
    if (semantic != "branch" && semantic != "spur") {
      throw std::invalid_argument("Only explicitly identified branches/spurs can be cut.");
    }
  }
}

CutCertificate evaluate_cut_gate(const CutObservation & observation, const CutConfig & config)
{
  CutCertificate certificate;
  auto & reasons = certificate.reasons;
  const bool live = observation.vision_source == "live";
  const bool held = observation.vision_source == "closure_hold";
  if (!live && !held) {
    reasons.emplace_back("invalid_vision_source");
  }
  if (!observation.vision_valid) {
    reasons.emplace_back("vision_invalid");
  }
  if (observation.target_id != config.selected_target_id) {
    reasons.emplace_back("wrong_target_id");
  }
  if (!contains(config.allowed_semantics, observation.target_semantic)) {
    reasons.emplace_back("target_not_branch");
  }
  if (observation.hazard_contact) {
    reasons.emplace_back("hazard_contact");
  }
  const double age = observation.time_s - observation.vision_timestamp_s;
  if (!std::isfinite(age) || !(0.0 <= age && age <= config.vision_max_age_s)) {
    reasons.emplace_back("vision_stale_or_future");
  }
  const double radius = observation.target_radius_m;
  if (!std::isfinite(radius) || radius <= 0.0) {
    reasons.emplace_back("invalid_branch_radius");
  } else if (radius > config.max_branch_radius_m) {
    reasons.emplace_back("branch_exceeds_demo_cut_limit");
  }

  const auto target = as_vector(observation.target_position_w, false);
  const auto mouth = as_vector(observation.mouth_position_w, false);
  const auto branch_axis = as_vector(observation.target_axis_w, true);
  const auto closing_axis = as_vector(observation.cutter_closing_axis_w, true);
  if (!target || !mouth || !branch_axis || !closing_axis) {
    reasons.emplace_back("invalid_geometry");
  } else {
    const Vec3 offset{(*target)[0] - (*mouth)[0], (*target)[1] - (*mouth)[1],
      (*target)[2] - (*mouth)[2]};
    certificate.mouth_distance_m = norm(offset);
    const double cosine = std::clamp(std::abs(dot(*branch_axis, *closing_axis)), 0.0, 1.0);
    certificate.perpendicularity_error_deg = std::asin(cosine) * (180.0 / M_PI);
  }
  certificate.safe_to_approach = reasons.empty();
  if (certificate.mouth_distance_m &&
    *certificate.mouth_distance_m > config.mouth_position_tolerance_m)
  {
    reasons.emplace_back("outside_mouth_tolerance");
  }
  if (certificate.perpendicularity_error_deg &&
    *certificate.perpendicularity_error_deg > config.alignment_tolerance_deg)
  {
    reasons.emplace_back("closing_axis_not_perpendicular");
  }
  certificate.ready_to_close = reasons.empty();
  certificate.vision_source = observation.vision_source;
  if (std::isfinite(age)) {
    certificate.vision_age_s = age;
  }
  if (held && observation.held_target_timestamp_s) {
    const double held_age = observation.time_s - *observation.held_target_timestamp_s;
    if (std::isfinite(held_age)) {
      certificate.held_target_age_s = held_age;
    }
  }
  if (held) {
    certificate.waived_checks = kClosureHoldWaivedChecks;
  }
  return certificate;
}

std::pair<Vec3, Vec3> tool_mouth_geometry(
  const std::array<double, 7> & pose,
  const Vec3 & mouth_offset_tool,
  const Vec3 & closing_axis_tool)
{
  for (const double value : pose) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("Pose, mouth offset, and nonzero closing axis must be finite.");
    }
  }
  const auto offset = as_vector(mouth_offset_tool, false);
  const auto axis = as_vector(closing_axis_tool, true);
  if (!offset || !axis) {
    throw std::invalid_argument("Pose, mouth offset, and nonzero closing axis must be finite.");
  }
  const double quaternion_norm =
    std::sqrt(pose[3] * pose[3] + pose[4] * pose[4] + pose[5] * pose[5] + pose[6] * pose[6]);
  if (quaternion_norm <= 1e-12) {
    throw std::invalid_argument("Tool orientation quaternion must be nonzero.");
  }
  const double w = pose[3] / quaternion_norm;
  const double x = pose[4] / quaternion_norm;
  const double y = pose[5] / quaternion_norm;
  const double z = pose[6] / quaternion_norm;
  const double r[3][3] = {
    {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
    {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
    {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)},
  };
  Vec3 mouth{};
  Vec3 closing{};
  for (int row = 0; row < 3; ++row) {
    mouth[row] = pose[row] +
      (r[row][0] * (*offset)[0] + r[row][1] * (*offset)[1] + r[row][2] * (*offset)[2]);
    closing[row] = r[row][0] * (*axis)[0] + r[row][1] * (*axis)[1] + r[row][2] * (*axis)[2];
  }
  return {mouth, closing};
}

SafetyGate::SafetyGate(
  CutConfig config, double minimum_clearance_m, int initial_missing,
  std::string initial_latched_reason)
: config_(std::move(config)),
  minimum_clearance_m_(minimum_clearance_m),
  missing_(initial_missing),
  latched_reason_(std::move(initial_latched_reason))
{
  config_.validate();
  if (!std::isfinite(minimum_clearance_m_) || minimum_clearance_m_ <= 0.0) {
    throw std::invalid_argument("minimum_clearance_m must be finite and positive");
  }
  if (missing_ < 0) {
    throw std::invalid_argument("initial_missing must be non-negative");
  }
}

void SafetyGate::latch(const std::string & reason)
{
  if (reason.empty()) {
    throw std::invalid_argument("A hold must record a nonempty reason");
  }
  if (latched_reason_.empty()) {
    latched_reason_ = reason;
  }
}

GateVerdict SafetyGate::step(const GateInputs & inputs)
{
  GateVerdict verdict;
  verdict.tof = sensor_guard(inputs.tof_ranges, inputs.tof_valid, missing_, minimum_clearance_m_);
  missing_ = verdict.tof.consecutive_missing;
  verdict.certificate = evaluate_cut_gate(inputs.observation, config_);

  std::string reason;
  if (inputs.operator_hold) {
    reason = kOperatorHold;
  } else if (!verdict.tof.reason.empty()) {
    reason = verdict.tof.reason;
  } else if (inputs.prior_detached) {
    // After detachment the target may leave view; only contact still stops.
    if (inputs.observation.hazard_contact) {
      reason = "hazard_contact";
    }
  } else if (!verdict.certificate.safe_to_approach) {
    reason = verdict.certificate.reasons.front();
  } else if (inputs.prior_cut_phase == "closing" && !verdict.certificate.ready_to_close) {
    reason = kGateLostDuringClosure;
  }
  if (!reason.empty()) {
    latch(reason);
  }
  verdict.passed = latched_reason_.empty();
  verdict.latched_reason = latched_reason_;
  if (!verdict.passed) {
    verdict.reason = reason.empty() ? latched_reason_ : reason;
  }
  verdict.closure_authorized =
    verdict.passed && (inputs.prior_detached || verdict.certificate.ready_to_close);
  return verdict;
}

}  // namespace pruning_sil_cpp
