// The safety gate every arm command passes through.
//
// Per control cycle (one sensor bundle, keyed by its stamp) the node collects:
//   /pruning/tof/{left,right}/range_image  sensor_msgs/Image 32FC1 8x8, NaN = no usable zone
//   /pruning/tool_pose                     geometry_msgs/PoseStamped, world frame
//   /pruning/contact/wrench                geometry_msgs/WrenchStamped, largest arm contact force
//   /pruning/perception/target             pruning_sil_msgs/TargetEstimate
//   /pruning/proposed/proposal             pruning_sil_msgs/ControllerProposal
// runs the two checks (SafetyGate: ToF clearance, then the cut check), and publishes
//   /pruning/gated/command_pose            the proposal if it passes, else the measured pose
//   /pruning/gated/jaw                     sensor_msgs/JointState, jaw closure in [0, 1]
//   /pruning/safety/decision               pruning_sil_msgs/SafetyDecision, with the reason
//   /pruning/safety/decision_json          std_msgs/String, the same call as JSON for the Isaac
//                                          recorder, whose bundled rclpy cannot load custom messages
// An operator hold (/pruning/operator/hold) latches a hold for the run. A missing,
// malformed or out-of-order input latches a hold too: the gate never guesses.
//
// Nothing here drives hardware. In this repository the arm it gates is the
// simulated UR5e in Isaac Sim, or a recorded capture being replayed.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "pruning_sil_cpp/safety_gates.hpp"
#include "pruning_sil_cpp/tof_deprojection.hpp"
#include "pruning_sil_msgs/msg/controller_proposal.hpp"
#include "pruning_sil_msgs/msg/operator_hold.hpp"
#include "pruning_sil_msgs/msg/safety_decision.hpp"
#include "pruning_sil_msgs/msg/target_estimate.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/string.hpp"

namespace pruning_sil_cpp
{

using geometry_msgs::msg::PoseStamped;
using geometry_msgs::msg::WrenchStamped;
using pruning_sil_msgs::msg::ControllerProposal;
using pruning_sil_msgs::msg::OperatorHold;
using pruning_sil_msgs::msg::SafetyDecision;
using pruning_sil_msgs::msg::TargetEstimate;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::JointState;

namespace
{

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr std::size_t kTofZones = 64;
const char * const kJawJoint = "mock_pruner__jaw_closure";

// Reasons and frames are identifiers, but escape anyway so the JSON is always valid.
std::string json_string(const std::string & value)
{
  std::string out = "\"";
  for (const char ch : value) {
    if (ch == '"' || ch == '\\') {
      out += '\\';
      out += ch;
    } else if (static_cast<unsigned char>(ch) < 0x20) {
      out += ' ';
    } else {
      out += ch;
    }
  }
  return out + "\"";
}

std::string json_number(double value)
{
  if (!std::isfinite(value)) {
    return "null";
  }
  std::ostringstream stream;
  stream.precision(17);
  stream << value;
  return stream.str();
}

std::string json_list(const std::vector<std::string> & values)
{
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    out += (i ? "," : "") + json_string(values[i]);
  }
  return out + "]";
}

int64_t stamp_ns(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}

double stamp_s(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

// Decode an 8x8 32FC1 range image; an empty optional means the message is malformed.
std::optional<std::vector<double>> decode_range_image(const Image & image)
{
  if (image.encoding != "32FC1" || image.width != 8 || image.height != 8 ||
    image.step < 8 * sizeof(float) || image.data.size() != image.height * image.step)
  {
    return std::nullopt;
  }
  const bool host_big_endian = false;  // x86 and ARM targets of this package are little-endian
  std::vector<double> ranges;
  ranges.reserve(kTofZones);
  for (uint32_t row = 0; row < image.height; ++row) {
    for (uint32_t col = 0; col < image.width; ++col) {
      unsigned char bytes[4];
      std::memcpy(bytes, &image.data[row * image.step + col * 4], 4);
      if (static_cast<bool>(image.is_bigendian) != host_big_endian) {
        std::swap(bytes[0], bytes[3]);
        std::swap(bytes[1], bytes[2]);
      }
      float value;
      std::memcpy(&value, bytes, 4);
      ranges.push_back(static_cast<double>(value));
    }
  }
  return ranges;
}

}  // namespace

class SafetyGateNode : public rclcpp::Node
{
public:
  explicit SafetyGateNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("pruning_safety_gate", options)
  {
    CutConfig config;
    config.selected_target_id = declare_parameter<std::string>("selected_target_id", "");
    config.allowed_semantics = declare_parameter<std::vector<std::string>>(
      "allowed_semantics", std::vector<std::string>{"branch", "spur"});
    config.max_branch_radius_m = declare_parameter<double>("max_branch_radius_m", 0.012);
    config.mouth_position_tolerance_m =
      declare_parameter<double>("mouth_position_tolerance_m", 0.008);
    config.alignment_tolerance_deg = declare_parameter<double>("alignment_tolerance_deg", 15.0);
    config.vision_max_age_s = declare_parameter<double>("vision_max_age_s", 0.25);
    const double clearance = declare_parameter<double>("minimum_clearance_m", kMinimumClearanceM);
    hazard_force_n_ = declare_parameter<double>("hazard_contact_force_n", kHazardContactForceN);
    const auto offset = declare_parameter<std::vector<double>>(
      "mouth_offset_tool", std::vector<double>{0.0, 0.0, 0.07});
    const auto closing = declare_parameter<std::vector<double>>(
      "closing_axis_tool", std::vector<double>{1.0, 0.0, 0.0});
    prior_phase_ = declare_parameter<std::string>("initial_cut_phase", "approach");
    prior_detached_ = declare_parameter<bool>("initial_detached", false);
    const int initial_missing = static_cast<int>(declare_parameter<int64_t>("initial_missing", 0));
    input_timeout_s_ = declare_parameter<double>("input_timeout_s", 0.25);
    world_frame_ = declare_parameter<std::string>("world_frame", "world");
    tof_frames_[0] = declare_parameter<std::string>("tof_left_frame", "mock_pruner__tof0");
    tof_frames_[1] = declare_parameter<std::string>("tof_right_frame", "mock_pruner__tof1");

    if (offset.size() != 3 || closing.size() != 3) {
      throw std::invalid_argument("mouth_offset_tool and closing_axis_tool need three values");
    }
    mouth_offset_ = {offset[0], offset[1], offset[2]};
    closing_axis_ = {closing[0], closing[1], closing[2]};
    if (!(input_timeout_s_ > 0.0) || !std::isfinite(input_timeout_s_)) {
      throw std::invalid_argument("input_timeout_s must be finite and positive");
    }
    gate_ = std::make_unique<SafetyGate>(config, clearance, initial_missing);

    // Commands and decisions must not be dropped, so every gate input is reliable;
    // a reliable publisher also serves the best-effort RViz and servo subscribers.
    const auto reliable = rclcpp::QoS(rclcpp::KeepLast(50)).reliable();
    const auto latched = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();

    tof_subscriptions_[0] = create_subscription<Image>(
      "/pruning/tof/left/range_image", reliable,
      [this](Image::ConstSharedPtr m) {store(stamp_ns(m->header.stamp)).tof[0] = m; process();});
    tof_subscriptions_[1] = create_subscription<Image>(
      "/pruning/tof/right/range_image", reliable,
      [this](Image::ConstSharedPtr m) {store(stamp_ns(m->header.stamp)).tof[1] = m; process();});
    pose_subscription_ = create_subscription<PoseStamped>(
      "/pruning/tool_pose", reliable,
      [this](PoseStamped::ConstSharedPtr m) {store(stamp_ns(m->header.stamp)).pose = m; process();});
    contact_subscription_ = create_subscription<WrenchStamped>(
      "/pruning/contact/wrench", reliable,
      [this](WrenchStamped::ConstSharedPtr m) {
        store(stamp_ns(m->header.stamp)).contact = m; process();
      });
    target_subscription_ = create_subscription<TargetEstimate>(
      "/pruning/perception/target", reliable,
      [this](TargetEstimate::ConstSharedPtr m) {
        store(stamp_ns(m->observation_stamp)).target = m; process();
      });
    proposal_subscription_ = create_subscription<ControllerProposal>(
      "/pruning/proposed/proposal", reliable,
      [this](ControllerProposal::ConstSharedPtr m) {
        store(stamp_ns(m->header.stamp)).proposal = m; process();
      });
    hold_subscription_ = create_subscription<OperatorHold>(
      "/pruning/operator/hold", latched, [this](OperatorHold::ConstSharedPtr m) {on_hold(*m);});

    decision_publisher_ = create_publisher<SafetyDecision>("/pruning/safety/decision", reliable);
    command_publisher_ = create_publisher<PoseStamped>("/pruning/gated/command_pose", reliable);
    jaw_publisher_ = create_publisher<JointState>("/pruning/gated/jaw", reliable);
    json_publisher_ = create_publisher<std_msgs::msg::String>("/pruning/safety/decision_json", reliable);
    timer_ = create_wall_timer(std::chrono::milliseconds(20), [this]() {process();});

    RCLCPP_INFO(
      get_logger(), "Safety gate for target %s: ToF clearance %.3f m, mouth tolerance %.4f m",
      config.selected_target_id.c_str(), clearance, config.mouth_position_tolerance_m);
  }

  ~SafetyGateNode() override
  {
    RCLCPP_INFO(
      get_logger(), "Gate summary: %lu cycles, %lu passed, %lu held, %lu stale inputs ignored, "
      "latched reason '%s'", static_cast<unsigned long>(cycles_),
      static_cast<unsigned long>(passed_), static_cast<unsigned long>(cycles_ - passed_),
      static_cast<unsigned long>(stale_inputs_), gate_->latched_reason().c_str());
  }

private:
  struct Pending
  {
    std::array<Image::ConstSharedPtr, 2> tof;
    PoseStamped::ConstSharedPtr pose;
    WrenchStamped::ConstSharedPtr contact;
    TargetEstimate::ConstSharedPtr target;
    ControllerProposal::ConstSharedPtr proposal;
    rclcpp::Time first_seen;
    rclcpp::Time last_seen;

    bool complete() const
    {
      return tof[0] && tof[1] && pose && contact && target && proposal;
    }

    std::string missing() const
    {
      if (!proposal) {return "proposal";}
      if (!pose) {return "tool_pose";}
      if (!tof[0]) {return "tof_left";}
      if (!tof[1]) {return "tof_right";}
      if (!contact) {return "contact";}
      return "target";
    }
  };

  Pending & store(int64_t stamp)
  {
    const auto now = get_clock()->now();
    if (last_processed_ && stamp <= *last_processed_) {
      // Already decided; a late or replayed input cannot change a past decision.
      ++stale_inputs_;
      scratch_ = Pending();
      return scratch_;
    }
    auto [it, inserted] = pending_.try_emplace(stamp);
    if (inserted) {
      it->second.first_seen = now;
    }
    it->second.last_seen = now;
    return it->second;
  }

  void on_hold(const OperatorHold & message)
  {
    if (!message.hold) {
      return;  // a hold is latched for the run; there is no remote release
    }
    operator_hold_ = true;
    operator_hold_stamp_ns_ = stamp_ns(message.header.stamp);
    gate_->latch(kOperatorHold);
    RCLCPP_WARN(get_logger(), "Operator hold from %s latched", message.source.c_str());
  }

  void process()
  {
    while (!pending_.empty()) {
      auto it = pending_.begin();
      Pending & cycle = it->second;
      const int64_t stamp = it->first;
      if (!cycle.complete()) {
        const double waited = (get_clock()->now() - cycle.first_seen).seconds();
        if (waited < input_timeout_s_) {
          return;  // keep waiting, in order: a later cycle never overtakes an earlier one
        }
        decide_fault(stamp, cycle, "input_missing:" + cycle.missing());
      } else {
        decide(stamp, cycle);
      }
      last_processed_ = stamp;
      pending_.erase(it);
    }
  }

  // The measured pose to hold at, or nothing when this cycle's pose is unusable.
  std::optional<PoseStamped> hold_pose(const Pending & cycle) const
  {
    if (cycle.pose && pose_usable(*cycle.pose)) {
      return *cycle.pose;
    }
    return last_good_pose_;
  }

  bool pose_usable(const PoseStamped & pose) const
  {
    const auto & p = pose.pose.position;
    const auto & q = pose.pose.orientation;
    const double values[7] = {p.x, p.y, p.z, q.w, q.x, q.y, q.z};
    for (const double value : values) {
      if (!std::isfinite(value)) {return false;}
    }
    const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    return pose.header.frame_id == world_frame_ && std::abs(norm - 1.0) <= 1e-3;
  }

  void decide_fault(int64_t stamp, const Pending & cycle, const std::string & reason)
  {
    gate_->latch(reason);
    SafetyDecision decision;
    decision.header.stamp = rclcpp::Time(stamp);
    decision.header.frame_id = world_frame_;
    decision.cycle = static_cast<uint32_t>(cycles_);
    decision.passed = false;
    decision.reason = reason;
    decision.latched_reason = gate_->latched_reason();
    decision.tof_missing_frames = gate_->consecutive_missing();
    decision.tof_min_range_m = kNan;
    decision.tof_nearest_point.x = decision.tof_nearest_point.y = decision.tof_nearest_point.z = kNan;
    decision.mouth_distance_m = decision.perpendicularity_error_deg = decision.vision_age_s = kNan;
    decision.prior_cut_phase = prior_phase_;
    decision.prior_detached = prior_detached_;
    decision.operator_hold = operator_hold_;
    decision.evaluation_latency_s = (get_clock()->now() - cycle.last_seen).seconds();
    publish(stamp, decision, hold_pose(cycle), std::nullopt);
    RCLCPP_WARN(get_logger(), "Cycle %ld held on input fault: %s", static_cast<long>(stamp),
      reason.c_str());
  }

  void decide(int64_t stamp, const Pending & cycle)
  {
    // Validate every input before trusting any of it; a malformed one latches a hold.
    const auto & pose_message = *cycle.pose;
    if (!pose_usable(pose_message)) {
      return decide_fault(stamp, cycle, "invalid_tool_pose");
    }
    std::vector<double> ranges;
    for (int side = 0; side < 2; ++side) {
      const auto & image = *cycle.tof[side];
      auto decoded = decode_range_image(image);
      if (!decoded || image.header.frame_id != tof_frames_[side]) {
        return decide_fault(stamp, cycle, side == 0 ? "invalid_tof_left" : "invalid_tof_right");
      }
      ranges.insert(ranges.end(), decoded->begin(), decoded->end());
    }
    const auto & force = cycle.contact->wrench.force;
    if (!std::isfinite(force.x) || !std::isfinite(force.y) || !std::isfinite(force.z)) {
      return decide_fault(stamp, cycle, "invalid_contact");
    }
    const auto & target = *cycle.target;
    if (target.header.frame_id != world_frame_) {
      return decide_fault(stamp, cycle, "target_frame_mismatch");
    }
    const auto & proposal = *cycle.proposal;
    if (proposal.has_pose && proposal.header.frame_id != world_frame_) {
      return decide_fault(stamp, cycle, "proposal_frame_mismatch");
    }

    GateInputs inputs;
    inputs.tof_ranges = ranges;
    inputs.tof_valid.reserve(ranges.size());
    for (const double range : ranges) {
      inputs.tof_valid.push_back(std::isfinite(range));
    }
    auto & observation = inputs.observation;
    observation.time_s = stamp_s(pose_message.header.stamp);
    observation.vision_timestamp_s = stamp_s(target.header.stamp);
    observation.target_id = target.target_id;
    observation.target_semantic = target.target_semantic;
    observation.vision_valid = target.vision_valid;
    observation.vision_source = target.vision_source;
    if (target.has_held_target_stamp) {
      observation.held_target_timestamp_s = stamp_s(target.held_target_stamp);
    }
    observation.target_position_w = {target.position.x, target.position.y, target.position.z};
    observation.target_axis_w = {target.axis.x, target.axis.y, target.axis.z};
    observation.target_radius_m = target.radius_m;
    const auto & p = pose_message.pose.position;
    const auto & q = pose_message.pose.orientation;
    const auto geometry =
      tool_mouth_geometry({p.x, p.y, p.z, q.w, q.x, q.y, q.z}, mouth_offset_, closing_axis_);
    observation.mouth_position_w = geometry.first;
    observation.cutter_closing_axis_w = geometry.second;
    const double force_n = std::sqrt(force.x * force.x + force.y * force.y + force.z * force.z);
    observation.hazard_contact = force_n > hazard_force_n_;
    inputs.prior_cut_phase = prior_phase_;
    inputs.prior_detached = prior_detached_;
    inputs.operator_hold = operator_hold_;

    auto verdict = gate_->step(inputs);
    last_good_pose_ = pose_message;

    // The jaw may advance only on a passing cycle whose certificate is ready to close.
    double jaw = gated_closure_;
    if (verdict.passed) {
      const double proposed = proposal.closure_progress;
      if (!std::isfinite(proposed) || proposed < 0.0 || proposed > 1.0) {
        gate_->latch("invalid_closure_progress");
        verdict = held_copy(verdict, "invalid_closure_progress");
      } else if (proposed > gated_closure_ && !verdict.closure_authorized) {
        gate_->latch("closure_without_ready_gate");
        verdict = held_copy(verdict, "closure_without_ready_gate");
        ++closure_rule_holds_;
      } else {
        jaw = proposed;
      }
    }

    SafetyDecision decision;
    decision.header.stamp = pose_message.header.stamp;
    decision.header.frame_id = world_frame_;
    decision.cycle = static_cast<uint32_t>(cycles_);
    decision.passed = verdict.passed;
    decision.reason = verdict.reason;
    decision.latched_reason = verdict.latched_reason;
    decision.tof_reason = verdict.tof.reason;
    decision.tof_missing_frames = verdict.tof.consecutive_missing;
    decision.tof_usable_zones = static_cast<uint32_t>(verdict.tof.usable_zones);
    decision.tof_min_range_m = verdict.tof.min_range_m;
    decision.tof_nearest_point.x = decision.tof_nearest_point.y = decision.tof_nearest_point.z = kNan;
    if (verdict.tof.min_index != static_cast<std::size_t>(-1)) {
      // Locate the nearest return with the ported deprojection, in its sensor's frame.
      const std::size_t side = verdict.tof.min_index / kTofZones;
      const std::size_t zone = verdict.tof.min_index % kTofZones;
      const std::vector<double> grid(
        ranges.begin() + side * kTofZones, ranges.begin() + (side + 1) * kTofZones);
      const std::vector<bool> valid(
        inputs.tof_valid.begin() + side * kTofZones, inputs.tof_valid.begin() + (side + 1) * kTofZones);
      const auto points = deproject_tof(grid, valid, {0.0, 0.0, 0.0}, 8, 8);
      decision.tof_nearest_point.x = points[zone].x;
      decision.tof_nearest_point.y = points[zone].y;
      decision.tof_nearest_point.z = points[zone].z;
      decision.tof_nearest_frame = tof_frames_[side];
    }
    const auto & certificate = verdict.certificate;
    decision.cut_safe_to_approach = certificate.safe_to_approach;
    decision.cut_ready_to_close = certificate.ready_to_close;
    decision.cut_reasons = certificate.reasons;
    decision.mouth_distance_m = certificate.mouth_distance_m.value_or(kNan);
    decision.perpendicularity_error_deg = certificate.perpendicularity_error_deg.value_or(kNan);
    decision.vision_age_s = certificate.vision_age_s.value_or(kNan);
    decision.vision_source = certificate.vision_source;
    decision.waived_checks = certificate.waived_checks;
    decision.prior_cut_phase = prior_phase_;
    decision.prior_detached = prior_detached_;
    decision.operator_hold = operator_hold_;
    decision.closure_authorized = verdict.closure_authorized;
    decision.evaluation_latency_s = (get_clock()->now() - cycle.last_seen).seconds();

    // The cutter's state after this cycle is the "before" state of the next one.
    prior_phase_ = proposal.cut_phase.empty() ? prior_phase_ : proposal.cut_phase;
    prior_detached_ = proposal.cut_detached;

    std::optional<PoseStamped> command;
    if (verdict.passed && proposal.has_pose) {
      PoseStamped forwarded;
      forwarded.header = pose_message.header;
      forwarded.pose = proposal.pose;
      command = forwarded;
    } else {
      command = pose_message;  // hold where the arm is
    }
    publish(stamp, decision, command, jaw);
  }

  static GateVerdict held_copy(GateVerdict verdict, const std::string & reason)
  {
    verdict.passed = false;
    verdict.reason = reason;
    if (verdict.latched_reason.empty()) {
      verdict.latched_reason = reason;
    }
    verdict.closure_authorized = false;
    return verdict;
  }

  void publish(
    int64_t stamp, const SafetyDecision & decision, const std::optional<PoseStamped> & command,
    std::optional<double> jaw)
  {
    ++cycles_;
    passed_ += decision.passed ? 1 : 0;
    if (jaw) {
      gated_closure_ = *jaw;
    }
    const rclcpp::Time time(stamp);
    if (command) {
      PoseStamped out = *command;
      out.header.stamp = time;
      out.header.frame_id = world_frame_;
      command_publisher_->publish(out);
    }
    JointState jaw_state;
    jaw_state.header.stamp = time;
    jaw_state.name = {kJawJoint};
    jaw_state.position = {gated_closure_};
    jaw_publisher_->publish(jaw_state);
    decision_publisher_->publish(decision);
    publish_json(stamp, decision);
  }

  void publish_json(int64_t stamp, const SafetyDecision & d)
  {
    const auto wall = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
    std::ostringstream out;
    out << "{\"stamp_ns\":" << stamp << ",\"cycle\":" << d.cycle
        << ",\"passed\":" << (d.passed ? "true" : "false")
        << ",\"reason\":" << json_string(d.reason)
        << ",\"latched_reason\":" << json_string(d.latched_reason)
        << ",\"tof_reason\":" << json_string(d.tof_reason)
        << ",\"tof_missing_frames\":" << d.tof_missing_frames
        << ",\"tof_usable_zones\":" << d.tof_usable_zones
        << ",\"tof_min_range_m\":" << json_number(d.tof_min_range_m)
        << ",\"cut_safe_to_approach\":" << (d.cut_safe_to_approach ? "true" : "false")
        << ",\"cut_ready_to_close\":" << (d.cut_ready_to_close ? "true" : "false")
        << ",\"cut_reasons\":" << json_list(d.cut_reasons)
        << ",\"mouth_distance_m\":" << json_number(d.mouth_distance_m)
        << ",\"vision_age_s\":" << json_number(d.vision_age_s)
        << ",\"prior_cut_phase\":" << json_string(d.prior_cut_phase)
        << ",\"operator_hold\":" << (d.operator_hold ? "true" : "false")
        << ",\"closure_authorized\":" << (d.closure_authorized ? "true" : "false")
        << ",\"gated_jaw\":" << json_number(gated_closure_)
        << ",\"evaluation_latency_s\":" << json_number(d.evaluation_latency_s)
        << ",\"operator_hold_stamp_ns\":" << operator_hold_stamp_ns_
        << ",\"publish_wall_ns\":" << wall << "}";
    std_msgs::msg::String message;
    message.data = out.str();
    json_publisher_->publish(message);
  }

  std::unique_ptr<SafetyGate> gate_;
  double hazard_force_n_{kHazardContactForceN};
  Vec3 mouth_offset_{};
  Vec3 closing_axis_{};
  std::string prior_phase_;
  bool prior_detached_{false};
  bool operator_hold_{false};
  int64_t operator_hold_stamp_ns_{0};
  double input_timeout_s_{0.25};
  std::string world_frame_;
  std::array<std::string, 2> tof_frames_;
  double gated_closure_{0.0};

  std::map<int64_t, Pending> pending_;
  Pending scratch_;
  std::optional<int64_t> last_processed_;
  std::optional<PoseStamped> last_good_pose_;
  std::size_t cycles_{0};
  std::size_t passed_{0};
  std::size_t stale_inputs_{0};
  std::size_t closure_rule_holds_{0};

  std::array<rclcpp::Subscription<Image>::SharedPtr, 2> tof_subscriptions_;
  rclcpp::Subscription<PoseStamped>::SharedPtr pose_subscription_;
  rclcpp::Subscription<WrenchStamped>::SharedPtr contact_subscription_;
  rclcpp::Subscription<TargetEstimate>::SharedPtr target_subscription_;
  rclcpp::Subscription<ControllerProposal>::SharedPtr proposal_subscription_;
  rclcpp::Subscription<OperatorHold>::SharedPtr hold_subscription_;
  rclcpp::Publisher<SafetyDecision>::SharedPtr decision_publisher_;
  rclcpp::Publisher<PoseStamped>::SharedPtr command_publisher_;
  rclcpp::Publisher<JointState>::SharedPtr jaw_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr json_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace pruning_sil_cpp

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  int status = 0;
  try {
    auto node = std::make_shared<pruning_sil_cpp::SafetyGateNode>();
    rclcpp::spin(node);
  } catch (const std::exception & error) {
    RCLCPP_FATAL(rclcpp::get_logger("pruning_safety_gate"), "%s", error.what());
    status = 1;
  }
  rclcpp::shutdown();
  return status;
}
