// The C++ gates must make the same call as the Python checks.
//
// The fixture (test/safety_gate_fixture.txt) holds contiguous segments of control
// cycles rebuilt from recorded Isaac captures by tools/export_gate_parity.py. Its
// expected values come from the Python implementation
// (isaaclab_pruning.task.safety_gate), so this compares the port against what it
// claims to reproduce. The unit tests below pin each rule on its own.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pruning_sil_cpp/safety_gates.hpp"

namespace
{

using pruning_sil_cpp::CutConfig;
using pruning_sil_cpp::CutObservation;
using pruning_sil_cpp::GateInputs;
using pruning_sil_cpp::SafetyGate;

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

CutObservation good_observation()
{
  CutObservation observation;
  observation.time_s = 5.0;
  observation.vision_timestamp_s = 5.0;
  observation.target_id = "tree0_SPUR_component_530";
  observation.target_semantic = "branch";
  observation.vision_valid = true;
  observation.target_position_w = {0.25, 0.63, 0.81};
  observation.target_axis_w = {0.0, 0.0, 1.0};
  observation.target_radius_m = 0.005;
  observation.mouth_position_w = {0.252, 0.63, 0.81};
  observation.cutter_closing_axis_w = {1.0, 0.0, 0.0};
  return observation;
}

CutConfig demo_config()
{
  CutConfig config;
  config.selected_target_id = "tree0_SPUR_component_530";
  return config;
}

GateInputs clear_inputs()
{
  GateInputs inputs;
  inputs.tof_ranges.assign(128, 0.3);
  inputs.tof_valid.assign(128, true);
  inputs.observation = good_observation();
  return inputs;
}

}  // namespace

TEST(SensorGuard, MatchesThePythonCases)
{
  // The same cases tests/test_isaac_render_runner.py pins for the Python guard.
  using pruning_sil_cpp::sensor_guard;
  auto a = sensor_guard({0.1, 0.3}, {true, true}, 3);
  EXPECT_EQ(a.reason, "");
  EXPECT_EQ(a.consecutive_missing, 0);
  auto b = sensor_guard({0.04, 0.3}, {true, true}, 0);
  EXPECT_EQ(b.reason, "tof_minimum_clearance");
  EXPECT_EQ(b.consecutive_missing, 0);
  auto c = sensor_guard({0.04, 0.3}, {false, true}, 0);
  EXPECT_EQ(c.reason, "");
  auto d = sensor_guard({INFINITY, kNan}, {false, false}, 2);
  EXPECT_EQ(d.reason, "");
  EXPECT_EQ(d.consecutive_missing, 3);
  auto e = sensor_guard({INFINITY, kNan}, {false, false}, 3);
  EXPECT_EQ(e.reason, "both_tof_missing_4_frames");
  EXPECT_EQ(e.consecutive_missing, 4);
}

TEST(SensorGuard, IgnoresNonPositiveAndNonFiniteZonesEvenWhenFlaggedValid)
{
  auto result = pruning_sil_cpp::sensor_guard({0.0, -0.02, kNan, INFINITY}, {true, true, true, true}, 0);
  EXPECT_EQ(result.usable_zones, 0u);
  EXPECT_EQ(result.consecutive_missing, 1);
  EXPECT_TRUE(std::isnan(result.min_range_m));
}

TEST(CutGate, PassesAGoodObservationAndListsEveryFailure)
{
  const auto config = demo_config();
  auto certificate = pruning_sil_cpp::evaluate_cut_gate(good_observation(), config);
  EXPECT_TRUE(certificate.safe_to_approach);
  EXPECT_TRUE(certificate.ready_to_close);
  EXPECT_NEAR(*certificate.mouth_distance_m, 0.002, 1e-12);
  EXPECT_NEAR(*certificate.perpendicularity_error_deg, 0.0, 1e-12);

  auto bad = good_observation();
  bad.vision_valid = false;
  bad.target_id = "another_branch";
  bad.target_semantic = "trunk";
  bad.hazard_contact = true;
  bad.vision_timestamp_s = 4.0;  // one second old
  bad.target_radius_m = 0.02;
  certificate = pruning_sil_cpp::evaluate_cut_gate(bad, config);
  const std::vector<std::string> expected{
    "vision_invalid", "wrong_target_id", "target_not_branch", "hazard_contact",
    "vision_stale_or_future", "branch_exceeds_demo_cut_limit"};
  EXPECT_EQ(certificate.reasons, expected);
  EXPECT_FALSE(certificate.safe_to_approach);
}

TEST(CutGate, FutureVisionAndNanGeometryFailClosed)
{
  auto observation = good_observation();
  observation.vision_timestamp_s = 5.1;  // a timestamp from the future
  observation.target_position_w = {kNan, kNan, kNan};
  const auto certificate = pruning_sil_cpp::evaluate_cut_gate(observation, demo_config());
  const std::vector<std::string> expected{"vision_stale_or_future", "invalid_geometry"};
  EXPECT_EQ(certificate.reasons, expected);
  EXPECT_FALSE(certificate.mouth_distance_m.has_value());
}

TEST(CutGate, ClosureHoldListsWaivedChecks)
{
  auto observation = good_observation();
  observation.vision_source = "closure_hold";
  observation.held_target_timestamp_s = 4.8;
  const auto certificate = pruning_sil_cpp::evaluate_cut_gate(observation, demo_config());
  EXPECT_EQ(certificate.waived_checks.size(), 4u);
  EXPECT_NEAR(*certificate.held_target_age_s, 0.2, 1e-12);
  observation.vision_source = "made_up";
  EXPECT_EQ(
    pruning_sil_cpp::evaluate_cut_gate(observation, demo_config()).reasons.front(),
    "invalid_vision_source");
}

TEST(SafetyGate, LatchesTheFirstHoldWithTheRecorderPrecedence)
{
  SafetyGate gate(demo_config());
  auto inputs = clear_inputs();
  EXPECT_TRUE(gate.step(inputs).passed);

  inputs.tof_ranges[17] = 0.05;           // too close
  inputs.observation.vision_valid = false;  // and vision lost in the same cycle
  auto verdict = gate.step(inputs);
  EXPECT_FALSE(verdict.passed);
  EXPECT_EQ(verdict.reason, "tof_minimum_clearance");  // ToF is checked first

  inputs = clear_inputs();
  verdict = gate.step(inputs);
  EXPECT_FALSE(verdict.passed) << "a hold must latch";
  EXPECT_EQ(verdict.latched_reason, "tof_minimum_clearance");
}

TEST(SafetyGate, AfterDetachmentOnlyContactStops)
{
  SafetyGate gate(demo_config());
  auto inputs = clear_inputs();
  inputs.prior_detached = true;
  inputs.observation.vision_valid = false;  // the cut piece fell out of view
  EXPECT_TRUE(gate.step(inputs).passed);
  inputs.observation.hazard_contact = true;
  EXPECT_EQ(gate.step(inputs).reason, "hazard_contact");
}

TEST(SafetyGate, LosingReadinessDuringClosureStops)
{
  SafetyGate gate(demo_config());
  auto inputs = clear_inputs();
  inputs.prior_cut_phase = "closing";
  inputs.observation.mouth_position_w = {0.27, 0.63, 0.81};  // 20 mm off the mouth
  const auto verdict = gate.step(inputs);
  EXPECT_EQ(verdict.reason, "gate_lost_during_closure");
  EXPECT_FALSE(verdict.closure_authorized);
}

TEST(SafetyGate, OperatorHoldWinsAndLatches)
{
  SafetyGate gate(demo_config());
  auto inputs = clear_inputs();
  inputs.operator_hold = true;
  EXPECT_EQ(gate.step(inputs).reason, "operator_hold");
  inputs.operator_hold = false;
  EXPECT_FALSE(gate.step(inputs).passed);
}

TEST(ToolMouthGeometry, RotatesTheOffsetAndAxis)
{
  // 90 degrees about z: tool x maps to world y.
  const double half = std::sqrt(0.5);
  const auto [mouth, axis] = pruning_sil_cpp::tool_mouth_geometry(
    {1.0, 2.0, 3.0, half, 0.0, 0.0, half}, {0.1, 0.0, 0.0}, {2.0, 0.0, 0.0});
  EXPECT_NEAR(mouth[0], 1.0, 1e-12);
  EXPECT_NEAR(mouth[1], 2.1, 1e-12);
  EXPECT_NEAR(axis[1], 1.0, 1e-12);
  EXPECT_THROW(
    pruning_sil_cpp::tool_mouth_geometry({0, 0, 0, 0, 0, 0, 0}, {0, 0, 0}, {1, 0, 0}),
    std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Parity with the Python checks on recorded cycles.

namespace
{

std::string word(std::istringstream & in)
{
  std::string value;
  in >> value;
  return value == "-" ? std::string() : value;
}

double number(std::istringstream & in)
{
  std::string value;
  in >> value;
  return std::stod(value);  // strtod reads "nan"
}

bool flag(std::istringstream & in)
{
  return word(in) == "1";
}

void expect_close(double actual, double expected, const std::string & what)
{
  if (std::isnan(expected)) {
    EXPECT_TRUE(std::isnan(actual)) << what;
  } else {
    EXPECT_NEAR(actual, expected, 1e-12 * std::max(1.0, std::abs(expected))) << what;
  }
}

double or_nan(const std::optional<double> & value)
{
  return value ? *value : kNan;
}

}  // namespace

TEST(SafetyGateParity, MatchesThePythonChecksOnRecordedCycles)
{
  const std::string path = std::string(PRUNING_SIL_CPP_FIXTURE_DIR) + "/safety_gate_fixture.txt";
  std::ifstream stream(path);
  ASSERT_TRUE(stream.is_open()) << "fixture not found: " << path;

  std::string line;
  std::getline(stream, line);
  ASSERT_EQ(line, "pruning_safety_gate_fixture 1");

  std::size_t segments = 0;
  std::size_t cycles = 0;
  std::size_t holds = 0;
  std::unique_ptr<SafetyGate> gate;
  CutConfig config;
  double clearance = pruning_sil_cpp::kMinimumClearanceM;
  std::string label;
  int initial_missing = 0;
  std::string initial_latch;
  GateInputs inputs;
  long index = -1;

  while (std::getline(stream, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    std::istringstream in(line);
    const std::string kind = word(in);
    if (kind == "segment") {
      label = word(in);
      word(in);  // first index
      word(in);  // cycle count
      config = CutConfig{};
      config.selected_target_id = word(in);
      initial_missing = std::stoi(word(in));
      initial_latch = word(in);
      ++segments;
    } else if (kind == "config") {
      config.max_branch_radius_m = number(in);
      config.mouth_position_tolerance_m = number(in);
      config.alignment_tolerance_deg = number(in);
      config.vision_max_age_s = number(in);
      clearance = number(in);
      gate = std::make_unique<SafetyGate>(config, clearance, initial_missing, initial_latch);
    } else if (kind == "cycle") {
      ASSERT_TRUE(gate) << "cycle before config in " << label;
      inputs = GateInputs{};
      index = std::stol(word(in));
      auto & observation = inputs.observation;
      observation.time_s = number(in);
      observation.vision_timestamp_s = number(in);
      const double held = number(in);
      if (!std::isnan(held)) {
        observation.held_target_timestamp_s = held;
      }
      observation.target_id = word(in);
      observation.target_semantic = word(in);
      observation.vision_valid = flag(in);
      observation.vision_source = word(in);
      for (auto & v : observation.target_position_w) {v = number(in);}
      for (auto & v : observation.target_axis_w) {v = number(in);}
      observation.target_radius_m = number(in);
      std::array<double, 7> pose{};
      for (auto & v : pose) {v = number(in);}
      pruning_sil_cpp::Vec3 offset{};
      pruning_sil_cpp::Vec3 closing{};
      for (auto & v : offset) {v = number(in);}
      for (auto & v : closing) {v = number(in);}
      const auto geometry = pruning_sil_cpp::tool_mouth_geometry(pose, offset, closing);
      observation.mouth_position_w = geometry.first;
      observation.cutter_closing_axis_w = geometry.second;
      observation.hazard_contact = number(in) > pruning_sil_cpp::kHazardContactForceN;
      inputs.prior_cut_phase = word(in);
      inputs.prior_detached = flag(in);
    } else if (kind == "tof") {
      inputs.tof_ranges.clear();
      inputs.tof_valid.clear();
      for (int zone = 0; zone < 128; ++zone) {
        const double range = number(in);
        inputs.tof_ranges.push_back(range);
        inputs.tof_valid.push_back(std::isfinite(range));  // as the node reads a range image
      }
    } else if (kind == "expect") {
      const std::string where = label + " frame " + std::to_string(index);
      const auto verdict = gate->step(inputs);
      ++cycles;
      holds += verdict.passed ? 0 : 1;
      EXPECT_EQ(verdict.passed, flag(in)) << where;
      EXPECT_EQ(verdict.reason, word(in)) << where;
      EXPECT_EQ(verdict.latched_reason, word(in)) << where;
      EXPECT_EQ(verdict.tof.reason, word(in)) << where;
      EXPECT_EQ(verdict.tof.consecutive_missing, std::stoi(word(in))) << where;
      expect_close(verdict.tof.min_range_m, number(in), where + " tof_min_range_m");
      EXPECT_EQ(verdict.certificate.safe_to_approach, flag(in)) << where;
      EXPECT_EQ(verdict.certificate.ready_to_close, flag(in)) << where;
      EXPECT_EQ(verdict.closure_authorized, flag(in)) << where;
      expect_close(or_nan(verdict.certificate.mouth_distance_m), number(in), where + " mouth");
      expect_close(
        or_nan(verdict.certificate.perpendicularity_error_deg), number(in), where + " alignment");
      expect_close(or_nan(verdict.certificate.vision_age_s), number(in), where + " age");
      const int count = std::stoi(word(in));
      std::vector<std::string> reasons;
      for (int i = 0; i < count; ++i) {
        reasons.push_back(word(in));
      }
      EXPECT_EQ(verdict.certificate.reasons, reasons) << where;
    }
  }
  EXPECT_GT(segments, 0u);
  EXPECT_GT(cycles, 0u);
  EXPECT_GT(holds, 0u) << "the fixture must exercise at least one hold";
  std::cout << "[parity] " << segments << " segments, " << cycles << " cycles, " << holds
            << " holds\n";
}
