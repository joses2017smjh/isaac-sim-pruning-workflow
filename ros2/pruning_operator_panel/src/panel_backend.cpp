#include "pruning_operator_panel/panel_backend.hpp"

#include <QMetaObject>
#include <QMutexLocker>

#include <cmath>
#include <sstream>

namespace pruning_operator_panel
{
namespace
{

int64_t stamp_ns(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}

std::string quoted(const QString & value)
{
  std::string out = "\"";
  for (const char ch : value.toStdString()) {
    if (ch == '"' || ch == '\\') {
      out += '\\';
    }
    out += (static_cast<unsigned char>(ch) < 0x20) ? ' ' : ch;
  }
  return out + "\"";
}

std::string number(double value)
{
  if (!std::isfinite(value)) {
    return "null";
  }
  std::ostringstream stream;
  stream.precision(12);
  stream << value;
  return stream.str();
}

}  // namespace

PanelBackend::PanelBackend(WristImageProvider * provider, const std::string & record_path, QObject * parent)
: QObject(parent), provider_(provider)
{
  if (!record_path.empty()) {
    record_.open(record_path, std::ios::out | std::ios::trunc);
  }
}

PanelBackend::~PanelBackend()
{
  stop();
}

void PanelBackend::write(const std::string & line)
{
  QMutexLocker lock(&record_mutex_);
  if (record_.is_open()) {
    record_ << line << "\n";
    record_.flush();
  }
}

void PanelBackend::noteEvent(const std::string & json_object)
{
  write(json_object);
}

void PanelBackend::start()
{
  node_ = std::make_shared<rclcpp::Node>("pruning_operator_panel");
  // The gate's inputs and outputs are reliable; a reliable reader keeps every decision.
  const auto reliable = rclcpp::QoS(rclcpp::KeepLast(50)).reliable();
  const auto latched = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  hold_publisher_ = node_->create_publisher<pruning_sil_msgs::msg::OperatorHold>("/pruning/operator/hold", latched);

  subscriptions_.push_back(node_->create_subscription<pruning_sil_msgs::msg::SafetyDecision>(
    "/pruning/safety/decision", reliable,
    [this](pruning_sil_msgs::msg::SafetyDecision::ConstSharedPtr message) {
      const int64_t received = wall_ns();
      QMetaObject::invokeMethod(this, [this, message, received]() {applyDecision(*message, received);},
        Qt::QueuedConnection);
    }));
  for (int side = 0; side < 2; ++side) {
    const char * topic = side == 0 ? "/pruning/tof/left/range_image" : "/pruning/tof/right/range_image";
    subscriptions_.push_back(node_->create_subscription<sensor_msgs::msg::Image>(
      topic, reliable, [this, side](sensor_msgs::msg::Image::ConstSharedPtr message) {
        const auto summary = summarize_range_image(*message);
        QMetaObject::invokeMethod(this, [this, side, summary]() {applyRange(side, summary);},
          Qt::QueuedConnection);
      }));
  }
  subscriptions_.push_back(node_->create_subscription<sensor_msgs::msg::Image>(
    "/pruning/wrist/image_raw", reliable, [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
      const int64_t received = wall_ns();
      QImage image = to_qimage(*message);
      const int64_t stamp = stamp_ns(message->header.stamp);
      QMetaObject::invokeMethod(this, [this, image, stamp, received]() {applyFrame(image, stamp, received);},
        Qt::QueuedConnection);
    }));
  subscriptions_.push_back(node_->create_subscription<std_msgs::msg::String>(
    "/pruning/sim/cycle", reliable, [this](std_msgs::msg::String::ConstSharedPtr message) {
      const int64_t received = wall_ns();
      const std::string data = message->data;
      QMetaObject::invokeMethod(this, [this, data, received]() {applyCycleMarker(data, received);},
        Qt::QueuedConnection);
    }));

  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
  executor_->add_node(node_);
  spinner_ = std::thread([this]() {executor_->spin();});
  write("{\"event\":\"start\",\"wall_ns\":" + std::to_string(wall_ns()) + "}");
}

void PanelBackend::stop()
{
  if (executor_) {
    executor_->cancel();
  }
  if (spinner_.joinable()) {
    spinner_.join();
  }
  executor_.reset();
  subscriptions_.clear();
  hold_publisher_.reset();
  node_.reset();
}

void PanelBackend::requestHold(const QString & source)
{
  const int64_t now = wall_ns();
  if (hold_publisher_) {
    pruning_sil_msgs::msg::OperatorHold hold;
    hold.header.stamp.sec = static_cast<int32_t>(now / 1000000000LL);
    hold.header.stamp.nanosec = static_cast<uint32_t>(now % 1000000000LL);
    hold.hold = true;
    hold.source = "operator_panel";
    hold_publisher_->publish(hold);
  }
  hold_requested_ = true;
  Q_EMIT holdChanged();
  write("{\"event\":\"hold_pressed\",\"wall_ns\":" + std::to_string(now) + ",\"source\":" + quoted(source) +
    ",\"shown_decision_stamp_ns\":" + std::to_string(shown_stamp_ns_) + "}");
}

void PanelBackend::applyDecision(const pruning_sil_msgs::msg::SafetyDecision & decision, int64_t receive_ns)
{
  const bool changed = !have_decision_ || passed_ != decision.passed;
  have_decision_ = true;
  passed_ = decision.passed;
  reason_ = QString::fromStdString(decision.reason);
  latched_reason_ = QString::fromStdString(decision.latched_reason);
  cycle_ = static_cast<int>(decision.cycle);
  mouth_distance_m_ = decision.mouth_distance_m;
  operator_hold_seen_ = operator_hold_seen_ || decision.operator_hold;
  shown_stamp_ns_ = stamp_ns(decision.header.stamp);
  ++decisions_shown_;
  Q_EMIT decisionChanged();
  write("{\"event\":\"decision\",\"stamp_ns\":" + std::to_string(shown_stamp_ns_) + ",\"cycle\":" +
    std::to_string(decision.cycle) + ",\"passed\":" + (decision.passed ? "true" : "false") + ",\"reason\":" +
    quoted(reason_) + ",\"operator_hold\":" + (decision.operator_hold ? "true" : "false") +
    ",\"pass_hold_changed\":" + (changed ? "true" : "false") + ",\"mouth_distance_m\":" +
    number(decision.mouth_distance_m) + ",\"receive_wall_ns\":" +
    std::to_string(receive_ns) + ",\"applied_wall_ns\":" + std::to_string(wall_ns()) + "}");
}

void PanelBackend::applyRange(int side, const RangeSummary & summary)
{
  tof_m_[side] = summary.min_m;
  Q_EMIT tofChanged();
}

void PanelBackend::applyFrame(const QImage & image, int64_t stamp_ns_value, int64_t receive_ns)
{
  if (provider_ && !image.isNull()) {
    provider_->set(image);
  }
  ++frame_counter_;
  Q_EMIT frameChanged();
  write("{\"event\":\"frame\",\"stamp_ns\":" + std::to_string(stamp_ns_value) + ",\"valid\":" +
    (image.isNull() ? "false" : "true") + ",\"receive_wall_ns\":" + std::to_string(receive_ns) +
    ",\"applied_wall_ns\":" + std::to_string(wall_ns()) + "}");
}

void PanelBackend::applyCycleMarker(const std::string & json, int64_t receive_ns)
{
  // The marker is the simulator's JSON; the panel needs two fields, so it reads them plainly.
  if (json.find("\"end\": true") != std::string::npos || json.find("\"end\":true") != std::string::npos) {
    write("{\"event\":\"simulation_end\",\"wall_ns\":" + std::to_string(receive_ns) + "}");
    Q_EMIT simulationEnded();
    return;
  }
  const auto key = json.find("\"delay_ms\"");
  if (key != std::string::npos) {
    const auto colon = json.find(':', key);
    if (colon != std::string::npos) {
      try {
        const double value = std::stod(json.substr(colon + 1));
        if (std::isfinite(value) && value != delay_ms_) {
          delay_ms_ = value;
          Q_EMIT delayChanged();
        }
      } catch (const std::exception &) {
      }
    }
  }
}

void PanelBackend::noteRendered()
{
  write("{\"event\":\"rendered\",\"wall_ns\":" + std::to_string(wall_ns()) + ",\"shown_decision_stamp_ns\":" +
    std::to_string(shown_stamp_ns_) + ",\"passed\":" + (passed_ ? "true" : "false") + ",\"frame\":" +
    std::to_string(frame_counter_) + "}");
}

}  // namespace pruning_operator_panel
