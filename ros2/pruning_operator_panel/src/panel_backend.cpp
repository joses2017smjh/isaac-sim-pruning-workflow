#include "pruning_operator_panel/panel_backend.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMetaObject>
#include <QMutexLocker>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace pruning_operator_panel
{
namespace
{

const char * const kDecisionTopic = "/pruning/safety/decision";
const char * const kDecisionJsonTopic = "/pruning/safety/decision_json";
const char * const kImageTopic = "/pruning/wrist/image_raw";
const char * const kCycleTopic = "/pruning/sim/cycle";
const char * const kRangeTopics[2] = {"/pruning/tof/left/range_image", "/pruning/tof/right/range_image"};

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

double json_number(const QJsonValue & value)
{
  return value.isDouble() ? value.toDouble() : kNan;
}

DecisionRow row_from_json(const QJsonObject & object)
{
  DecisionRow row;
  row.stamp_ns = static_cast<int64_t>(object.value(QStringLiteral("stamp_ns")).toDouble());
  row.cycle = object.value(QStringLiteral("cycle")).toInt();
  row.passed = object.value(QStringLiteral("passed")).toBool();
  row.reason = object.value(QStringLiteral("reason")).toString();
  row.latched_reason = object.value(QStringLiteral("latched_reason")).toString();
  row.tof_reason = object.value(QStringLiteral("tof_reason")).toString();
  row.tof_min_m = json_number(object.value(QStringLiteral("tof_min_range_m")));
  row.mouth_distance_m = json_number(object.value(QStringLiteral("mouth_distance_m")));
  row.operator_hold = object.value(QStringLiteral("operator_hold")).toBool();
  row.closure_authorized = object.value(QStringLiteral("closure_authorized")).toBool();
  for (const auto & reason : object.value(QStringLiteral("cut_reasons")).toArray()) {
    row.cut_reasons.push_back(reason.toString());
  }
  return row;
}

QVariantList zone_list(const std::vector<double> & zones)
{
  QVariantList out;
  for (const double value : zones) {
    out.push_back(value);
  }
  return out;
}

double zone_min(const std::vector<double> & zones)
{
  double best = kNan;
  for (const double value : zones) {
    if (std::isfinite(value) && value > 0.0 && (std::isnan(best) || value < best)) {
      best = value;
    }
  }
  return best;
}

}  // namespace

PanelBackend::PanelBackend(
  WristImageProvider * provider, const std::string & record_path, QObject * parent, double stale_after_s)
: QObject(parent), provider_(provider), log_(this), filter_(this), series_(this), stale_after_s_(stale_after_s)
{
  filter_.setSourceModel(&log_);
  if (!record_path.empty()) {
    record_.open(record_path, std::ios::out | std::ios::trunc);
  }
  connect(&diagnostics_timer_, &QTimer::timeout, this, [this]() {refreshDiagnostics(wall_ns());});
  diagnostics_timer_.start(250);
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
  const auto reliable = rclcpp::QoS(rclcpp::KeepLast(50)).reliable();
  const auto latched = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
  hold_publisher_ = node_->create_publisher<pruning_sil_msgs::msg::OperatorHold>("/pruning/operator/hold", latched);

  // Callbacks never touch the screen: each hands its data to the GUI thread.
  subscriptions_.push_back(node_->create_subscription<pruning_sil_msgs::msg::SafetyDecision>(
    kDecisionTopic, reliable, [this](pruning_sil_msgs::msg::SafetyDecision::ConstSharedPtr message) {
      const int64_t received = wall_ns();
      QMetaObject::invokeMethod(this, [this, message, received]() {applyDecision(*message, received);},
        Qt::QueuedConnection);
    }));
  subscriptions_.push_back(node_->create_subscription<std_msgs::msg::String>(
    kDecisionJsonTopic, reliable, [this](std_msgs::msg::String::ConstSharedPtr message) {
      const int64_t received = wall_ns();
      const std::string data = message->data;
      QMetaObject::invokeMethod(this, [this, data, received]() {applyDecisionJson(data, received);},
        Qt::QueuedConnection);
    }));
  for (int side = 0; side < 2; ++side) {
    subscriptions_.push_back(node_->create_subscription<sensor_msgs::msg::Image>(
      kRangeTopics[side], reliable, [this, side](sensor_msgs::msg::Image::ConstSharedPtr message) {
        const int64_t received = wall_ns();
        const auto summary = summarize_range_image(*message);
        const int64_t stamp = stamp_ns(message->header.stamp);
        QMetaObject::invokeMethod(this, [this, side, summary, stamp, received]() {
            applyRange(side, summary, stamp, received);
          }, Qt::QueuedConnection);
      }));
  }
  subscriptions_.push_back(node_->create_subscription<sensor_msgs::msg::Image>(
    kImageTopic, reliable, [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
      const int64_t received = wall_ns();
      QImage image = to_qimage(*message);
      const int64_t stamp = stamp_ns(message->header.stamp);
      QMetaObject::invokeMethod(this, [this, image, stamp, received]() {applyFrame(image, stamp, received);},
        Qt::QueuedConnection);
    }));
  subscriptions_.push_back(node_->create_subscription<std_msgs::msg::String>(
    kCycleTopic, reliable, [this](std_msgs::msg::String::ConstSharedPtr message) {
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
  if (review_mode_) {
    return;  // a recording cannot be stopped
  }
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

void PanelBackend::show(const DecisionRow & row)
{
  have_decision_ = true;
  passed_ = row.passed;
  reason_ = row.reason;
  latched_reason_ = row.latched_reason;
  cut_reasons_ = row.cut_reasons.join(QStringLiteral(", "));
  cycle_ = row.cycle;
  mouth_distance_m_ = row.mouth_distance_m;
  operator_hold_seen_ = review_mode_ ? row.operator_hold : (operator_hold_seen_ || row.operator_hold);
  shown_stamp_ns_ = row.stamp_ns;
  Q_EMIT decisionChanged();
}

void PanelBackend::applyDecision(const pruning_sil_msgs::msg::SafetyDecision & decision, int64_t receive_ns)
{
  DecisionRow row;
  row.stamp_ns = stamp_ns(decision.header.stamp);
  row.cycle = static_cast<int>(decision.cycle);
  row.passed = decision.passed;
  row.reason = QString::fromStdString(decision.reason);
  row.latched_reason = QString::fromStdString(decision.latched_reason);
  row.tof_reason = QString::fromStdString(decision.tof_reason);
  row.tof_min_m = decision.tof_min_range_m;
  row.mouth_distance_m = decision.mouth_distance_m;
  row.operator_hold = decision.operator_hold;
  row.closure_authorized = decision.closure_authorized;
  for (const auto & reason : decision.cut_reasons) {
    row.cut_reasons.push_back(QString::fromStdString(reason));
  }
  const bool changed = !have_decision_ || passed_ != row.passed;
  if (last_cycle_ >= 0 && row.cycle > last_cycle_ + 1) {
    missed_decisions_ += row.cycle - last_cycle_ - 1;  // the gate numbers its decisions
  }
  last_cycle_ = std::max(last_cycle_, row.cycle);
  last_decision_receive_ns_ = receive_ns;
  ++decisions_shown_;
  log_.append(row);
  const double t = static_cast<double>(row.stamp_ns) * 1e-9;
  series_.append(QStringLiteral("mouth_mm"), t, row.mouth_distance_m * 1000.0);
  series_.append(QStringLiteral("hold"), t, row.passed ? 0.0 : 1.0);
  stats_.note(kDecisionTopic, receive_ns);
  receive_wall_by_stamp_[row.stamp_ns] = receive_ns;
  noteLatency(row.stamp_ns);
  show(row);
  write("{\"event\":\"decision\",\"stamp_ns\":" + std::to_string(row.stamp_ns) + ",\"cycle\":" +
    std::to_string(row.cycle) + ",\"passed\":" + (row.passed ? "true" : "false") + ",\"reason\":" +
    quoted(row.reason) + ",\"operator_hold\":" + (row.operator_hold ? "true" : "false") +
    ",\"pass_hold_changed\":" + (changed ? "true" : "false") + ",\"mouth_distance_m\":" +
    number(row.mouth_distance_m) + ",\"receive_wall_ns\":" + std::to_string(receive_ns) +
    ",\"applied_wall_ns\":" + std::to_string(wall_ns()) + "}");
}

void PanelBackend::applyDecisionJson(const std::string & json, int64_t receive_ns)
{
  const QJsonObject object = QJsonDocument::fromJson(QByteArray::fromStdString(json)).object();
  if (object.isEmpty()) {
    return;
  }
  const auto stamp = static_cast<int64_t>(object.value(QStringLiteral("stamp_ns")).toDouble());
  const auto published = static_cast<int64_t>(object.value(QStringLiteral("publish_wall_ns")).toDouble());
  stats_.note(kDecisionJsonTopic, receive_ns);
  if (published > 0) {
    publish_wall_by_stamp_[stamp] = published;
    noteLatency(stamp);
  }
}

void PanelBackend::noteLatency(int64_t stamp)
{
  const auto published = publish_wall_by_stamp_.find(stamp);
  const auto received = receive_wall_by_stamp_.find(stamp);
  if (published != publish_wall_by_stamp_.end() && received != receive_wall_by_stamp_.end()) {
    // Gate publish to panel receive: both clocks are this node's wall clock.
    latencies_ms_.push_back(static_cast<double>(received->second - published->second) * 1e-6);
    while (latencies_ms_.size() > 500) {
      latencies_ms_.pop_front();
    }
    publish_wall_by_stamp_.erase(published);
    receive_wall_by_stamp_.erase(received);
  }
  while (publish_wall_by_stamp_.size() > 1000) {
    publish_wall_by_stamp_.erase(publish_wall_by_stamp_.begin());
  }
  while (receive_wall_by_stamp_.size() > 1000) {
    receive_wall_by_stamp_.erase(receive_wall_by_stamp_.begin());
  }
}

void PanelBackend::applyRange(int side, const RangeSummary & summary, int64_t stamp, int64_t receive_ns)
{
  tof_m_[side] = summary.min_m;
  zones_[side] = zone_list(summary.zones);
  series_.append(side == 0 ? QStringLiteral("tof_left_cm") : QStringLiteral("tof_right_cm"),
    static_cast<double>(stamp) * 1e-9, summary.min_m * 100.0);
  stats_.note(kRangeTopics[side], receive_ns);
  Q_EMIT tofChanged();
}

void PanelBackend::applyFrame(const QImage & image, int64_t stamp_ns_value, int64_t receive_ns)
{
  if (provider_ && !image.isNull()) {
    provider_->set(image);
  }
  ++frame_counter_;
  stats_.note(kImageTopic, receive_ns);
  Q_EMIT frameChanged();
  write("{\"event\":\"frame\",\"stamp_ns\":" + std::to_string(stamp_ns_value) + ",\"valid\":" +
    (image.isNull() ? "false" : "true") + ",\"receive_wall_ns\":" + std::to_string(receive_ns) +
    ",\"applied_wall_ns\":" + std::to_string(wall_ns()) + "}");
}

void PanelBackend::applyCycleMarker(const std::string & json, int64_t receive_ns)
{
  const QJsonObject object = QJsonDocument::fromJson(QByteArray::fromStdString(json)).object();
  stats_.note(kCycleTopic, receive_ns);
  if (object.value(QStringLiteral("end")).toBool()) {
    simulation_ended_ = true;
    write("{\"event\":\"simulation_end\",\"wall_ns\":" + std::to_string(receive_ns) + "}");
    Q_EMIT diagnosticsChanged();
    Q_EMIT simulationEnded();
    return;
  }
  const double delay = json_number(object.value(QStringLiteral("delay_ms")));
  if (std::isfinite(delay) && delay != delay_ms_) {
    delay_ms_ = delay;
    Q_EMIT delayChanged();
  }
}

void PanelBackend::noteRendered()
{
  const int64_t now = wall_ns();
  renders_.push_back(now);
  while (!renders_.empty() && now - renders_.front() > 2000000000LL) {
    renders_.pop_front();
  }
  write("{\"event\":\"rendered\",\"wall_ns\":" + std::to_string(now) + ",\"shown_decision_stamp_ns\":" +
    std::to_string(shown_stamp_ns_) + ",\"passed\":" + (passed_ ? "true" : "false") + ",\"frame\":" +
    std::to_string(frame_counter_) + "}");
}

void PanelBackend::refreshDiagnostics(int64_t now_ns)
{
  decision_age_s_ = have_decision_ && !review_mode_ ?
    static_cast<double>(now_ns - last_decision_receive_ns_) * 1e-9 : kNan;
  // A screen that keeps showing PASS after the gate went quiet is lying; say so.
  stale_ = have_decision_ && !review_mode_ && !simulation_ended_ && decision_age_s_ > stale_after_s_;
  topic_rows_ = stats_.rows(now_ns);
  if (node_) {
    for (auto & entry : topic_rows_) {
      QVariantMap row = entry.toMap();
      row[QStringLiteral("publishers")] =
        static_cast<int>(node_->count_publishers(row[QStringLiteral("topic")].toString().toStdString()));
      entry = row;
    }
  }
  const std::vector<double> latencies(latencies_ms_.begin(), latencies_ms_.end());
  latency_p50_ms_ = percentile(latencies, 50);
  latency_p95_ms_ = percentile(latencies, 95);
  int recent = 0;
  for (const int64_t t : renders_) {
    recent += now_ns - t <= 1000000000LL ? 1 : 0;
  }
  render_fps_ = recent;
  Q_EMIT diagnosticsChanged();
}

bool PanelBackend::loadReview(const QString & run_dir)
{
  const QDir dir(run_dir);
  QFile decisions(dir.filePath(QStringLiteral("ros2_loop_decisions.jsonl")));
  if (!decisions.open(QIODevice::ReadOnly | QIODevice::Text)) {
    return false;
  }
  std::vector<DecisionRow> rows;
  while (!decisions.atEnd()) {
    const QByteArray line = decisions.readLine().trimmed();
    if (!line.isEmpty()) {
      rows.push_back(row_from_json(QJsonDocument::fromJson(line).object()));
    }
  }
  if (rows.empty()) {
    return false;
  }
  std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {return a.stamp_ns < b.stamp_ns;});

  // Per-frame ToF grids from the recorder's frames.json (null = unusable zone).
  review_zones_[0].clear();
  review_zones_[1].clear();
  QFile frames(dir.filePath(QStringLiteral("frames.json")));
  if (frames.open(QIODevice::ReadOnly)) {
    const QJsonObject document = QJsonDocument::fromJson(frames.readAll()).object();
    review_fps_ = std::max(1, document.value(QStringLiteral("fps")).toInt(10));
    for (const auto & frame_value : document.value(QStringLiteral("frames")).toArray()) {
      const QJsonObject frame = frame_value.toObject();
      for (int side = 0; side < 2; ++side) {
        std::vector<double> zones;
        const auto key = side == 0 ? QStringLiteral("tof_left_m") : QStringLiteral("tof_right_m");
        const auto valid_key = side == 0 ? QStringLiteral("tof_left_valid") : QStringLiteral("tof_right_valid");
        const QJsonArray grid = frame.value(key).toArray();
        const QJsonArray valid = frame.value(valid_key).toArray();
        for (int r = 0; r < grid.size(); ++r) {
          const QJsonArray row = grid[r].toArray();
          const QJsonArray valid_row = valid[r].toArray();
          for (int c = 0; c < row.size(); ++c) {
            const bool ok = c < valid_row.size() && valid_row[c].toBool();
            zones.push_back(ok ? json_number(row[c]) : kNan);
          }
        }
        review_zones_[side].push_back(zones);
      }
    }
  }

  review_mode_ = true;
  review_dir_ = run_dir;
  review_run_ = QFileInfo(run_dir).fileName();
  review_rows_ = rows;
  log_.clear();
  series_.clear();
  for (const auto & row : review_rows_) {
    log_.append(row);
    const double t = static_cast<double>(row.stamp_ns) * 1e-9;
    series_.append(QStringLiteral("mouth_mm"), t, row.mouth_distance_m * 1000.0);
    series_.append(QStringLiteral("hold"), t, row.passed ? 0.0 : 1.0);
    // The bundle stamped k / fps was captured after frame k - 1's physics.
    const int frame = static_cast<int>(std::llround(t * review_fps_)) - 1;
    for (int side = 0; side < 2; ++side) {
      const auto & zones = review_zones_[side];
      const double nearest = frame >= 0 && frame < static_cast<int>(zones.size()) ? zone_min(zones[frame]) : kNan;
      series_.append(side == 0 ? QStringLiteral("tof_left_cm") : QStringLiteral("tof_right_cm"), t, nearest * 100.0);
    }
  }
  review_index_ = -1;
  setReviewIndex(0);
  return true;
}

void PanelBackend::seekReviewTime(double time_s)
{
  if (!review_mode_ || review_rows_.empty()) {
    return;
  }
  int best = 0;
  double best_gap = std::numeric_limits<double>::infinity();
  for (int i = 0; i < static_cast<int>(review_rows_.size()); ++i) {
    const double gap = std::abs(static_cast<double>(review_rows_[i].stamp_ns) * 1e-9 - time_s);
    if (gap < best_gap) {
      best_gap = gap;
      best = i;
    }
  }
  setReviewIndex(best);
}

void PanelBackend::setReviewIndex(int index)
{
  if (!review_mode_ || review_rows_.empty()) {
    return;
  }
  index = std::clamp(index, 0, static_cast<int>(review_rows_.size()) - 1);
  if (index == review_index_) {
    return;
  }
  review_index_ = index;
  const DecisionRow & row = review_rows_[index];
  const int frame = static_cast<int>(std::llround(static_cast<double>(row.stamp_ns) * 1e-9 * review_fps_)) - 1;
  for (int side = 0; side < 2; ++side) {
    const auto & zones = review_zones_[side];
    const std::vector<double> grid = frame >= 0 && frame < static_cast<int>(zones.size()) ? zones[frame] :
      std::vector<double>(64, kNan);
    zones_[side] = zone_list(grid);
    tof_m_[side] = zone_min(grid);
  }
  const QDir dir(review_dir_);
  const QString image_path = frame < 0 ? dir.filePath(QStringLiteral("preview_wrist.png")) :
    dir.filePath(QStringLiteral("frames/wrist_%1.png").arg(frame, 5, 10, QLatin1Char('0')));
  QImage image(image_path);
  if (provider_ && !image.isNull()) {
    provider_->set(image.convertToFormat(QImage::Format_RGB888));
  }
  ++frame_counter_;
  show(row);
  Q_EMIT tofChanged();
  Q_EMIT frameChanged();
  Q_EMIT reviewChanged();
}

}  // namespace pruning_operator_panel
