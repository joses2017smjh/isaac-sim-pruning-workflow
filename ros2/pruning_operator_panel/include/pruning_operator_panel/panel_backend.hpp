// C++ backend of the Qt/QML operator panel.
//
// Live mode: an rclcpp node, spun on its own thread, subscribes to the wrist camera,
// both time-of-flight range images, the safety gate's decisions (typed, and the JSON
// mirror that carries the gate's publish time) and the simulator's cycle markers.
// Each message is handed to the GUI thread (queued), where it updates the properties,
// models and series the QML screen binds to. requestHold() publishes a
// pruning_sil_msgs/OperatorHold, which the C++ safety gate latches for the run.
//
// Review mode: no graph. loadReview() reads a recorded live-loop run (the gate's
// decision log, frames.json and the wrist images) and setReviewIndex() scrubs it.
//
// Every received decision and camera frame, every repaint and every hold press is
// written to a JSONL record with wall-clock times.

#ifndef PRUNING_OPERATOR_PANEL__PANEL_BACKEND_HPP_
#define PRUNING_OPERATOR_PANEL__PANEL_BACKEND_HPP_

#include <QImage>
#include <QMutex>
#include <QObject>
#include <QQuickImageProvider>
#include <QQuickWindow>
#include <QString>
#include <QTimer>
#include <QVariantList>

#include <atomic>
#include <cstdint>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "pruning_operator_panel/panel_items.hpp"
#include "pruning_sil_msgs/msg/operator_hold.hpp"
#include "pruning_sil_msgs/msg/safety_decision.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"

namespace pruning_operator_panel
{

/// Nearest usable return of one 8x8 32FC1 range image; NaN zones are not usable.
struct RangeSummary
{
  double min_m{std::numeric_limits<double>::quiet_NaN()};
  int usable_zones{0};
  bool valid_message{false};
  std::vector<double> zones;  // 64 values, row-major; NaN = unusable
};

RangeSummary summarize_range_image(const sensor_msgs::msg::Image & image);

/// Deep-copied RGB888 image of an rgb8 message; a null image when it is malformed.
QImage to_qimage(const sensor_msgs::msg::Image & image);

int64_t wall_ns();

/// The window's pixels. QQuickWindow::grabWindow() returns null under the software
/// backend on the offscreen platform (its backing store has no toImage()), so this
/// falls back to the screen's copy of the same backing store. Null if not exposed.
QImage grab(QQuickWindow * window);

/// The p-th percentile (0-100) of values; NaN when empty.
double percentile(std::vector<double> values, double p);

/// Serves the latest wrist frame to QML as image://wrist/<n>.
class WristImageProvider : public QQuickImageProvider
{
public:
  WristImageProvider();
  QImage requestImage(const QString & id, QSize * size, const QSize & requested) override;
  void set(const QImage & image);

private:
  QMutex mutex_;
  QImage latest_;
};

class PanelBackend : public QObject
{
  Q_OBJECT
  Q_PROPERTY(bool haveDecision READ haveDecision NOTIFY decisionChanged)
  Q_PROPERTY(bool passed READ passed NOTIFY decisionChanged)
  Q_PROPERTY(QString reason READ reason NOTIFY decisionChanged)
  Q_PROPERTY(QString latchedReason READ latchedReason NOTIFY decisionChanged)
  Q_PROPERTY(int cycle READ cycle NOTIFY decisionChanged)
  Q_PROPERTY(double timeS READ timeS NOTIFY decisionChanged)
  Q_PROPERTY(double mouthDistanceM READ mouthDistanceM NOTIFY decisionChanged)
  Q_PROPERTY(bool operatorHoldSeen READ operatorHoldSeen NOTIFY decisionChanged)
  Q_PROPERTY(QString cutReasons READ cutReasons NOTIFY decisionChanged)
  Q_PROPERTY(double tofLeftM READ tofLeftM NOTIFY tofChanged)
  Q_PROPERTY(double tofRightM READ tofRightM NOTIFY tofChanged)
  Q_PROPERTY(QVariantList tofLeftZones READ tofLeftZones NOTIFY tofChanged)
  Q_PROPERTY(QVariantList tofRightZones READ tofRightZones NOTIFY tofChanged)
  Q_PROPERTY(double clearanceM READ clearanceM CONSTANT)
  Q_PROPERTY(double mouthToleranceM READ mouthToleranceM CONSTANT)
  Q_PROPERTY(double delayMs READ delayMs NOTIFY delayChanged)
  Q_PROPERTY(int frameCounter READ frameCounter NOTIFY frameChanged)
  Q_PROPERTY(bool holdRequested READ holdRequested NOTIFY holdChanged)
  Q_PROPERTY(QObject * decisions READ decisions CONSTANT)
  Q_PROPERTY(int decisionCount READ decisionCount NOTIFY decisionChanged)
  Q_PROPERTY(int holdCount READ holdCount NOTIFY decisionChanged)
  Q_PROPERTY(QObject * series READ series CONSTANT)
  Q_PROPERTY(QVariantList topicStats READ topicStats NOTIFY diagnosticsChanged)
  Q_PROPERTY(double decisionAgeS READ decisionAgeS NOTIFY diagnosticsChanged)
  Q_PROPERTY(bool stale READ stale NOTIFY diagnosticsChanged)
  Q_PROPERTY(double staleAfterS READ staleAfterS CONSTANT)
  Q_PROPERTY(double latencyP50Ms READ latencyP50Ms NOTIFY diagnosticsChanged)
  Q_PROPERTY(double latencyP95Ms READ latencyP95Ms NOTIFY diagnosticsChanged)
  Q_PROPERTY(int missedDecisions READ missedDecisions NOTIFY diagnosticsChanged)
  Q_PROPERTY(double renderFps READ renderFps NOTIFY diagnosticsChanged)
  Q_PROPERTY(bool simulationEnded READ simulationEndedFlag NOTIFY diagnosticsChanged)
  Q_PROPERTY(bool reviewMode READ reviewMode NOTIFY reviewChanged)
  Q_PROPERTY(int reviewIndex READ reviewIndex WRITE setReviewIndex NOTIFY reviewChanged)
  Q_PROPERTY(int reviewCount READ reviewCount NOTIFY reviewChanged)
  Q_PROPERTY(QString reviewRun READ reviewRun NOTIFY reviewChanged)
  Q_PROPERTY(double cursorS READ cursorS NOTIFY reviewChanged)

public:
  PanelBackend(
    WristImageProvider * provider, const std::string & record_path, QObject * parent = nullptr,
    double stale_after_s = 10.0);
  ~PanelBackend() override;

  /// Create the node and spin it on a background thread (live mode).
  void start();
  void stop();

  /// Publish an operator hold. Bound to the QML Hold button and its shortcut.
  Q_INVOKABLE void requestHold(const QString & source = QStringLiteral("button"));
  /// Load a recorded live-loop run directory; returns false and leaves the panel live on failure.
  Q_INVOKABLE bool loadReview(const QString & run_dir);
  Q_INVOKABLE void stepReview(int delta) {setReviewIndex(review_index_ + delta);}
  /// Scrub to the recorded decision nearest `time_s` (a list row, whatever the filter).
  Q_INVOKABLE void seekReviewTime(double time_s);

  /// Called after each repaint, with the decision stamp the screen was showing.
  void noteRendered();
  void noteEvent(const std::string & json_object);
  /// Recompute ages, rates and staleness at `now_ns` (a timer calls it with the wall clock).
  void refreshDiagnostics(int64_t now_ns);

  bool haveDecision() const {return have_decision_;}
  bool passed() const {return passed_;}
  QString reason() const {return reason_;}
  QString latchedReason() const {return latched_reason_;}
  int cycle() const {return cycle_;}
  double timeS() const {return static_cast<double>(shown_stamp_ns_) * 1e-9;}
  double mouthDistanceM() const {return mouth_distance_m_;}
  bool operatorHoldSeen() const {return operator_hold_seen_;}
  QString cutReasons() const {return cut_reasons_;}
  double tofLeftM() const {return tof_m_[0];}
  double tofRightM() const {return tof_m_[1];}
  QVariantList tofLeftZones() const {return zones_[0];}
  QVariantList tofRightZones() const {return zones_[1];}
  double clearanceM() const {return 0.06;}
  double mouthToleranceM() const {return 0.008;}
  double delayMs() const {return delay_ms_;}
  int frameCounter() const {return frame_counter_;}
  bool holdRequested() const {return hold_requested_;}
  QObject * decisions() {return &filter_;}
  int decisionCount() const {return log_.count();}
  int holdCount() const {return log_.holdCount();}
  QObject * series() {return &series_;}
  DecisionLogModel & log() {return log_;}
  SeriesStore & seriesStore() {return series_;}
  QVariantList topicStats() const {return topic_rows_;}
  double decisionAgeS() const {return decision_age_s_;}
  bool stale() const {return stale_;}
  double staleAfterS() const {return stale_after_s_;}
  double latencyP50Ms() const {return latency_p50_ms_;}
  double latencyP95Ms() const {return latency_p95_ms_;}
  int missedDecisions() const {return missed_decisions_;}
  double renderFps() const {return render_fps_;}
  bool simulationEndedFlag() const {return simulation_ended_;}
  bool reviewMode() const {return review_mode_;}
  int reviewIndex() const {return review_index_;}
  int reviewCount() const {return static_cast<int>(review_rows_.size());}
  QString reviewRun() const {return review_run_;}
  double cursorS() const {return review_mode_ ? timeS() : kNan;}
  void setReviewIndex(int index);
  int decisionsShown() const {return decisions_shown_;}
  int64_t shownStampNs() const {return shown_stamp_ns_;}

  // Applied on the GUI thread. Public so tests can drive the model without a graph.
  void applyDecision(const pruning_sil_msgs::msg::SafetyDecision & decision, int64_t receive_ns);
  void applyDecisionJson(const std::string & json, int64_t receive_ns);
  void applyRange(int side, const RangeSummary & summary, int64_t stamp_ns, int64_t receive_ns);
  void applyFrame(const QImage & image, int64_t stamp_ns, int64_t receive_ns);
  void applyCycleMarker(const std::string & json, int64_t receive_ns);

Q_SIGNALS:
  void decisionChanged();
  void tofChanged();
  void delayChanged();
  void frameChanged();
  void holdChanged();
  void diagnosticsChanged();
  void reviewChanged();
  void simulationEnded();

private:
  void write(const std::string & line);
  void show(const DecisionRow & row);
  void noteLatency(int64_t stamp_ns);

  WristImageProvider * provider_;
  std::ofstream record_;
  QMutex record_mutex_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
  std::thread spinner_;
  rclcpp::Publisher<pruning_sil_msgs::msg::OperatorHold>::SharedPtr hold_publisher_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions_;
  QTimer diagnostics_timer_;

  DecisionLogModel log_;
  HoldFilterModel filter_;
  SeriesStore series_;
  TopicStats stats_;
  QVariantList topic_rows_;

  bool have_decision_{false};
  bool passed_{true};
  QString reason_;
  QString latched_reason_;
  QString cut_reasons_;
  int cycle_{0};
  int last_cycle_{-1};
  double mouth_distance_m_{kNan};
  bool operator_hold_seen_{false};
  double tof_m_[2]{kNan, kNan};
  QVariantList zones_[2];
  double delay_ms_{0.0};
  int frame_counter_{0};
  bool hold_requested_{false};
  int decisions_shown_{0};
  int64_t shown_stamp_ns_{-1};
  int64_t last_decision_receive_ns_{0};

  double stale_after_s_;
  double decision_age_s_{kNan};
  bool stale_{false};
  bool simulation_ended_{false};
  std::map<int64_t, int64_t> publish_wall_by_stamp_;
  std::map<int64_t, int64_t> receive_wall_by_stamp_;
  std::deque<double> latencies_ms_;
  double latency_p50_ms_{kNan};
  double latency_p95_ms_{kNan};
  int missed_decisions_{0};
  std::deque<int64_t> renders_;
  double render_fps_{0.0};

  bool review_mode_{false};
  int review_index_{-1};
  QString review_run_;
  QString review_dir_;
  std::vector<DecisionRow> review_rows_;
  std::vector<std::vector<double>> review_zones_[2];
  int review_fps_{10};
};

}  // namespace pruning_operator_panel

#endif  // PRUNING_OPERATOR_PANEL__PANEL_BACKEND_HPP_
