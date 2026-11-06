// C++ backend of the Qt/QML operator panel.
//
// An rclcpp node, spun on its own thread, subscribes to the wrist camera, both
// time-of-flight range images, the safety gate's decisions and the simulator's
// cycle markers. Each message is handed to the GUI thread (queued), where it
// updates the properties the QML screen binds to. requestHold() publishes a
// pruning_sil_msgs/OperatorHold, which the C++ safety gate latches for the run.
//
// Every received decision and camera frame, every repaint and every hold press is
// written to a JSONL record with wall-clock times, so "the panel kept up" and
// "Hold stopped the arm within one cycle" are measured, not asserted.

#ifndef PRUNING_OPERATOR_PANEL__PANEL_BACKEND_HPP_
#define PRUNING_OPERATOR_PANEL__PANEL_BACKEND_HPP_

#include <QImage>
#include <QMutex>
#include <QObject>
#include <QQuickImageProvider>
#include <QQuickWindow>
#include <QString>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <thread>

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
};

RangeSummary summarize_range_image(const sensor_msgs::msg::Image & image);

/// Deep-copied RGB888 image of an rgb8 message; a null image when it is malformed.
QImage to_qimage(const sensor_msgs::msg::Image & image);

int64_t wall_ns();

/// The window's pixels. QQuickWindow::grabWindow() returns null under the software
/// backend on the offscreen platform (its backing store has no toImage()), so this
/// falls back to the screen's copy of the same backing store. Null if not exposed.
QImage grab(QQuickWindow * window);

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
  Q_PROPERTY(double mouthDistanceM READ mouthDistanceM NOTIFY decisionChanged)
  Q_PROPERTY(bool operatorHoldSeen READ operatorHoldSeen NOTIFY decisionChanged)
  Q_PROPERTY(double tofLeftM READ tofLeftM NOTIFY tofChanged)
  Q_PROPERTY(double tofRightM READ tofRightM NOTIFY tofChanged)
  Q_PROPERTY(double clearanceM READ clearanceM CONSTANT)
  Q_PROPERTY(double delayMs READ delayMs NOTIFY delayChanged)
  Q_PROPERTY(int frameCounter READ frameCounter NOTIFY frameChanged)
  Q_PROPERTY(bool holdRequested READ holdRequested NOTIFY holdChanged)

public:
  PanelBackend(WristImageProvider * provider, const std::string & record_path, QObject * parent = nullptr);
  ~PanelBackend() override;

  /// Create the node and spin it on a background thread.
  void start();
  void stop();

  /// Publish an operator hold. Bound to the QML Hold button.
  Q_INVOKABLE void requestHold(const QString & source = QStringLiteral("button"));

  /// Called after each repaint, with the decision stamp the screen was showing.
  void noteRendered();
  void noteEvent(const std::string & json_object);

  bool haveDecision() const {return have_decision_;}
  bool passed() const {return passed_;}
  QString reason() const {return reason_;}
  QString latchedReason() const {return latched_reason_;}
  int cycle() const {return cycle_;}
  double mouthDistanceM() const {return mouth_distance_m_;}
  bool operatorHoldSeen() const {return operator_hold_seen_;}
  double tofLeftM() const {return tof_m_[0];}
  double tofRightM() const {return tof_m_[1];}
  double clearanceM() const {return 0.06;}
  double delayMs() const {return delay_ms_;}
  int frameCounter() const {return frame_counter_;}
  bool holdRequested() const {return hold_requested_;}
  int decisionsShown() const {return decisions_shown_;}
  int64_t shownStampNs() const {return shown_stamp_ns_;}

  // Applied on the GUI thread. Public so tests can drive the model without a graph.
  void applyDecision(const pruning_sil_msgs::msg::SafetyDecision & decision, int64_t receive_ns);
  void applyRange(int side, const RangeSummary & summary);
  void applyFrame(const QImage & image, int64_t stamp_ns, int64_t receive_ns);
  void applyCycleMarker(const std::string & json, int64_t receive_ns);

Q_SIGNALS:
  void decisionChanged();
  void tofChanged();
  void delayChanged();
  void frameChanged();
  void holdChanged();
  void simulationEnded();

private:
  void write(const std::string & line);

  WristImageProvider * provider_;
  std::ofstream record_;
  QMutex record_mutex_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::executors::SingleThreadedExecutor::SharedPtr executor_;
  std::thread spinner_;
  rclcpp::Publisher<pruning_sil_msgs::msg::OperatorHold>::SharedPtr hold_publisher_;
  std::vector<rclcpp::SubscriptionBase::SharedPtr> subscriptions_;

  bool have_decision_{false};
  bool passed_{true};
  QString reason_;
  QString latched_reason_;
  int cycle_{0};
  double mouth_distance_m_{std::numeric_limits<double>::quiet_NaN()};
  bool operator_hold_seen_{false};
  double tof_m_[2]{std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
  double delay_ms_{0.0};
  int frame_counter_{0};
  bool hold_requested_{false};
  int decisions_shown_{0};
  int64_t shown_stamp_ns_{-1};
};

}  // namespace pruning_operator_panel

#endif  // PRUNING_OPERATOR_PANEL__PANEL_BACKEND_HPP_
