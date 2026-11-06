// The panel's model and its QML screen, without a ROS graph.
//
// The QML test loads the real screen offscreen and clicks the real Hold button with
// synthesized mouse events, the same way --auto-hold-after-cycles does in a live run.

#include <gtest/gtest.h>

#include <QEventLoop>
#include <QGuiApplication>
#include <QMouseEvent>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickItem>
#include <QQuickWindow>
#include <QTemporaryDir>
#include <QTimer>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pruning_operator_panel/panel_backend.hpp"

using pruning_operator_panel::PanelBackend;
using pruning_operator_panel::RangeSummary;
using pruning_operator_panel::WristImageProvider;

namespace
{

sensor_msgs::msg::Image range_image(const std::vector<float> & values, uint32_t width = 8)
{
  sensor_msgs::msg::Image image;
  image.encoding = "32FC1";
  image.width = width;
  image.height = static_cast<uint32_t>(values.size() / width);
  image.step = width * 4;
  image.data.resize(values.size() * 4);
  std::memcpy(image.data.data(), values.data(), image.data.size());
  return image;
}

QGuiApplication & app()
{
  static int argc = 1;
  static char name[] = "test_panel_model";
  static char * argv[] = {name, nullptr};
  static QGuiApplication application(argc, argv);
  return application;
}

std::string read(const std::string & path)
{
  std::ifstream stream(path);
  std::stringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

}  // namespace

TEST(RangeSummary, NearestUsableZoneIgnoresNanZeroAndNegative)
{
  std::vector<float> values(64, NAN);
  values[3] = 0.25f;
  values[9] = 0.051f;
  values[10] = 0.0f;
  values[11] = -0.2f;
  values[12] = INFINITY;
  const RangeSummary summary = pruning_operator_panel::summarize_range_image(range_image(values));
  EXPECT_TRUE(summary.valid_message);
  EXPECT_EQ(summary.usable_zones, 2);
  EXPECT_NEAR(summary.min_m, 0.051, 1e-6);
}

TEST(RangeSummary, MalformedImagesShowNoReading)
{
  auto wrong_size = range_image(std::vector<float>(16, 0.3f), 4);
  EXPECT_FALSE(pruning_operator_panel::summarize_range_image(wrong_size).valid_message);
  auto wrong_encoding = range_image(std::vector<float>(64, 0.3f));
  wrong_encoding.encoding = "mono8";
  const auto summary = pruning_operator_panel::summarize_range_image(wrong_encoding);
  EXPECT_FALSE(summary.valid_message);
  EXPECT_TRUE(std::isnan(summary.min_m));
}

TEST(WristImage, RespectsRowPaddingAndRejectsShortPayloads)
{
  sensor_msgs::msg::Image image;
  image.encoding = "rgb8";
  image.width = 2;
  image.height = 2;
  image.step = 8;  // 6 bytes of pixels plus 2 of padding per row
  image.data = {255, 0, 0, 0, 255, 0, 9, 9, 0, 0, 255, 10, 20, 30, 9, 9};
  const QImage converted = pruning_operator_panel::to_qimage(image);
  ASSERT_FALSE(converted.isNull());
  EXPECT_EQ(converted.pixelColor(1, 0), QColor(0, 255, 0));
  EXPECT_EQ(converted.pixelColor(1, 1), QColor(10, 20, 30));
  image.data.resize(10);
  EXPECT_TRUE(pruning_operator_panel::to_qimage(image).isNull());
}

TEST(PanelBackend, DecisionsDriveTheBannerAndAreRecorded)
{
  app();
  QTemporaryDir directory;
  const std::string record = directory.filePath("record.jsonl").toStdString();
  {
    PanelBackend backend(nullptr, record);
    EXPECT_FALSE(backend.haveDecision());
    pruning_sil_msgs::msg::SafetyDecision decision;
    decision.header.stamp.sec = 3;
    decision.cycle = 30;
    decision.passed = true;
    backend.applyDecision(decision, 1);
    EXPECT_TRUE(backend.haveDecision());
    EXPECT_TRUE(backend.passed());
    decision.header.stamp.sec = 4;
    decision.passed = false;
    decision.reason = "tof_minimum_clearance";
    decision.latched_reason = "tof_minimum_clearance";
    backend.applyDecision(decision, 2);
    EXPECT_FALSE(backend.passed());
    EXPECT_EQ(backend.reason().toStdString(), "tof_minimum_clearance");
    EXPECT_EQ(backend.decisionsShown(), 2);
    backend.applyCycleMarker("{\"delay_ms\": 33.333333, \"index\": 4}", 3);
    EXPECT_NEAR(backend.delayMs(), 33.333333, 1e-9);
  }
  const std::string text = read(record);
  EXPECT_NE(text.find("\"pass_hold_changed\":true"), std::string::npos);
  EXPECT_NE(text.find("\"reason\":\"tof_minimum_clearance\""), std::string::npos);
}

TEST(PanelScreen, LoadsOffscreenAndTheHoldButtonReachesTheBackend)
{
  app();
  QTemporaryDir directory;
  const std::string record = directory.filePath("record.jsonl").toStdString();
  auto * provider = new WristImageProvider();
  PanelBackend backend(provider, record);
  QQmlApplicationEngine engine;
  engine.addImageProvider(QStringLiteral("wrist"), provider);
  engine.rootContext()->setContextProperty(QStringLiteral("backend"), &backend);
  engine.load(QUrl(QStringLiteral("qrc:/qml/main.qml")));
  ASSERT_FALSE(engine.rootObjects().isEmpty());
  auto * window = qobject_cast<QQuickWindow *>(engine.rootObjects().first());
  ASSERT_NE(window, nullptr);
  window->show();
  // The offscreen platform exposes the window asynchronously; grab and click only after that.
  for (int i = 0; i < 100 && !window->isExposed(); ++i) {
    QEventLoop wait;
    QTimer::singleShot(20, &wait, &QEventLoop::quit);
    wait.exec();
  }
  ASSERT_TRUE(window->isExposed());

  auto * button = window->findChild<QQuickItem *>("holdButton");
  ASSERT_NE(button, nullptr);
  const QPointF centre = button->mapToScene(QPointF(button->width() / 2.0, button->height() / 2.0));
  QMouseEvent press(QEvent::MouseButtonPress, centre, centre, window->mapToGlobal(centre.toPoint()),
    Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
  QMouseEvent release(QEvent::MouseButtonRelease, centre, centre, window->mapToGlobal(centre.toPoint()),
    Qt::LeftButton, Qt::NoButton, Qt::NoModifier);
  QGuiApplication::sendEvent(window, &press);
  QGuiApplication::sendEvent(window, &release);
  QGuiApplication::processEvents();
  EXPECT_TRUE(backend.holdRequested()) << "the click did not reach PanelBackend::requestHold";
  EXPECT_NE(read(record).find("\"event\":\"hold_pressed\""), std::string::npos);

  const QImage shot = pruning_operator_panel::grab(window);
  EXPECT_FALSE(shot.isNull());
  EXPECT_EQ(shot.width(), 1040);
}

TEST(PanelScreen, RepaintsOffscreenAfterADecision)
{
  // R5 counts repaints (frameSwapped); they must happen without a display.
  app();
  QTemporaryDir directory;
  const std::string record = directory.filePath("record.jsonl").toStdString();
  auto * provider = new WristImageProvider();
  PanelBackend backend(provider, record);
  QQmlApplicationEngine engine;
  engine.addImageProvider(QStringLiteral("wrist"), provider);
  engine.rootContext()->setContextProperty(QStringLiteral("backend"), &backend);
  engine.load(QUrl(QStringLiteral("qrc:/qml/main.qml")));
  ASSERT_FALSE(engine.rootObjects().isEmpty());
  auto * window = qobject_cast<QQuickWindow *>(engine.rootObjects().first());
  ASSERT_NE(window, nullptr);
  int swaps = 0;
  QObject::connect(window, &QQuickWindow::frameSwapped, [&]() {
      ++swaps;
      backend.noteRendered();
    });
  window->show();
  pruning_sil_msgs::msg::SafetyDecision decision;
  decision.header.stamp.sec = 7;
  decision.passed = false;
  decision.reason = "operator_hold";
  decision.operator_hold = true;
  backend.applyDecision(decision, 1);
  QEventLoop loop;
  QTimer::singleShot(600, &loop, &QEventLoop::quit);
  loop.exec();
  EXPECT_GT(swaps, 0) << "no repaint offscreen";
  EXPECT_TRUE(backend.operatorHoldSeen());
  EXPECT_NE(read(record).find("\"event\":\"rendered\",\"wall_ns\""), std::string::npos);
  EXPECT_NE(read(record).find("\"shown_decision_stamp_ns\":7000000000"), std::string::npos);
}
