// The panel's models, items and QML screen, without a ROS graph.
//
// The screen tests load the real QML offscreen: they click the real Hold button with
// synthesized mouse events (as --auto-hold-after-cycles does in a live run), render
// every tab, and check the responsive layout at compact, medium and wide sizes.

#include <gtest/gtest.h>

#include <QEventLoop>
#include <QFile>
#include <QGuiApplication>
#include <QMouseEvent>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQmlProperty>
#include <QQuickItem>
#include <QQuickWindow>
#include <QTemporaryDir>
#include <QTimer>

#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "pruning_operator_panel/panel_backend.hpp"
#include "pruning_operator_panel/panel_items.hpp"

using pruning_operator_panel::DecisionLogModel;
using pruning_operator_panel::DecisionRow;
using pruning_operator_panel::HoldFilterModel;
using pruning_operator_panel::PanelBackend;
using pruning_operator_panel::RangeSummary;
using pruning_operator_panel::SeriesStore;
using pruning_operator_panel::TopicStats;
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
  static bool registered = false;
  if (!registered) {
    pruning_operator_panel::registerPanelTypes();
    registered = true;
  }
  return application;
}

std::string read(const std::string & path)
{
  std::ifstream stream(path);
  std::stringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

void settle(int ms)
{
  QEventLoop loop;
  QTimer::singleShot(ms, &loop, &QEventLoop::quit);
  loop.exec();
}

pruning_sil_msgs::msg::SafetyDecision decision(int sec, bool passed, const std::string & reason = "")
{
  pruning_sil_msgs::msg::SafetyDecision message;
  message.header.stamp.sec = sec;
  message.cycle = static_cast<uint32_t>(sec);
  message.passed = passed;
  message.reason = reason;
  message.mouth_distance_m = 0.02;
  return message;
}

/// The real screen, offscreen, at a given size.
struct Screen
{
  QTemporaryDir directory;
  std::string record;
  WristImageProvider * provider;
  std::unique_ptr<PanelBackend> backend;
  QQmlApplicationEngine engine;
  QQuickWindow * window{nullptr};

  explicit Screen(int width = 1280, int height = 760)
  {
    app();
    record = directory.filePath("record.jsonl").toStdString();
    provider = new WristImageProvider();
    backend = std::make_unique<PanelBackend>(provider, record);
    engine.addImageProvider(QStringLiteral("wrist"), provider);
    engine.rootContext()->setContextProperty(QStringLiteral("backend"), backend.get());
    engine.load(QUrl(QStringLiteral("qrc:/qml/main.qml")));
    if (engine.rootObjects().isEmpty()) {
      return;
    }
    window = qobject_cast<QQuickWindow *>(engine.rootObjects().first());
    window->resize(width, height);
    window->show();
    for (int i = 0; i < 100 && !window->isExposed(); ++i) {
      settle(20);
    }
    settle(150);
  }

  QQuickItem * item(const char * name) const {return window->findChild<QQuickItem *>(name);}

  QRectF rect(const char * name) const
  {
    auto * found = item(name);
    return found ? found->mapRectToScene(QRectF(0, 0, found->width(), found->height())) : QRectF();
  }
};

}  // namespace

// ----------------------------------------------------------------------------- helpers

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
  ASSERT_EQ(summary.zones.size(), 64u);
  EXPECT_TRUE(std::isnan(summary.zones[10]));
  EXPECT_NEAR(summary.zones[3], 0.25, 1e-6);
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

TEST(Percentile, InterpolatesAndHandlesEmpty)
{
  EXPECT_TRUE(std::isnan(pruning_operator_panel::percentile({}, 50)));
  EXPECT_DOUBLE_EQ(pruning_operator_panel::percentile({4, 1, 3, 2}, 50), 2.5);
  EXPECT_DOUBLE_EQ(pruning_operator_panel::percentile({1, 2, 3, 4, 5}, 100), 5.0);
}

// ----------------------------------------------------------------------------- models

TEST(DecisionLogModel, ExposesRolesCountsHoldsAndACap)
{
  DecisionLogModel model(nullptr, 3);
  for (int i = 0; i < 4; ++i) {
    DecisionRow row;
    row.stamp_ns = i * 100000000LL;
    row.cycle = i;
    row.passed = i != 2;
    row.reason = row.passed ? "" : "tof_minimum_clearance";
    row.tof_min_m = 0.12;
    model.append(row);
  }
  EXPECT_EQ(model.rowCount(), 3) << "the oldest row is dropped at the cap";
  EXPECT_EQ(model.holdCount(), 1);
  const auto index = model.index(1, 0);
  EXPECT_EQ(model.data(index, DecisionLogModel::CycleRole).toInt(), 2);
  EXPECT_FALSE(model.data(index, DecisionLogModel::PassedRole).toBool());
  EXPECT_NEAR(model.data(index, DecisionLogModel::TofMinCmRole).toDouble(), 12.0, 1e-9);
  EXPECT_EQ(model.roleNames().value(DecisionLogModel::ReasonRole), QByteArray("reason"));
  EXPECT_EQ(model.indexOfStamp(300000000LL), 2);
}

TEST(HoldFilterModel, ShowsOnlyHoldsWhenAsked)
{
  DecisionLogModel model;
  HoldFilterModel filter;
  filter.setSourceModel(&model);
  for (int i = 0; i < 5; ++i) {
    DecisionRow row;
    row.cycle = i;
    row.passed = i % 2 == 0;
    model.append(row);
  }
  EXPECT_EQ(filter.rowCount(), 5);
  filter.setHoldsOnly(true);
  EXPECT_EQ(filter.rowCount(), 2);
  EXPECT_EQ(filter.data(filter.index(0, 0), DecisionLogModel::CycleRole).toInt(), 1);
}

TEST(SeriesStore, KeepsTheNewestPointsPerChannel)
{
  SeriesStore store(nullptr, 3);
  for (int i = 0; i < 5; ++i) {
    store.append(QStringLiteral("a"), i, i * 10.0);
  }
  const auto points = store.points(QStringLiteral("a"));
  ASSERT_EQ(points.size(), 3);
  EXPECT_DOUBLE_EQ(points.front().x(), 2.0);
  EXPECT_DOUBLE_EQ(store.latestTime(), 4.0);
  EXPECT_TRUE(store.points(QStringLiteral("missing")).isEmpty());
}

TEST(TopicStats, RateCountAndAge)
{
  TopicStats stats;
  for (int i = 0; i < 11; ++i) {
    stats.note("/t", i * 100000000LL);  // 10 Hz for one second
  }
  EXPECT_NEAR(stats.rateHz("/t"), 10.0, 1e-9);
  EXPECT_EQ(stats.count("/t"), 11);
  const auto rows = stats.rows(1500000000LL);
  ASSERT_EQ(rows.size(), 1);
  EXPECT_NEAR(rows[0].toMap()["ageS"].toDouble(), 0.5, 1e-9);
}

TEST(PanelBackend, DecisionsDriveTheBannerAreRecordedAndCountMissedCycles)
{
  app();
  QTemporaryDir directory;
  const std::string record = directory.filePath("record.jsonl").toStdString();
  {
    PanelBackend backend(nullptr, record);
    EXPECT_FALSE(backend.haveDecision());
    backend.applyDecision(decision(3, true), 1);
    EXPECT_TRUE(backend.passed());
    backend.applyDecision(decision(5, false, "tof_minimum_clearance"), 2);  // cycle 4 never arrived
    EXPECT_FALSE(backend.passed());
    EXPECT_EQ(backend.reason().toStdString(), "tof_minimum_clearance");
    EXPECT_EQ(backend.decisionsShown(), 2);
    EXPECT_EQ(backend.holdCount(), 1);
    backend.refreshDiagnostics(3);
    EXPECT_EQ(backend.missedDecisions(), 1);
    backend.applyCycleMarker("{\"delay_ms\": 33.333333, \"index\": 4}", 3);
    EXPECT_NEAR(backend.delayMs(), 33.333333, 1e-9);
  }
  const std::string text = read(record);
  EXPECT_NE(text.find("\"pass_hold_changed\":true"), std::string::npos);
  EXPECT_NE(text.find("\"reason\":\"tof_minimum_clearance\""), std::string::npos);
}

TEST(PanelBackend, GoesStaleWhenTheGateFallsSilentButNotAfterTheRunEnds)
{
  app();
  PanelBackend backend(nullptr, "", nullptr, 2.0);
  backend.applyDecision(decision(1, true), 1000000000LL);
  backend.refreshDiagnostics(2500000000LL);
  EXPECT_FALSE(backend.stale());
  backend.refreshDiagnostics(3500000000LL);
  EXPECT_TRUE(backend.stale()) << "2.5 s without a decision, threshold 2 s";
  EXPECT_NEAR(backend.decisionAgeS(), 2.5, 1e-9);
  backend.applyCycleMarker("{\"end\": true}", 3600000000LL);
  backend.refreshDiagnostics(9000000000LL);
  EXPECT_FALSE(backend.stale()) << "a finished run is quiet, not stale";
}

TEST(PanelBackend, MeasuresGateToPanelLatencyFromTheJsonMirror)
{
  app();
  PanelBackend backend(nullptr, "");
  backend.applyDecisionJson("{\"stamp_ns\": 1000000000, \"publish_wall_ns\": 5000000}", 1);
  backend.applyDecision(decision(1, true), 7000000);  // received 2 ms after publish
  backend.refreshDiagnostics(8000000);
  EXPECT_NEAR(backend.latencyP50Ms(), 2.0, 1e-9);
}

TEST(PanelBackend, ReviewModeScrubsARecordedRun)
{
  app();
  QTemporaryDir run;
  {
    QFile log(run.filePath("ros2_loop_decisions.jsonl"));
    ASSERT_TRUE(log.open(QIODevice::WriteOnly));
    for (int k = 0; k < 4; ++k) {
      const bool passed = k < 3;
      log.write(QStringLiteral("{\"stamp_ns\": %1, \"cycle\": %2, \"passed\": %3, \"reason\": \"%4\", "
        "\"latched_reason\": \"%4\", \"mouth_distance_m\": 0.01, \"operator_hold\": %5, \"cut_reasons\": []}\n")
        .arg(k * 100000000LL).arg(k).arg(passed ? "true" : "false").arg(passed ? "" : "operator_hold")
        .arg(passed ? "false" : "true").toUtf8());
    }
    QFile frames(run.filePath("frames.json"));
    ASSERT_TRUE(frames.open(QIODevice::WriteOnly));
    QString grid = "[";
    QString valid = "[";
    for (int r = 0; r < 8; ++r) {
      grid += QString(r ? "," : "") + "[0.2,0.05,null,0.3,0.3,0.3,0.3,0.3]";
      valid += QString(r ? "," : "") + "[true,true,false,true,true,true,true,true]";
    }
    grid += "]";
    valid += "]";
    QString body = "{\"fps\": 10, \"frames\": [";
    for (int f = 0; f < 4; ++f) {
      body += QString(f ? "," : "") + "{\"index\": " + QString::number(f) + ", \"tof_left_m\": " + grid +
        ", \"tof_left_valid\": " + valid + ", \"tof_right_m\": " + grid + ", \"tof_right_valid\": " + valid + "}";
    }
    frames.write((body + "]}").toUtf8());
  }
  PanelBackend backend(nullptr, "");
  ASSERT_TRUE(backend.loadReview(run.path()));
  EXPECT_TRUE(backend.reviewMode());
  EXPECT_EQ(backend.reviewCount(), 4);
  EXPECT_EQ(backend.reviewIndex(), 0);
  backend.setReviewIndex(3);
  EXPECT_FALSE(backend.passed());
  EXPECT_EQ(backend.reason().toStdString(), "operator_hold");
  EXPECT_NEAR(backend.tofLeftM(), 0.05, 1e-9) << "cycle 3 shows frame 2's grid";
  EXPECT_EQ(backend.tofLeftZones().size(), 64);
  backend.seekReviewTime(0.11);
  EXPECT_EQ(backend.reviewIndex(), 1);
  backend.requestHold();
  EXPECT_FALSE(backend.holdRequested()) << "a recording cannot be held";
}

// ----------------------------------------------------------------------------- screen

TEST(PanelScreen, TheHoldButtonReachesTheBackend)
{
  Screen screen;
  ASSERT_NE(screen.window, nullptr);
  auto * button = screen.item("holdButton");
  ASSERT_NE(button, nullptr);
  const QPointF centre = button->mapToScene(QPointF(button->width() / 2.0, button->height() / 2.0));
  QMouseEvent press(QEvent::MouseButtonPress, centre, centre, screen.window->mapToGlobal(centre.toPoint()),
    Qt::LeftButton, Qt::LeftButton, Qt::NoModifier);
  QMouseEvent release(QEvent::MouseButtonRelease, centre, centre, screen.window->mapToGlobal(centre.toPoint()),
    Qt::LeftButton, Qt::NoButton, Qt::NoModifier);
  QGuiApplication::sendEvent(screen.window, &press);
  QGuiApplication::sendEvent(screen.window, &release);
  QGuiApplication::processEvents();
  EXPECT_TRUE(screen.backend->holdRequested()) << "the click did not reach PanelBackend::requestHold";
  EXPECT_NE(read(screen.record).find("\"event\":\"hold_pressed\""), std::string::npos);
  const QImage shot = pruning_operator_panel::grab(screen.window);
  ASSERT_FALSE(shot.isNull());
  EXPECT_EQ(shot.width(), 1280);
}

TEST(PanelScreen, RepaintsOffscreenAfterADecision)
{
  Screen screen;
  ASSERT_NE(screen.window, nullptr);
  int swaps = 0;
  QObject::connect(screen.window, &QQuickWindow::frameSwapped, [&]() {
      ++swaps;
      screen.backend->noteRendered();
    });
  auto hold = decision(7, false, "operator_hold");
  hold.operator_hold = true;
  screen.backend->applyDecision(hold, 1);
  settle(600);
  EXPECT_GT(swaps, 0) << "no repaint offscreen";
  EXPECT_TRUE(screen.backend->operatorHoldSeen());
  EXPECT_NE(read(screen.record).find("\"shown_decision_stamp_ns\":7000000000"), std::string::npos);
}

class ResponsiveLayout : public ::testing::TestWithParam<std::tuple<int, int, const char *>> {};

TEST_P(ResponsiveLayout, SafetyControlAndCallStayOnScreenAndApart)
{
  const auto [width, height, mode] = GetParam();
  Screen screen(width, height);
  ASSERT_NE(screen.window, nullptr);
  EXPECT_EQ(QQmlProperty::read(screen.window, "layoutMode").toString().toStdString(), mode);
  const QRectF window(0, 0, width, height);
  const QRectF hold = screen.rect("holdButton");
  const QRectF banner = screen.rect("banner");
  ASSERT_FALSE(hold.isEmpty());
  ASSERT_FALSE(banner.isEmpty());
  EXPECT_TRUE(window.contains(hold)) << "Hold must be fully visible without scrolling";
  EXPECT_GE(hold.height(), 64) << "Hold keeps its minimum touch target";
  EXPECT_TRUE(window.contains(banner)) << "the gate's call must be visible on open";
  EXPECT_FALSE(hold.intersects(banner));
  const QRectF camera = screen.rect("cameraView");
  EXPECT_LE(camera.right(), width + 0.5) << "nothing spills off the side";
  EXPECT_NEAR(camera.width() / camera.height(), 1.5, 0.02) << "the camera keeps its aspect ratio";
}

INSTANTIATE_TEST_SUITE_P(
  Sizes, ResponsiveLayout,
  ::testing::Values(
    std::make_tuple(1440, 860, "wide"), std::make_tuple(960, 720, "medium"),
    std::make_tuple(420, 860, "compact")));

TEST(PanelScreen, EveryTabRendersAndTheChartsPaint)
{
  Screen screen(1280, 760);
  ASSERT_NE(screen.window, nullptr);
  for (int k = 0; k < 40; ++k) {
    std::vector<float> zones(64, 0.25f - 0.004f * k);
    RangeSummary left = pruning_operator_panel::summarize_range_image(range_image(zones));
    screen.backend->applyRange(0, left, k * 100000000LL, k);
    screen.backend->applyRange(1, left, k * 100000000LL, k);
    screen.backend->applyDecision(decision(k, k < 30, k < 30 ? "" : "tof_minimum_clearance"), k);
  }
  for (int tab = 0; tab < 3; ++tab) {
    QQmlProperty::write(screen.window, "currentTab", tab);
    settle(200);
    const QImage shot = pruning_operator_panel::grab(screen.window);
    ASSERT_FALSE(shot.isNull()) << "tab " << tab;
  }
  QQmlProperty::write(screen.window, "currentTab", 1);
  settle(200);
  // The ToF chart must have drawn its series: count pixels in the left-sensor colour.
  const QImage shot = pruning_operator_panel::grab(screen.window);
  const QRectF chart = screen.rect("tofChart");
  ASSERT_FALSE(chart.isEmpty());
  int series_pixels = 0;
  for (int y = static_cast<int>(chart.top()); y < static_cast<int>(chart.bottom()); ++y) {
    for (int x = static_cast<int>(chart.left()); x < static_cast<int>(chart.right()); ++x) {
      const QColor c = shot.pixelColor(x * shot.width() / 1280, y * shot.height() / 760);
      series_pixels += std::abs(c.red() - 0x58) < 20 && std::abs(c.green() - 0xa6) < 20 && std::abs(c.blue() - 0xff) < 20;
    }
  }
  EXPECT_GT(series_pixels, 20) << "the left ToF series was not drawn";
  auto * list = screen.window->findChild<QObject *>("decisionList");
  ASSERT_NE(list, nullptr);
  EXPECT_EQ(QQmlProperty::read(list, "count").toInt(), 40);
  auto * holds_only = screen.window->findChild<QObject *>("holdsOnly");
  ASSERT_NE(holds_only, nullptr);
  qobject_cast<pruning_operator_panel::HoldFilterModel *>(screen.backend->decisions())->setHoldsOnly(true);
  settle(50);
  EXPECT_EQ(QQmlProperty::read(list, "count").toInt(), 10);
}
