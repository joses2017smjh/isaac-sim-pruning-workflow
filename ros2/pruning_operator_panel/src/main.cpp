// Qt/QML operator panel for the live pruning loop.
//
//   operator_panel [--record FILE] [--screenshot-dir DIR] [--screenshot-every N]
//                  [--auto-hold-after-cycles N] [--exit-after-s S] [--stale-after-s S]
//                  [--size WxH] [--tab N] [--screenshot-all-tabs]
//                  [--review RUN_DIR] [--review-index N]
//
// Live mode subscribes to the ROS 2 graph. --review opens a recorded live-loop run
// instead (no graph): the gate's decision log, frames.json and the wrist images.
// --auto-hold-after-cycles synthesizes a mouse click on the QML Hold button after N
// gate decisions, so a headless run exercises the real button and its handler.
// Headless: QT_QPA_PLATFORM=offscreen QT_QUICK_BACKEND=software.

#include <QCommandLineParser>
#include <QDir>
#include <QEventLoop>
#include <QGuiApplication>
#include <QMouseEvent>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQmlProperty>
#include <QQuickItem>
#include <QQuickWindow>
#include <QTimer>

#include <memory>
#include <string>
#include <vector>

#include "pruning_operator_panel/panel_backend.hpp"
#include "pruning_operator_panel/panel_items.hpp"

using pruning_operator_panel::PanelBackend;
using pruning_operator_panel::WristImageProvider;
using pruning_operator_panel::wall_ns;

namespace
{

bool click(QQuickWindow * window, const char * object_name)
{
  auto * item = window->findChild<QQuickItem *>(object_name);
  if (!item || !item->isVisible() || !item->isEnabled()) {
    return false;
  }
  const QPointF centre = item->mapToScene(QPointF(item->width() / 2.0, item->height() / 2.0));
  const QPointF global = window->mapToGlobal(centre.toPoint());
  QMouseEvent press(QEvent::MouseButtonPress, centre, centre, global, Qt::LeftButton, Qt::LeftButton,
    Qt::NoModifier);
  QMouseEvent release(QEvent::MouseButtonRelease, centre, centre, global, Qt::LeftButton, Qt::NoButton,
    Qt::NoModifier);
  QGuiApplication::sendEvent(window, &press);
  QGuiApplication::sendEvent(window, &release);
  return true;
}

void settle(int ms)
{
  QEventLoop loop;
  QTimer::singleShot(ms, &loop, &QEventLoop::quit);
  loop.exec();
}

}  // namespace

int main(int argc, char ** argv)
{
  // ROS arguments go to rclcpp; the rest (including Qt's -platform) to Qt.
  std::vector<std::string> remaining = rclcpp::init_and_remove_ros_arguments(argc, argv);
  std::vector<char *> qt_argv;
  for (auto & argument : remaining) {
    qt_argv.push_back(argument.data());
  }
  int qt_argc = static_cast<int>(qt_argv.size());
  QGuiApplication app(qt_argc, qt_argv.data());
  app.setApplicationName("pruning_operator_panel");

  QCommandLineParser parser;
  parser.addHelpOption();
  const QCommandLineOption record(QStringLiteral("record"), "JSONL record of decisions, frames, repaints, holds.", "file");
  const QCommandLineOption shots(QStringLiteral("screenshot-dir"), "Save window screenshots here.", "dir");
  const QCommandLineOption every(QStringLiteral("screenshot-every"), "Screenshot every N decisions (0: off).", "n", "25");
  const QCommandLineOption hold(QStringLiteral("auto-hold-after-cycles"), "Click Hold after N decisions (0: never).", "n", "0");
  const QCommandLineOption limit(QStringLiteral("exit-after-s"), "Quit after S seconds (0: never).", "s", "0");
  const QCommandLineOption stale(QStringLiteral("stale-after-s"), "Warn when no decision for S seconds.", "s", "10");
  const QCommandLineOption size(QStringLiteral("size"), "Window size, e.g. 1280x760.", "WxH", "1280x760");
  const QCommandLineOption tab(QStringLiteral("tab"), "Tab to open: 0 live, 1 timeline, 2 diagnostics.", "n", "0");
  const QCommandLineOption all_tabs(QStringLiteral("screenshot-all-tabs"), "Screenshots of every tab, not only the open one.");
  const QCommandLineOption review(QStringLiteral("review"), "Open a recorded live-loop run instead of the graph.", "dir");
  const QCommandLineOption review_index(QStringLiteral("review-index"), "Decision to open in review (-1: last).", "n", "0");
  parser.addOptions({record, shots, every, hold, limit, stale, size, tab, all_tabs, review, review_index});
  parser.process(app);
  const QString shot_dir = parser.value(shots);
  const int shot_every = parser.value(every).toInt();
  const int hold_after = parser.value(hold).toInt();
  if (!shot_dir.isEmpty()) {
    QDir().mkpath(shot_dir);
  }

  pruning_operator_panel::registerPanelTypes();
  auto * provider = new WristImageProvider();  // owned by the engine
  PanelBackend backend(provider, parser.value(record).toStdString(), nullptr, parser.value(stale).toDouble());
  QQmlApplicationEngine engine;
  engine.addImageProvider(QStringLiteral("wrist"), provider);
  engine.rootContext()->setContextProperty(QStringLiteral("backend"), &backend);
  engine.load(QUrl(QStringLiteral("qrc:/qml/main.qml")));
  if (engine.rootObjects().isEmpty()) {
    return 2;
  }
  auto * window = qobject_cast<QQuickWindow *>(engine.rootObjects().first());
  if (!window) {
    return 2;
  }
  const QStringList dims = parser.value(size).split(QLatin1Char('x'));
  if (dims.size() == 2) {
    window->resize(dims[0].toInt(), dims[1].toInt());
  }
  QQmlProperty::write(window, QStringLiteral("currentTab"), parser.value(tab).toInt());

  const QStringList tab_names{QStringLiteral("live"), QStringLiteral("timeline"), QStringLiteral("diagnostics")};
  auto save = [&](const QString & name) {
      const QString path = shot_dir + "/" + name + ".png";
      const QImage shot = pruning_operator_panel::grab(window);
      if (shot.isNull() || !shot.save(path)) {
        backend.noteEvent("{\"event\":\"screenshot_failed\",\"wall_ns\":" + std::to_string(wall_ns()) +
          ",\"file\":\"" + name.toStdString() + ".png\"}");
        return;
      }
      backend.noteEvent("{\"event\":\"screenshot\",\"wall_ns\":" + std::to_string(wall_ns()) + ",\"file\":\"" +
        name.toStdString() + ".png\",\"shown_decision_stamp_ns\":" + std::to_string(backend.shownStampNs()) + "}");
    };
  auto grab = [&](const QString & name) {
      if (shot_dir.isEmpty()) {
        return;
      }
      if (!parser.isSet(all_tabs)) {
        save(name);
        return;
      }
      const int open = QQmlProperty::read(window, QStringLiteral("currentTab")).toInt();
      for (int i = 0; i < tab_names.size(); ++i) {
        QQmlProperty::write(window, QStringLiteral("currentTab"), i);
        settle(150);
        save(name + "_" + tab_names[i]);
      }
      QQmlProperty::write(window, QStringLiteral("currentTab"), open);
    };

  QObject::connect(window, &QQuickWindow::frameSwapped, &backend, [&backend]() {backend.noteRendered();});
  const double exit_after = parser.value(limit).toDouble();
  if (exit_after > 0) {
    QTimer::singleShot(static_cast<int>(exit_after * 1000), &app, &QGuiApplication::quit);
  }

  if (parser.isSet(review)) {
    if (!backend.loadReview(parser.value(review))) {
      return 3;
    }
    const int index = parser.value(review_index).toInt();
    backend.setReviewIndex(index < 0 ? backend.reviewCount() - 1 : index);
    if (!shot_dir.isEmpty()) {
      // Wait for the offscreen window to be exposed and painted, then capture.
      for (int i = 0; i < 100 && !window->isExposed(); ++i) {
        settle(20);
      }
      settle(300);
      grab(QStringLiteral("review_%1").arg(backend.reviewIndex(), 4, 10, QLatin1Char('0')));
    }
    const int status = exit_after > 0 || shot_dir.isEmpty() ? app.exec() : 0;
    rclcpp::shutdown();
    return status;
  }

  bool clicked = false;
  bool hold_shot = false;
  QObject::connect(&backend, &PanelBackend::decisionChanged, &app, [&]() {
      const int shown = backend.decisionsShown();
      if (shown == 1) {
        grab(QStringLiteral("panel_first_decision"));
      }
      if (shot_every > 0 && shown % shot_every == 0) {
        save(QStringLiteral("panel_cycle_%1").arg(backend.cycle(), 4, 10, QLatin1Char('0')));
      }
      if (hold_after > 0 && !clicked && shown >= hold_after) {
        clicked = true;
        backend.noteEvent("{\"event\":\"synthesized_click\",\"wall_ns\":" + std::to_string(wall_ns()) +
          ",\"after_decisions\":" + std::to_string(shown) + "}");
        // The button lives in the footer of every tab, so this works whichever tab is open.
        if (!click(window, "holdButton")) {
          backend.noteEvent("{\"event\":\"click_failed\",\"wall_ns\":" + std::to_string(wall_ns()) + "}");
        }
      }
      if (backend.operatorHoldSeen() && !hold_shot) {
        hold_shot = true;
        grab(QStringLiteral("panel_operator_hold"));
      }
    });
  QObject::connect(&backend, &PanelBackend::simulationEnded, &app, [&]() {
      grab(QStringLiteral("panel_final"));
      QTimer::singleShot(200, &app, &QGuiApplication::quit);
    });
  QTimer watchdog;
  QObject::connect(&watchdog, &QTimer::timeout, &app, [&]() {
      if (!rclcpp::ok()) {
        app.quit();  // SIGINT from the bring-up
      }
    });
  watchdog.start(200);

  backend.start();
  const int status = app.exec();
  backend.noteEvent("{\"event\":\"exit\",\"wall_ns\":" + std::to_string(wall_ns()) + ",\"decisions_shown\":" +
    std::to_string(backend.decisionsShown()) + "}");
  backend.stop();
  rclcpp::shutdown();
  return status;
}
