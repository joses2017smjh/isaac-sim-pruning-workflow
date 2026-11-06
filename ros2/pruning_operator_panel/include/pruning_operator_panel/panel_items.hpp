// Models and custom Qt Quick items behind the operator panel.
//
//   DecisionLogModel   every gate decision, as a QAbstractListModel for QML views
//   HoldFilterModel    a proxy that can show only holds
//   SeriesStore        rolling time series (ToF per sensor, mouth distance, holds)
//   TimeSeriesItem     a QQuickPaintedItem chart of SeriesStore channels with a limit line
//   ZoneGridItem       a QQuickPaintedItem 8x8 heatmap of one ToF sensor's zones
//   TopicStats         per-topic message counts, rates and ages
//
// Painted with QPainter, so they render the same under the OpenGL and the software
// Qt Quick backends (the batch jobs run offscreen on the software backend).

#ifndef PRUNING_OPERATOR_PANEL__PANEL_ITEMS_HPP_
#define PRUNING_OPERATOR_PANEL__PANEL_ITEMS_HPP_

#include <QAbstractListModel>
#include <QColor>
#include <QHash>
#include <QPointF>
#include <QQuickPaintedItem>
#include <QSortFilterProxyModel>
#include <QString>
#include <QStringList>
#include <QVariantList>
#include <QVector>

#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <string>

namespace pruning_operator_panel
{

constexpr double kNan = std::numeric_limits<double>::quiet_NaN();

struct DecisionRow
{
  int64_t stamp_ns{0};
  int cycle{0};
  bool passed{true};
  QString reason;
  QString latched_reason;
  QString tof_reason;
  double tof_min_m{kNan};
  double mouth_distance_m{kNan};
  bool operator_hold{false};
  bool closure_authorized{false};
  QStringList cut_reasons;
};

class DecisionLogModel : public QAbstractListModel
{
  Q_OBJECT
  Q_PROPERTY(int count READ count NOTIFY countChanged)
  Q_PROPERTY(int holdCount READ holdCount NOTIFY countChanged)

public:
  enum Role
  {
    CycleRole = Qt::UserRole + 1,
    TimeRole,
    PassedRole,
    ReasonRole,
    LatchedRole,
    TofMinCmRole,
    MouthMmRole,
    OperatorHoldRole,
    ClosureRole,
    CutReasonsRole,
  };

  explicit DecisionLogModel(QObject * parent = nullptr, int capacity = 5000);
  int rowCount(const QModelIndex & parent = QModelIndex()) const override;
  QVariant data(const QModelIndex & index, int role) const override;
  QHash<int, QByteArray> roleNames() const override;

  void append(const DecisionRow & row);
  void clear();
  int count() const {return static_cast<int>(rows_.size());}
  int holdCount() const {return hold_count_;}
  const DecisionRow & row(int index) const {return rows_.at(index);}
  /// Row index of the decision with this stamp, or -1.
  int indexOfStamp(int64_t stamp_ns) const;

Q_SIGNALS:
  void countChanged();

private:
  std::deque<DecisionRow> rows_;
  int capacity_;
  int hold_count_{0};
};

class HoldFilterModel : public QSortFilterProxyModel
{
  Q_OBJECT
  Q_PROPERTY(bool holdsOnly READ holdsOnly WRITE setHoldsOnly NOTIFY holdsOnlyChanged)

public:
  explicit HoldFilterModel(QObject * parent = nullptr);
  bool holdsOnly() const {return holds_only_;}
  void setHoldsOnly(bool value);

Q_SIGNALS:
  void holdsOnlyChanged();

protected:
  bool filterAcceptsRow(int source_row, const QModelIndex & source_parent) const override;

private:
  bool holds_only_{false};
};

/// Named channels of (time s, value) points; capped so a long run stays light.
class SeriesStore : public QObject
{
  Q_OBJECT

public:
  explicit SeriesStore(QObject * parent = nullptr, int capacity = 6000);
  void append(const QString & channel, double t, double value);
  QVector<QPointF> points(const QString & channel) const;
  double latestTime() const {return latest_;}
  void clear();

Q_SIGNALS:
  void changed();

private:
  std::map<QString, std::deque<QPointF>> channels_;
  int capacity_;
  double latest_{0.0};
};

class TimeSeriesItem : public QQuickPaintedItem
{
  Q_OBJECT
  Q_PROPERTY(QObject * store READ store WRITE setStore NOTIFY storeChanged)
  Q_PROPERTY(QStringList channels MEMBER channels_ NOTIFY appearanceChanged)
  Q_PROPERTY(QStringList labels MEMBER labels_ NOTIFY appearanceChanged)
  Q_PROPERTY(QVariantList colors MEMBER colors_ NOTIFY appearanceChanged)
  Q_PROPERTY(QString holdChannel MEMBER hold_channel_ NOTIFY appearanceChanged)
  Q_PROPERTY(double yMin MEMBER y_min_ NOTIFY appearanceChanged)
  Q_PROPERTY(double yMax MEMBER y_max_ NOTIFY appearanceChanged)
  Q_PROPERTY(double limit MEMBER limit_ NOTIFY appearanceChanged)
  Q_PROPERTY(QString limitLabel MEMBER limit_label_ NOTIFY appearanceChanged)
  Q_PROPERTY(QString unit MEMBER unit_ NOTIFY appearanceChanged)
  Q_PROPERTY(QString title MEMBER title_ NOTIFY appearanceChanged)
  Q_PROPERTY(double windowS MEMBER window_s_ NOTIFY appearanceChanged)
  Q_PROPERTY(double cursorS MEMBER cursor_s_ NOTIFY appearanceChanged)

public:
  explicit TimeSeriesItem(QQuickItem * parent = nullptr);
  void paint(QPainter * painter) override;
  QObject * store() const {return store_;}
  void setStore(QObject * store);

Q_SIGNALS:
  void storeChanged();
  void appearanceChanged();

private:
  SeriesStore * store_{nullptr};
  QStringList channels_;
  QStringList labels_;
  QVariantList colors_;
  QString hold_channel_;
  double y_min_{0.0};
  double y_max_{1.0};
  double limit_{kNan};
  QString limit_label_;
  QString unit_;
  QString title_;
  double window_s_{0.0};
  double cursor_s_{kNan};
};

class ZoneGridItem : public QQuickPaintedItem
{
  Q_OBJECT
  Q_PROPERTY(QVariantList zones MEMBER zones_ NOTIFY changed)
  Q_PROPERTY(double clearance MEMBER clearance_ NOTIFY changed)
  Q_PROPERTY(double fullScale MEMBER full_scale_ NOTIFY changed)
  Q_PROPERTY(QString title MEMBER title_ NOTIFY changed)

public:
  explicit ZoneGridItem(QQuickItem * parent = nullptr);
  void paint(QPainter * painter) override;
  /// The fill colour of one zone: warm when near, cool when far, dark when unusable.
  static QColor zoneColor(double range_m, double full_scale_m);

Q_SIGNALS:
  void changed();

private:
  QVariantList zones_;
  double clearance_{0.06};
  double full_scale_{0.5};
  QString title_;
};

/// Per-topic message counts, rates over the last messages, and time since the last one.
class TopicStats
{
public:
  void note(const std::string & topic, int64_t wall_ns);
  /// Rows of {topic, count, hz, ageS} for QML, at time `now_ns`.
  QVariantList rows(int64_t now_ns) const;
  double rateHz(const std::string & topic) const;
  int count(const std::string & topic) const;

private:
  struct Entry
  {
    int count{0};
    std::deque<int64_t> recent;
  };
  std::map<std::string, Entry> topics_;
};

/// Register TimeSeries and ZoneGrid as QML types in the Pruning.Panel 1.0 module.
void registerPanelTypes();

}  // namespace pruning_operator_panel

#endif  // PRUNING_OPERATOR_PANEL__PANEL_ITEMS_HPP_
