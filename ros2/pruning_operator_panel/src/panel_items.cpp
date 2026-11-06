#include "pruning_operator_panel/panel_items.hpp"

#include <QFont>
#include <QFontMetrics>
#include <QPainter>
#include <QPainterPath>
#include <QtQml>

#include <algorithm>
#include <cmath>

namespace pruning_operator_panel
{

// ----------------------------------------------------------------------------- decisions

DecisionLogModel::DecisionLogModel(QObject * parent, int capacity)
: QAbstractListModel(parent), capacity_(capacity) {}

int DecisionLogModel::rowCount(const QModelIndex & parent) const
{
  return parent.isValid() ? 0 : static_cast<int>(rows_.size());
}

QVariant DecisionLogModel::data(const QModelIndex & index, int role) const
{
  if (!index.isValid() || index.row() < 0 || index.row() >= count()) {
    return {};
  }
  const DecisionRow & row = rows_[index.row()];
  switch (role) {
    case CycleRole: return row.cycle;
    case TimeRole: return static_cast<double>(row.stamp_ns) * 1e-9;
    case PassedRole: return row.passed;
    case ReasonRole: return row.reason;
    case LatchedRole: return row.latched_reason;
    case TofMinCmRole: return row.tof_min_m * 100.0;
    case MouthMmRole: return row.mouth_distance_m * 1000.0;
    case OperatorHoldRole: return row.operator_hold;
    case ClosureRole: return row.closure_authorized;
    case CutReasonsRole: return row.cut_reasons.join(QStringLiteral(", "));
    default: return {};
  }
}

QHash<int, QByteArray> DecisionLogModel::roleNames() const
{
  return {
    {CycleRole, "cycle"}, {TimeRole, "timeS"}, {PassedRole, "passed"}, {ReasonRole, "reason"},
    {LatchedRole, "latchedReason"}, {TofMinCmRole, "tofMinCm"}, {MouthMmRole, "mouthMm"},
    {OperatorHoldRole, "operatorHold"}, {ClosureRole, "closureAuthorized"}, {CutReasonsRole, "cutReasons"},
  };
}

void DecisionLogModel::append(const DecisionRow & row)
{
  if (count() >= capacity_) {
    beginRemoveRows(QModelIndex(), 0, 0);
    hold_count_ -= rows_.front().passed ? 0 : 1;
    rows_.pop_front();
    endRemoveRows();
  }
  beginInsertRows(QModelIndex(), count(), count());
  rows_.push_back(row);
  hold_count_ += row.passed ? 0 : 1;
  endInsertRows();
  Q_EMIT countChanged();
}

void DecisionLogModel::clear()
{
  beginResetModel();
  rows_.clear();
  hold_count_ = 0;
  endResetModel();
  Q_EMIT countChanged();
}

int DecisionLogModel::indexOfStamp(int64_t stamp_ns) const
{
  for (int i = count() - 1; i >= 0; --i) {
    if (rows_[i].stamp_ns == stamp_ns) {
      return i;
    }
  }
  return -1;
}

HoldFilterModel::HoldFilterModel(QObject * parent)
: QSortFilterProxyModel(parent) {}

void HoldFilterModel::setHoldsOnly(bool value)
{
  if (value == holds_only_) {
    return;
  }
  holds_only_ = value;
  invalidateFilter();
  Q_EMIT holdsOnlyChanged();
}

bool HoldFilterModel::filterAcceptsRow(int source_row, const QModelIndex & source_parent) const
{
  if (!holds_only_) {
    return true;
  }
  const QModelIndex index = sourceModel()->index(source_row, 0, source_parent);
  return !sourceModel()->data(index, DecisionLogModel::PassedRole).toBool();
}

// ----------------------------------------------------------------------------- series

SeriesStore::SeriesStore(QObject * parent, int capacity)
: QObject(parent), capacity_(capacity) {}

void SeriesStore::append(const QString & channel, double t, double value)
{
  auto & points = channels_[channel];
  points.emplace_back(t, value);
  while (static_cast<int>(points.size()) > capacity_) {
    points.pop_front();
  }
  latest_ = std::max(latest_, t);
  Q_EMIT changed();
}

QVector<QPointF> SeriesStore::points(const QString & channel) const
{
  const auto it = channels_.find(channel);
  if (it == channels_.end()) {
    return {};
  }
  return QVector<QPointF>(it->second.begin(), it->second.end());
}

void SeriesStore::clear()
{
  channels_.clear();
  latest_ = 0.0;
  Q_EMIT changed();
}

// ----------------------------------------------------------------------------- chart

TimeSeriesItem::TimeSeriesItem(QQuickItem * parent)
: QQuickPaintedItem(parent)
{
  setAntialiasing(true);
  connect(this, &TimeSeriesItem::appearanceChanged, this, [this]() {update();});
}

void TimeSeriesItem::setStore(QObject * store)
{
  auto * typed = qobject_cast<SeriesStore *>(store);
  if (typed == store_) {
    return;
  }
  if (store_) {
    disconnect(store_, nullptr, this, nullptr);
  }
  store_ = typed;
  if (store_) {
    connect(store_, &SeriesStore::changed, this, [this]() {update();});
  }
  Q_EMIT storeChanged();
  update();
}

void TimeSeriesItem::paint(QPainter * painter)
{
  const QRectF bounds = boundingRect();
  painter->fillRect(bounds, QColor("#1b2027"));
  QFont font = painter->font();
  font.setPixelSize(12);
  painter->setFont(font);
  const QFontMetrics metrics(font);
  const double left = metrics.horizontalAdvance(QStringLiteral("0000.0")) + 10;
  const QRectF plot(left, 26, std::max(10.0, bounds.width() - left - 12), std::max(10.0, bounds.height() - 48));

  painter->setPen(QColor("#c9d1d9"));
  painter->drawText(QPointF(8, 17), title_);

  double t_end = store_ ? store_->latestTime() : 0.0;
  double t_start = window_s_ > 0 ? t_end - window_s_ : 0.0;
  if (store_ && window_s_ > 0) {
    // A window longer than the run so far starts at the first point, not before it.
    double first = std::numeric_limits<double>::infinity();
    for (const auto & channel : channels_) {
      const auto points = store_->points(channel);
      if (!points.isEmpty()) {
        first = std::min(first, points.front().x());
      }
    }
    if (std::isfinite(first)) {
      t_start = std::max(t_start, first);
    }
  }
  if (store_ && window_s_ <= 0) {
    t_start = std::numeric_limits<double>::infinity();
    for (const auto & channel : channels_) {
      const auto points = store_->points(channel);
      if (!points.isEmpty()) {
        t_start = std::min(t_start, points.front().x());
      }
    }
    if (!std::isfinite(t_start)) {
      t_start = 0.0;
    }
  }
  if (t_end <= t_start) {
    t_end = t_start + 1.0;
  }
  const double span = y_max_ - y_min_ > 0 ? y_max_ - y_min_ : 1.0;
  auto x_of = [&](double t) {return plot.left() + (t - t_start) / (t_end - t_start) * plot.width();};
  auto y_of = [&](double v) {
      const double clamped = std::clamp(v, y_min_, y_max_);
      return plot.bottom() - (clamped - y_min_) / span * plot.height();
    };

  // Grid and axis labels.
  painter->setPen(QPen(QColor("#2d343e"), 1));
  for (int i = 0; i <= 4; ++i) {
    const double v = y_min_ + span * i / 4.0;
    const double y = y_of(v);
    painter->drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));
    painter->setPen(QColor("#8b949e"));
    painter->drawText(QRectF(0, y - 8, left - 6, 16), Qt::AlignRight | Qt::AlignVCenter,
      QString::number(v, 'f', span < 10 ? 1 : 0));
    painter->setPen(QPen(QColor("#2d343e"), 1));
  }
  painter->setPen(QColor("#8b949e"));
  painter->drawText(QRectF(plot.left(), plot.bottom() + 4, plot.width(), 16), Qt::AlignLeft,
    QStringLiteral("%1 s").arg(t_start, 0, 'f', 1));
  painter->drawText(QRectF(plot.left(), plot.bottom() + 4, plot.width(), 16), Qt::AlignRight,
    QStringLiteral("%1 s   %2").arg(t_end, 0, 'f', 1).arg(unit_));

  // Hold periods shaded across the plot.
  if (store_ && !hold_channel_.isEmpty()) {
    const auto holds = store_->points(hold_channel_);
    for (int i = 0; i < holds.size(); ++i) {
      if (holds[i].y() < 0.5 || holds[i].x() < t_start) {
        continue;
      }
      const double next = i + 1 < holds.size() ? holds[i + 1].x() : t_end;
      painter->fillRect(QRectF(QPointF(x_of(holds[i].x()), plot.top()), QPointF(x_of(next), plot.bottom())),
        QColor(211, 47, 47, 60));
    }
  }

  // The limit the gate enforces.
  if (std::isfinite(limit_) && limit_ >= y_min_ && limit_ <= y_max_) {
    painter->setPen(QPen(QColor("#ffcc00"), 1.5, Qt::DashLine));
    const double y = y_of(limit_);
    painter->drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y));
    painter->drawText(QPointF(plot.left() + 6, y - 4), limit_label_);
  }

  // Channels, with gaps where a value is missing.
  double legend_x = plot.right();
  for (int c = channels_.size() - 1; c >= 0; --c) {
    const QColor color = c < colors_.size() ? colors_[c].value<QColor>() : QColor("#58a6ff");
    if (store_) {
      QPainterPath path;
      bool open = false;
      for (const auto & point : store_->points(channels_[c])) {
        if (point.x() < t_start) {
          continue;
        }
        if (!std::isfinite(point.y())) {
          open = false;
          continue;
        }
        const QPointF p(x_of(point.x()), y_of(point.y()));
        if (open) {
          path.lineTo(p);
        } else {
          path.moveTo(p);
          open = true;
        }
      }
      painter->setPen(QPen(color, 2));
      painter->drawPath(path);
    }
    const QString label = c < labels_.size() ? labels_[c] : channels_[c];
    legend_x -= metrics.horizontalAdvance(label) + 26;
    painter->fillRect(QRectF(legend_x, 8, 12, 4), color);
    painter->setPen(QColor("#c9d1d9"));
    painter->drawText(QPointF(legend_x + 16, 15), label);
  }

  // The review cursor.
  if (std::isfinite(cursor_s_) && cursor_s_ >= t_start && cursor_s_ <= t_end) {
    painter->setPen(QPen(QColor("#ffffff"), 1));
    const double x = x_of(cursor_s_);
    painter->drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()));
  }
  painter->setPen(QPen(QColor("#3a414b"), 1));
  painter->drawRect(plot);
}

// ----------------------------------------------------------------------------- zone grid

ZoneGridItem::ZoneGridItem(QQuickItem * parent)
: QQuickPaintedItem(parent)
{
  setAntialiasing(true);
  connect(this, &ZoneGridItem::changed, this, [this]() {update();});
}

QColor ZoneGridItem::zoneColor(double range_m, double full_scale_m)
{
  if (!std::isfinite(range_m) || range_m <= 0.0) {
    return QColor("#2a2f37");
  }
  // Near is warm, far is cool: 0 -> red, mid -> amber, far -> teal.
  const double t = std::clamp(range_m / std::max(1e-6, full_scale_m), 0.0, 1.0);
  const QColor near("#e5533d");
  const QColor mid("#f2b134");
  const QColor far("#1f8a8a");
  auto mix = [](const QColor & a, const QColor & b, double u) {
      return QColor::fromRgbF(a.redF() + (b.redF() - a.redF()) * u, a.greenF() + (b.greenF() - a.greenF()) * u,
             a.blueF() + (b.blueF() - a.blueF()) * u);
    };
  return t < 0.5 ? mix(near, mid, t * 2.0) : mix(mid, far, (t - 0.5) * 2.0);
}

void ZoneGridItem::paint(QPainter * painter)
{
  const QRectF bounds = boundingRect();
  QFont font = painter->font();
  font.setPixelSize(12);
  painter->setFont(font);
  const double title_h = title_.isEmpty() ? 0 : 18;
  painter->setPen(QColor("#c9d1d9"));
  if (!title_.isEmpty()) {
    painter->drawText(QPointF(0, 13), title_);
  }
  const double side = std::max(8.0, std::min(bounds.width(), bounds.height() - title_h));
  const double cell = side / 8.0;
  const QPointF origin((bounds.width() - side) / 2.0, title_h);
  int nearest = -1;
  double nearest_m = std::numeric_limits<double>::infinity();
  for (int i = 0; i < 64 && i < zones_.size(); ++i) {
    const double v = zones_[i].toDouble();
    if (std::isfinite(v) && v > 0.0 && v < nearest_m) {
      nearest_m = v;
      nearest = i;
    }
  }
  font.setPixelSize(std::max(7, static_cast<int>(cell * 0.28)));
  painter->setFont(font);
  for (int i = 0; i < 64; ++i) {
    const double v = i < zones_.size() ? zones_[i].toDouble() : kNan;
    const QRectF rect(origin.x() + (i % 8) * cell, origin.y() + (i / 8) * cell, cell - 2, cell - 2);
    painter->fillRect(rect, zoneColor(v, full_scale_));
    const bool usable = std::isfinite(v) && v > 0.0;
    if (!usable) {
      painter->setPen(QPen(QColor("#3a414b"), 1));
      painter->drawLine(rect.topLeft(), rect.bottomRight());
      continue;
    }
    if (v < clearance_) {
      painter->setPen(QPen(QColor("#ff1744"), 3));
      painter->drawRect(rect.adjusted(1.5, 1.5, -1.5, -1.5));
    } else if (i == nearest) {
      painter->setPen(QPen(QColor("#ffffff"), 2));
      painter->drawRect(rect.adjusted(1, 1, -1, -1));
    }
    if (cell >= 22) {
      painter->setPen(QColor("#0d1117"));
      painter->drawText(rect, Qt::AlignCenter, QString::number(v * 100.0, 'f', 0));
    }
  }
}

// ----------------------------------------------------------------------------- topic stats

void TopicStats::note(const std::string & topic, int64_t wall_ns)
{
  auto & entry = topics_[topic];
  ++entry.count;
  entry.recent.push_back(wall_ns);
  while (entry.recent.size() > 50) {
    entry.recent.pop_front();
  }
}

double TopicStats::rateHz(const std::string & topic) const
{
  const auto it = topics_.find(topic);
  if (it == topics_.end() || it->second.recent.size() < 2) {
    return 0.0;
  }
  const auto & recent = it->second.recent;
  const double span = static_cast<double>(recent.back() - recent.front()) * 1e-9;
  return span > 0 ? static_cast<double>(recent.size() - 1) / span : 0.0;
}

int TopicStats::count(const std::string & topic) const
{
  const auto it = topics_.find(topic);
  return it == topics_.end() ? 0 : it->second.count;
}

QVariantList TopicStats::rows(int64_t now_ns) const
{
  QVariantList out;
  for (const auto & [topic, entry] : topics_) {
    QVariantMap row;
    row[QStringLiteral("topic")] = QString::fromStdString(topic);
    row[QStringLiteral("count")] = entry.count;
    row[QStringLiteral("hz")] = rateHz(topic);
    row[QStringLiteral("ageS")] =
      entry.recent.empty() ? kNan : static_cast<double>(now_ns - entry.recent.back()) * 1e-9;
    out.push_back(row);
  }
  return out;
}

void registerPanelTypes()
{
  qmlRegisterType<TimeSeriesItem>("Pruning.Panel", 1, 0, "TimeSeries");
  qmlRegisterType<ZoneGridItem>("Pruning.Panel", 1, 0, "ZoneGrid");
}

}  // namespace pruning_operator_panel
