// Pure helpers of the panel: no node, no window, so they test without a graph.

#include <QMutexLocker>
#include <QPixmap>
#include <QScreen>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

#include "pruning_operator_panel/panel_backend.hpp"

namespace pruning_operator_panel
{

int64_t wall_ns()
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::system_clock::now().time_since_epoch()).count();
}

double percentile(std::vector<double> values, double p)
{
  if (values.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::sort(values.begin(), values.end());
  const double rank = std::clamp(p, 0.0, 100.0) / 100.0 * static_cast<double>(values.size() - 1);
  const auto low = static_cast<std::size_t>(std::floor(rank));
  const auto high = static_cast<std::size_t>(std::ceil(rank));
  return values[low] + (values[high] - values[low]) * (rank - static_cast<double>(low));
}

QImage grab(QQuickWindow * window)
{
  if (!window || !window->isExposed()) {
    return QImage();
  }
  QImage image = window->grabWindow();
  if (image.isNull() && window->screen()) {
    image = window->screen()->grabWindow(window->winId()).toImage();
  }
  return image;
}

RangeSummary summarize_range_image(const sensor_msgs::msg::Image & image)
{
  RangeSummary summary;
  if (image.encoding != "32FC1" || image.width != 8 || image.height != 8 || image.step < 32 ||
    image.data.size() != image.height * image.step || image.is_bigendian)
  {
    return summary;  // malformed: no reading, never a plausible one
  }
  summary.valid_message = true;
  summary.zones.reserve(64);
  for (uint32_t row = 0; row < image.height; ++row) {
    for (uint32_t col = 0; col < image.width; ++col) {
      float value;
      std::memcpy(&value, &image.data[row * image.step + col * 4], 4);
      const bool usable = std::isfinite(value) && value > 0.0f;
      summary.zones.push_back(usable ? static_cast<double>(value) : std::numeric_limits<double>::quiet_NaN());
      if (!usable) {
        continue;
      }
      ++summary.usable_zones;
      if (std::isnan(summary.min_m) || value < summary.min_m) {
        summary.min_m = value;
      }
    }
  }
  return summary;
}

QImage to_qimage(const sensor_msgs::msg::Image & image)
{
  if (image.encoding != "rgb8" || image.width == 0 || image.height == 0 ||
    image.step < image.width * 3 || image.data.size() != image.height * image.step)
  {
    return QImage();
  }
  QImage view(image.data.data(), static_cast<int>(image.width), static_cast<int>(image.height),
    static_cast<int>(image.step), QImage::Format_RGB888);
  return view.copy();  // the message buffer does not outlive the callback
}

WristImageProvider::WristImageProvider()
: QQuickImageProvider(QQuickImageProvider::Image) {}

QImage WristImageProvider::requestImage(const QString &, QSize * size, const QSize & requested)
{
  QMutexLocker lock(&mutex_);
  QImage image = latest_.isNull() ? QImage(480, 320, QImage::Format_RGB888) : latest_;
  if (latest_.isNull()) {
    image.fill(Qt::black);
  }
  if (size) {
    *size = image.size();
  }
  if (requested.isValid() && !requested.isEmpty()) {
    return image.scaled(requested, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  }
  return image;
}

void WristImageProvider::set(const QImage & image)
{
  QMutexLocker lock(&mutex_);
  latest_ = image;
}

}  // namespace pruning_operator_panel
