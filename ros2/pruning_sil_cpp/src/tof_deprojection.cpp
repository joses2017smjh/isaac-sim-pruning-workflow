#include "pruning_sil_cpp/tof_deprojection.hpp"

#include <limits>
#include <stdexcept>

namespace pruning_sil_cpp
{

std::vector<Point3> pinhole_rays(std::size_t width, std::size_t height, double diagonal_fov_deg)
{
  if (width == 0 || height == 0) {
    throw std::invalid_argument("Time-of-flight grid must have a non-zero extent");
  }
  const double diagonal = diagonal_fov_deg * M_PI / 180.0;
  const double tangent = std::tan(diagonal / 2.0);
  if (!(tangent > 0.0)) {
    throw std::invalid_argument("Diagonal field of view must be positive and below 180 degrees");
  }
  const double focal =
    0.5 * std::sqrt(static_cast<double>(width * width + height * height)) / tangent;

  std::vector<Point3> rays;
  rays.reserve(width * height);
  for (std::size_t row = 0; row < height; ++row) {
    for (std::size_t col = 0; col < width; ++col) {
      const double u = static_cast<double>(col) + 0.5;
      const double v = static_cast<double>(row) + 0.5;
      Point3 ray;
      ray.x = (u - 0.5 * static_cast<double>(width)) / focal;
      ray.y = (v - 0.5 * static_cast<double>(height)) / focal;
      ray.z = 1.0;
      const double norm = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);
      ray.x /= norm;
      ray.y /= norm;
      ray.z /= norm;
      rays.push_back(ray);
    }
  }
  return rays;
}

std::vector<Point3> deproject_tof(
  const std::vector<double> & ranges_m,
  const std::vector<bool> & valid,
  const std::array<double, 3> & offset_m,
  std::size_t width,
  std::size_t height,
  double diagonal_fov_deg)
{
  const std::size_t zones = width * height;
  if (ranges_m.size() != zones || valid.size() != zones) {
    throw std::invalid_argument("Ranges and validity must both hold width * height elements");
  }
  const auto rays = pinhole_rays(width, height, diagonal_fov_deg);
  const double nan = std::numeric_limits<double>::quiet_NaN();

  std::vector<Point3> points;
  points.reserve(zones);
  for (std::size_t index = 0; index < zones; ++index) {
    if (!valid[index]) {
      // No measurement. Say so, rather than emitting the offset as if it were one.
      points.push_back(Point3{nan, nan, nan});
      continue;
    }
    const double range = ranges_m[index];
    Point3 point;
    point.x = range * rays[index].x + offset_m[0];
    point.y = range * rays[index].y + offset_m[1];
    point.z = range * rays[index].z + offset_m[2];
    points.push_back(point);
  }
  return points;
}

}  // namespace pruning_sil_cpp
