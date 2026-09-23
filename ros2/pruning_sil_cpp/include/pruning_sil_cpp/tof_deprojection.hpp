// Deproject an 8x8 time-of-flight grid into the frame its offset is expressed in.
//
// This is a port of isaaclab_pruning.baselines.tof_servo.deproject_tof, kept
// deliberately small and dependency-free so it can be checked against the Python
// original on recorded frames rather than trusted. The parity test in
// test/test_tof_deprojection.cpp reads vectors produced by that Python function.
//
// Invalid zones produce NaN. They never produce a plausible-looking point: a
// downstream gate must be able to tell "no measurement" from "a measurement at
// the origin", and silently substituting zero would erase that difference.

#ifndef PRUNING_SIL_CPP__TOF_DEPROJECTION_HPP_
#define PRUNING_SIL_CPP__TOF_DEPROJECTION_HPP_

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace pruning_sil_cpp
{

/// Default diagonal field of view of the VL53L8CX, in degrees.
inline constexpr double kDefaultDiagonalFovDeg = 65.0;

struct Point3
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

/// Unit ray per zone, matching the Python pinhole construction exactly:
/// focal = 0.5 * sqrt(w^2 + h^2) / tan(dfov / 2), pixel centres at +0.5.
std::vector<Point3> pinhole_rays(
  std::size_t width, std::size_t height, double diagonal_fov_deg = kDefaultDiagonalFovDeg);

/// Deproject ranges into the parent frame of `offset_m`.
///
/// `ranges_m` and `valid` are row-major, `height * width` elements each.
/// An invalid zone yields a point whose components are all NaN.
std::vector<Point3> deproject_tof(
  const std::vector<double> & ranges_m,
  const std::vector<bool> & valid,
  const std::array<double, 3> & offset_m,
  std::size_t width,
  std::size_t height,
  double diagonal_fov_deg = kDefaultDiagonalFovDeg);

/// True when a point carries no measurement.
inline bool is_missing(const Point3 & point)
{
  return std::isnan(point.x) || std::isnan(point.y) || std::isnan(point.z);
}

}  // namespace pruning_sil_cpp

#endif  // PRUNING_SIL_CPP__TOF_DEPROJECTION_HPP_
