cmake_minimum_required(VERSION 3.8)
project(pruning_sil_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pruning_sil_msgs REQUIRED)

# Geometry and gate logic: plain C++17, no ROS, so they test without a graph.
add_library(tof_deprojection src/tof_deprojection.cpp)
target_include_directories(tof_deprojection PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

add_library(safety_gates src/safety_gates.cpp)
target_include_directories(safety_gates PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# The node every arm command passes through.
add_executable(safety_gate_node src/safety_gate_node.cpp)
target_link_libraries(safety_gate_node safety_gates tof_deprojection)
ament_target_dependencies(safety_gate_node rclcpp geometry_msgs sensor_msgs std_msgs pruning_sil_msgs)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS tof_deprojection safety_gates
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(TARGETS safety_gate_node DESTINATION lib/${PROJECT_NAME})
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  # Fixtures are read from the source tree so the reference values stay visible and
  # reviewable next to the tests that consume them.
  ament_add_gtest(test_tof_deprojection test/test_tof_deprojection.cpp)
  target_link_libraries(test_tof_deprojection tof_deprojection)
  target_compile_definitions(test_tof_deprojection PRIVATE
    PRUNING_SIL_CPP_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")

  ament_add_gtest(test_safety_gates test/test_safety_gates.cpp)
  target_link_libraries(test_safety_gates safety_gates)
  target_compile_definitions(test_safety_gates PRIVATE
    PRUNING_SIL_CPP_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")
endif()

ament_package()
