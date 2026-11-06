cmake_minimum_required(VERSION 3.8)
project(pruning_operator_panel)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pruning_sil_msgs REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Gui Qml Quick)

# Qt keywords (signals, slots, emit) stay off so they cannot collide with ROS headers.
add_library(panel_backend
  src/panel_backend.cpp
  src/panel_model.cpp
  include/pruning_operator_panel/panel_backend.hpp)
target_include_directories(panel_backend PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_definitions(panel_backend PUBLIC QT_NO_KEYWORDS)
target_link_libraries(panel_backend Qt5::Core Qt5::Gui Qt5::Qml Qt5::Quick)
ament_target_dependencies(panel_backend rclcpp sensor_msgs std_msgs pruning_sil_msgs)

add_executable(operator_panel src/main.cpp qml/qml.qrc)
target_link_libraries(operator_panel panel_backend)

install(TARGETS panel_backend ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS operator_panel DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_panel_model test/test_panel_model.cpp qml/qml.qrc
    ENV QT_QPA_PLATFORM=offscreen QT_QUICK_BACKEND=software)
  target_link_libraries(test_panel_model panel_backend)
endif()

ament_package()
