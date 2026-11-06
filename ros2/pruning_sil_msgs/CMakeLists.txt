cmake_minimum_required(VERSION 3.8)
project(pruning_sil_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ControllerProposal.msg"
  "msg/OperatorHold.msg"
  "msg/SafetyDecision.msg"
  "msg/TargetEstimate.msg"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
