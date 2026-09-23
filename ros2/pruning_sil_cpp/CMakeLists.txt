cmake_minimum_required(VERSION 3.8)
project(pruning_sil_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)

add_library(tof_deprojection src/tof_deprojection.cpp)
target_include_directories(tof_deprojection PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS tof_deprojection
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_tof_deprojection test/test_tof_deprojection.cpp)
  target_link_libraries(test_tof_deprojection tof_deprojection)
  # The parity fixture is read from the source tree so the reference values stay
  # visible and reviewable next to the test that consumes them.
  target_compile_definitions(test_tof_deprojection PRIVATE
    PRUNING_SIL_CPP_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test")
endif()

ament_package()
