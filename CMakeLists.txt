cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(trajopt VERSION 1.0)

# Export compile commands for LSP.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
# Find ifopt
find_package(ifopt REQUIRED)
# Find robo_spline
find_package(robo_spline 1.0 REQUIRED)

add_subdirectory(src)

add_subdirectory(test)
