cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(trajopt VERSION 1.0)

# Export compile commands for LSP.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(USE_RobotDART "Use RobotDART for visualization" ON)

# Find Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# Find ifopt
find_package(ifopt REQUIRED)

# Include RobotDART with Mangum for graphics.
if (USE_RobotDART)
    find_package(RobotDART QUIET OPTIONAL_COMPONENTS Magnum)
endif()

add_subdirectory(src)

add_subdirectory(test)
