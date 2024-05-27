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

# Create executable
add_executable(${PROJECT_NAME} main.cpp)

# Link libraries and add compiler options
target_link_libraries(${PROJECT_NAME} PUBLIC
  Eigen3::Eigen
  robo_spline::robo_spline
  ifopt::ifopt_ipopt
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME} PRIVATE "-Wall;-Wextra;-O3") # -Werror
target_compile_definitions(${PROJECT_NAME} PUBLIC "SRCPATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

# add_subdirectory(test)
