# Create executable
add_executable(${PROJECT_NAME}_test jac_test.cpp)

# Link libraries and add compiler options
target_link_libraries(${PROJECT_NAME}_test PUBLIC
  Eigen3::Eigen
  robo_spline::robo_spline
  ifopt::ifopt_ipopt
)

target_compile_features(${PROJECT_NAME}_test PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME}_test PRIVATE "-Wall;-Wextra;-O3") # -Werror
target_compile_definitions(${PROJECT_NAME}_test PUBLIC "SRCPATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

