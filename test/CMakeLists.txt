# Create executable
add_executable(${PROJECT_NAME}_jac_test jac_test.cpp)

# Link jacobian_test
target_link_libraries(${PROJECT_NAME}_jac_test PUBLIC
  Eigen3::Eigen
  robo_spline::robo_spline
  ifopt::ifopt_ipopt
)

target_compile_features(${PROJECT_NAME}_jac_test PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME}_jac_test PRIVATE "-Wall;-Wextra;-O3") # -Werror
target_compile_definitions(${PROJECT_NAME}_jac_test PUBLIC "SRCPATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

add_executable(${PROJECT_NAME}_terrain_test terrain_test.cpp)

# Add terrain test
target_link_libraries(${PROJECT_NAME}_terrain_test PUBLIC
  Eigen3::Eigen
  robo_spline::robo_spline
  ifopt::ifopt_ipopt
)

target_compile_features(${PROJECT_NAME}_terrain_test PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME}_terrain_test PRIVATE "-Wall;-Wextra;-O3") # -Werror
target_compile_definitions(${PROJECT_NAME}_terrain_test PUBLIC "SRCPATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
