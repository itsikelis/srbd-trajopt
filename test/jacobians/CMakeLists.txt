add_executable(jac_test_phased jac_test_phased.cpp)

target_include_directories(jac_test_phased PRIVATE ${PROJECT_SOURCE_DIR})

# Link libraries and add compiler options
target_link_libraries(jac_test_phased PUBLIC
  Eigen3::Eigen
  ifopt::ifopt_ipopt
)

target_compile_features(jac_test_phased PRIVATE cxx_std_20)
target_compile_options(jac_test_phased PRIVATE "-Wall;-Wextra;-O3") # -Werror
target_compile_definitions(jac_test_phased PUBLIC "SRCPATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

# Contact implicit Jacobian tests
add_executable(jac_test_implicit jac_test_implicit.cpp)

target_include_directories(jac_test_implicit PRIVATE ${PROJECT_SOURCE_DIR})

# Link libraries and add compiler options
target_link_libraries(jac_test_implicit PUBLIC
  Eigen3::Eigen
  ifopt::ifopt_ipopt
)

target_compile_features(jac_test_implicit PRIVATE cxx_std_20)
target_compile_options(jac_test_implicit PRIVATE "-Wall;-Wextra;-O3") # -Werror
target_compile_definitions(jac_test_implicit PUBLIC "SRCPATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
