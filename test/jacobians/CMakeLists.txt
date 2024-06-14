add_executable(jac_test_phased jac_test_phased.cpp)
target_include_directories(jac_test_phased PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Link libraries and add compiler options
target_link_libraries(jac_test_phased PUBLIC ${PROJECT_NAME})
target_compile_options(jac_test_phased PRIVATE "-Wall;-Wextra;-O3")

# Contact implicit Jacobian tests
add_executable(jac_test_implicit jac_test_implicit.cpp)
target_include_directories(jac_test_implicit PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Link libraries and add compiler options
target_link_libraries(jac_test_implicit PUBLIC ${PROJECT_NAME})
target_compile_options(jac_test_implicit PRIVATE "-Wall;-Wextra;-O3")

# Augmented Lagrangian Jacobian tests
add_executable(jac_test_alm jac_test_alm.cpp)
target_include_directories(jac_test_alm PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Link libraries and add compiler options
target_link_libraries(jac_test_alm PUBLIC ${PROJECT_NAME})
target_compile_options(jac_test_alm PRIVATE "-Wall;-Wextra;-O3")
