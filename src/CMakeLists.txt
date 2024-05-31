# Create executable
add_executable(${PROJECT_NAME}_phased trajopt_phased.cpp)

target_include_directories(${PROJECT_NAME}_phased PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Link libraries and add compiler options
target_link_libraries(${PROJECT_NAME}_phased PRIVATE
  Eigen3::Eigen
  ifopt::ifopt_ipopt
)

target_compile_features(${PROJECT_NAME}_phased PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME}_phased PRIVATE "-Wall;-Wextra;-O3;-Werror")

# Add contact implicit example
add_executable(${PROJECT_NAME}_implicit trajopt_contact_implicit.cpp)

target_include_directories(${PROJECT_NAME}_implicit PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(${PROJECT_NAME}_implicit PRIVATE
  Eigen3::Eigen
  ifopt::ifopt_ipopt
)

target_compile_features(${PROJECT_NAME}_implicit PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME}_implicit PRIVATE "-Wall;-Wextra;-O3;-Werror")
