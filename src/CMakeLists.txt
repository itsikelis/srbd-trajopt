# Create executable
add_executable(${PROJECT_NAME}_phased trajopt_phased.cpp)

target_include_directories(${PROJECT_NAME}_phased PRIVATE ${PROJECT_SOURCE_DIR})

# Link libraries and add compiler options
target_link_libraries(${PROJECT_NAME}_phased PRIVATE
  Eigen3::Eigen
  ifopt::ifopt_ipopt
)

target_compile_features(${PROJECT_NAME}_phased PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME}_phased PRIVATE "-Wall;-Wextra;-O3;-Werror")
