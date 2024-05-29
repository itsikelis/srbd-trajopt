# Create executable
add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR})

# Link libraries and add compiler options
target_link_libraries(${PROJECT_NAME} PRIVATE
  Eigen3::Eigen
  ifopt::ifopt_ipopt
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME} PRIVATE "-Wall;-Wextra;-O3") # -Werror
