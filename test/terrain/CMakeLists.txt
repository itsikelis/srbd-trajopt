add_executable(test_terrain test_terrain.cpp)

target_include_directories(test_terrain PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Link libraries and add compiler options
target_link_libraries(test_terrain PUBLIC
  Eigen3::Eigen
)

target_compile_features(test_terrain PRIVATE cxx_std_20)
target_compile_options(test_terrain PRIVATE "-Wall;-Wextra;-O3") # -Werror
target_compile_definitions(test_terrain PUBLIC "SRCPATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
