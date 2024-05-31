add_executable(spline_test_regular spline_test_regular.cpp)

target_include_directories(spline_test_regular PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(spline_test_regular PRIVATE
  Eigen3::Eigen
)

target_compile_features(spline_test_regular PRIVATE cxx_std_20)
target_compile_options(spline_test_regular PRIVATE "-Wall;-Wextra;-O3")


add_executable(spline_test_phased spline_test_phased.cpp)

target_include_directories(spline_test_phased PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(spline_test_phased PRIVATE
  Eigen3::Eigen
)

target_compile_features(spline_test_phased PRIVATE cxx_std_20)
target_compile_options(spline_test_phased PRIVATE "-Wall;-Wextra;-O3")
