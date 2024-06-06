# Add trajopt library
file(GLOB_RECURSE SRCS trajopt/*.cpp)

add_library(${PROJECT_NAME} ${SRCS})
 
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_compile_options(${PROJECT_NAME} PUBLIC "-Wall;-Wextra;-O3;-Werror")

target_link_libraries(${PROJECT_NAME} PUBLIC
  Eigen3::Eigen
  ifopt::ifopt_ipopt
)

# Add phased based example
add_executable(${PROJECT_NAME}_phased examples/trajopt_phased.cpp)

target_include_directories(${PROJECT_NAME}_phased PRIVATE ${PROJECT_SOURCE_DIR}/src)

# Link libraries and add compiler options
target_link_libraries(${PROJECT_NAME}_phased PRIVATE ${PROJECT_NAME})

# Add contact implicit example
add_executable(${PROJECT_NAME}_implicit examples/trajopt_contact_implicit.cpp)

target_include_directories(${PROJECT_NAME}_implicit PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME}_implicit PRIVATE 
  ${PROJECT_NAME}
)

if (RobotDART_FOUND AND RobotDART_Magnum_FOUND)
  target_compile_definitions(${PROJECT_NAME}_implicit PUBLIC "-DVIZ=1")
  target_link_libraries(${PROJECT_NAME}_implicit PRIVATE 
    RobotDART::Simu
    RobotDART::Magnum
  )
endif()

target_compile_definitions(${PROJECT_NAME}_implicit PUBLIC "SRCPATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
