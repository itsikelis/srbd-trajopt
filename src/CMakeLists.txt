# Add trajopt library
file(GLOB_RECURSE SRCS trajopt/*.cpp)

add_library(${PROJECT_NAME} ${SRCS})
 
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME} PUBLIC
  Eigen3::Eigen
  ifopt::ifopt_ipopt
)

if (RobotDART_FOUND AND RobotDART_Magnum_FOUND)
  set_target_properties(RobotDART::Simu PROPERTIES INTERFACE_COMPILE_OPTIONS "")
  set_target_properties(RobotDART::Magnum PROPERTIES INTERFACE_COMPILE_OPTIONS "")

  target_compile_definitions(${PROJECT_NAME} PUBLIC "-DVIZ=1;SRCPATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
  target_link_libraries(${PROJECT_NAME} PUBLIC 
    RobotDART::Simu
    RobotDART::Magnum
  )
endif()

target_compile_options(${PROJECT_NAME} PUBLIC "-Wall;-Wextra;-O3;-std=c++20") # ;-Werror


# Add contact ipopt example
add_executable(${PROJECT_NAME}_ipopt examples/trajopt_ipopt.cpp)
target_include_directories(${PROJECT_NAME}_ipopt PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME}_ipopt PRIVATE  ${PROJECT_NAME})

# Add Augmented Lagrangian example
add_executable(${PROJECT_NAME}_alm examples/trajopt_alm.cpp)
target_include_directories(${PROJECT_NAME}_alm PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME}_alm PRIVATE ${PROJECT_NAME})

# Add CEM example
add_executable(${PROJECT_NAME}_cem examples/trajopt_cem.cpp)
target_include_directories(${PROJECT_NAME}_cem PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME}_cem PRIVATE
    ${PROJECT_NAME}
)
target_include_directories(${PROJECT_NAME}_cem PUBLIC /usr/local/include)
