# Add trajopt library
file(GLOB_RECURSE SRCS trajopt/*.cpp)

add_library(${PROJECT_NAME} ${SRCS})
 
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_compile_options(${PROJECT_NAME} PUBLIC "-Wall;-Wextra;-O3;-std=c++20") # ;-Werror

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

target_link_libraries(${PROJECT_NAME}_implicit PRIVATE  ${PROJECT_NAME})

if (RobotDART_FOUND AND RobotDART_Magnum_FOUND)
  set_target_properties(RobotDART::Simu PROPERTIES INTERFACE_COMPILE_OPTIONS "")
  set_target_properties(RobotDART::Magnum PROPERTIES INTERFACE_COMPILE_OPTIONS "")

  target_compile_definitions(${PROJECT_NAME}_implicit PUBLIC "-DVIZ=1")
  target_link_libraries(${PROJECT_NAME}_implicit PRIVATE 
    RobotDART::Simu
    RobotDART::Magnum
  )

  target_compile_definitions(${PROJECT_NAME}_phased PUBLIC "-DVIZ=1")
  target_link_libraries(${PROJECT_NAME}_phased PRIVATE 
    RobotDART::Simu
    RobotDART::Magnum
  )
endif()

target_compile_definitions(${PROJECT_NAME}_implicit PUBLIC "SRCPATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

# Add Augmented Lagrangian example
add_executable(${PROJECT_NAME}_alm examples/trajopt_alm.cpp)

target_include_directories(${PROJECT_NAME}_alm PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME}_alm PRIVATE ${PROJECT_NAME})

if (RobotDART_FOUND AND RobotDART_Magnum_FOUND)
  set_target_properties(RobotDART::Simu PROPERTIES INTERFACE_COMPILE_OPTIONS "")
  set_target_properties(RobotDART::Magnum PROPERTIES INTERFACE_COMPILE_OPTIONS "")

  target_compile_definitions(${PROJECT_NAME}_alm PUBLIC "-DVIZ=1")
  target_link_libraries(${PROJECT_NAME}_alm PRIVATE 
    RobotDART::Simu
    RobotDART::Magnum
  )
endif()

target_compile_definitions(${PROJECT_NAME}_alm PUBLIC "SRCPATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

# Add Phased Augmented Lagrangian example
add_executable(${PROJECT_NAME}_phased_alm examples/trajopt_phased_alm.cpp)

target_include_directories(${PROJECT_NAME}_phased_alm PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME}_phased_alm PRIVATE ${PROJECT_NAME})

if (RobotDART_FOUND AND RobotDART_Magnum_FOUND)
  set_target_properties(RobotDART::Simu PROPERTIES INTERFACE_COMPILE_OPTIONS "")
  set_target_properties(RobotDART::Magnum PROPERTIES INTERFACE_COMPILE_OPTIONS "")

  target_compile_definitions(${PROJECT_NAME}_phased_alm PUBLIC "-DVIZ=1")
  target_link_libraries(${PROJECT_NAME}_phased_alm PRIVATE 
    RobotDART::Simu
    RobotDART::Magnum
  )
endif()

target_compile_definitions(${PROJECT_NAME}_phased_alm PUBLIC "SRCPATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

# Add DI example
add_executable(${PROJECT_NAME}_di_alm examples/trajopt_di_alm.cpp)

target_include_directories(${PROJECT_NAME}_di_alm PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME}_di_alm PRIVATE ${PROJECT_NAME})

if (RobotDART_FOUND AND RobotDART_Magnum_FOUND)
  set_target_properties(RobotDART::Simu PROPERTIES INTERFACE_COMPILE_OPTIONS "")
  set_target_properties(RobotDART::Magnum PROPERTIES INTERFACE_COMPILE_OPTIONS "")

  target_compile_definitions(${PROJECT_NAME}_di_alm PUBLIC "-DVIZ=1")
  target_link_libraries(${PROJECT_NAME}_di_alm PRIVATE 
    RobotDART::Simu
    RobotDART::Magnum
  )
endif()

target_compile_definitions(${PROJECT_NAME}_di_alm PUBLIC "SRCPATH=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
