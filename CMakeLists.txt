cmake_minimum_required(VERSION 3.10)
project(acados_cpp VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT APPLE)
    link_libraries(stdc++fs)
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sys; print(sys.prefix)"
    OUTPUT_VARIABLE PYTHON_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))"
    OUTPUT_VARIABLE Python3_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
    OUTPUT_VARIABLE Python3_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(Python3_FIND_STRATEGY LOCATION)
find_package(Python3 COMPONENTS Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
include_directories(${NUMPY_INCLUDE_DIR})

message(STATUS "Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python include: ${Python3_INCLUDE_DIRS}")
message(STATUS "NumPy include: ${NUMPY_INCLUDE_DIR}")

include_directories(${CMAKE_SOURCE_DIR}/external/matplotlib-cpp)
include_directories(${CMAKE_SOURCE_DIR}/external/spline/src)

if(DEFINED ENV{ACADOS_SOURCE_DIR})
    set(ACADOS_INSTALL_DIR "$ENV{ACADOS_SOURCE_DIR}")
else()
    set(ACADOS_INSTALL_DIR "$ENV{HOME}/Desktop/Robotics/github/acados")
endif()
message(STATUS "Using ACADOS_INSTALL_DIR: ${ACADOS_INSTALL_DIR}")

include_directories(${ACADOS_INSTALL_DIR}/include)
include_directories(${ACADOS_INSTALL_DIR}/include/acados)
include_directories(${ACADOS_INSTALL_DIR}/include/blasfeo/include)
include_directories(${ACADOS_INSTALL_DIR}/include/hpipm/include)

include_directories(${CMAKE_SOURCE_DIR}/include)

if(APPLE)
    set(LIB_EXT "dylib")
else()
    set(LIB_EXT "so")
endif()

set(ACADOS_GENERATED_DIR "${CMAKE_SOURCE_DIR}/acados_generated")
if(EXISTS ${ACADOS_GENERATED_DIR}/libacados_ocp_solver_differential_drive.${LIB_EXT})
    file(COPY ${ACADOS_GENERATED_DIR}/libacados_ocp_solver_differential_drive.${LIB_EXT}
         DESTINATION ${CMAKE_BINARY_DIR})
endif()

set(ACADOS_GENERATED_BICYCLE_DIR "${CMAKE_SOURCE_DIR}/acados_generated_bicycle")
if(EXISTS ${ACADOS_GENERATED_BICYCLE_DIR}/libacados_ocp_solver_bicycle.${LIB_EXT})
    file(COPY ${ACADOS_GENERATED_BICYCLE_DIR}/libacados_ocp_solver_bicycle.${LIB_EXT}
         DESTINATION ${CMAKE_BINARY_DIR})
endif()

set(ACADOS_GENERATED_QUADROTOR_DIR "${CMAKE_SOURCE_DIR}/acados_generated_quadrotor")
if(EXISTS ${ACADOS_GENERATED_QUADROTOR_DIR}/libacados_ocp_solver_quadrotor.${LIB_EXT})
    file(COPY ${ACADOS_GENERATED_QUADROTOR_DIR}/libacados_ocp_solver_quadrotor.${LIB_EXT}
         DESTINATION ${CMAKE_BINARY_DIR})
endif()

set(ACADOS_GENERATED_ROCKET_DIR "${CMAKE_SOURCE_DIR}/acados_generated_rocket")
if(EXISTS ${ACADOS_GENERATED_ROCKET_DIR}/libacados_ocp_solver_rocket.${LIB_EXT})
    file(COPY ${ACADOS_GENERATED_ROCKET_DIR}/libacados_ocp_solver_rocket.${LIB_EXT}
         DESTINATION ${CMAKE_BINARY_DIR})
endif()

add_executable(mpc_controller
    src/mpc_controller.cpp
    src/acados_mpc_solver.cpp
)

target_include_directories(mpc_controller PRIVATE ${CMAKE_SOURCE_DIR}/external)

target_link_libraries(mpc_controller
    ${ACADOS_INSTALL_DIR}/lib/libacados.${LIB_EXT}
    ${ACADOS_INSTALL_DIR}/lib/libblasfeo.${LIB_EXT}
    ${ACADOS_INSTALL_DIR}/lib/libhpipm.${LIB_EXT}
    ${Python3_LIBRARIES}
    ${CMAKE_DL_LIBS}
)

target_compile_definitions(mpc_controller PRIVATE
    WITHOUT_NUMPY
    MPL_strcasecmp=strcasecmp
)

get_filename_component(PYTHON_LIB_DIR ${Python3_LIBRARIES} DIRECTORY)
set_target_properties(mpc_controller PROPERTIES
    INSTALL_RPATH "${CMAKE_BINARY_DIR};${ACADOS_INSTALL_DIR}/lib;${PYTHON_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    BUILD_RPATH "${CMAKE_BINARY_DIR};${ACADOS_INSTALL_DIR}/lib;${PYTHON_LIB_DIR}"
)
