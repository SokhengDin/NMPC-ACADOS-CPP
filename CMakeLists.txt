cmake_minimum_required(VERSION 3.10)
project(acados_cpp VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Link file system library
link_libraries(stdc++fs)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

# Find matplotlib
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import matplotlib; print(matplotlib.__file__)"
    OUTPUT_VARIABLE MATPLOTLIB_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
get_filename_component(MATPLOTLIB_PATH ${MATPLOTLIB_PATH} DIRECTORY)

# Try to find backend_agg.h
find_path(MATPLOTLIB_INCLUDE_DIR
    NAMES backend_agg.h agg_backend.h
    PATHS
        ${MATPLOTLIB_PATH}
        ${MATPLOTLIB_PATH}/backends
        ${MATPLOTLIB_PATH}/backends/qt_compat
        ${Python3_SITELIB}/matplotlib/backends
        ${Python3_SITELIB}/matplotlib/backends/qt_compat
    PATH_SUFFIXES matplotlib/backends
)

if(NOT MATPLOTLIB_INCLUDE_DIR)
    message(WARNING "matplotlib backend header not found. Plotting functionality might be limited.")
else()
    message(STATUS "Found matplotlib backend header at: ${MATPLOTLIB_INCLUDE_DIR}")
    include_directories(${MATPLOTLIB_INCLUDE_DIR})
    include_directories(${MATPLOTLIB_PATH})
    include_directories(${MATPLOTLIB_PATH}/core)
endif()

include_directories(${MATPLOTLIB_PATH})
include_directories(${MATPLOTLIB_PATH}/core)

# Set path
set(ACADOS_INSTALL_DIR "/home/eroxii/ocp_ws/acados" CACHE PATH "Path to acados installation")

# Add acados include directories
include_directories(${ACADOS_INSTALL_DIR}/include)
include_directories(${ACADOS_INSTALL_DIR}/include/acados)
include_directories(${ACADOS_INSTALL_DIR}/include/blasfeo/include)
include_directories(${ACADOS_INSTALL_DIR}/include/hpipm/include)

# Add the path to generated acados code
set(ACADOS_GENERATED_DIR "${CMAKE_SOURCE_DIR}/acados_generated")
include_directories(${ACADOS_GENERATED_DIR})
include_directories(${ACADOS_GENERATED_DIR}/differential_drive_model)
include_directories(${ACADOS_GENERATED_DIR}/differential_drive_cost)

# Add the include directory
include_directories(${CMAKE_SOURCE_DIR}/include)

file(COPY ${ACADOS_GENERATED_DIR}/libacados_ocp_solver_differential_drive.so
     DESTINATION ${CMAKE_BINARY_DIR})

add_executable(differential_drive_solver
    src/main.cpp
    src/acados_cpp_wrapper.cpp
    ${ACADOS_GENERATED_DIR}/acados_solver_differential_drive.c
    ${ACADOS_GENERATED_DIR}/differential_drive_model/differential_drive_expl_ode_fun.c
    ${ACADOS_GENERATED_DIR}/differential_drive_model/differential_drive_expl_vde_forw.c
    ${ACADOS_GENERATED_DIR}/differential_drive_model/differential_drive_expl_vde_adj.c
    ${ACADOS_GENERATED_DIR}/differential_drive_cost/differential_drive_cost_y_0_fun.c
    ${ACADOS_GENERATED_DIR}/differential_drive_cost/differential_drive_cost_y_0_fun_jac_ut_xt.c
    ${ACADOS_GENERATED_DIR}/differential_drive_cost/differential_drive_cost_y_0_hess.c
    ${ACADOS_GENERATED_DIR}/differential_drive_cost/differential_drive_cost_y_e_fun.c
    ${ACADOS_GENERATED_DIR}/differential_drive_cost/differential_drive_cost_y_e_fun_jac_ut_xt.c
    ${ACADOS_GENERATED_DIR}/differential_drive_cost/differential_drive_cost_y_e_hess.c
    ${ACADOS_GENERATED_DIR}/differential_drive_cost/differential_drive_cost_y_fun.c
    ${ACADOS_GENERATED_DIR}/differential_drive_cost/differential_drive_cost_y_fun_jac_ut_xt.c
    ${ACADOS_GENERATED_DIR}/differential_drive_cost/differential_drive_cost_y_hess.c
)

target_include_directories(differential_drive_solver PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link against the acados libraries
target_link_libraries(differential_drive_solver
    ${ACADOS_INSTALL_DIR}/lib/libacados.so
    ${ACADOS_INSTALL_DIR}/lib/libblasfeo.so
    ${ACADOS_INSTALL_DIR}/lib/libhpipm.so
    ${CMAKE_BINARY_DIR}/libacados_ocp_solver_differential_drive.so
    ${Python3_LIBRARIES}
    ${Python3_NumPy_LIBRARIES}
)

# Add matplotlib-cpp
target_include_directories(differential_drive_solver PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(differential_drive_solver 
    ${Python3_LIBRARIES}
    ${Python3_NumPy_LIBRARIES}
)
target_compile_definitions(differential_drive_solver PRIVATE
    WITHOUT_NUMPY
    MPL_strcasecmp=strcasecmp
)

# Set RPATH for the executable
set_target_properties(differential_drive_solver PROPERTIES
    INSTALL_RPATH "${CMAKE_BINARY_DIR}:${ACADOS_INSTALL_DIR}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Print debug information
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
message(STATUS "Python3_NumPy_INCLUDE_DIRS: ${Python3_NumPy_INCLUDE_DIRS}")
message(STATUS "Python3_NumPy_LIBRARIES: ${Python3_NumPy_LIBRARIES}")
message(STATUS "MATPLOTLIB_PATH: ${MATPLOTLIB_PATH}")
message(STATUS "MATPLOTLIB_INCLUDE_DIR: ${MATPLOTLIB_INCLUDE_DIR}")