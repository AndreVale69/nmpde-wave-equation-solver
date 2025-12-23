cmake_minimum_required(VERSION 3.12.0)
project(lab06 LANGUAGES CXX C)

include(cmake-common.cmake)

add_executable(wave_solver src/wave_solver.cpp src/Wave.cpp)
deal_ii_setup_target(wave_solver)

add_executable(wave_convergence src/wave_convergence.cpp src/Wave.cpp)
deal_ii_setup_target(wave_convergence)

add_executable(wave_dispersion src/wave_dispersion.cpp src/Wave.cpp)
deal_ii_setup_target(wave_dispersion)

add_custom_target(clean-build
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/*
    COMMENT "Pulendo tutti i file nella directory build"
)
