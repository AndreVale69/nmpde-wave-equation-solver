cmake_minimum_required(VERSION 3.12.0)
project(waveEquation LANGUAGES CXX C)

include(./cmake-common.cmake)

# Basic wave solver
add_executable(wave-solver 
  src/wave-solver.cpp 
  src/Wave.cpp)
deal_ii_setup_target(wave-solver)

# Convergence study
add_executable(wave-convergence 
  src/wave-convergence.cpp 
  src/Wave.cpp)
deal_ii_setup_target(wave-convergence)

# Dispersion/dissipation analysis
add_executable(wave-dispersion 
  src/wave-dispersion.cpp 
  src/Wave.cpp)
deal_ii_setup_target(wave-dispersion)

