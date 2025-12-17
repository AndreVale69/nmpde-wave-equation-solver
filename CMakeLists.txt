# Minimum CMake version required.
cmake_minimum_required(VERSION 3.12)
# Project name and language.
project(nm4pde LANGUAGES CXX C)

# Add MK modules path (https://github.com/pcafrica/mk).
# 1) gcc-glibc/11.2.0  11) eigen/3.3.9      21) scipy/1.7.0
# 2) adol-c/2.7.2      12) glpk/5.0         22) scotch/6.1.1
# 3) arborx/1.5        13) superlu/5.3.0    23) openblas/0.3.15
# 4) gsl/2.7           14) hypre/2.22.0     24) metis/5.1.0
# 5) arpack/3.8.0      15) matio/1.5.21     25) suitesparse/5.10.1
# 6) fftw/3.3.9        16) mumps/5.4.0      26) tbb/2021.4.0
# 7) p4est/2.3.2       17) hdf5/1.12.0      27) trilinos/15.0.0
# 8) petsc/3.15.1      18) netcdf/4.9.2     28) dealii/9.5.1
# 9) blacs/1.1         19) pnetcdf/1.12.3
# 10) boost/1.76.0     20) scalapack/2.1.0
#
# Optional: MK-specific include path list for Politecnico environment.
# These absolute include directories come from the Politecnico "mk" environment
# and the HPC module layout. Standard users building with system-installed
# libraries can comment out or remove this block if it's not applicable.
include_directories(
        # gcc
        /u/sw/toolchains/gcc-glibc/11.2.0/base/include
        # deal.II dependencies
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/adol-c/2.7.2/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/arborx/1.5/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/gsl/2.7/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/arpack/3.8.0/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/fftw/3.3.9/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/p4est/2.3.2/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/petsc/3.15.1/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/blacs/1.1/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/eigen/3.3.9/include/eigen3
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/glpk/5.0/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/superlu/5.3.0/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/hypre/2.22.0/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/matio/1.5.21/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/mumps/5.4.0/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/hdf5/1.12.0/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/netcdf/4.9.2/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/pnetcdf/1.12.3/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/scalapack/2.1.0/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/scipy/1.7.0/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/scotch/6.1.1/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/openblas/0.3.15/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/metis/5.1.0/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/suitesparse/5.10.1/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/tbb/2021.4.0/include
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/15.0.0/include
        # deal.II
        /u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.5.1/include
)

# Include common CMake settings if present (Politecnico course fragment).
if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/../common/cmake-common.cmake")
    include(${CMAKE_CURRENT_LIST_DIR}/../common/cmake-common.cmake)
else()
    message(STATUS "Course common CMake fragment not found; continuing with a standard CMake setup.")
    # If you want to remove course-specific references entirely, open this file
    # and delete or comment out the include above. Standard CMake-based builds
    # will still work with system-installed dependencies and a proper
    # environment (CMAKE_PREFIX_PATH, DEAL_II_DIR, BOOST_ROOT, etc.).
endif()

# Locate MPI compiler.
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")

# Locate Boost.
find_package(Boost 1.72.0 REQUIRED
        COMPONENTS filesystem iostreams serialization
        HINTS ${BOOST_DIR} $ENV{BOOST_DIR} $ENV{mkBoostPrefix})
message(STATUS "Using the Boost-${Boost_VERSION} configuration found at ${Boost_DIR}")
message(STATUS)
include_directories(${Boost_INCLUDE_DIRS})

# Locate deal.II and initialize its variables.
find_package(deal.II 9.3.1 REQUIRED
        HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()

# Add useful compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion -Wmissing-braces -Wnon-virtual-dtor")

# Print current directories.
message(STATUS "CMake source dir: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMake binary dir: ${CMAKE_BINARY_DIR}")
message(STATUS "CMake current dir: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS)

# Add the executable target.
add_executable(
        nm4pde
        src/main.cpp
        src/Wave.cpp
        src/Wave.hpp
        src/time_integrator.hpp
        src/theta_integrator.hpp
        src/theta_integrator.cpp
        src/time_scheme.hpp
        src/parameters.hpp
        src/progress_bar.hpp
)
deal_ii_setup_target(nm4pde)