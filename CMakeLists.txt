cmake_minimum_required(VERSION 3.16)

project(nonlocal)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

option(ENABLE_OMP "Enable OpenMP parallel regions" ON)
if(ENABLE_OMP)
    # Required when enabled: fail fast if not available.
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP enabled")
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
enable_testing()

# Conan 2 integration (CMakeDeps + CMakeToolchain)
# The toolchain file is passed from the "conan install" step (see README).
# After that we just rely on find_package with CONFIG mode to load dependencies.
find_package(Eigen3 REQUIRED CONFIG)

add_compile_options(
    $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>:-Wno-missing-template-arg-list-after-template-kw>
)

# libraries
add_subdirectory(libraries)

# NonLocFEM
add_subdirectory(NonLocFEM)