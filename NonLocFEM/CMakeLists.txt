cmake_minimum_required(VERSION 3.16)

project(NonLocFEM)

add_executable(NonLocFEM 
    main.cpp
)
target_include_directories(
    NonLocFEM
    PRIVATE #
        ${LIBRARIES_DIR}
)
target_link_libraries(
    NonLocFEM
    PRIVATE #
        slae_solver_lib
)
