cmake_minimum_required(VERSION 3.16)
project(wb_sort_cpu C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(
    wb_sort_main
    main.c
)

target_link_libraries(wb_sort_main PRIVATE wgpu_native)
target_link_libraries(wb_sort_main PRIVATE wb_sort_gpu_headers)
