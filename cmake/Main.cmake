cmake_minimum_required(VERSION 3.16)
project(merge_sort_cpu C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(
    merge_sort_main
    main.c
)

target_link_libraries(merge_sort_main PRIVATE wgpu_native)
target_link_libraries(merge_sort_main PRIVATE merge_sort_gpu_headers)
