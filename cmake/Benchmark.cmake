cmake_minimum_required(VERSION 3.16)
project(wb_benchmark C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(
    wb_benchmark
    benchmark.c
)

target_link_libraries(wb_benchmark PRIVATE wgpu_native)
target_link_libraries(wb_benchmark PRIVATE wb_sort_cpu_headers)
target_link_libraries(wb_benchmark PRIVATE wb_sort_gpu_headers)
