cmake_minimum_required(VERSION 3.16)

add_executable(
    wb_benchmark_throughput
    benchmark_throughput.c
)

target_link_libraries(wb_benchmark_throughput PRIVATE webgpu_backend)
target_link_libraries(wb_benchmark_throughput PRIVATE wb_sort_gpu_headers)
