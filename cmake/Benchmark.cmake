cmake_minimum_required(VERSION 3.16)
project(wb_benchmark C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(
    wb_benchmark
    benchmark.c
)

target_link_libraries(wb_benchmark PRIVATE webgpu_backend)
target_link_libraries(wb_benchmark PRIVATE wb_sort_cpu_headers)
target_link_libraries(wb_benchmark PRIVATE wb_sort_gpu_headers)

if (DEFINED DEBUG)
    set(_cpu_build_type "debug")
else()
    set(_cpu_build_type "release")
endif()

target_compile_definitions(
    wb_benchmark PRIVATE
    "BENCH_CPU_RELEASE_TYPE=\"${_cpu_build_type}\""
)
