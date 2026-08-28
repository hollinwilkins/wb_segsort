cmake_minimum_required(VERSION 3.16)
project(merge_sort_gpu C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(
    merge_sort_gpu OBJECT
    cpu.h
)
set_source_files_properties(cpu.h PROPERTIES LANGUAGE C)

target_compile_definitions(
    merge_sort_gpu PRIVATE

    MERGE_SORT_GPU_IMPLEMENTATION
)

add_library(merge_sort_gpu_headers INTERFACE)
target_include_directories(
    merge_sort_gpu_headers INTERFACE
    "${CMAKE_SOURCE_DIR}"
)
