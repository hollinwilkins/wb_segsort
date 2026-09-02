cmake_minimum_required(VERSION 3.16)
project(wb_sort C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_EXPORT_COMPILE_COMMANDS on)

option(USE_DAWN "Use Dawn instead of wgpu-native" ON)

if (USE_DAWN)
    include(Dawn)
    add_library(webgpu_backend ALIAS dawn)
else()
    include(WgpuNative)
    add_library(webgpu_backend ALIAS wgpu_native)
endif()

include(Cpu)
include(Gpu)
include(Benchmark)
include(BenchmarkKernel)
include(BenchmarkThroughput)

enable_testing()

add_subdirectory("dependencies/unity")
add_subdirectory("tests")
