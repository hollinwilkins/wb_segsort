cmake_minimum_required(VERSION 3.16)
project(merge_sort C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_EXPORT_COMPILE_COMMANDS on)

include(WgpuNative)
include(Cpu)
include(Gpu)
include(Main)

enable_testing()

add_subdirectory("dependencies/unity")
add_subdirectory("tests")
