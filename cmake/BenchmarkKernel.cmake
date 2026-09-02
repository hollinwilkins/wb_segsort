cmake_minimum_required(VERSION 3.16)
project(wb_benchmark_kernel C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(
    wb_benchmark_kernel
    benchmark_kernel.c
)

target_link_libraries(wb_benchmark_kernel PRIVATE webgpu_backend)
target_link_libraries(wb_benchmark_kernel PRIVATE wb_sort_cpu_headers)
target_link_libraries(wb_benchmark_kernel PRIVATE wb_sort_gpu_headers)

if (DEFINED DEBUG)
    set(_cpu_build_type "debug")
else()
    set(_cpu_build_type "release")
endif()

# Capture the source revision at configure time so a benchmark run can be tied
# to the exact code that produced it. Note: this refreshes on (re)configure,
# not on every build, so re-run cmake to pick up new commits.
find_program(GIT_EXECUTABLE git)
set(_bench_git_commit "unknown")
if (GIT_EXECUTABLE)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE _git_hash
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if (_git_hash)
        set(_bench_git_commit "${_git_hash}")

        execute_process(
            COMMAND ${GIT_EXECUTABLE} status --porcelain --untracked-files=no
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE _git_dirty
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if (_git_dirty)
            set(_bench_git_commit "${_bench_git_commit}-dirty")
        endif()
    endif()
endif()
message(STATUS "[benchmark] git commit: ${_bench_git_commit}")

target_compile_definitions(
    wb_benchmark_kernel PRIVATE
    "BENCH_CPU_RELEASE_TYPE=\"${_cpu_build_type}\""
    "BENCH_GIT_COMMIT=\"${_bench_git_commit}\""
)
