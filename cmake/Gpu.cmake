cmake_minimum_required(VERSION 3.16)
project(wb_sort_gpu C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(
    wb_sort_gpu OBJECT
    gpu.h
)
set_source_files_properties(gpu.h PROPERTIES LANGUAGE C)

target_link_libraries(wb_sort_gpu PRIVATE webgpu_backend)

target_compile_definitions(
    wb_sort_gpu PRIVATE

    WB_SORT_GPU_IMPLEMENTATION
    HWDS_IMPLEMENTATION
    HWDS_MEMS_ENABLED
)

add_library(wb_sort_gpu_headers INTERFACE)
target_include_directories(
    wb_sort_gpu_headers INTERFACE
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/dependencies/hw"
)

target_link_libraries(wb_sort_gpu PRIVATE wb_sort_gpu_headers)

find_program(XXD_EXECUTABLE xxd REQUIRED)

set(MSG_SHADER_GEN_DIR "${CMAKE_SOURCE_DIR}/shaders")
file(MAKE_DIRECTORY "${MSG_SHADER_GEN_DIR}")

function(msg_embed_shader SHADER OUT_VAR)
    get_filename_component(_dir  "${SHADER}" DIRECTORY)
    get_filename_component(_name "${SHADER}" NAME)
    set(_out "${MSG_SHADER_GEN_DIR}/${_name}.h")
    add_custom_command(
        OUTPUT  "${_out}"
        COMMAND sh -c "'${XXD_EXECUTABLE}' -i '${_name}' > '${_out}'"
        DEPENDS "${SHADER}"
        WORKING_DIRECTORY "${_dir}"
        COMMENT "Embedding shader ${_name}"
        VERBATIM
    )
    set(${OUT_VAR} "${_out}" PARENT_SCOPE)
endfunction()

msg_embed_shader("${CMAKE_SOURCE_DIR}/shaders/wb_bin.wgsl" WB_BIN_HEADER)
msg_embed_shader("${CMAKE_SOURCE_DIR}/shaders/wbm_schedule.wgsl" WBM_SCHEDULE_HEADER)
msg_embed_shader("${CMAKE_SOURCE_DIR}/shaders/wbm_merge.wgsl" WBM_MERGE_HEADER)
msg_embed_shader("${CMAKE_SOURCE_DIR}/shaders/sort_kernels/segsort_tile_n2048_m256.wgsl" WBM_SORT_HEADER)

target_sources(wb_sort_gpu PRIVATE "${WB_BIN_HEADER}")
target_sources(wb_sort_gpu PRIVATE "${WBM_SCHEDULE_HEADER}")
target_sources(wb_sort_gpu PRIVATE "${WBM_MERGE_HEADER}")
target_sources(wb_sort_gpu PRIVATE "${WBM_SORT_HEADER}")
