cmake_minimum_required(VERSION 3.16)
project(merge_sort_gpu C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(
    merge_sort_gpu OBJECT
    gpu.h
)
set_source_files_properties(gpu.h PROPERTIES LANGUAGE C)

target_link_libraries(merge_sort_gpu PRIVATE wgpu_native)

target_compile_definitions(
    merge_sort_gpu PRIVATE

    MERGE_SORT_GPU_IMPLEMENTATION
    HWDS_IMPLEMENTATION
    HWDS_MEMS_ENABLED
)

add_library(merge_sort_gpu_headers INTERFACE)
target_include_directories(
    merge_sort_gpu_headers INTERFACE
    "${CMAKE_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/dependencies/hw"
)

target_link_libraries(merge_sort_gpu PRIVATE merge_sort_gpu_headers)

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

msg_embed_shader("${CMAKE_SOURCE_DIR}/shaders/merge_bin.wgsl" MERGE_BIN_HEADER)

target_sources(merge_sort_gpu PRIVATE "${MERGE_BIN_HEADER}")