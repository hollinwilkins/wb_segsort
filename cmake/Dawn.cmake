if (NOT DEFINED SCRIBLZ_DAWN_VERSION)
    set(SCRIBLZ_DAWN_VERSION "v20260828.215121"
        CACHE STRING "Dawn release tag from https://github.com/google/dawn/releases")
endif()

if (NOT DEFINED SCRIBLZ_DAWN_COMMIT)
    set(SCRIBLZ_DAWN_COMMIT "bddf1a04f7c262107a9aae301c45fc49e15c7fef"
        CACHE STRING "Dawn source commit embedded in the release asset filename")
endif()

if (NOT DEFINED SCRIBLZ_DAWN_CONFIG)
    set(SCRIBLZ_DAWN_CONFIG "Release"
        CACHE STRING "Dawn prebuilt config: Release or Debug")
endif()
set(_dawn_config "${SCRIBLZ_DAWN_CONFIG}")

if (APPLE)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(_dawn_platform "macos-latest")
    else()
        set(_dawn_platform "macos-15-intel")
    endif()
    set(_dawn_asset "Dawn-${SCRIBLZ_DAWN_COMMIT}-${_dawn_platform}-${_dawn_config}.tar.gz")
else()
    message(FATAL_ERROR
        "[dawn] Dawn.cmake currently only wires up macOS prebuilts. "
        "Add the appropriate asset name for ${CMAKE_SYSTEM_NAME} from "
        "https://github.com/google/dawn/releases")
endif()

set(_dawn_url
    "https://github.com/google/dawn/releases/download/${SCRIBLZ_DAWN_VERSION}/${_dawn_asset}")

message(STATUS "[dawn] fetching prebuilt: ${_dawn_url}")

include(FetchContent)
FetchContent_Declare(
    dawn_dist
    URL "${_dawn_url}"
)
FetchContent_MakeAvailable(dawn_dist)

file(GLOB_RECURSE _dawn_config_files
    "${dawn_dist_SOURCE_DIR}/*DawnConfig.cmake")
if (NOT _dawn_config_files)
    message(FATAL_ERROR
        "[dawn] Could not find DawnConfig.cmake under ${dawn_dist_SOURCE_DIR}. "
        "Inspect the extracted tree and adjust Dawn.cmake (library name / layout "
        "may have changed).")
endif()
list(GET _dawn_config_files 0 _dawn_config_file)
get_filename_component(_dawn_config_dir "${_dawn_config_file}" DIRECTORY)

find_package(Threads REQUIRED)

find_package(Dawn REQUIRED
    PATHS "${_dawn_config_dir}"
    NO_DEFAULT_PATH
)

add_library(dawn INTERFACE)
target_link_libraries(dawn INTERFACE dawn::webgpu_dawn)

if (APPLE)
    target_link_libraries(dawn INTERFACE c++)
else()
    target_link_libraries(dawn INTERFACE stdc++)
endif()

target_compile_definitions(dawn INTERFACE HWGUTIL_WEBGPU_BACKEND_DAWN)

add_library(dawn_headers INTERFACE)
get_target_property(_dawn_inc dawn::webgpu_dawn INTERFACE_INCLUDE_DIRECTORIES)
if (_dawn_inc)
    target_include_directories(dawn_headers SYSTEM INTERFACE ${_dawn_inc})
endif()

function(dawn_copy_binaries _target)
    get_target_property(_dawn_type dawn::webgpu_dawn TYPE)
    if (_dawn_type STREQUAL "SHARED_LIBRARY")
        add_custom_command(TARGET ${_target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:dawn::webgpu_dawn>
                $<TARGET_FILE_DIR:${_target}>
            COMMENT "[dawn] copying $<TARGET_FILE:dawn::webgpu_dawn> next to ${_target}")
    endif()
endfunction()
