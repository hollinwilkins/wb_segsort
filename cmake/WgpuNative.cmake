# download and make wgpu-native library available

if (NOT DEFINED SCRIBLZ_WGPU_NATIVE_VERSION)
    set(SCRIBLZ_WGPU_NATIVE_VERSION "v29.0.1.1" CACHE STRING "wgpu-native version from https://github.com/gfx-rs/wgpu-native/releases")
endif()

if (APPLE)
    set(_wgpu_native_lib_file "lib/libwgpu_native.a")
    set(_wgpu_native_include_dir "include")
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(_wgpu_native_platform "macos-aarch64")
    else()
        set(_wgpu_native_platform "macos-x86_64")
    endif()
elseif (WIN32)
    if (MINGW)
        set(_wgpu_native_lib_file "lib/libwgpu_native.a")
        set(_wgpu_native_include_dir "include")
        set(_wgpu_native_platform "windows-x86_64-gnu")
    else()
        set(_wgpu_native_lib_file "lib/wgpu_native.lib")
        set(_wgpu_native_include_dir "include")
        set(_wgpu_native_platform "windows-x86_64-msvc")
    endif()
elseif (UNIX)
    set(_wgpu_native_lib_file "lib/libwgpu_native.a")
    set(_wgpu_native_include_dir "include")
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(_wgpu_native_platform "linux-aarch64")
    else()
        set(_wgpu_native_platform "linux-x86_64")
    endif()
else()
    message(FATAL_ERROR "[wgpu-native] Unsupported platform ${CMAKE_SYSTEM_NAME}")
endif()

if (DEFINED DEBUG)
    set(_wgpu_native_type "debug")
else()
    set(_wgpu_native_type "release")
endif()

set(_wgpu_native_url "https://github.com/gfx-rs/wgpu-native/releases/download/${SCRIBLZ_WGPU_NATIVE_VERSION}/wgpu-${_wgpu_native_platform}-${_wgpu_native_type}.zip")

include(FetchContent)
FetchContent_Declare(
    wgpu_native
    URL "${_wgpu_native_url}"
)

FetchContent_MakeAvailable(wgpu_native)

add_library(wgpu_native UNKNOWN IMPORTED GLOBAL)
set_target_properties(
    wgpu_native PROPERTIES
    IMPORTED_LOCATION "${wgpu_native_SOURCE_DIR}/${_wgpu_native_lib_file}"
)
target_include_directories(
    wgpu_native SYSTEM INTERFACE
    "${wgpu_native_SOURCE_DIR}/${_wgpu_native_include_dir}"
)

add_library(wgpu_headers INTERFACE)
target_include_directories(
    wgpu_headers SYSTEM INTERFACE
    "${wgpu_native_SOURCE_DIR}/${_wgpu_native_include_dir}"
)

if (APPLE)
    # transitively link required frameworks
    set_property(
        TARGET wgpu_native
        APPEND PROPERTY INTERFACE_LINK_LIBRARIES
        "-framework QuartzCore"
        "-framework CoreFoundation"
        "-framework Metal"
        "-framework IOKit"
        "-framework IOSurface"
    )
elseif (WIN32)
    # transitively link required DLLs
    set_property(
        TARGET wgpu_native
        APPEND PROPERTY INTERFACE_LINK_LIBRARIES
        ws2_32
        d3d11
        d3d12
        dxgi
        dxguid
        ntdll
        bcrypt
    )
elseif (UNIX)
    set_property(
        TARGET wgpu_native
        APPEND PROPERTY INTERFACE_LINK_LIBRARIES
        dl
        m
        pthread
    )
endif()

# define for hw_gutil.h to use wgpu-native as the WebGPU backend
target_compile_definitions(
    wgpu_native
    INTERFACE HWGUTIL_WEBGPU_BACKEND_WGPU
)
