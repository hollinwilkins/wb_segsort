// Copyright (C) 2026 Hollin Wilkins
// SPDX-License-Identifier: Zlib

#ifndef HWGUTIL_H
#define HWGUTIL_H

/*

hw_gutil.h v0.1.0
by Hollin Wilkins

SECURITY
    This library provides NO SECURITY GUARANTEES whatsoever.

WHAT DOES THIS LIBRARY DO?
    Graphics utilities library. Helps with integrating GLFW and WebGPU,
    as well as get HDR range information.

COMPILING & LINKING
    This library requires C99, though it may be refactored to support C89 in the future.

    In one C/C++ file that includes this file, do this:
        #define HWGUTIL_IMPLEMENTATION
    before the include, that will create the implementation in that file.

    If you also do this:
        #define HWGUTIL_STATIC
    then all of the functions will be declared as static.

CONFIGURATION
    HWGUTIL_IMPLEMENTATION          - define in one .c file to compile the implementation
    HWGUTIL_STATIC                  - make all symbols static
    HWGUTIL_GLFW3WEBGPU_ENABLED     - enable GLFW3 + WebGPU surface creation
    HWGUTIL_WEBGPU_ENABLED          - enable WebGPU only (auto-set by HWGUTIL_GLFW3WEBGPU_ENABLED)
    HWGUTIL_GLFW3_ENABLED           - enable GLFW3 only (auto-set by HWGUTIL_GLFW3WEBGPU_ENABLED)
*/

#ifndef HWGUTIL_NO_CRT
#   include <stddef.h>
#   include <stdbool.h>
#   include <stdio.h>
#   include <stdint.h>
#endif

#ifndef HWGUTIL_NO_CRT
#   ifndef hwgutil_size
#       define hwgutil_size size_t
#   endif
#   ifndef hwgutil_u32
#       define hwgutil_u32 uint32_t
#   endif
#   ifndef hwgutil_bool
#       define hwgutil_bool bool
#   endif
#   ifndef hwgutil_true
#       define hwgutil_true true
#   endif
#   ifndef hwgutil_false
#       define hwgutil_false false
#   endif
#   ifndef HWGUTIL_NULL
#      define HWGUTIL_NULL NULL
#   endif
#else
#   ifndef hwgutil_size
#      if defined(__SIZE_TYPE__)
#          define hwgutil_size __SIZE_TYPE__
#      elif defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || \
       defined(__LP64__) || defined(_LP64) || (defined(__WORDSIZE) && __WORDSIZE == 64)
#          define hwgutil_size unsigned long long
#      elif defined(__i386__) || defined(_M_IX86) || defined(__arm__) || \
       defined(__ILP32__) || (defined(__WORDSIZE) && __WORDSIZE == 32)
#          define hwgutil_size unsigned int
#      else
#          define hwgutil_size unsigned long
#      endif
#   endif
#   ifndef hwgutil_u32
#       define hwgutil_u32 unsigned
#   endif
#   ifndef hwgutil_bool
#      define hwgutil_bool int
#   endif
#   ifndef hwgutil_true
#      define hwgutil_true 1
#   endif
#   ifndef hwgutil_false
#      define hwgutil_false 0
#   endif
#   ifndef HWGUTIL_NULL
#      define HWGUTIL_NULL 0
#   endif
#endif

#ifndef HWGUTIL_EXPORT
#   ifdef HWGUTIL_STATIC
#       define HWGUTIL_EXPORT static
#   else
#       ifdef __cplusplus
#           define HWGUTIL_EXPORT extern "C"
#       else
#           define HWGUTIL_EXPORT extern
#       endif
#   endif
#endif

#ifdef _MSC_VER
#   define HWGUTIL_INLINE static __forceinline
#   define HWGUTIL_NOINLINE static __declspec(noinline)
#   define HWGUTIL_ALIGNOF __alignof
#else
#   ifdef __has_attribute
#       if __has_attribute(always_inline)
#           define HWGUTIL_INLINE static inline __attribute__((always_inline))
#       endif
#       if __has_attribute(noinline)
#           define HWGUTIL_NOINLINE static __attribute__((noinline))
#       endif
#   endif
#   define HWGUTIL_ALIGNOF __alignof__
#endif

#ifndef HWGUTIL_INLINE
#  define HWGUTIL_INLINE static inline
#endif

#ifndef HWGUTIL_USE
#   define HWGUTIL_USE(value) ((void)value)
#endif

#ifndef HWGUTIL_NO_CRT
#   ifndef HWGUTIL_ABORT
#       define HWGUTIL_ABORT() abort()
#   endif
#   ifndef HWGUTIL_LOG_ERROR
#       define HWGUTIL_LOG_ERROR(...) fprintf(stderr, __VA_ARGS__)
#   endif
#else
#endif

#ifdef HWGUTIL_MEMS_ENABLED
#   include "hw_mems.h"
#endif

#ifdef HWGUTIL_GLFW3WEBGPU_ENABLED

#ifndef HWGUTIL_WEBGPU_ENABLED
#   define HWGUTIL_WEBGPU_ENABLED
#endif

#ifndef HWGUTIL_GLFW3_ENABLED
#   define HWGUTIL_GLFW3_ENABLED
#endif

#include <GLFW/glfw3.h>

#ifdef __EMSCRIPTEN__
#   include <GLFW/emscripten_glfw3.h>
#else
#   include <GLFW/glfw3native.h>
#endif

#ifdef GLFW_EXPOSE_NATIVE_COCOA
#  include <Foundation/Foundation.h>
#  include <QuartzCore/CAMetalLayer.h>
#endif

#endif

#ifdef HWGUTIL_WEBGPU_ENABLED
#   include <webgpu/webgpu.h>
#   ifdef HWGUTIL_WEBGPU_BACKEND_WGPU
#       include <webgpu/wgpu.h>
#   endif
#endif

#define HWGUTIL_SDR_NITS 203.0f
#define HWGUTIL_PQ_PEAK_NITS 10000.0f
#define HWGUTIL_DEFAULT_SDR_REFERENCE_NITS 100.0f

typedef hwgutil_u32 hwgutil_color_space;
typedef enum hwgutil_color_space_enum
{
    hwgutil_color_space_linear_srgba = 0,
    hwgutil_color_space_srgba = 1,
    hwgutil_color_space_extended_linear_srgba = 2,
    hwgutil_color_space_extended_srgba = 3,
} hwgutil_color_space_enum;

HWGUTIL_EXPORT hwgutil_bool hwgutil_color_space_is_extended(hwgutil_color_space color_space);

#ifdef __cplusplus
extern "C" {
#endif


#ifdef HWGUTIL_WEBGPU_ENABLED

typedef struct hwgutil_wgpu_context
{
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
} hwgutil_wgpu_context;

typedef struct hwgutil_wgpu_blit_bindings
{
    WGPURenderPipeline pipeline;
    WGPUBindGroup ubo_binding;
    WGPUBindGroup src_texture_binding;
    WGPUBuffer ubo;
    WGPUBuffer instances;
} hwgutil_wgpu_blit_bindings;

typedef struct hwgutil_wgpu_blit_instance
{
    float src_bounds[4];
    float dst_bounds[4];
    uint32_t flags;
} hwgutil_wgpu_blit_instance;

HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_init(
    WGPUInstance * instance,
    WGPUAdapter * adapter,
    WGPUDevice * device,
    WGPUQueue * queue
);

HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_context_init(hwgutil_wgpu_context * context);
HWGUTIL_EXPORT void hwgutil_wgpu_context_add_ref(hwgutil_wgpu_context * context);
HWGUTIL_EXPORT void hwgutil_wgpu_context_release(hwgutil_wgpu_context * context);
HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_read_texture(
    WGPUInstance instance,
    WGPUDevice device,
    WGPUQueue queue,
    WGPUTexture texture,
    WGPUExtent3D extent,
    hwgutil_size bytes_per_pixel,
    hwgutil_size buffer_len,
    void * buffer,
    hwgutil_size * required_size
);
HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_write_texture(
    WGPUQueue queue,
    WGPUTexture texture,
    WGPUOrigin3D origin,
    WGPUExtent3D extent,
    hwgutil_size bytes_per_pixel,
    const void * pixels,
    hwgutil_size buffer_len,
    void * buffer,
    hwgutil_size * required_size
);

HWGUTIL_EXPORT WGPURenderPipeline hwgutil_wgpu_create_blit_pipeline(
    WGPUDevice device,
    WGPUTextureFormat format
);

HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_blit_bindings_init(
    hwgutil_wgpu_blit_bindings * bindings,
    WGPUDevice device,
    WGPURenderPipeline pipeline,
    hwgutil_size capacity,
    WGPUTextureView src_view
);

HWGUTIL_EXPORT void hwgutil_wgpu_blit_bindings_deinit(hwgutil_wgpu_blit_bindings * bindings);

HWGUTIL_EXPORT void hwgutil_wgpu_blit(
    const hwgutil_wgpu_blit_bindings * bindings,
    WGPUQueue queue,
    WGPURenderPassEncoder render_pass,
    hwgutil_u32 width, hwgutil_u32 height,
    hwgutil_size instances_count,
    hwgutil_wgpu_blit_instance * instances
);

#ifdef HWGUTIL_MEMS_ENABLED

HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_read_texture_alloc(
    WGPUInstance instance,
    WGPUDevice device,
    WGPUQueue queue,
    WGPUTexture texture,
    WGPUExtent3D extent,
    hwgutil_size bytes_per_pixel,
    const mems_allocator * allocator,
    void ** buffer
);

HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_write_texture_alloc(
    WGPUQueue queue,
    WGPUTexture texture,
    WGPUOrigin3D origin,
    WGPUExtent3D extent,
    hwgutil_size bytes_per_pixel,
    const void * pixels,
    const mems_allocator * allocator
);

#endif // HWGUTIL_MEMS_ENABLED

#endif // HWGUTIL_WEBGPU_ENABLED

#ifdef HWGUTIL_GLFW3_ENABLED

#include <GLFW/glfw3.h>

HWGUTIL_EXPORT hwgutil_bool hwgutil_glfw3_get_hdr_component_max(
    GLFWwindow * window,
    float * hdr_component_max
);
HWGUTIL_EXPORT hwgutil_bool hwgutil_glfw3_get_hdr_nits(
    GLFWwindow * window,
    float * sdr_reference_nits,
    float * hdr_peak_nits
);
void hwgutil_glfw3_window_configure_color_space(GLFWwindow * window, hwgutil_color_space color_space);

#endif // HWGUTIL_WEBGPU_ENABLED

#ifdef HWGUTIL_GLFW3WEBGPU_ENABLED

HWGUTIL_EXPORT WGPUSurface hwgutil_glfw3wgpu_create_surface(WGPUInstance instance, GLFWwindow * window);

#endif // HWGUTIL_GLFW3WEBGPU_ENABLED

#endif

#ifdef __cplusplus
}
#endif

#ifdef HWGUTIL_IMPLEMENTATION
#ifndef HWGUTIL_IMPLEMENTED
#define HWGUTIL_IMPLEMENTED

HWGUTIL_EXPORT hwgutil_bool hwgutil_color_space_is_extended(const hwgutil_color_space color_space)
{
    switch (color_space)
    {
        case hwgutil_color_space_extended_linear_srgba:
        case hwgutil_color_space_extended_srgba: return hwgutil_true;
        default: return hwgutil_false;
    }
}

#ifdef HWGUTIL_WEBGPU_ENABLED

#include "hw_gutil_webgpu_blit.h"

typedef struct hwgutil__request_adapter_response
{
    WGPURequestAdapterStatus status;
    WGPUAdapter adapter;
} hwgutil__request_adapter_response;

static void hwgutil__init_on_request_adapter(
    const WGPURequestAdapterStatus status,
    WGPUAdapter const adapter,
    const WGPUStringView message,
    void * const userdata1,
    void * const userdata2
)
{
    HWGUTIL_USE(userdata2);

    hwgutil__request_adapter_response * const response = (hwgutil__request_adapter_response *)userdata1;
    response->status = status;
    response->adapter = adapter;

    if (message.length > 0) HWGUTIL_LOG_ERROR("On Request Adapter: %.*s\n", (int)message.length, message.data);
}

typedef struct hwgutil__request_device_response
{
    WGPURequestDeviceStatus status;
    WGPUDevice device;
} hwgutil__request_device_response;

static void hwgutil__init_on_request_device(
    const WGPURequestDeviceStatus status,
    WGPUDevice const device,
    const WGPUStringView message,
    void * const userdata1,
    void * const userdata2
)
{
    HWGUTIL_USE(device);
    HWGUTIL_USE(userdata2);

    hwgutil__request_device_response * const response = (hwgutil__request_device_response *)userdata1;
    response->status = status;
    response->device = device;

    if (message.length > 0) HWGUTIL_LOG_ERROR("On Request Device: %.*s\n", (int)message.length, message.data);
}

static void hwgutil__on_unhandled_error(
    WGPUDevice const * const device,
    const WGPUErrorType error_type,
    const WGPUStringView message,
    void * const userdata1,
    void * const userdata2
)
{
    HWGUTIL_USE(device);
    HWGUTIL_USE(userdata1);
    HWGUTIL_USE(userdata2);

    const char * error_type_name = HWGUTIL_NULL;

    switch (error_type)
    {
        case WGPUErrorType_Internal:
        {
            error_type_name = "internal";
            break;
        }
        case WGPUErrorType_NoError:
        {
            error_type_name = "no_error";
            break;
        }
        case WGPUErrorType_OutOfMemory:
        {
            error_type_name = "out_of_memory";
            break;
        }
        case WGPUErrorType_Validation:
        {
            error_type_name = "validation";
            break;
        }
        case WGPUErrorType_Unknown:
        {
            error_type_name = "unknown";
            break;
        }
        default:
        {
            error_type_name = "(unknown)";
            break;
        }
    }

    HWGUTIL_LOG_ERROR("Uncaptured WebGPU error (%s): %.*s\n", error_type_name, (int)message.length, message.data);
}

#ifdef __EMSCRIPTEN__
#   define HWGUTIL_CALLBACK_MODE WGPUCallbackMode_WaitAnyOnly
#   define HWGUTIL_CALLBACK_WAIT
#else
#   define HWGUTIL_CALLBACK_MODE WGPUCallbackMode_AllowSpontaneous
#endif

HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_init(
    WGPUInstance * const instance,
    WGPUAdapter * const adapter,
    WGPUDevice * const device,
    WGPUQueue * const queue
)
{
    WGPUInstanceDescriptor instance_descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    WGPUInstanceFeatureName instance_features[] = { WGPUInstanceFeatureName_TimedWaitAny };

    instance_descriptor.requiredFeatureCount = 1;
    instance_descriptor.requiredFeatures = instance_features;

    *instance = wgpuCreateInstance(&instance_descriptor);
    if (*instance == HWGUTIL_NULL) return hwgutil_false;

    WGPURequestAdapterOptions adapter_options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    adapter_options.powerPreference = WGPUPowerPreference_HighPerformance;
    adapter_options.forceFallbackAdapter = WGPU_FALSE;

    hwgutil__request_adapter_response adapter_response = (hwgutil__request_adapter_response){
        .status = WGPURequestAdapterStatus_Error,
    };
    WGPURequestAdapterCallbackInfo request_adapter_cb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    request_adapter_cb.mode = HWGUTIL_CALLBACK_MODE;
    request_adapter_cb.callback = hwgutil__init_on_request_adapter;
    request_adapter_cb.userdata1 = &adapter_response;

#ifdef HWGUTIL_CALLBACK_WAIT
    WGPUFutureWaitInfo future_info = WGPU_FUTURE_WAIT_INFO_INIT;
    WGPUFuture future = wgpuInstanceRequestAdapter(*instance, &adapter_options, request_adapter_cb);
    future_info.future = future;
    WGPUWaitStatus future_status = wgpuInstanceWaitAny(*instance, 1, &future_info, (uint64_t)5 * 1000000000);

    if (future_status != WGPUWaitStatus_Success) return hwgutil_false;
#else
    wgpuInstanceRequestAdapter(*instance, &adapter_options, request_adapter_cb);
#endif

    if (adapter_response.status != WGPURequestAdapterStatus_Success) return hwgutil_false;
    if (adapter_response.adapter == HWGUTIL_NULL) return hwgutil_false;
    *adapter = adapter_response.adapter;

    WGPUDeviceDescriptor device_descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    device_descriptor.uncapturedErrorCallbackInfo.callback = hwgutil__on_unhandled_error;

#ifdef HWGUTIL_WEBGPU_BACKEND_DAWN

    const char* const dawn_toggles[] = { "use_user_defined_labels_in_backend" };
    WGPUDawnTogglesDescriptor toggles = WGPU_DAWN_TOGGLES_DESCRIPTOR_INIT;
    toggles.chain.sType = WGPUSType_DawnTogglesDescriptor;
    toggles.enabledToggleCount = 1;
    toggles.enabledToggles = dawn_toggles;
    device_descriptor.nextInChain = &toggles.chain;

#endif

    hwgutil__request_device_response device_response = (hwgutil__request_device_response){
        .status = WGPURequestDeviceStatus_Error,
    };
    WGPURequestDeviceCallbackInfo request_device_cb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    request_device_cb.mode = HWGUTIL_CALLBACK_MODE;
    request_device_cb.callback = hwgutil__init_on_request_device;
    request_device_cb.userdata1 = &device_response;

#ifdef HWGUTIL_CALLBACK_WAIT
   future_info = WGPU_FUTURE_WAIT_INFO_INIT;
   future = wgpuAdapterRequestDevice(*adapter, &device_descriptor, request_device_cb);
   future_info.future = future;
   future_status = wgpuInstanceWaitAny(*instance, 1, &future_info, (uint64_t)5 * 1000000000);

    if (future_status != WGPUWaitStatus_Success) return hwgutil_false;
#else
    wgpuAdapterRequestDevice(*adapter, &device_descriptor, request_device_cb);
#endif

    if (device_response.status != WGPURequestDeviceStatus_Success) return hwgutil_false;
    if (device_response.device == HWGUTIL_NULL) return hwgutil_false;
    *device = device_response.device;

    *queue = wgpuDeviceGetQueue(*device);
    if (*queue == HWGUTIL_NULL) return hwgutil_false;

    return hwgutil_true;
}

HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_context_init(hwgutil_wgpu_context * const context)
{
    return hwgutil_wgpu_init(
        &context->instance,
        &context->adapter,
        &context->device,
        &context->queue
    );
}

HWGUTIL_EXPORT void hwgutil_wgpu_context_add_ref(hwgutil_wgpu_context * const context)
{
    wgpuInstanceAddRef(context->instance);
    wgpuAdapterAddRef(context->adapter);
    wgpuDeviceAddRef(context->device);
    wgpuQueueAddRef(context->queue);
}

HWGUTIL_EXPORT void hwgutil_wgpu_context_release(hwgutil_wgpu_context * const context)
{
    wgpuInstanceRelease(context->instance);
    wgpuAdapterRelease(context->adapter);
    wgpuDeviceRelease(context->device);
    wgpuQueueRelease(context->queue);
}

static void hwgutil__map_buffer_cb(
    WGPUMapAsyncStatus status,
    WGPUStringView message,
    WGPU_NULLABLE void * userdata1,
    WGPU_NULLABLE void * userdata2
)
{
    HWGUTIL_USE(status);
    HWGUTIL_USE(message);
    HWGUTIL_USE(userdata2);
#ifdef HWGUTIL_WEBGPU_BACKEND_WGPU
    if (userdata1 != HWGUTIL_NULL) *(hwgutil_bool *)userdata1 = hwgutil_true;
#else
    HWGUTIL_USE(userdata1);
#endif
}

HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_read_texture(
    WGPUInstance const instance,
    WGPUDevice const device,
    WGPUQueue const queue,
    WGPUTexture const texture,
    const WGPUExtent3D extent,
    const hwgutil_size bytes_per_pixel,
    const hwgutil_size buffer_len,
    void * const buffer,
    hwgutil_size * const required_size
)
{
    const hwgutil_size bytes_per_row_unaligned = bytes_per_pixel * extent.width;
    const hwgutil_size bytes_per_row = (bytes_per_row_unaligned + 255) & ~(hwgutil_size)255;
    const hwgutil_size read_size = bytes_per_pixel * extent.width * extent.height;
    const hwgutil_size gpu_buffer_size = bytes_per_row * extent.height;

    *required_size = read_size;

    if (buffer_len < *required_size) return false;

    WGPUBufferDescriptor read_buffer_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    read_buffer_descriptor.label = (WGPUStringView){
        .data = "HWGutil Texture Read Buffer",
        .length = WGPU_STRLEN,
    };
    read_buffer_descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    read_buffer_descriptor.size = (uint64_t)gpu_buffer_size;

    WGPUBuffer read_buffer = wgpuDeviceCreateBuffer(device, &read_buffer_descriptor);

    WGPUCommandEncoderDescriptor ce_desc = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    ce_desc.label = (WGPUStringView){
        .data = "HWGutil Texture Read Command Encoder",
        .length = WGPU_STRLEN,
    };

    WGPUCommandEncoder ce = wgpuDeviceCreateCommandEncoder(device, &ce_desc);

    WGPUTexelCopyTextureInfo texture_info = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    texture_info.texture = texture;
    texture_info.origin = WGPU_ORIGIN_3D_INIT;

    WGPUTexelCopyBufferInfo buffer_info = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    buffer_info.buffer = read_buffer;
    buffer_info.layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    buffer_info.layout.bytesPerRow = bytes_per_row;
    buffer_info.layout.rowsPerImage = extent.height;

    wgpuCommandEncoderCopyTextureToBuffer(
        ce,
        &texture_info,
        &buffer_info,
        &extent
    );

    WGPUCommandBufferDescriptor commands_desc = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    commands_desc.label = (WGPUStringView){
        .data = "HWGutil Texture Read Commands",
        .length = WGPU_STRLEN,
    };

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(ce, &commands_desc);
    wgpuQueueSubmit(queue, 1, &commands);

    WGPUBufferMapCallbackInfo map_cb_info = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    map_cb_info.mode = HWGUTIL_CALLBACK_MODE;
    map_cb_info.callback = hwgutil__map_buffer_cb;

#ifdef HWGUTIL_WEBGPU_BACKEND_WGPU
    HWGUTIL_USE(instance);
    hwgutil_bool map_completed = hwgutil_false;
    map_cb_info.userdata1 = &map_completed;
#endif

    WGPUFuture future = wgpuBufferMapAsync(
        read_buffer,
        WGPUMapMode_Read,
        0,
        gpu_buffer_size,
        map_cb_info
    );

#ifdef HWGUTIL_WEBGPU_BACKEND_WGPU
    while (!map_completed) wgpuDevicePoll(device, WGPU_TRUE, HWGUTIL_NULL);
#else
    WGPUFutureWaitInfo future_info = WGPU_FUTURE_WAIT_INFO_INIT;
    future_info.future = future;
    const WGPUWaitStatus status = wgpuInstanceWaitAny(instance, 1, &future_info, (uint64_t)5 * 1000000000);
    if (status != WGPUWaitStatus_Success) return false;
#endif

    HWGUTIL_USE(future);

    const void * const mapped_read_buffer = wgpuBufferGetConstMappedRange(read_buffer, 0, gpu_buffer_size);
    for (hwgutil_size row = 0; row < extent.height; row++) {
        memcpy(
            (uint8_t *)buffer + row * bytes_per_row_unaligned,
            (const uint8_t *)mapped_read_buffer + row * bytes_per_row,
            bytes_per_row_unaligned
        );
    }
    wgpuBufferUnmap(read_buffer);

    wgpuCommandBufferRelease(commands);
    wgpuBufferRelease(read_buffer);
    wgpuCommandEncoderRelease(ce);

    return true;
}

HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_write_texture(
    WGPUQueue const queue,
    const WGPUTexture texture,
    const WGPUOrigin3D origin,
    const WGPUExtent3D extent,
    const hwgutil_size bytes_per_pixel,
    const void * pixels,
    const hwgutil_size buffer_len,
    void * const buffer,
    hwgutil_size * const required_size
)
{
    const hwgutil_size bytes_per_row = extent.width * bytes_per_pixel;
    const hwgutil_size aligned_bytes_per_row = (bytes_per_row + 255u) & ~255u;
    const hwgutil_bool needs_padding = aligned_bytes_per_row != bytes_per_row;

    if (needs_padding)
    {
        *required_size = aligned_bytes_per_row * extent.height;
    }
    else
    {
        *required_size = 0;
    }

    if (buffer_len < *required_size) return hwgutil_false;

    if (needs_padding)
    {
        for (uint32_t i = 0; i < extent.height; i++)
        {
            memcpy(
                (uint8_t *)buffer + aligned_bytes_per_row * i,
                (uint8_t *)pixels + bytes_per_row * i,
                bytes_per_row
            );
        }

        pixels = buffer;
    }

    WGPUTexelCopyTextureInfo copy_info = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    copy_info.texture = texture;
    copy_info.origin = origin;

    WGPUTexelCopyBufferLayout buffer_layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
    buffer_layout.bytesPerRow = aligned_bytes_per_row;
    buffer_layout.rowsPerImage = extent.height;

    wgpuQueueWriteTexture(
        queue,
        &copy_info,
        pixels,
        aligned_bytes_per_row * extent.height,
        &buffer_layout,
        &extent
    );

    return hwgutil_true;
}

HWGUTIL_EXPORT WGPURenderPipeline hwgutil_wgpu_create_blit_pipeline(
    WGPUDevice const device,
    const WGPUTextureFormat format
)
{
    WGPUBindGroupLayoutDescriptor bind0_layout_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bind0_layout_desc.label = (WGPUStringView){
        .data = "HWGutil Blit UBO Binding Layout",
        .length = WGPU_STRLEN,
    };
    WGPUBindGroupLayoutEntry bind0_layout_desc_entry0 = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    bind0_layout_desc_entry0.binding = 0;
    bind0_layout_desc_entry0.visibility = WGPUShaderStage_Vertex;
    bind0_layout_desc_entry0.buffer = WGPU_BUFFER_BINDING_LAYOUT_INIT;
    bind0_layout_desc_entry0.buffer.type = WGPUBufferBindingType_Uniform;
    bind0_layout_desc.entryCount = 1;
    bind0_layout_desc.entries = (WGPUBindGroupLayoutEntry[]){
        bind0_layout_desc_entry0
    };

    WGPUBindGroupLayout bind0_layout = wgpuDeviceCreateBindGroupLayout(device, &bind0_layout_desc);

    WGPUBindGroupLayoutDescriptor bind1_layout_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bind1_layout_desc.label = (WGPUStringView){
        .data = "HWGutil Blit Source Texture Binding Layout",
        .length = WGPU_STRLEN,
    };
    WGPUBindGroupLayoutEntry bind1_layout_desc_entry0 = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    bind1_layout_desc_entry0.binding = 0;
    bind1_layout_desc_entry0.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    bind1_layout_desc_entry0.texture = WGPU_TEXTURE_BINDING_LAYOUT_INIT;
    bind1_layout_desc_entry0.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
    bind1_layout_desc_entry0.texture.viewDimension = WGPUTextureViewDimension_2D;
    WGPUBindGroupLayoutEntry bind1_layout_desc_entry1 = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    bind1_layout_desc_entry1.binding = 1;
    bind1_layout_desc_entry1.visibility = WGPUShaderStage_Fragment;
    bind1_layout_desc_entry1.sampler = WGPU_SAMPLER_BINDING_LAYOUT_INIT;
    bind1_layout_desc_entry1.sampler.type = WGPUSamplerBindingType_NonFiltering;
    bind1_layout_desc.entryCount = 2;
    bind1_layout_desc.entries = (WGPUBindGroupLayoutEntry[]){
        bind1_layout_desc_entry0,
        bind1_layout_desc_entry1
    };

    WGPUBindGroupLayout bind1_layout = wgpuDeviceCreateBindGroupLayout(device, &bind1_layout_desc);

    WGPUPipelineLayoutDescriptor layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    layout_desc.label = (WGPUStringView){
        .data = "HWGutil Blit Pipeline Layout",
        .length = WGPU_STRLEN,
    };
    layout_desc.bindGroupLayoutCount = 2;
    layout_desc.bindGroupLayouts = (WGPUBindGroupLayout[]){
        bind0_layout,
        bind1_layout
    };

    WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(device, &layout_desc);

    WGPUShaderSourceWGSL shader_source_wgsl = (WGPUShaderSourceWGSL){
        .chain = (WGPUChainedStruct){
            .next = HWGUTIL_NULL,
            .sType = WGPUSType_ShaderSourceWGSL
        },
        .code = (WGPUStringView){
            .data = (const char *)hw_gutil_webgpu_blit_wgsl,
            .length = hw_gutil_webgpu_blit_wgsl_len,
        },
    };
    WGPUShaderModuleDescriptor shader_module_desc = (WGPUShaderModuleDescriptor){
        .label = (WGPUStringView){
            .data = "HWGutil Blit Shader",
            .length = WGPU_STRLEN,
        },
        .nextInChain = (WGPUChainedStruct *)(&shader_source_wgsl),
    };
    WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(
        device,
        &shader_module_desc
    );

    WGPURenderPipelineDescriptor pipeline_desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    pipeline_desc.layout = layout;
    pipeline_desc.primitive = WGPU_PRIMITIVE_STATE_INIT;
    pipeline_desc.primitive.topology = WGPUPrimitiveTopology_TriangleStrip;
    pipeline_desc.vertex = WGPU_VERTEX_STATE_INIT;
    pipeline_desc.vertex.module = shader_module;
    pipeline_desc.vertex.entryPoint = (WGPUStringView){
        .data = "vs_main",
        .length = WGPU_STRLEN,
    };
    WGPUVertexBufferLayout vertex_layout = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
    vertex_layout.stepMode = WGPUVertexStepMode_Instance;
    vertex_layout.arrayStride = sizeof(hwgutil_wgpu_blit_instance);
    vertex_layout.attributeCount = 3;
    vertex_layout.attributes = (WGPUVertexAttribute[]){
        (WGPUVertexAttribute){
            .offset = 0,
            .shaderLocation = 0,
            .format = WGPUVertexFormat_Float32x4,
        },
        (WGPUVertexAttribute){
            .offset = sizeof(float[4]),
            .shaderLocation = 1,
            .format = WGPUVertexFormat_Float32x4,
        },
        (WGPUVertexAttribute){
            .offset = sizeof(float[4]) * 2,
            .shaderLocation = 2,
            .format = WGPUVertexFormat_Uint32,
        },
    };
    pipeline_desc.vertex.bufferCount = 1;
    pipeline_desc.vertex.buffers = (WGPUVertexBufferLayout[]){
        vertex_layout
    };

    WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
    fragment.module = shader_module;
    fragment.entryPoint = (WGPUStringView){
        .data = "fs_main",
        .length = WGPU_STRLEN,
    };
    WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
    target.format = format;
    target.blend = &(WGPUBlendState){
        .color = (WGPUBlendComponent){
            .operation = WGPUBlendOperation_Add,
            .srcFactor  = WGPUBlendFactor_SrcAlpha,
            .dstFactor  = WGPUBlendFactor_OneMinusSrcAlpha,
        },
        .alpha = (WGPUBlendComponent){
            .operation = WGPUBlendOperation_Add,
            .srcFactor  = WGPUBlendFactor_One,
            .dstFactor  = WGPUBlendFactor_OneMinusSrcAlpha,
        },
    };
    fragment.targetCount = 1;
    fragment.targets = (WGPUColorTargetState[]){
        target
    };
    pipeline_desc.fragment = &fragment;

    WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(device, &pipeline_desc);

    wgpuBindGroupLayoutRelease(bind0_layout);
    wgpuBindGroupLayoutRelease(bind1_layout);
    wgpuPipelineLayoutRelease(layout);
    wgpuShaderModuleRelease(shader_module);

    return pipeline;
}

HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_blit_bindings_init(
    hwgutil_wgpu_blit_bindings * const bindings,
    WGPUDevice const device,
    WGPURenderPipeline const pipeline,
    const hwgutil_size capacity,
    WGPUTextureView src_view
)
{
    WGPUBindGroupLayout ubo_layout = wgpuRenderPipelineGetBindGroupLayout(pipeline, 0);
    WGPUBindGroupLayout src_texture_layout = wgpuRenderPipelineGetBindGroupLayout(pipeline, 1);

    WGPUBufferDescriptor ubo_buffer_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    ubo_buffer_desc.label = (WGPUStringView){
        .data = "HWGutil Blit UBO Buffer",
        .length = WGPU_STRLEN,
    };
    ubo_buffer_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
    ubo_buffer_desc.size = sizeof(int) * 2;

    WGPUBuffer ubo_buffer = wgpuDeviceCreateBuffer(device, &ubo_buffer_desc);

    WGPUBufferDescriptor instances_buffer_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    instances_buffer_desc.label = (WGPUStringView){
        .data = "HWGutil Blit Instances Buffer",
        .length = WGPU_STRLEN,
    };
    instances_buffer_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex;
    instances_buffer_desc.size = capacity * sizeof(hwgutil_wgpu_blit_instance);

    WGPUBuffer instances_buffer = wgpuDeviceCreateBuffer(device, &instances_buffer_desc);

    WGPUBindGroupDescriptor ubo_bind_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    ubo_bind_desc.label = (WGPUStringView){
        .data = "HWGutil Blit UBO Bind Group",
        .length = WGPU_STRLEN,
    };
    ubo_bind_desc.layout = ubo_layout;
    WGPUBindGroupEntry ubo_bind_entry0 = WGPU_BIND_GROUP_ENTRY_INIT;
    ubo_bind_entry0.binding = 0;
    ubo_bind_entry0.buffer = ubo_buffer;
    ubo_bind_desc.entryCount = 1;
    ubo_bind_desc.entries = (WGPUBindGroupEntry[]){
        ubo_bind_entry0
    };

    WGPUSamplerDescriptor sampler_desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    sampler_desc.label = (WGPUStringView){
        .data = "HWGutil Blit Src Sampler",
        .length = WGPU_STRLEN,
    };
    sampler_desc.minFilter = WGPUFilterMode_Nearest;
    sampler_desc.magFilter = WGPUFilterMode_Nearest;
    sampler_desc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    sampler_desc.addressModeU = WGPUAddressMode_ClampToEdge;
    sampler_desc.addressModeV = WGPUAddressMode_ClampToEdge;
    sampler_desc.addressModeW = WGPUAddressMode_ClampToEdge;

    WGPUSampler sampler = wgpuDeviceCreateSampler(device, &sampler_desc);

    WGPUBindGroupDescriptor src_texture_bind_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    src_texture_bind_desc.label = (WGPUStringView){
        .data = "HWGutil Blit Src Texture Bind Group",
        .length = WGPU_STRLEN,
    };
    src_texture_bind_desc.layout = src_texture_layout;
    WGPUBindGroupEntry src_texture_entry0 = WGPU_BIND_GROUP_ENTRY_INIT;
    src_texture_entry0.binding = 0;
    src_texture_entry0.textureView = src_view;
    WGPUBindGroupEntry src_texture_entry1 = WGPU_BIND_GROUP_ENTRY_INIT;
    src_texture_entry1.binding = 1;
    src_texture_entry1.sampler = sampler;
    src_texture_bind_desc.entryCount = 2;
    src_texture_bind_desc.entries = (WGPUBindGroupEntry[]){
        src_texture_entry0,
        src_texture_entry1,
    };

    WGPUBindGroup ubo_bind = wgpuDeviceCreateBindGroup(device, &ubo_bind_desc);
    WGPUBindGroup src_texture_bind = wgpuDeviceCreateBindGroup(device, &src_texture_bind_desc);

    wgpuBindGroupLayoutRelease(ubo_layout);
    wgpuBindGroupLayoutRelease(src_texture_layout);
    wgpuSamplerRelease(sampler);
    wgpuRenderPipelineAddRef(pipeline);

    *bindings = (hwgutil_wgpu_blit_bindings){
        .pipeline = pipeline,
        .instances = instances_buffer,
        .ubo = ubo_buffer,
        .ubo_binding = ubo_bind,
        .src_texture_binding = src_texture_bind,
    };

    return hwgutil_true;
}

HWGUTIL_EXPORT void hwgutil_wgpu_blit_bindings_deinit(hwgutil_wgpu_blit_bindings * const bindings)
{
    wgpuRenderPipelineRelease(bindings->pipeline);
    wgpuBufferDestroy(bindings->instances);
    wgpuBufferRelease(bindings->instances);
    wgpuBindGroupRelease(bindings->src_texture_binding);
    wgpuBufferDestroy(bindings->ubo);
    wgpuBufferRelease(bindings->ubo);
}

HWGUTIL_EXPORT void hwgutil_wgpu_blit(
    const hwgutil_wgpu_blit_bindings * const bindings,
    WGPUQueue const queue,
    WGPURenderPassEncoder const render_pass,
    const hwgutil_u32 width, const hwgutil_u32 height,
    const hwgutil_size instances_count,
    hwgutil_wgpu_blit_instance * const instances
)
{
    const hwgutil_u32 target_dimensions[2] = { width, height };
    wgpuQueueWriteBuffer(queue, bindings->ubo, 0, target_dimensions, 8);
    wgpuQueueWriteBuffer(queue, bindings->instances, 0, instances, instances_count * sizeof(hwgutil_wgpu_blit_instance));

    wgpuRenderPassEncoderSetPipeline(render_pass, bindings->pipeline);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 0, bindings->ubo_binding, 0, 0);
    wgpuRenderPassEncoderSetBindGroup(render_pass, 1, bindings->src_texture_binding, 0, 0);
    wgpuRenderPassEncoderSetVertexBuffer(render_pass, 0, bindings->instances, 0, WGPU_WHOLE_SIZE);
    wgpuRenderPassEncoderDraw(render_pass, 4, instances_count, 0, 0);
}

#ifdef HWGUTIL_MEMS_ENABLED

HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_read_texture_alloc(
    WGPUInstance const instance,
    WGPUDevice const device,
    WGPUQueue const queue,
    WGPUTexture const texture,
    const WGPUExtent3D extent,
    const hwgutil_size bytes_per_pixel,
    const mems_allocator * allocator,
    void ** const buffer
)
{
    allocator = allocator == HWGUTIL_NULL ? &mems_system_allocator : allocator;

    hwgutil_size required_size;
    if (hwgutil_wgpu_read_texture(
        HWGUTIL_NULL,
        HWGUTIL_NULL,
        HWGUTIL_NULL,
        HWGUTIL_NULL,
        extent,
        bytes_per_pixel,
        0,
        HWGUTIL_NULL,
        &required_size
    )) return hwgutil_false;

    *buffer = mems_allocator_alloc(allocator, MEMS_ALIGN_DEFAULT, required_size);
    if (*buffer == HWGUTIL_NULL) return hwgutil_false;

    if (!hwgutil_wgpu_read_texture(
        instance,
        device,
        queue,
        texture,
        extent,
        bytes_per_pixel,
        required_size,
        *buffer,
        &required_size
    )) return hwgutil_false;

    return hwgutil_true;
}

HWGUTIL_EXPORT hwgutil_bool hwgutil_wgpu_write_texture_alloc(
    WGPUQueue const queue,
    const WGPUTexture texture,
    const WGPUOrigin3D origin,
    const WGPUExtent3D extent,
    const hwgutil_size bytes_per_pixel,
    const void * const pixels,
    const mems_allocator * allocator
)
{
    allocator = allocator == HWGUTIL_NULL ? &mems_system_allocator : allocator;

    hwgutil_size required_size;
    if (hwgutil_wgpu_write_texture(
        queue,
        texture,
        origin,
        extent,
        bytes_per_pixel,
        pixels,
        0,
        HWGUTIL_NULL,
        &required_size
    )) return hwgutil_true;

    void * const buffer = mems_allocator_alloc(allocator, 0, required_size);

    const hwgutil_bool result = hwgutil_wgpu_write_texture(
        queue,
        texture,
        origin,
        extent,
        bytes_per_pixel,
        pixels,
        required_size,
        buffer,
        &required_size
    );

    mems_allocator_free(allocator, buffer);

    return result;
}

#endif // HWGUTIL_MEMS_ENABLED

#endif // HWGUTIL_WEBGPU_ENABLED

#ifdef HWGUTIL_GLFW3_ENABLED

#ifdef __EMSCRIPTEN__

EM_JS(float, hwgutil__wgpu_emscripten_hdr_component_max, (void), {
    if (window.matchMedia && window.matchMedia('(dynamic-range: high)').matches) {
        return 4.0;
    }
    return 1.0;
})

#endif

HWGUTIL_EXPORT hwgutil_bool hwgutil_glfw3_get_hdr_component_max(
    GLFWwindow * const window,
    float * const hdr_component_max
)
{
    HWGUTIL_USE(window);

#ifndef __EMSCRIPTEN__
    switch (glfwGetPlatform())
    {
#else
    switch (GLFW_PLATFORM_EMSCRIPTEN)
    {
#endif

#ifdef GLFW_EXPOSE_NATIVE_COCOA
        case GLFW_PLATFORM_COCOA:
        {
            NSWindow * ns_window = glfwGetCocoaWindow(window);
            *hdr_component_max = ns_window.screen.maximumExtendedDynamicRangeColorComponentValue;

            return hwgutil_true;
        }
#endif

#ifdef GLFW_EXPOSE_NATIVE_EMSCRIPTEN
        case GLFW_PLATFORM_EMSCRIPTEN:
        {
            *hdr_component_max = hwgutil__wgpu_emscripten_hdr_component_max();
            return hwgutil_true;
        }
#endif

        default:
        {
            *hdr_component_max = 1.0f;
            return hwgutil_false;
        }
    }
}

HWGUTIL_EXPORT hwgutil_bool hwgutil_glfw3_get_hdr_nits(
    GLFWwindow * const window,
    float * const sdr_reference_nits,
    float * const hdr_peak_nits
)
{
    HWGUTIL_USE(window);

#ifndef __EMSCRIPTEN__
    switch (glfwGetPlatform())
    {
#else
    switch (GLFW_PLATFORM_EMSCRIPTEN)
    {
#endif

#ifdef GLFW_EXPOSE_NATIVE_COCOA
        case GLFW_PLATFORM_COCOA:
        {
            NSWindow * ns_window = glfwGetCocoaWindow(window);

            // maximumPotentialExtendedDynamicRangeColorComponentValue is the ratio of the
            // display's peak HDR luminance to its SDR reference white. When the display can
            // reach the PQ 10000-nit ceiling (potential_max > 1), we recover the SDR white
            // by dividing HWGUTIL_PQ_PEAK_NITS by that ratio.
            // potential_max <= 1 means no EDR headroom (SDR-only or brightness-limited);
            // the formula would give nonsensical values so we fall back to the default.
            float potential_max = ns_window.screen.maximumPotentialExtendedDynamicRangeColorComponentValue;
            if (potential_max > 1.0f)
            {
                *sdr_reference_nits = HWGUTIL_PQ_PEAK_NITS / potential_max;
                const float hdr_component_max = ns_window.screen.maximumExtendedDynamicRangeColorComponentValue;
                *hdr_peak_nits = *sdr_reference_nits * hdr_component_max;
            }
            else
            {
                *sdr_reference_nits = HWGUTIL_DEFAULT_SDR_REFERENCE_NITS;
                *hdr_peak_nits = HWGUTIL_DEFAULT_SDR_REFERENCE_NITS;
            }


            return hwgutil_true;
        }
#endif

        default:
        {
            *sdr_reference_nits = HWGUTIL_DEFAULT_SDR_REFERENCE_NITS;
            *hdr_peak_nits = HWGUTIL_DEFAULT_SDR_REFERENCE_NITS;
            return hwgutil_false;
        }
    }
}

void hwgutil_glfw3_window_configure_color_space(GLFWwindow * const window, const hwgutil_color_space color_space) {
    HWGUTIL_USE(window);
    HWGUTIL_USE(color_space);

#ifndef __EMSCRIPTEN__
    switch (glfwGetPlatform()) {
#else
    // glfwGetPlatform is not available in older versions of emscripten
    switch (GLFW_PLATFORM_EMSCRIPTEN) {
#endif

#ifdef GLFW_EXPOSE_NATIVE_COCOA
    case GLFW_PLATFORM_COCOA: {
        NSWindow* ns_window = glfwGetCocoaWindow(window);
        CAMetalLayer *metalLayer = (CAMetalLayer *)ns_window.contentView.layer;
        metalLayer.wantsExtendedDynamicRangeContent = hwgutil_color_space_is_extended(color_space);
        CFStringRef name;
        switch (color_space)
        {
            case hwgutil_color_space_linear_srgba:
            {
                name = kCGColorSpaceLinearSRGB;
            } break;
            case hwgutil_color_space_srgba:
            {
                name = kCGColorSpaceSRGB;
            } break;
            case hwgutil_color_space_extended_linear_srgba:
            {
                name = kCGColorSpaceExtendedLinearSRGB;
            } break;
            case hwgutil_color_space_extended_srgba:
            {
                name = kCGColorSpaceExtendedSRGB;
            } break;
        }
        CGColorSpaceRef colorspace = CGColorSpaceCreateWithName(name);
        metalLayer.colorspace = colorspace;
        CGColorSpaceRelease(colorspace);
        return;
    }
#endif // GLFW_EXPOSE_NATIVE_COCOA

    default:
        // Unsupported platform
        return;
    }
}

#endif

#ifdef HWGUTIL_GLFW3WEBGPU_ENABLED

HWGUTIL_EXPORT WGPUSurface hwgutil_glfw3wgpu_create_surface(WGPUInstance instance, GLFWwindow * window)
{
    HWGUTIL_USE(instance);
    HWGUTIL_USE(window);

#ifndef __EMSCRIPTEN__
    switch (glfwGetPlatform())
    {
#else
    // glfwGetPlatform is not available in older versions of emscripten
    switch (GLFW_PLATFORM_EMSCRIPTEN)
    {
#endif

#ifdef GLFW_EXPOSE_NATIVE_X11
    case GLFW_PLATFORM_X11: {
        Display* x11_display = glfwGetX11Display();
        Window x11_window = glfwGetX11Window(window);

        WGPUSurfaceSourceXlibWindow fromXlibWindow;
        fromXlibWindow.chain.sType = WGPUSType_SurfaceSourceXlibWindow;
        fromXlibWindow.chain.next = HWGUTIL_NULL;
        fromXlibWindow.display = x11_display;
        fromXlibWindow.window = x11_window;

        WGPUSurfaceDescriptor surfaceDescriptor;
        surfaceDescriptor.nextInChain = &fromXlibWindow.chain;
        surfaceDescriptor.label = (WGPUStringView){ HWGUTIL_NULL, WGPU_STRLEN };

        return wgpuInstanceCreateSurface(instance, &surfaceDescriptor);
    }
#endif // GLFW_EXPOSE_NATIVE_X11

#ifdef GLFW_EXPOSE_NATIVE_WAYLAND
    case GLFW_PLATFORM_WAYLAND: {
        struct wl_display* wayland_display = glfwGetWaylandDisplay();
        struct wl_surface* wayland_surface = glfwGetWaylandWindow(window);

        WGPUSurfaceSourceWaylandSurface fromWaylandSurface;
        fromWaylandSurface.chain.sType = WGPUSType_SurfaceSourceWaylandSurface;
        fromWaylandSurface.chain.next = HWGUTIL_NULL;
        fromWaylandSurface.display = wayland_display;
        fromWaylandSurface.surface = wayland_surface;

        WGPUSurfaceDescriptor surfaceDescriptor;
        surfaceDescriptor.nextInChain = &fromWaylandSurface.chain;
        surfaceDescriptor.label = (WGPUStringView){ HWGUTIL_NULL, WGPU_STRLEN };

        return wgpuInstanceCreateSurface(instance, &surfaceDescriptor);
    }
#endif // GLFW_EXPOSE_NATIVE_WAYLAND

#ifdef GLFW_EXPOSE_NATIVE_COCOA
    case GLFW_PLATFORM_COCOA: {
        id metal_layer = [CAMetalLayer layer];
        [metal_layer setContentsGravity:kCAGravityTopLeft];
        NSWindow* ns_window = glfwGetCocoaWindow(window);
        [metal_layer setContentsScale:[ns_window backingScaleFactor]];
        [ns_window.contentView setWantsLayer : YES] ;
        [ns_window.contentView setLayer : metal_layer] ;

        WGPUSurfaceSourceMetalLayer fromMetalLayer;
        fromMetalLayer.chain.sType = WGPUSType_SurfaceSourceMetalLayer;
        fromMetalLayer.chain.next = HWGUTIL_NULL;
        fromMetalLayer.layer = metal_layer;

        WGPUSurfaceDescriptor surfaceDescriptor;
        surfaceDescriptor.nextInChain = &fromMetalLayer.chain;
        surfaceDescriptor.label = (WGPUStringView){ HWGUTIL_NULL, WGPU_STRLEN };

        return wgpuInstanceCreateSurface(instance, &surfaceDescriptor);
    }
#endif // GLFW_EXPOSE_NATIVE_COCOA

#ifdef GLFW_EXPOSE_NATIVE_WIN32
    case GLFW_PLATFORM_WIN32: {
        HWND hwnd = glfwGetWin32Window(window);
        HINSTANCE hinstance = GetModuleHandle(HWGUTIL_NULL);

        WGPUSurfaceSourceWindowsHWND fromWindowsHWND;
        fromWindowsHWND.chain.sType = WGPUSType_SurfaceSourceWindowsHWND;
        fromWindowsHWND.chain.next = HWGUTIL_NULL;
        fromWindowsHWND.hinstance = hinstance;
        fromWindowsHWND.hwnd = hwnd;

        WGPUSurfaceDescriptor surfaceDescriptor;
        surfaceDescriptor.nextInChain = &fromWindowsHWND.chain;
        surfaceDescriptor.label = (WGPUStringView){ HWGUTIL_NULL, WGPU_STRLEN };

        return wgpuInstanceCreateSurface(instance, &surfaceDescriptor);
    }
#endif // GLFW_EXPOSE_NATIVE_WIN32

#ifdef GLFW_EXPOSE_NATIVE_EMSCRIPTEN
    case GLFW_PLATFORM_EMSCRIPTEN: {
#  ifdef WEBGPU_BACKEND_EMDAWNWEBGPU
        WGPUEmscriptenSurfaceSourceCanvasHTMLSelector fromCanvasHTMLSelector;
        fromCanvasHTMLSelector.chain.sType = WGPUSType_EmscriptenSurfaceSourceCanvasHTMLSelector;
        fromCanvasHTMLSelector.selector = (WGPUStringView){ "canvas", WGPU_STRLEN };
#  else
        WGPUSurfaceDescriptorFromCanvasHTMLSelector fromCanvasHTMLSelector;
        fromCanvasHTMLSelector.chain.sType = WGPUSType_SurfaceDescriptorFromCanvasHTMLSelector;
        fromCanvasHTMLSelector.selector = "canvas";
#  endif
        fromCanvasHTMLSelector.chain.next = HWGUTIL_NULL;

        WGPUSurfaceDescriptor surfaceDescriptor;
        surfaceDescriptor.nextInChain = &fromCanvasHTMLSelector.chain;
#  ifdef WEBGPU_BACKEND_EMDAWNWEBGPU
        surfaceDescriptor.label = (WGPUStringView){ HWGUTIL_NULL, WGPU_STRLEN };
#  else
        surfaceDescriptor.label = HWGUTIL_NULL;
#  endif
        return wgpuInstanceCreateSurface(instance, &surfaceDescriptor);
    }
#endif // GLFW_EXPOSE_NATIVE_EMSCRIPTEN

    default:
        // Unsupported platform
        return HWGUTIL_NULL;
    }
}

#endif // HWGUTIL_GLFW3WEBGPU_ENABLED

#endif
#endif

#ifndef HWGUTIL_LICENSE
#define HWGUTIL_LICENSE

/*
Zlib License

Copyright (c) 2026 Hollin Wilkins

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#endif // HWGUTIL_LICENSE
