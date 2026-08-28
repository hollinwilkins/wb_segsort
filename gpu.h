#ifndef MERGE_SORT_GPU_H
#define MERGE_SORT_GPU_H

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include <webgpu/webgpu.h>

#include "common.h"

#include "shaders/merge_bin.wgsl.h"

typedef struct msg_gpu_config
{
    uint32_t segments_len;
} msg_gpu_config;

typedef struct msg_buffers_options
{
    size_t max_segments;
    bool is_initialized;
} msg_buffers_options;

typedef struct msg_buffers
{
    WGPUBuffer config;
    WGPUBuffer segments;
    WGPUBuffer bin_histogram;
} msg_buffers;

typedef struct msg_bindings
{
    WGPUBindGroup bin_hist_binding;
} msg_bindings;

typedef struct msg_dispatch_size
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
} msg_dispatch_size;

#define MSG_DISPATCH_SIZE_ONCE { 1, 1, 1 }
#define MSG_DISPATCH_SIZE_DEFAULT { 16, 16, 1 }

typedef struct msg_options
{
    msg_dispatch_size bin_hist_dispatch_size;
    bool is_initialized;
} msg_options;

typedef struct msg_pipeline
{
    msg_options options;
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    WGPUComputePipeline bin_hist_kernel;
} msg_pipeline;

MERGE_EXPORT bool msg_pipeline_init(
    msg_pipeline * pipeline,
    const msg_options * options,
    WGPUInstance instance,
    WGPUAdapter adapter,
    WGPUDevice device,
    WGPUQueue queue
);

MERGE_EXPORT void msg_buffers_init(
    msg_buffers * buffers,
    const msg_buffers_options * options,
    WGPUDevice device
);

MERGE_EXPORT void msg_bindings_init(
    msg_bindings * bindings,
    const msg_pipeline * pipeline,
    const msg_buffers * buffers
);

MERGE_EXPORT void msg_prepare(
    WGPUQueue queue,
    const msg_buffers * buffers,
    size_t segments_len, const uint32_t * segments,
    msg_gpu_config * config
);

MERGE_EXPORT void msg_run_bin_histogram_kernel(
    const msg_pipeline * pipeline,
    const msg_bindings * bindings,
    const msg_gpu_config * config,
    WGPUComputePassEncoder encoder
);

#endif

#ifdef MERGE_SORT_GPU_IMPLEMENTATION
#ifndef MERGE_SORT_GPU_IMPLEMENTED
#define MERGE_SORT_GPU_IMPLEMENTED

#define MSG_MAX_WORKGROUP_DIMENSION 65535u

static msg_dispatch_size msg_dispatch_size_for_len(const msg_dispatch_size * const size, const size_t len)
{
    const uint32_t workgroup_items = size->x * size->y;
    uint32_t x = (len + workgroup_items - 1) / workgroup_items;
    uint32_t y = 1;
    if (size->x > MSG_MAX_WORKGROUP_DIMENSION)
    {
        y = (len + MSG_MAX_WORKGROUP_DIMENSION - 1u) / MSG_MAX_WORKGROUP_DIMENSION;
        x = MSG_MAX_WORKGROUP_DIMENSION;
    }

    return (msg_dispatch_size){ x, y, 1 };
}

static WGPUComputePipeline msg__create_bin_hist_kernel(WGPUDevice const device, const msg_dispatch_size * const dispatch_size)
{
    WGPUBindGroupLayoutDescriptor layout0_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    layout0_desc.label = (WGPUStringView){
        .data = "Merge Sort: Bin Histogram Kernel (segments/bin histogram)",
        .length = WGPU_STRLEN,
    };
    layout0_desc.entryCount = 3;
    layout0_desc.entries = (WGPUBindGroupLayoutEntry[]){
        (WGPUBindGroupLayoutEntry){
            .binding = 0, // config
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Uniform,
            }
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 1, // segments
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
            }
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 2, // bin_histogram
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
            },
        },
    };

    WGPUBindGroupLayout layout0 = wgpuDeviceCreateBindGroupLayout(device, &layout0_desc);

    WGPUPipelineLayoutDescriptor pipeline_layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipeline_layout_desc.label = (WGPUStringView){
        .data = "Merge Sort: Bin Histogram Pipeline Layout",
        .length = WGPU_STRLEN,
    };
    pipeline_layout_desc.bindGroupLayoutCount = 1;
    pipeline_layout_desc.bindGroupLayouts = (WGPUBindGroupLayout[]){
        layout0,
    };

    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_desc);

    WGPUShaderSourceWGSL shader_source_wgsl = (WGPUShaderSourceWGSL){
        .chain = (WGPUChainedStruct){
            .sType = WGPUSType_ShaderSourceWGSL
        },
        .code = (WGPUStringView){
            .data = (const char *)merge_bin_wgsl,
            .length = merge_bin_wgsl_len,
        },
    };
    WGPUShaderModuleDescriptor shader_module_desc = (WGPUShaderModuleDescriptor){
        .label = (WGPUStringView){
            .data = "RenderStrip shader module",
            .length = WGPU_STRLEN,
        },
        .nextInChain = (WGPUChainedStruct *)(&shader_source_wgsl),
    };
    WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(
        device,
        &shader_module_desc
    );

    WGPUComputePipelineDescriptor pipeline_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pipeline_desc.label = (WGPUStringView){
        .data = "Merge Sort: Bin Histogram Pipeline",
        .length = WGPU_STRLEN,
    };
    pipeline_desc.layout = pipeline_layout;
    pipeline_desc.compute.entryPoint = (WGPUStringView){
        .data = "main_histogram",
        .length = WGPU_STRLEN,
    };
    pipeline_desc.compute.module = shader_module;
    pipeline_desc.compute.constantCount = 2;
    pipeline_desc.compute.constants = (WGPUConstantEntry[]){
        (WGPUConstantEntry){
            .key = (WGPUStringView){
                .data = "WORKGROUP_SIZE_X",
                .length = WGPU_STRLEN
            },
            .value = (float)dispatch_size->x,
        },
        (WGPUConstantEntry){
            .key = (WGPUStringView){
                .data = "WORKGROUP_SIZE_Y",
                .length = WGPU_STRLEN
            },
            .value = (float)dispatch_size->y,
        },
    };

    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &pipeline_desc);

    wgpuShaderModuleRelease(shader_module);
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuBindGroupLayoutRelease(layout0);

    return pipeline;
}

static bool msg__dispatch_size_is_zero(const msg_dispatch_size * const size)
{
    return size->x == 0 ||
        size->y == 0 ||
        size->z == 0;
}

static void msg__options_init(msg_options * const options)
{
    if (options->is_initialized) return;

    options->bin_hist_dispatch_size = msg__dispatch_size_is_zero(&options->bin_hist_dispatch_size) ?
        (msg_dispatch_size)MSG_DISPATCH_SIZE_DEFAULT :
        options->bin_hist_dispatch_size;

    options->is_initialized = true;
}

MERGE_EXPORT bool msg_pipeline_init(
    msg_pipeline * const pipeline,
    const msg_options * const options,
    WGPUInstance const instance,
    WGPUAdapter const adapter,
    WGPUDevice const device,
    WGPUQueue const queue
)
{
    msg_options options2 = options == NULL ? (msg_options){0} : *options;
    msg__options_init(&options2);

    wgpuInstanceAddRef(instance);
    wgpuAdapterAddRef(adapter);
    wgpuDeviceAddRef(device);
    wgpuQueueAddRef(queue);

    *pipeline = (msg_pipeline){
        .options = options2,
        .instance = instance,
        .adapter = adapter,
        .device = device,
        .queue = queue,
    };

    pipeline->bin_hist_kernel = msg__create_bin_hist_kernel(device, &options2.bin_hist_dispatch_size);

    return true;
}

#define MSG_BUFFERS_OPTIONS_DEFAULT_MAX_SEGMENTS 1024

static void msg_buffers_options_init(msg_buffers_options * const options)
{
    if (options->is_initialized) return;

    options->max_segments = options->max_segments == 0 ? MSG_BUFFERS_OPTIONS_DEFAULT_MAX_SEGMENTS : options->max_segments;

    options->is_initialized = true;
}

MERGE_EXPORT void msg_buffers_init(
    msg_buffers * const buffers,
    const msg_buffers_options * const options,
    WGPUDevice const device
)
{
    msg_buffers_options options2 = options == NULL ? (msg_buffers_options){0} : *options;
    msg_buffers_options_init(&options2);

    *buffers = (msg_buffers){0};

    WGPUBufferDescriptor config_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    config_desc.label = (WGPUStringView){
        .data = "Merge Sort: Config",
        .length = WGPU_STRLEN,
    };
    config_desc.size = sizeof(msg_gpu_config);
    config_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;

    buffers->config = wgpuDeviceCreateBuffer(device, &config_desc);

    WGPUBufferDescriptor segments_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    segments_desc.label = (WGPUStringView){
        .data = "Merge Sort: Segments",
        .length = WGPU_STRLEN,
    };
    segments_desc.size = options2.max_segments * sizeof(uint32_t);
    segments_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    buffers->segments = wgpuDeviceCreateBuffer(device, &segments_desc);

    WGPUBufferDescriptor bin_histogram_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bin_histogram_desc.label = (WGPUStringView){
        .data = "Merge Sort: Bin Histogram",
        .length = WGPU_STRLEN,
    };
    bin_histogram_desc.size = 13 * sizeof(uint32_t);
    bin_histogram_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    buffers->bin_histogram = wgpuDeviceCreateBuffer(device, &bin_histogram_desc);
}

MERGE_EXPORT void msg_bindings_init(
    msg_bindings * const bindings,
    const msg_pipeline * const pipeline,
    const msg_buffers * const buffers
)
{
    WGPUBindGroupLayout bin_hist_layout0 = wgpuComputePipelineGetBindGroupLayout(pipeline->bin_hist_kernel, 0);

    WGPUBindGroupDescriptor bin_hist_binding_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bin_hist_binding_desc.label = (WGPUStringView){
        .data = "Merge Sort: Bin Histogram Binding",
        .length = WGPU_STRLEN,
    };
    bin_hist_binding_desc.layout = bin_hist_layout0;
    bin_hist_binding_desc.entryCount = 3;
    bin_hist_binding_desc.entries = (WGPUBindGroupEntry[]){
        (WGPUBindGroupEntry){
            .binding = 0, // config
            .buffer = buffers->config,
            .size = sizeof(msg_gpu_config),
        },
        (WGPUBindGroupEntry){
            .binding = 1, // segments
            .buffer = buffers->segments,
            .size = WGPU_WHOLE_SIZE,
        },
        (WGPUBindGroupEntry){
            .binding = 2, // bin_histogram
            .buffer = buffers->bin_histogram,
            .size = WGPU_WHOLE_SIZE,
        },
    };

    WGPUBindGroup bin_hist_binding = wgpuDeviceCreateBindGroup(pipeline->device, &bin_hist_binding_desc);

    wgpuBindGroupLayoutRelease(bin_hist_layout0);

    *bindings = (msg_bindings){
        .bin_hist_binding = bin_hist_binding,
    };
}

MERGE_EXPORT void msg_prepare(
    WGPUQueue const queue,
    const msg_buffers * const buffers,
    size_t segments_len, const uint32_t * segments,
    msg_gpu_config * const config
)
{
    *config = (msg_gpu_config){
        .segments_len = segments_len,
    };

    wgpuQueueWriteBuffer(queue, buffers->config, 0, config, sizeof(msg_gpu_config));
    wgpuQueueWriteBuffer(queue, buffers->segments, 0, segments, segments_len * sizeof(uint32_t));
}

MERGE_EXPORT void msg_run_bin_histogram_kernel(
    const msg_pipeline * const pipeline,
    const msg_bindings * const bindings,
    const msg_gpu_config * const config,
    WGPUComputePassEncoder const encoder
)
{
    const msg_dispatch_size bin_hist_dispatch_size = msg_dispatch_size_for_len(&pipeline->options.bin_hist_dispatch_size, config->segments_len);

    wgpuComputePassEncoderSetPipeline(encoder, pipeline->bin_hist_kernel);
    wgpuComputePassEncoderSetBindGroup(encoder, 0, bindings->bin_hist_binding, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(encoder, bin_hist_dispatch_size.x, bin_hist_dispatch_size.y, bin_hist_dispatch_size.z);
}

#endif
#endif
