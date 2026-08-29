#ifndef MERGE_SORT_GPU_H
#define MERGE_SORT_GPU_H

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include <webgpu/webgpu.h>

#include "common.h"

#include "hw_ds.h"
#include "hw_mems.h"

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
    WGPUBuffer bin_workgroup_size;
    WGPUBuffer bin_histogram;
    WGPUBuffer bin_offsets;
    WGPUBuffer bin_indices;
    WGPUBuffer dispatch;
} msg_buffers;

typedef struct msg_kernels
{
    WGPUComputePipeline bin_histogram;
    WGPUComputePipeline schedule;
    WGPUComputePipeline group;
} msg_kernels;

typedef struct msg_bindings
{
    WGPUBindGroup bin;
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

typedef struct msg_sort_kernel_key
{
    uint32_t n;
    uint32_t m;
    bool is_register;
} msg_sort_kernel_key;

static inline uint32_t msg_sort_kernel_key_hash(const msg_sort_kernel_key key)
{
    return hwds_jenkins32(&key, sizeof(msg_sort_kernel_key));
}

static inline bool msg_sort_kernel_key_equal(const msg_sort_kernel_key a, const msg_sort_kernel_key b)
{
    return a.n == b.n && a.m == b.m && a.is_register == b.is_register;
}

#define HWDS_HM_DECLARATION
#define HWDS_NAME msg_sort_kernel_map
#define HWDS_KEY msg_sort_kernel_key
#define HWDS_VALUE WGPUComputePipeline
#define HWDS_HASH msg_sort_kernel_key_hash
#define HWDS_EQUAL msg_sort_kernel_key_equal
#include "hw_ds.h"

typedef struct msg_pipeline
{
    msg_options options;
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPUDevice device;
    WGPUQueue queue;
    msg_kernels kernels;
    msg_sort_kernel_map sort_kernels;
} msg_pipeline;

MERGE_EXPORT void msg_pipeline_init(
    msg_pipeline * pipeline,
    const msg_options * options,
    WGPUInstance instance,
    WGPUAdapter adapter,
    WGPUDevice device,
    WGPUQueue queue,
    const mems_allocator * allocator
);

MERGE_EXPORT void msg_buffers_init(
    msg_buffers * buffers,
    const msg_buffers_options * options,
    WGPUDevice device,
    WGPUQueue queue
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

MERGE_EXPORT void msg_run_bin_histogram(
    const msg_pipeline * pipeline,
    const msg_bindings * bindings,
    const msg_gpu_config * config,
    WGPUComputePassEncoder encoder
);

MERGE_EXPORT void msg_bin_run_schedule(
    const msg_pipeline * pipeline,
    const msg_bindings * bindings,
    const msg_gpu_config * config,
    WGPUComputePassEncoder encoder
);

MERGE_EXPORT void msg_bin_run_group(
    const msg_pipeline * pipeline,
    const msg_bindings * bindings,
    const msg_gpu_config * config,
    WGPUComputePassEncoder encoder
);

#endif

#ifdef MERGE_SORT_GPU_IMPLEMENTATION
#ifndef MERGE_SORT_GPU_IMPLEMENTED
#define MERGE_SORT_GPU_IMPLEMENTED

#define HWDS_HM_IMPLEMENTATION
#define HWDS_NAME msg_sort_kernel_map
#define HWDS_KEY msg_sort_kernel_key
#define HWDS_VALUE WGPUComputePipeline
#define HWDS_HASH msg_sort_kernel_key_hash
#define HWDS_EQUAL msg_sort_kernel_key_equal
#include "hw_ds.h"

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

static void msg__kernels_init(
    msg_kernels * const kernels,
    WGPUDevice const device,
    const msg_dispatch_size * const dispatch_size
)
{
    WGPUBindGroupLayoutDescriptor layout0_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    layout0_desc.label = (WGPUStringView){
        .data = "Merge Sort: Bin Histogram Kernel (segments/bin histogram)",
        .length = WGPU_STRLEN,
    };
    layout0_desc.entryCount = 7;
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
            .binding = 2, // bin_workgroup_size
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
            }
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 3, // bin_histogram
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
            },
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 4, // bin_offsets
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
            },
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 5, // bin_indices
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
            },
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 6, // dispatch
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

    WGPUComputePipelineDescriptor bin_histogram_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    bin_histogram_desc.label = (WGPUStringView){
        .data = "Merge Sort: Bin Histogram Pipeline",
        .length = WGPU_STRLEN,
    };
    bin_histogram_desc.layout = pipeline_layout;
    bin_histogram_desc.compute.entryPoint = (WGPUStringView){
        .data = "main_histogram",
        .length = WGPU_STRLEN,
    };
    bin_histogram_desc.compute.module = shader_module;
    bin_histogram_desc.compute.constantCount = 2;
    bin_histogram_desc.compute.constants = (WGPUConstantEntry[]){
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

    WGPUComputePipeline bin_histogram_kernel = wgpuDeviceCreateComputePipeline(device, &bin_histogram_desc);

    WGPUComputePipelineDescriptor schedule_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    schedule_desc.label = (WGPUStringView){
        .data = "Merge Sort: Dispatch Pipeline",
        .length = WGPU_STRLEN,
    };
    schedule_desc.layout = pipeline_layout;
    schedule_desc.compute.entryPoint = (WGPUStringView){
        .data = "main_schedule",
        .length = WGPU_STRLEN,
    };
    schedule_desc.compute.module = shader_module;

    WGPUComputePipeline scheduler_kernel = wgpuDeviceCreateComputePipeline(device, &schedule_desc);

    WGPUComputePipelineDescriptor group_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    group_desc.label = (WGPUStringView){
        .data = "Merge Sort: Group Pipeline",
        .length = WGPU_STRLEN,
    };
    group_desc.layout = pipeline_layout;
    group_desc.compute.entryPoint = (WGPUStringView){
        .data = "main_group",
        .length = WGPU_STRLEN,
    };
    group_desc.compute.module = shader_module;
    group_desc.compute.constantCount = 2;
    group_desc.compute.constants = (WGPUConstantEntry[]){
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

    WGPUComputePipeline group_kernel = wgpuDeviceCreateComputePipeline(device, &group_desc);

    wgpuShaderModuleRelease(shader_module);
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuBindGroupLayoutRelease(layout0);

    *kernels = (msg_kernels){
        .bin_histogram = bin_histogram_kernel,
        .schedule = scheduler_kernel,
        .group = group_kernel,
    };
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

// subgroups_sizes and segment_sizes must match what is in kernel_generator.py
static const uint32_t msg__sort_kernels_subgroup_sizes[] = {
    8, 16, 32, 64, 128
};
static const uint32_t msg__sort_kernels_subgroup_sizes_len = sizeof(msg__sort_kernels_subgroup_sizes) / sizeof(uint32_t);

static const uint32_t msg__sort_kernels_segment_sizes[] = {
    2, 4, 6, 8,
    16, 32, 64, 128,
    256, 512, 1024, 2048
};

static const uint32_t msg__sort_kernels_segment_sizes_len = sizeof(msg__sort_kernels_segment_sizes) / sizeof(uint32_t);

static const uint32_t msg__sort_kernels_len = msg__sort_kernels_subgroup_sizes_len * msg__sort_kernels_segment_sizes_len;

MERGE_EXPORT void msg_pipeline_init(
    msg_pipeline * const pipeline,
    const msg_options * const options,
    WGPUInstance const instance,
    WGPUAdapter const adapter,
    WGPUDevice const device,
    WGPUQueue const queue,
    const mems_allocator * const allocator
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

    msg__kernels_init(
        &pipeline->kernels,
        pipeline->device,
        &options2.bin_hist_dispatch_size
    );

    if (!msg_sort_kernel_map_init_alloc(
        &pipeline->sort_kernels,
        allocator,
        msg__sort_kernels_len * 4,
        msg__sort_kernels_len
    )) abort();
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
    WGPUDevice const device,
    WGPUQueue const queue
)
{
    msg_buffers_options options2 = options == NULL ? (msg_buffers_options){0} : *options;
    msg_buffers_options_init(&options2);

    WGPUBufferDescriptor config_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    config_desc.label = (WGPUStringView){
        .data = "Merge Sort: Config",
        .length = WGPU_STRLEN,
    };
    config_desc.size = sizeof(msg_gpu_config);
    config_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;

    WGPUBufferDescriptor bin_workgroup_size_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bin_workgroup_size_desc.label = (WGPUStringView){
        .data = "Merge Sort: Bin Workgroup Size",
        .length = WGPU_STRLEN,
    };
    bin_workgroup_size_desc.size = 13 * sizeof(uint32_t);
    bin_workgroup_size_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    WGPUBufferDescriptor segments_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    segments_desc.label = (WGPUStringView){
        .data = "Merge Sort: Segments",
        .length = WGPU_STRLEN,
    };
    segments_desc.size = options2.max_segments * sizeof(uint32_t);
    segments_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    WGPUBufferDescriptor bin_histogram_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bin_histogram_desc.label = (WGPUStringView){
        .data = "Merge Sort: Bin Histogram",
        .length = WGPU_STRLEN,
    };
    bin_histogram_desc.size = 13 * sizeof(uint32_t);
    bin_histogram_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    WGPUBufferDescriptor bin_offsets_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bin_offsets_desc.label = (WGPUStringView){
        .data = "Merge Sort: Bin Offsets",
        .length = WGPU_STRLEN,
    };
    bin_offsets_desc.size = 13 * sizeof(uint32_t);
    bin_offsets_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    WGPUBufferDescriptor bin_indices_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bin_indices_desc.label = (WGPUStringView){
        .data = "Merge Sort: Bin Indices",
        .length = WGPU_STRLEN,
    };
    bin_indices_desc.size = options2.max_segments * sizeof(uint32_t);
    bin_indices_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    WGPUBufferDescriptor dispatch_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    dispatch_desc.label = (WGPUStringView){
        .data = "Merge Sort: Dispatch Args",
        .length = WGPU_STRLEN,
    };
    dispatch_desc.size = 13 * sizeof(msg_dispatch_size);
    dispatch_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    *buffers = (msg_buffers){
        .config = wgpuDeviceCreateBuffer(device, &config_desc),
        .bin_workgroup_size = wgpuDeviceCreateBuffer(device, &bin_workgroup_size_desc),
        .segments = wgpuDeviceCreateBuffer(device, &segments_desc),
        .bin_histogram = wgpuDeviceCreateBuffer(device, &bin_histogram_desc),
        .bin_offsets = wgpuDeviceCreateBuffer(device, &bin_offsets_desc),
        .bin_indices = wgpuDeviceCreateBuffer(device, &bin_indices_desc),
        .dispatch = wgpuDeviceCreateBuffer(device, &dispatch_desc),
    };

    wgpuQueueWriteBuffer(
        queue,
        buffers->bin_workgroup_size,
        0,
        (uint32_t[13]){
            0, 128, 64, 32,
            16, 8, 4, 2,
            1, 1, 1, 1,
            1
        },
        13 * sizeof(uint32_t)
    );
}

MERGE_EXPORT void msg_bindings_init(
    msg_bindings * const bindings,
    const msg_pipeline * const pipeline,
    const msg_buffers * const buffers
)
{
    WGPUBindGroupLayout bin_hist_layout0 = wgpuComputePipelineGetBindGroupLayout(pipeline->kernels.bin_histogram, 0);

    WGPUBindGroupDescriptor bin_hist_binding_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bin_hist_binding_desc.label = (WGPUStringView){
        .data = "Merge Sort: Bin Histogram Binding",
        .length = WGPU_STRLEN,
    };
    bin_hist_binding_desc.layout = bin_hist_layout0;
    bin_hist_binding_desc.entryCount = 7;
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
            .binding = 2, // bin_workgroup_size
            .buffer = buffers->bin_workgroup_size,
            .size = WGPU_WHOLE_SIZE,
        },
        (WGPUBindGroupEntry){
            .binding = 3, // bin_histogram
            .buffer = buffers->bin_histogram,
            .size = WGPU_WHOLE_SIZE,
        },
        (WGPUBindGroupEntry){
            .binding = 4, // bin_offsets
            .buffer = buffers->bin_offsets,
            .size = WGPU_WHOLE_SIZE,
        },
        (WGPUBindGroupEntry){
            .binding = 5, // bin_indices
            .buffer = buffers->bin_indices,
            .size = WGPU_WHOLE_SIZE,
        },
        (WGPUBindGroupEntry){
            .binding = 6, // dispatch
            .buffer = buffers->dispatch,
            .size = WGPU_WHOLE_SIZE,
        },
    };

    WGPUBindGroup bin_binding = wgpuDeviceCreateBindGroup(pipeline->device, &bin_hist_binding_desc);

    wgpuBindGroupLayoutRelease(bin_hist_layout0);

    *bindings = (msg_bindings){
        .bin = bin_binding,
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

MERGE_EXPORT void msg_run_bin_histogram(
    const msg_pipeline * const pipeline,
    const msg_bindings * const bindings,
    const msg_gpu_config * const config,
    WGPUComputePassEncoder const encoder
)
{
    const msg_dispatch_size bin_hist_dispatch_size = msg_dispatch_size_for_len(&pipeline->options.bin_hist_dispatch_size, config->segments_len);

    wgpuComputePassEncoderSetPipeline(encoder, pipeline->kernels.bin_histogram);
    wgpuComputePassEncoderSetBindGroup(encoder, 0, bindings->bin, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(encoder, bin_hist_dispatch_size.x, bin_hist_dispatch_size.y, bin_hist_dispatch_size.z);
}

MERGE_EXPORT void msg_bin_run_schedule(
    const msg_pipeline * const pipeline,
    const msg_bindings * const bindings,
    const msg_gpu_config * const config,
    WGPUComputePassEncoder const encoder
)
{
    wgpuComputePassEncoderSetPipeline(encoder, pipeline->kernels.schedule);
    wgpuComputePassEncoderSetBindGroup(encoder, 0, bindings->bin, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(encoder, 1, 1, 1);
}

MERGE_EXPORT void msg_bin_run_group(
    const msg_pipeline * const pipeline,
    const msg_bindings * const bindings,
    const msg_gpu_config * const config,
    WGPUComputePassEncoder const encoder
)
{
    const msg_dispatch_size bin_hist_dispatch_size = msg_dispatch_size_for_len(&pipeline->options.bin_hist_dispatch_size, config->segments_len);

    wgpuComputePassEncoderSetPipeline(encoder, pipeline->kernels.group);
    wgpuComputePassEncoderSetBindGroup(encoder, 0, bindings->bin, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(encoder, bin_hist_dispatch_size.x, bin_hist_dispatch_size.y, bin_hist_dispatch_size.z);
}

#endif
#endif
