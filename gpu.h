#ifndef MERGE_SORT_GPU_H
#define MERGE_SORT_GPU_H

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#include <webgpu/webgpu.h>

#include "common.h"

#include "hw_ds.h"
#include "hw_mems.h"

#include "shaders/merge_bin.wgsl.h"

typedef struct msg_gpu_config
{
    uint32_t segments_len;
} msg_gpu_config;

typedef uint32_t msg_bin_flag;
typedef enum msg_bin_flag_enum
{
    // uses registers for sorting
    msg_bin_flag_is_register = 1u << 0u,

    // is variable sized (merge sort)
    msg_bin_flag_is_variable = 1u << 1u,
} msg_bin_flag_enum;

typedef struct msg_gpu_bin
{
    uint32_t n;
    uint32_t m;
    uint32_t wg;
    msg_bin_flag flags;
} msg_gpu_bin;

typedef struct msg_buffers_options
{
    size_t max_segments;
    size_t max_items;
    size_t value_size;
    bool is_initialized;
} msg_buffers_options;

typedef struct msg_buffers
{
    WGPUBuffer config;
    WGPUBuffer segments;
    WGPUBuffer bin_config;
    WGPUBuffer bin_histogram;
    WGPUBuffer bin_offsets;
    WGPUBuffer bin_indices;
    WGPUBuffer dispatch;

    // sort buffers
    WGPUBuffer value_indices;
    WGPUBuffer values;
    WGPUBuffer keys;
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
    WGPUBindGroup sort;
} msg_bindings;

typedef struct msg_dispatch_size
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
} msg_dispatch_size;

#define MSG_DISPATCH_SIZE_ONCE { 1, 1, 1 }
#define MSG_DISPATCH_SIZE_DEFAULT { 16, 16, 1 }

typedef struct msg_sort_layouts
{
    WGPUBindGroupLayout layout0;
    WGPUPipelineLayout pipeline_layout;
} msg_sort_layouts;

typedef struct msg_options
{
    const char * sort_kernels_root_dir;
    msg_dispatch_size bin_hist_dispatch_size;
    size_t wpt_threshold;
    size_t target_wg_size;
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
    uint32_t max_invocations;
    uint32_t max_smem_size;
    uint32_t subgroup_size;
    bool has_subgroups;
    msg_kernels kernels;
    msg_sort_kernel_map sort_kernels;
    msg_sort_layouts sort_layouts;

    msg_gpu_bin bins[13];
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
    const msg_pipeline * pipeline,
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
    size_t keys_len, const uint32_t * keys,
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

MERGE_EXPORT void msg_sort(
    msg_pipeline * pipeline,
    const msg_bindings * bindings,
    const msg_buffers * buffers,
    const msg_gpu_config * config,
    WGPUComputePassEncoder encoder,
    const mems_allocator * allocator
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

#define MSG_OPTIONS_DEFAULT_WPT_THRESHOLD 8u
#define MSG_OPTIONS_DEFAULT_TARGET_WG_SIZE 256u

static void msg__options_init(msg_options * const options)
{
    if (options->is_initialized) return;

    options->bin_hist_dispatch_size = msg__dispatch_size_is_zero(&options->bin_hist_dispatch_size) ?
        (msg_dispatch_size)MSG_DISPATCH_SIZE_DEFAULT :
        options->bin_hist_dispatch_size;

    options->wpt_threshold = options->wpt_threshold == 0 ? MSG_OPTIONS_DEFAULT_WPT_THRESHOLD : options->wpt_threshold;
    options->target_wg_size = options->target_wg_size == 0 ? MSG_OPTIONS_DEFAULT_TARGET_WG_SIZE : options->target_wg_size;

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


static void msg__sort_layouts_init(
    msg_sort_layouts * const layouts,
    WGPUDevice const device
)
{
    WGPUBindGroupLayoutDescriptor layout0_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    layout0_desc.label = (WGPUStringView){
        .data = "Merge Sort: Sort Bindings",
        .length = WGPU_STRLEN,
    };
    layout0_desc.entryCount = 5;
    layout0_desc.entries = (WGPUBindGroupLayoutEntry[]){
        (WGPUBindGroupLayoutEntry){
            .binding = 0, // global_keys
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
            }
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 1, // global_value_indices
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
            }
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 2, // segments
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
            }
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 3, // bin_offsets
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
            },
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 4, // bin_indices
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
            },
        },
    };

    WGPUBindGroupLayout layout0 = wgpuDeviceCreateBindGroupLayout(device, &layout0_desc);

    WGPUPipelineLayoutDescriptor pipeline_layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipeline_layout_desc.label = (WGPUStringView){
        .data = "Merge Sort: Sort Pipeline Layout",
        .length = WGPU_STRLEN,
    };
    pipeline_layout_desc.bindGroupLayoutCount = 1;
    pipeline_layout_desc.bindGroupLayouts = (WGPUBindGroupLayout[]){
        layout0,
    };

    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_desc);

    *layouts = (msg_sort_layouts){
        .layout0 = layout0,
        .pipeline_layout = pipeline_layout,
    };
}

static size_t msg__round_pow2(size_t a) {
    if (a <= 1) return 1;
    a--;
    a |= a >> 1; a |= a >> 2; a |= a >> 4;
    a |= a >> 8; a |= a >> 16; a |= a >> 32;
    return a + 1;
}

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

    WGPULimits limits = WGPU_LIMITS_INIT;
    if (wgpuDeviceGetLimits(device, &limits) != WGPUStatus_Success) abort();

    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    if (wgpuAdapterGetInfo(adapter, &info) != WGPUStatus_Success) abort();

    const WGPUBool has_subgroups = wgpuAdapterHasFeature(adapter, WGPUFeatureName_Subgroups);

    *pipeline = (msg_pipeline){
        .options = options2,
        .instance = instance,
        .adapter = adapter,
        .device = device,
        .queue = queue,
        .max_invocations = limits.maxComputeInvocationsPerWorkgroup,
        .max_smem_size = limits.maxComputeWorkgroupStorageSize,
        .subgroup_size = info.subgroupMinSize,
        .has_subgroups = has_subgroups == WGPU_TRUE,
    };

    msg__kernels_init(
        &pipeline->kernels,
        pipeline->device,
        &options2.bin_hist_dispatch_size
    );

    msg__sort_layouts_init(
        &pipeline->sort_layouts,
        device
    );

    if (!msg_sort_kernel_map_init_alloc(
        &pipeline->sort_kernels,
        allocator,
        msg__sort_kernels_len * 4,
        msg__sort_kernels_len
    )) abort();

        for (uint32_t i = 0; i < 13; i++)
    {
        const uint32_t N = 1u << i;
        uint32_t M = 0;
        bool is_register = false;

        if (pipeline->has_subgroups)
        {
            is_register = true;
            M = MSG_MIN(N, pipeline->subgroup_size);
            const uint32_t wpt = N / M;

            if (wpt > pipeline->options.wpt_threshold) M = 0;
        }

        if (M == 0)
        {
            is_register = false;
            M = MSG_MIN(N, msg__round_pow2(((N + pipeline->options.wpt_threshold - 1) / pipeline->options.wpt_threshold)));

            if (N * 8u > pipeline->options.wpt_threshold) M = 0;
        }

        const uint32_t wpt = N / M;
        uint32_t wg;
        if (is_register)
        {
            wg = MSG_MIN(pipeline->options.target_wg_size, pipeline->max_invocations);
            wg = (wg / pipeline->subgroup_size) * pipeline->subgroup_size;
            if (wg < pipeline->subgroup_size) wg = pipeline->subgroup_size;
        }
        else
        {
            uint32_t smem_cap = pipeline->max_smem_size / (wpt * 2u * 4u); // keys and values (keys + values) * sizeof(uint32_t)
            wg = MSG_MIN(
                pipeline->options.target_wg_size,
                MSG_MIN(
                    pipeline->max_invocations,
                    smem_cap
                )
            );
            wg = (wg / M) * M;
            if (wg < M) wg = M;
        }

        const bool is_variable = M == 0;

        msg_bin_flag flags = 0;
        flags |= (uint32_t)is_register * msg_bin_flag_is_register;
        flags |= (uint32_t)is_variable * msg_bin_flag_is_variable;

        pipeline->bins[i] = (msg_gpu_bin){
            .n = N,
            .m = M,
            .wg = wg,
            .flags = flags,
        };
    }
}

#define MSG_BUFFERS_OPTIONS_DEFAULT_MAX_SEGMENTS 1024
#define MSG_BUFFERS_OPTIONS_DEFAULT_MAX_ITEMS 1024 * 256

static void msg_buffers_options_init(msg_buffers_options * const options)
{
    if (options->is_initialized) return;

    options->max_segments = options->max_segments == 0 ? MSG_BUFFERS_OPTIONS_DEFAULT_MAX_SEGMENTS : options->max_segments;
    options->max_items = options->max_items == 0 ? MSG_BUFFERS_OPTIONS_DEFAULT_MAX_ITEMS : options->max_items;

    options->is_initialized = true;
}

MERGE_EXPORT void msg_buffers_init(
    const msg_pipeline * const pipeline,
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

    WGPUBufferDescriptor bin_config_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bin_config_desc.label = (WGPUStringView){
        .data = "Merge Sort: Bin Config",
        .length = WGPU_STRLEN,
    };
    bin_config_desc.size = 13 * sizeof(msg_gpu_bin);
    bin_config_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

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

    WGPUBufferDescriptor value_indices_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    value_indices_desc.label = (WGPUStringView){
        .data = "Merge Sort: Value Indices",
        .length = WGPU_STRLEN,
    };
    value_indices_desc.size = options2.max_items * sizeof(msg_dispatch_size);
    value_indices_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    WGPUBufferDescriptor values_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    values_desc.label = (WGPUStringView){
        .data = "Merge Sort: Values",
        .length = WGPU_STRLEN,
    };
    values_desc.size = options2.max_items * options2.value_size;
    values_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    WGPUBufferDescriptor keys_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    keys_desc.label = (WGPUStringView){
        .data = "Merge Sort: Keys",
        .length = WGPU_STRLEN,
    };
    keys_desc.size = options2.max_items * sizeof(uint32_t);
    keys_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    *buffers = (msg_buffers){
        .config = wgpuDeviceCreateBuffer(device, &config_desc),
        .bin_config = wgpuDeviceCreateBuffer(device, &bin_config_desc),
        .segments = wgpuDeviceCreateBuffer(device, &segments_desc),
        .bin_histogram = wgpuDeviceCreateBuffer(device, &bin_histogram_desc),
        .bin_offsets = wgpuDeviceCreateBuffer(device, &bin_offsets_desc),
        .bin_indices = wgpuDeviceCreateBuffer(device, &bin_indices_desc),
        .dispatch = wgpuDeviceCreateBuffer(device, &dispatch_desc),
        .value_indices = wgpuDeviceCreateBuffer(device, &value_indices_desc),
        .values = wgpuDeviceCreateBuffer(device, &values_desc),
        .keys = wgpuDeviceCreateBuffer(device, &keys_desc),
    };

    wgpuQueueWriteBuffer(
        queue,
        buffers->bin_config,
        0,
        pipeline->bins,
        13 * sizeof(msg_gpu_bin)
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
            .buffer = buffers->bin_config,
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

    WGPUBindGroupDescriptor sort_binding_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    sort_binding_desc.label = (WGPUStringView){
        .data = "Merge Sort: Sort Binding",
        .length = WGPU_STRLEN,
    };
    sort_binding_desc.layout = pipeline->sort_layouts.layout0;
    sort_binding_desc.entryCount = 5;
    sort_binding_desc.entries = (WGPUBindGroupEntry[]){
        (WGPUBindGroupEntry){
            .binding = 0, // global_keys
            .buffer = buffers->keys,
            .size = sizeof(msg_gpu_config),
        },
        (WGPUBindGroupEntry){
            .binding = 1, // global_value_indices
            .buffer = buffers->value_indices,
            .size = WGPU_WHOLE_SIZE,
        },
        (WGPUBindGroupEntry){
            .binding = 2, // segments
            .buffer = buffers->segments,
            .size = WGPU_WHOLE_SIZE,
        },
        (WGPUBindGroupEntry){
            .binding = 3, // bin_offsets
            .buffer = buffers->bin_offsets,
            .size = WGPU_WHOLE_SIZE,
        },
        (WGPUBindGroupEntry){
            .binding = 4, // bin_indices
            .buffer = buffers->bin_indices,
            .size = WGPU_WHOLE_SIZE,
        },
    };

    WGPUBindGroup sort_binding = wgpuDeviceCreateBindGroup(pipeline->device, &sort_binding_desc);

    *bindings = (msg_bindings){
        .bin = bin_binding,
        .sort = sort_binding,
    };
}

MERGE_EXPORT void msg_prepare(
    WGPUQueue const queue,
    const msg_buffers * const buffers,
    size_t segments_len, const uint32_t * segments,
    size_t keys_len, const uint32_t * keys,
    msg_gpu_config * const config
)
{
    *config = (msg_gpu_config){
        .segments_len = segments_len,
    };

    wgpuQueueWriteBuffer(queue, buffers->config, 0, config, sizeof(msg_gpu_config));
    wgpuQueueWriteBuffer(queue, buffers->segments, 0, segments, segments_len * sizeof(uint32_t));
    wgpuQueueWriteBuffer(queue, buffers->segments, 0, keys, keys_len * sizeof(uint32_t));
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

static char * msg__read_file(
    const char * const path,
    const mems_allocator * const allocator,
    size_t * const buffer_len
)
{
    FILE * const f = fopen(path, "rb");
    if (f == NULL) abort();

    if (fseek(f, 0, SEEK_END) != 0) abort();

    long file_size = ftell(f);
    if (file_size < 0) abort();

    if (fseek(f, 0, SEEK_SET) != 0) abort();

    char * const buffer = (char *)mems_allocator_alloc(allocator, MEMS_ALIGN_DEFAULT, file_size + 1);
    size_t bytes_read = fread(buffer, 1, file_size, f);
    if (bytes_read < (size_t)file_size) {
        if (ferror(f)) {
            perror("Error reading file");
            free(buffer);
            fclose(f);
            abort();
        }
        file_size = bytes_read;
    }

    *buffer_len = file_size;
    buffer[file_size] = 0;

    fclose(f);

    return buffer;
}

static WGPUComputePipeline msg__pipeline_get_sort_kernel(
    msg_pipeline * const pipeline,
    const msg_gpu_bin * const bin,
    const mems_allocator * const allocator
)
{
    const bool is_register = (bin->flags & msg_bin_flag_is_variable) != 0;

    msg_sort_kernel_map_gop_result gop;
    msg_sort_kernel_map_get_or_put(
        &pipeline->sort_kernels,
        (msg_sort_kernel_key){ bin->n, bin->m, is_register },
        &gop
    );

    if (!gop.found_existing)
    {
        const char * const memory = is_register ? "reg" : "wg";

        static char KERNEL_NAME[2048];
        snprintf(KERNEL_NAME, 2048, "segsort_%s_n%u_m%u",
            memory,
            bin->n, bin->m
        );

        static char KERNEL_FILE_PATH[2048];
        snprintf(KERNEL_FILE_PATH, 2048, "%s/%s.wgsl",
            pipeline->options.sort_kernels_root_dir,
            KERNEL_NAME
        );

        size_t source_len;
        char * const source = msg__read_file(KERNEL_FILE_PATH, allocator, &source_len);

        WGPUShaderSourceWGSL shader_source_wgsl = (WGPUShaderSourceWGSL){
            .chain = (WGPUChainedStruct){
                .sType = WGPUSType_ShaderSourceWGSL
            },
            .code = (WGPUStringView){
                .data = source,
                .length = source_len,
            },
        };
        WGPUShaderModuleDescriptor shader_module_desc = (WGPUShaderModuleDescriptor){
            .label = (WGPUStringView){
                .data = KERNEL_NAME,
                .length = WGPU_STRLEN,
            },
            .nextInChain = (WGPUChainedStruct *)(&shader_source_wgsl),
        };
        WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(
            pipeline->device,
            &shader_module_desc
        );

        static char PIPELINE_NAME[2048];
        snprintf(PIPELINE_NAME, 2048, "Sort %s", KERNEL_FILE_PATH);
        WGPUComputePipelineDescriptor sort_kernel_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        sort_kernel_desc.label = (WGPUStringView){
            .data = PIPELINE_NAME,
            .length = WGPU_STRLEN,
        };
        sort_kernel_desc.layout = pipeline->sort_layouts.pipeline_layout;
        sort_kernel_desc.compute.entryPoint = (WGPUStringView){
            .data = KERNEL_NAME,
            .length = WGPU_STRLEN,
        };
        sort_kernel_desc.compute.module = shader_module;
        sort_kernel_desc.compute.constantCount = 1;
        sort_kernel_desc.compute.constants = (WGPUConstantEntry[]){
            (WGPUConstantEntry){
                .key = (WGPUStringView){
                    .data = "WG",
                    .length = WGPU_STRLEN
                },
                .value = (float)bin->wg,
            },
        };

        *gop.value = wgpuDeviceCreateComputePipeline(pipeline->device, &sort_kernel_desc);
    }

    return *gop.value;
}

MERGE_EXPORT void msg_sort(
    msg_pipeline * const pipeline,
    const msg_bindings * const bindings,
    const msg_buffers * const buffers,
    const msg_gpu_config * const config,
    WGPUComputePassEncoder const encoder,
    const mems_allocator * const allocator
)
{
    wgpuComputePassEncoderSetBindGroup(encoder, 0, bindings->sort, 0, NULL);
      
    for (uint32_t i = 1u; i < 12u; i++)
    {
        const msg_gpu_bin * const bin = &pipeline->bins[i];

        WGPUComputePipeline const sort_kernel = msg__pipeline_get_sort_kernel(
            pipeline,
            bin,
            allocator
        );

        wgpuComputePassEncoderSetPipeline(encoder, sort_kernel);
        wgpuComputePassEncoderDispatchWorkgroupsIndirect(
            encoder,
            buffers->dispatch,
            (uint64_t)i * sizeof(msg_dispatch_size)
        );
    }
}

#endif
#endif
