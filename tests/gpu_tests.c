#include <stdlib.h>
#include <string.h>
#include <webgpu/webgpu.h>

#include "unity.h"
#include "unity_internals.h"

#define HWGUTIL_WEBGPU_ENABLED
#define HWGUTIL_MEMS_ENABLED
#define HWDS_MEMS_ENABLED

#define MERGE_SORT_GPU_IMPLEMENTATION
#define HWGUTIL_IMPLEMENTATION
#define HWDS_IMPLEMENTATION
#define MEMS_IMPLEMENTATION

#include "gpu.h"
#include "hw_gutil.h"
#include "hw_mems.h"

hwgutil_wgpu_context context;
msg_pipeline pipeline;
msg_buffers buffers;
msg_bindings bindings;

void setUp(void)
{
}

void tearDown(void)
{
}

void test__sort_kernel(
    const uint32_t max_n,
    const uint32_t segments_len,
    const unsigned int seed
)
{
    srand(seed);

    uint32_t * const segments = (uint32_t *)malloc(segments_len * sizeof(uint32_t));

    uint32_t len = 0;
    for (uint32_t i = 0; i < segments_len; i++)
    {
        const uint32_t segment_len = rand() % max_n;
        len += segment_len;
        segments[i] = len;
    }

    uint32_t * const keys = (uint32_t *)malloc(len * sizeof(uint32_t));
    uint32_t * const values = (uint32_t *)malloc(len * sizeof(uint32_t));

    for (uint32_t i = 0; i < len; i++)
    {
        keys[i] = rand() % 100;
        values[i] = rand() % 712893;
    }

    msg_gpu_config config = {0};
    msg_prepare(
        pipeline.queue,
        &buffers,
        segments_len, segments,
        len, keys,
        &config);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(pipeline.device, &(WGPUCommandEncoderDescriptor){
        .label = (WGPUStringView){
            .data = "Merge Sort: Command Encoder",
            .length = WGPU_STRLEN,
        }
    });

    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(encoder, &(WGPUComputePassDescriptor){
        .label = (WGPUStringView){
            .data = "Merge Sort: Compute Pass Encoder",
        },
    });

    msg_run_bin_histogram(
        &pipeline,
        &bindings,
        &config,
        compute_pass
    );

    msg_bin_run_schedule(
        &pipeline,
        &bindings,
        &config,
        compute_pass
    );

    msg_bin_run_group(
        &pipeline,
        &bindings,
        &config,
        compute_pass
    );

    msg_sort(
        &pipeline,
        &bindings,
        &buffers,
        &config,
        compute_pass,
        &mems_system_allocator
    );

    wgpuComputePassEncoderEnd(compute_pass);

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &(WGPUCommandBufferDescriptor){
        .label = (WGPUStringView){
            .data = "Merge Sort: Command Buffer",
            .length = WGPU_STRLEN,
        }
    });

    wgpuQueueSubmit(pipeline.queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuCommandEncoderRelease(encoder);
}

void test_sort_fixed_4096_seed1337()
{
    test__sort_kernel(2048, 4096, 1337);
}

// void test_bin(void)
// {
//     msg_gpu_config config;

//     const uint32_t segments[] = {
//         0, 1, 3, 6, 10, 17,
//         25, 40, 72, 136, 264, 520,
//         1032, 2056, 4104, 8200, 17088
//     };
//     const size_t segments_len = sizeof(segments) / sizeof(uint32_t);

//     msg_prepare(pipeline.queue, &buffers, segments_len, segments, &config);

//     WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(pipeline.device, &(WGPUCommandEncoderDescriptor){
//         .label = (WGPUStringView){
//             .data = "Merge Sort: Command Encoder",
//             .length = WGPU_STRLEN,
//         }
//     });

//     WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(encoder, &(WGPUComputePassDescriptor){
//         .label = (WGPUStringView){
//             .data = "Merge Sort: Compute Pass Encoder",
//         },
//     });

//     msg_run_bin_histogram(
//         &pipeline,
//         &bindings,
//         &config,
//         compute_pass
//     );

//     msg_bin_run_schedule(
//         &pipeline,
//         &bindings,
//         &config,
//         compute_pass
//     );

//     msg_bin_run_group(
//         &pipeline,
//         &bindings,
//         &config,
//         compute_pass
//     );

//     wgpuComputePassEncoderEnd(compute_pass);

//     WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &(WGPUCommandBufferDescriptor){
//         .label = (WGPUStringView){
//             .data = "Merge Sort: Command Buffer",
//             .length = WGPU_STRLEN,
//         }
//     });

//     wgpuQueueSubmit(pipeline.queue, 1, &commands);

//     wgpuCommandBufferRelease(commands);
//     wgpuComputePassEncoderRelease(compute_pass);
//     wgpuCommandEncoderRelease(encoder);

//     uint32_t * bin_histogram;
//     uint32_t * bin_offsets;
//     uint32_t * bin_indices;

//     if (!hwgutil_wgpu_read_buffer_alloc(
//         pipeline.instance,
//         pipeline.device,
//         pipeline.queue,
//         buffers.bin_histogram,
//         &mems_system_allocator,
//         (void **)&bin_histogram
//     )) abort();

//     if (!hwgutil_wgpu_read_buffer_alloc(
//         pipeline.instance,
//         pipeline.device,
//         pipeline.queue,
//         buffers.bin_offsets,
//         &mems_system_allocator,
//         (void **)&bin_offsets
//     )) abort();

//     if (!hwgutil_wgpu_read_buffer_alloc(
//         pipeline.instance,
//         pipeline.device,
//         pipeline.queue,
//         buffers.bin_indices,
//         &mems_system_allocator,
//         (void **)&bin_indices
//     )) abort();

//     const uint32_t expected_histogram[] = {
//         0, 2, 3, 5,
//         7, 8, 9, 10,
//         11, 12, 13, 14,
//         15
//     };

//     const uint32_t expected_offsets[] = {
//         2, 3, 5, 7,
//         8, 9, 10, 11,
//         12, 13, 14, 15,
//         17
//     };

//     const uint32_t expected_indices[] = {
//         1, 0, 2, 4,
//         3, 6, 5, 7,
//         8, 9, 10, 11,
//         12, 13, 14, 16, 15
//     };

//     TEST_ASSERT_EQUAL_UINT32_ARRAY(
//         expected_histogram,
//         bin_histogram,
//         13
//     );

//     TEST_ASSERT_EQUAL_UINT32_ARRAY(
//         expected_offsets,
//         bin_offsets,
//         13
//     );

//     TEST_ASSERT_EQUAL_UINT32_ARRAY(
//         expected_indices,
//         bin_indices,
//         17
//     );

//     mems_allocator_free(&mems_system_allocator, bin_histogram);
//     mems_allocator_free(&mems_system_allocator, bin_offsets);
//     mems_allocator_free(&mems_system_allocator, bin_indices);
// }

int main(void)
{
    if (!hwgutil_wgpu_context_init(
        &context,
        1, (WGPUInstanceFeatureName[]){ WGPUInstanceFeatureName_TimedWaitAny },
        1, (WGPUFeatureName[]){ WGPUFeatureName_Subgroups }
    )) abort();

    msg_pipeline_init(
        &pipeline,
        &(msg_options){
            .sort_kernels_root_dir = "shaders/sort_kernels"
        },
        context.instance,
        context.adapter,
        context.device, 
        context.queue,
        &mems_system_allocator
    );

    msg_buffers_init(
        &pipeline,
        &buffers,
        NULL,
        context.device,
        context.queue
    );

    msg_bindings_init(
        &bindings,
        &pipeline,
        &buffers
    );

    UNITY_BEGIN();
    RUN_TEST(test_sort_fixed_4096_seed1337);
    return UNITY_END();
}
