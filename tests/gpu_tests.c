#include <stdlib.h>
#include <string.h>
#include <webgpu/webgpu.h>

#include "hw_mems.h"
#include "unity.h"
#include "unity_internals.h"

#define HWGUTIL_WEBGPU_ENABLED
#define HWGUTIL_MEMS_ENABLED

#define MERGE_SORT_GPU_IMPLEMENTATION
#define HWGUTIL_IMPLEMENTATION
#define MEMS_IMPLEMENTATION

#include "gpu.h"
#include "hw_gutil.h"

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

void test_bin(void)
{
    msg_gpu_config config;

    const uint32_t segments[] = {
        0, 1, 3, 6, 10, 17,
        25, 40, 72, 136, 264, 520,
        1032, 2056, 4104, 8200, 17088
    };
    const size_t segments_len = sizeof(segments) / sizeof(uint32_t);

    msg_prepare(pipeline.queue, &buffers, segments_len, segments, &config);

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

    uint32_t * bin_histogram;
    if (!hwgutil_wgpu_read_buffer_alloc(
        pipeline.instance,
        pipeline.device,
        pipeline.queue,
        buffers.bin_histogram,
        &mems_system_allocator,
        (void **)&bin_histogram
    )) abort();

    const uint32_t expected[] = {
        2, 3, 5, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
        17
    };

    TEST_ASSERT_EQUAL_UINT32_ARRAY(
        expected,
        bin_histogram,
        13
    );

    mems_allocator_free(&mems_system_allocator, bin_histogram);
}

int main(void)
{
    if (!hwgutil_wgpu_context_init(&context)) abort();

    if (!msg_pipeline_init(
        &pipeline,
        NULL,
        context.instance,
        context.adapter,
        context.device, 
        context.queue
    )) abort();

    msg_buffers_init(
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
    RUN_TEST(test_bin);
    return UNITY_END();
}
