#include <stdlib.h>
#include <string.h>
#include <webgpu/webgpu.h>

#include "unity.h"
#include "unity_internals.h"

#define HWGUTIL_WEBGPU_ENABLED

#define MERGE_SORT_GPU_IMPLEMENTATION
#define HWGUTIL_IMPLEMENTATION

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

void test_bin_histogram(void)
{
    msg_gpu_config config;

    const uint32_t segments[] = {
        0, 1, 2, 3, 4, 7,
        8, 15, 32, 64, 128, 256,
        512, 1024, 2048, 4096, 8888
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

    msg_run_bin_histogram_kernel(
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

    TEST_ASSERT_TRUE(true);
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
        context.device
    );

    msg_bindings_init(
        &bindings,
        &pipeline,
        &buffers
    );

    UNITY_BEGIN();
    RUN_TEST(test_bin_histogram);
    return UNITY_END();
}
