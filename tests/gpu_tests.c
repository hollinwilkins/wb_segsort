#include <stdlib.h>
#include <string.h>
#include <webgpu/webgpu.h>

#include "unity.h"
#include "unity_internals.h"

#define HWGUTIL_WEBGPU_ENABLED
#define HWGUTIL_MEMS_ENABLED
#define HWDS_MEMS_ENABLED

#define WB_SORT_CPU_IMPLEMENTATION
#define WB_SORT_GPU_IMPLEMENTATION
#define HWGUTIL_IMPLEMENTATION
#define HWDS_IMPLEMENTATION
#define MEMS_IMPLEMENTATION

#include "cpu.h"
#include "gpu.h"
#include "hw_gutil.h"
#include "hw_mems.h"

hwgutil_wgpu_context context;
wbg_pipeline pipeline;
wbg_buffers buffers;
wbg_bindings bindings;

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

    uint32_t * expected_keys = (uint32_t *)malloc(len * sizeof(uint32_t));
    memcpy(expected_keys, keys, len * sizeof(uint32_t));
    wbc_segsort_alloc(len, expected_keys, segments_len, segments);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(pipeline.device, &(WGPUCommandEncoderDescriptor){
        .label = (WGPUStringView){
            .data = "Merge Sort: Command Encoder",
            .length = WGPU_STRLEN,
        }
    });

    wbg_sort(
        &pipeline,
        &bindings,
        &buffers,
        encoder,
        segments_len, segments,
        len, keys,
        NULL
    );

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &(WGPUCommandBufferDescriptor){
        .label = (WGPUStringView){
            .data = "WB Sort: Command Buffer",
            .length = WGPU_STRLEN,
        }
    });

    wgpuQueueSubmit(pipeline.queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);

    uint32_t *gpu_keys;
    if (!hwgutil_wgpu_read_buffer_alloc(
        pipeline.instance,
        pipeline.device,
        pipeline.queue,
        buffers.keys,
        &mems_system_allocator,
        (void **)&gpu_keys)
    ) abort();
    
    TEST_ASSERT_EQUAL_UINT32_ARRAY(expected_keys, gpu_keys, len);

    mems_allocator_free(&mems_system_allocator, gpu_keys);
    free(expected_keys);
    free(values);
    free(keys);
    free(segments);
}

void test_sort_fixed_4096_seed1337()
{
    test__sort_kernel(2048, 4096, 1337);
}

void test_sort_variable_seed99()
{
    test__sort_kernel(10000, 128, 99);
}

int main(void)
{
    if (!hwgutil_wgpu_context_init(
        &context,
        1, (WGPUInstanceFeatureName[]){ WGPUInstanceFeatureName_TimedWaitAny },
        1, (WGPUFeatureName[]){ WGPUFeatureName_Subgroups }
    )) abort();

    wbg_pipeline_init(
        &pipeline,
        &(wbg_options){
            .sort_kernels_root_dir = "shaders/sort_kernels"
        },
        context.instance,
        context.adapter,
        context.device, 
        context.queue,
        &mems_system_allocator
    );

    wbg_buffers_init(
        &pipeline,
        &buffers,
        NULL,
        context.device,
        context.queue
    );

    wbg_bindings_init(
        &bindings,
        &pipeline,
        &buffers
    );

    UNITY_BEGIN();
    RUN_TEST(test_sort_fixed_4096_seed1337);
    RUN_TEST(test_sort_variable_seed99);
    return UNITY_END();
}
