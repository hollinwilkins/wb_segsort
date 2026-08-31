#include <stdint.h>
#include <stdio.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <webgpu/webgpu.h>

#define HWDS_MEMS_ENABLED
#define HWGUTIL_MEMS_ENABLED
#define HWGUTIL_WEBGPU_ENABLED

#define WB_SORT_CPU_IMPLEMENTATION
#define WB_SORT_GPU_IMPLEMENTATION
#define HWSTATS_IMPLEMENTATION
#define HWGUTIL_IMPLEMENTATION
#define HWDS_IMPLEMENTATION
#define MEMS_IMPLEMENTATION

#include <webgpu/webgpu.h>

#include "hw_stats.h"
#include "hw_gutil.h"
#include "hw_mems.h"
#include "cpu.h"
#include "gpu.h"

#define PANIC(...) { \
    fprintf(stderr, __VA_ARGS__); \
    abort(); \
}

static hwstats_sampler * create_uniform_sampler(const uint64_t seed, const mems_allocator * const allocator)
{
    hwstats_x256pp * const x256pp = (hwstats_x256pp *)mems_allocator_alloc(
        allocator,
        MEMS_ALIGNOF(hwstats_x256pp),
        sizeof(hwstats_x256pp)
    );
    hwstats_randomizer * const r = (hwstats_randomizer *)mems_allocator_alloc(
        allocator,
        MEMS_ALIGNOF(hwstats_randomizer),
        sizeof(hwstats_randomizer)
    );
    hwstats_sampler * const sampler = (hwstats_sampler *)mems_allocator_alloc(
        allocator,
        MEMS_ALIGNOF(hwstats_sampler),
        sizeof(hwstats_sampler)
    );

    hwstats_x256pp_init(x256pp, seed);
    hwstats_x256pp_rand_init(x256pp, r);
    hwstats_uniform_sampler_init(sampler, r);

    return sampler;
}

static hwstats_sampler * create_sampler(
    const char * const name,
    const uint64_t seed,
    const mems_allocator * const allocator
)
{
    if (strcmp("uniform", name) == 0) return create_uniform_sampler(seed, allocator);
    return NULL;
}

static uint32_t sample_range(hwstats_sampler * const sampler, const uint32_t min, const uint32_t max)
{
    const uint32_t range = max - min;
    return min + (uint32_t)round(hwstats_sample(sampler) * (double)range);
}

#define QUERY_COUNT 8

static void benchmark(
    const wbg_pipeline * const pipeline,
    const wbg_buffers * const buffers,
    const wbg_bindings * const bindings,
    hwstats_sampler * const bin_sampler,
    hwstats_sampler * const key_sampler,
    const uint64_t n_segments,
    const uint64_t n_runs,
    const mems_allocator * const allocator
)
{
    uint32_t * const bin_counts = (uint32_t *)mems_allocator_alloc(allocator, MEMS_ALIGNOF(uint32_t), 13);
    memset(bin_counts, 0, 13 * sizeof(uint32_t));

    WGPUQuerySetDescriptor query_desc = WGPU_QUERY_SET_DESCRIPTOR_INIT;
    query_desc.label = (WGPUStringView){
        .data = "WB Sort: Timestamp Queries",
        .length = WGPU_STRLEN,
    };
    query_desc.type = WGPUQueryType_Timestamp;
    query_desc.count = QUERY_COUNT;

    WGPUQuerySet query = wgpuDeviceCreateQuerySet(pipeline->device, &query_desc);
    wbg_sort_timing timing = (wbg_sort_timing){
        .query = query,
    };

    WGPUBufferDescriptor query_buffer_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    query_buffer_desc.label = (WGPUStringView){
        .data = "WB Sort: Timestamp Queries",
        .length = WGPU_STRLEN,
    };
    query_buffer_desc.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    query_buffer_desc.size = QUERY_COUNT * sizeof(uint64_t);

    WGPUBuffer query_buffer = wgpuDeviceCreateBuffer(pipeline->device, &query_buffer_desc);

    uint32_t * const segments = (uint32_t *)malloc(n_segments * sizeof(uint32_t));

    for (uint32_t i = 0; i < 13; i++)
    {
        const uint32_t lo = i == 0 ? 0 : 1u << (i - 1u);
        const uint32_t hi = 1u << i;

        printf("Bin(%u): Lo(%u) -> Hi(%u)\n", i, lo, hi);
    }

    uint32_t len = 0;
    for (uint32_t i = 0; i < n_segments; i++)
    {
        const uint32_t bin = sample_range(bin_sampler, 0, 12);
        bin_counts[bin]++;

        const uint32_t lo = bin == 0 ? 0 : 1u << (bin - 1u);
        const uint32_t hi = 1u << bin;
        const uint32_t segment_len = sample_range(key_sampler, lo, hi);

        len += segment_len;
        segments[i] = len;
    }

    uint32_t * const keys = (uint32_t *)malloc(len * sizeof(uint32_t));

    for (uint32_t i = 0; i < len; i++)
    {
        keys[i] = rand() % 100;
    }

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(pipeline->device, &(WGPUCommandEncoderDescriptor){
        .label = (WGPUStringView){
            .data = "Merge Sort: Command Encoder",
            .length = WGPU_STRLEN,
        }
    });

    wbg_sort(
        pipeline,
        bindings,
        buffers,
        encoder,
        n_segments, segments,
        len, keys,
        &timing
    );

    wgpuCommandEncoderResolveQuerySet(
        encoder,
        timing.query,
        0,
        QUERY_COUNT,
        query_buffer,
        0
    );

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &(WGPUCommandBufferDescriptor){
        .label = (WGPUStringView){
            .data = "WB Sort: Command Buffer",
            .length = WGPU_STRLEN,
        }
    });

    wgpuQueueSubmit(pipeline->queue, 1, &commands);

    wgpuCommandBufferRelease(commands);
    wgpuCommandEncoderRelease(encoder);

    uint64_t * timestamps;
    hwgutil_wgpu_read_buffer_alloc(
        pipeline->instance,
        pipeline->device,
        pipeline->queue,
        query_buffer,
        &mems_system_allocator,
        (void **)&timestamps
    );

    for (uint32_t i = 0; i < 13; i++)
    {
        printf("Bin(%u) segment count %u\n", i, bin_counts[i]);
    }

    printf("\n\n");

    for (int i = 0; i < QUERY_COUNT; i++)
    {
        printf("Blah %llu\n", timestamps[i]);
    }
}

int main(int argc, char ** argv)
{
    char * endptr;

    const char * const sampler_name = argv[1];
    const uint64_t seed = strtoull(argv[2], &endptr, 10);
    const uint64_t n_segments = strtoull(argv[3], &endptr, 10);
    const uint64_t n_runs = strtoull(argv[4], &endptr, 10);

    hwgutil_wgpu_context context;
    wbg_pipeline pipeline;
    wbg_buffers buffers;
    wbg_bindings bindings;

    printf("Initializing WebGPU context...\n");

    if (!hwgutil_wgpu_context_init(
        &context,
        1, (WGPUInstanceFeatureName[]){ WGPUInstanceFeatureName_TimedWaitAny },
        2, (WGPUFeatureName[]){ WGPUFeatureName_Subgroups, WGPUFeatureName_TimestampQuery }
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

    static const size_t BUFFER_LEN = 1024 * 1024 * 4;
    void * const buffer = malloc(BUFFER_LEN);

    mems_bump bump;
    mems_bump_init(
        &bump,
        BUFFER_LEN,
        buffer
    );

    mems_allocator allocator;
    mems_bump_allocator_init(&bump, &allocator);

    hwstats_sampler * const bin_sampler = create_sampler(
        sampler_name,
        seed,
        &allocator
    );

    hwstats_sampler * const key_sampler = create_uniform_sampler(seed, &allocator);

    benchmark(
        &pipeline,
        &buffers,
        &bindings,
        bin_sampler,
        key_sampler,
        n_segments,
        n_runs,
        &allocator
    );

    return 0;
}
