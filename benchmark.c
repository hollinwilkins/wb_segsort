#include <stdio.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HWDS_MEMS_ENABLED

#define WB_SORT_CPU_IMPLEMENTATION
#define WB_SORT_GPU_IMPLEMENTATION
#define HWSTATS_IMPLEMENTATION
#define HWGUTIL_IMPLEMENTATION
#define HWDS_IMPLEMENTATION
#define MEMS_IMPLEMENTATION

#include "hw_stats.h"
#include "hw_gutil.h"
#include "hw_mems.h"
#include "cpu.h"
#include "gpu.h"

#define PANIC(...) { \
    fprintf(stderr, __VA_ARGS__); \
    abort(); \
}

static hwstats_sampler * create_normal_sampler(const uint64_t seed, const mems_allocator * const allocator)
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
    hwstats_normal * const normal = (hwstats_normal *)mems_allocator_alloc(
        allocator,
        MEMS_ALIGNOF(hwstats_normal),
        sizeof(hwstats_normal)
    );
    hwstats_sampler * const sampler = (hwstats_sampler *)mems_allocator_alloc(
        allocator,
        MEMS_ALIGNOF(hwstats_sampler),
        sizeof(hwstats_sampler)
    );

    hwstats_x256pp_init(x256pp, seed);
    hwstats_x256pp_rand_init(x256pp, r);
    hwstats_normal_init_standard(normal, r);
    hwstats_normal_sampler_init(sampler, normal);

    return sampler;
}

static hwstats_sampler * create_sampler(
    const char * const name,
    const uint64_t seed,
    const mems_allocator * const allocator
)
{
    if (strcmp("normal", name) == 0) return create_normal_sampler(seed, allocator);
    return NULL;
}

int main(void)
{
    hwgutil_wgpu_context context;
    wbg_pipeline pipeline;
    wbg_buffers buffers;
    wbg_bindings bindings;

    printf("Initializing WebGPU context...\n");

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

    return 0;
}
