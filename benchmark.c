#include <sys/utsname.h>

#if defined(__APPLE__)
#   include <sys/sysctl.h>
#endif

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

typedef struct benchmark_meta
{
    const char * os_string;
    const char * gpu_vendor;
    const char * gpu_architecture;
    const char * gpu_device;
    const char * gpu_description;
    const char * gpu_backend_type;
    const char * gpu_adapter_type;
    uint32_t gpu_vendor_id;
    uint32_t gpu_device_id;
    uint32_t subgroup_min_size;
    uint32_t subgroup_max_size;
    const wbg_gpu_bin * bins;
    const char * wgpu_backend_name;
    const char * wgpu_backend_version;
    const char * wgpu_backend_release_type;
    const char * cpu_release_type;
    const char * bin_sampler;
    const char * key_sampler;
    bool subgroups_enabled;
    size_t n_segments;
    size_t n_runs;
} benchmark_meta;

typedef struct benchmark_result
{
    uint32_t bin_counts[13];
    uint64_t timestamps[6];
} benchmark_result;

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

static void report_results(
    const size_t len,
    benchmark_result * const results
)
{
    for (size_t i = 0; i < len; i++)
    {

    }
}

#define QUERY_COUNT 8

static void benchmark_sample(
    const wbg_pipeline * const pipeline,
    const wbg_buffers * const buffers,
    const wbg_bindings * const bindings,
    hwstats_sampler * const bin_sampler,
    hwstats_sampler * const key_sampler,
    wbg_sort_timing * const timing,
    WGPUBuffer const query_buffer,
    const size_t segments_len, uint32_t * const segments,
    const size_t keys_len, uint32_t * const keys,
    benchmark_result * const result
)
{
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
        segments_len, segments,
        keys_len, keys,
        timing
    );

    wgpuCommandEncoderResolveQuerySet(
        encoder,
        timing->query,
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
}

static void benchmark(
    const wbg_pipeline * const pipeline,
    const wbg_buffers * const buffers,
    const wbg_bindings * const bindings,
    hwstats_sampler * const bin_sampler,
    hwstats_sampler * const key_sampler,
    const size_t n_segments,
    const size_t n_runs,
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

    benchmark_result * const results = (benchmark_result *)mems_allocator_alloc(
        allocator,
        MEMS_ALIGNOF(benchmark_result),
        sizeof(benchmark_result)
    );

    for (size_t i = 0; i < n_runs; i++)
    {
        timing.index = 0;

        benchmark_sample(
            pipeline,
            buffers,
            bindings,
            bin_sampler,
            key_sampler,
            &timing,
            query_buffer,
            n_segments, segments,
            len, keys,
            results + i
        );
    }

    report_results(n_runs, results);
}

static const char * copy_string(
    const char * const str,
    const mems_allocator * const allocator
)
{
    const size_t len = strlen(str);
    char * const copy_str = (char *)mems_allocator_alloc(
        allocator,
        MEMS_ALIGNOF(char),
        len + 1
    );
    memcpy(copy_str, str, len);
    copy_str[len] = 0;

    return copy_str;
}

static const char * copy_string_view(
    WGPUStringView view,
    const mems_allocator * const allocator
)
{
    const size_t len = view.length == WGPU_STRLEN ? strlen(view.data) : view.length;
    char * const copy_str = (char *)mems_allocator_alloc(
        allocator,
        MEMS_ALIGNOF(char),
        len + 1
    );
    memcpy(copy_str, view.data, len);
    copy_str[len] = 0;

    return copy_str;
}

static char OS_STRING[1024];
static const char * benchmark_os_string() {
    struct utsname u;
    uname(&u);

#if defined(__APPLE__)
    char product[256] = {0};
    size_t len = sizeof(product);
    if (sysctlbyname("kern.osproductversion", product, &len, NULL, 0) == 0)
    {
        const size_t required_len = snprintf(OS_STRING, sizeof(OS_STRING), "macOS %s (Darwin %s %s)",
            product,
            u.release,
            u.machine
        );

        if (required_len >= sizeof(OS_STRING)) PANIC("increase size of OS_STRING");
    }
    else
    {
        snprintf(OS_STRING, sizeof(OS_STRING), "%s %s %s", u.sysname, u.release, u.machine);
    }
#else
    snprintf(OS_STRING, sizeof(OS_STRING), "%s %s %s", u.sysname, u.release, u.machine);
#endif

    return OS_STRING;
}

static const char * benchmark_meta_gpu_backend_type_name(const WGPUBackendType t)
{
    switch (t)
    {
        case WGPUBackendType_WebGPU: return "WebGPU";
        case WGPUBackendType_D3D11: return "D3D11";
        case WGPUBackendType_D3D12: return "D3D12";
        case WGPUBackendType_Metal: return "Metal";
        case WGPUBackendType_OpenGL: return "OpenGL";
        case WGPUBackendType_OpenGLES: return "OpenGLES";
        case WGPUBackendType_Vulkan: return "Vulkan";
        default: PANIC("Unknown backend type");
    }
}

static const char * benchmark_meta_gpu_adapter_type_name(const WGPUAdapterType t)
{
    switch (t)
    {
        case WGPUAdapterType_DiscreteGPU: return "discrete_gpu";
        case WGPUAdapterType_IntegratedGPU: return "integrated_gpu";
        case WGPUAdapterType_CPU: return "cpu";
        default: PANIC("unknown adapter type name");
    }
}

static void benchmark_meta_init(
    benchmark_meta * const meta,
    const char * bin_sampler,
    const char * key_sampler,
    uint64_t seed,
    size_t n_segments,
    size_t n_runs,
    bool subgroups_enabled,
    const wbg_pipeline * const pipeline,
    const hwgutil_wgpu_context * const context,
    const mems_allocator * const allocator
)
{
    WGPUAdapterInfo adapter_info;
    wgpuAdapterGetInfo(context->adapter, &adapter_info);

    wbg_gpu_bin * const bins = (wbg_gpu_bin *)malloc(sizeof(pipeline->bins));
    memcpy(bins, pipeline->bins, sizeof(pipeline->bins));

    *meta = (benchmark_meta){
        .os_string = copy_string(benchmark_os_string(), allocator),
        .gpu_vendor = copy_string_view(adapter_info.vendor, allocator),
        .gpu_architecture = copy_string_view(adapter_info.architecture, allocator),
        .gpu_device = copy_string_view(adapter_info.device, allocator),
        .gpu_description = copy_string_view(adapter_info.description, allocator),
        .gpu_backend_type = copy_string(benchmark_meta_gpu_backend_type_name(adapter_info.backendType), allocator),
        .gpu_adapter_type = copy_string(benchmark_meta_gpu_adapter_type_name(adapter_info.adapterType), allocator),
        .gpu_vendor_id = adapter_info.vendorID,
        .gpu_device_id = adapter_info.deviceID,
        .subgroup_min_size = adapter_info.subgroupMinSize,
        .subgroup_max_size = adapter_info.subgroupMaxSize,
        .bins = bins,
        .wgpu_backend_name = BENCH_BACKEND_NAME,
        .wgpu_backend_version = BENCH_BACKEND_VERSION,
        .bin_sampler = copy_string(bin_sampler, allocator),
        .key_sampler = copy_string(key_sampler, allocator),
        .subgroups_enabled = subgroups_enabled,
        .n_segments = n_segments,
        .n_runs = n_runs,
    };
}

int main(int argc, char ** argv)
{
    char * endptr;

    const char * const sampler_name = argv[1];
    const uint64_t seed = strtoull(argv[2], &endptr, 10);
    const size_t n_segments = strtoull(argv[3], &endptr, 10);
    const size_t n_runs = strtoull(argv[4], &endptr, 10);

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

    static const size_t BUFFER_LEN = 1024 * 1024 * 32;
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
