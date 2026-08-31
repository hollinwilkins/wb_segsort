#include <stdbool.h>
#include <sys/utsname.h>
#include <sys/sysctl.h>
#include <sys/stat.h>

#include <time.h>
#include <stdint.h>
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
#include "cpu.h"
#include "gpu.h"

#define PANIC(...) { \
    fprintf(stderr, __VA_ARGS__); \
    abort(); \
}

typedef struct benchmark_meta
{
    const char * git_commit;
    const char * os_string;
    const char * cpu_string;
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
    uint32_t bin_counts[13];
    uint32_t bin_key_counts[13];
    const char * wgpu_backend_name;
    const char * wgpu_backend_version;
    const char * wgpu_backend_release_type;
    const char * cpu_release_type;
    const char * bin_sampler;
    const char * key_sampler;
    uint32_t max_key;
    bool subgroups_enabled;
    uint32_t merge_wg;
    uint32_t merge_tile_size;
    uint32_t merge_input_tile_size;
    uint32_t merge_max_passes;
    uint64_t seed;
    size_t key_bit_size;
    size_t n_segments;
    size_t n_keys;
    size_t n_warmup_runs;
    size_t n_runs;
} benchmark_meta;

typedef struct benchmark_result
{
    uint64_t timestamps[6];
    uint64_t wall_time;
} benchmark_result;

typedef struct benchmark_data
{
    uint32_t bin_counts[13];
    uint32_t bin_key_counts[13];
    size_t segments_len;
    const uint32_t * segments;
    size_t keys_len;
    const uint32_t * keys;
} benchmark_data;

static hwstats_sampler * create_uniform_sampler(const uint64_t seed)
{
    hwstats_x256pp * const x256pp = (hwstats_x256pp *)malloc(sizeof(hwstats_x256pp));
    hwstats_randomizer * const r = (hwstats_randomizer *)malloc(sizeof(hwstats_randomizer));
    hwstats_sampler * const sampler = (hwstats_sampler *)malloc(sizeof(hwstats_sampler));

    hwstats_x256pp_init(x256pp, seed);
    hwstats_x256pp_rand_init(x256pp, r);
    hwstats_uniform_sampler_init(sampler, r);

    return sampler;
}

static hwstats_sampler * create_sampler(
    const char * const name,
    const uint64_t seed
)
{
    if (strcmp("uniform", name) == 0) return create_uniform_sampler(seed);
    return NULL;
}

static uint32_t sample_range(hwstats_sampler * const sampler, const uint32_t min, const uint32_t max)
{
    const uint32_t range = max - min;
    return min + (uint32_t)round(hwstats_sample(sampler) * (double)range);
}

#define QUERY_COUNT 8

static void benchmark_sample(
    const wbg_pipeline * const pipeline,
    const wbg_buffers * const buffers,
    const wbg_bindings * const bindings,
    const benchmark_data * const data,
    wbg_sort_timing * const timing,
    WGPUBuffer const query_buffer,
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
        data->segments_len, data->segments,
        data->keys_len, data->keys,
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
    const size_t n_runs,
    const size_t n_warmup_runs,
    const benchmark_data * const data,
    benchmark_result * const results
)
{
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

    benchmark_result warmup_result;
    for (size_t i = 0; i < n_warmup_runs; i++)
    {
        timing.index = 0;

        benchmark_sample(
            pipeline,
            buffers,
            bindings,
            data,
            &timing,
            query_buffer,
            &warmup_result
        );
    }

    for (size_t i = 0; i < n_runs; i++)
    {
        timing.index = 0;

        benchmark_sample(
            pipeline,
            buffers,
            bindings,
            data,
            &timing,
            query_buffer,
            results + i
        );
    }
}

static const char * copy_string(
    const char * const str
)
{
    const size_t len = strlen(str);
    char * const copy_str = (char *)malloc(len + 1);
    memcpy(copy_str, str, len);
    copy_str[len] = 0;

    return copy_str;
}

static const char * copy_string_view(
    WGPUStringView view
)
{
    const size_t len = view.length == WGPU_STRLEN ? strlen(view.data) : view.length;
    char * const copy_str = (char *)malloc(len + 1);
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

static char CPU_STRING[1024];
static const char * benchmark_cpu_string() {
#if defined(__APPLE__)
    size_t len = sizeof(CPU_STRING);
    if (sysctlbyname("machdep.cpu.brand_string", CPU_STRING, &len, NULL, 0) != 0)
    {
        snprintf(CPU_STRING, sizeof(CPU_STRING), "unknown");
    }
#elif defined(__linux__)
    CPU_STRING[0] = '\0';

    FILE * f = fopen("/proc/cpuinfo", "r");
    if (f)
    {
        char line[256];
        while (fgets(line, sizeof(line), f))
        {
            char * c = strchr(line, ':');
            if (c && strncmp(line, "model name", 10) == 0)
            {
                c += 2;                       // skip ": "
                c[strcspn(c, "\n")] = '\0';   // trim newline
                snprintf(CPU_STRING, sizeof(CPU_STRING), "%s", c);
                break;
            }
        }
        fclose(f);
    }

    if (CPU_STRING[0] == '\0') snprintf(CPU_STRING, sizeof(CPU_STRING), "unknown");
#else
    snprintf(CPU_STRING, sizeof(CPU_STRING), "unknown");
#endif

    return CPU_STRING;
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
    const uint32_t * const bin_counts,
    const uint32_t * const bin_key_counts,
    const char * bin_sampler,
    const char * key_sampler,
    const uint64_t seed,
    const uint64_t max_key,
    const size_t n_segments,
    const size_t n_keys,
    const size_t n_warmup_runs,
    const size_t n_runs,
    const bool subgroups_enabled,
    const wbg_pipeline * const pipeline,
    const hwgutil_wgpu_context * const context
)
{
    WGPUAdapterInfo adapter_info;
    wgpuAdapterGetInfo(context->adapter, &adapter_info);

    wbg_gpu_bin * const bins = (wbg_gpu_bin *)malloc(sizeof(pipeline->bins));
    memcpy(bins, pipeline->bins, sizeof(pipeline->bins));

    *meta = (benchmark_meta){
        .git_commit = BENCH_GIT_COMMIT,
        .os_string = copy_string(benchmark_os_string()),
        .cpu_string = copy_string(benchmark_cpu_string()),
        .gpu_vendor = copy_string_view(adapter_info.vendor),
        .gpu_architecture = copy_string_view(adapter_info.architecture),
        .gpu_device = copy_string_view(adapter_info.device),
        .gpu_description = copy_string_view(adapter_info.description),
        .gpu_backend_type = copy_string(benchmark_meta_gpu_backend_type_name(adapter_info.backendType)),
        .gpu_adapter_type = copy_string(benchmark_meta_gpu_adapter_type_name(adapter_info.adapterType)),
        .gpu_vendor_id = adapter_info.vendorID,
        .gpu_device_id = adapter_info.deviceID,
        .subgroup_min_size = adapter_info.subgroupMinSize,
        .subgroup_max_size = adapter_info.subgroupMaxSize,
        .bins = bins,
        .wgpu_backend_name = BENCH_BACKEND_NAME,
        .wgpu_backend_version = BENCH_BACKEND_VERSION,
        .wgpu_backend_release_type = BENCH_BACKEND_RELEASE_TYPE,
        .cpu_release_type = BENCH_CPU_RELEASE_TYPE,
        .bin_sampler = copy_string(bin_sampler),
        .key_sampler = copy_string(key_sampler),
        .subgroups_enabled = subgroups_enabled,
        .max_key = max_key,
        .merge_wg = WBG_MERGE_WG,
        .merge_tile_size = WBG_MERGE_TILE_SIZE,
        .merge_input_tile_size = WBG_MERGE_TILE_SIZE,
        .merge_max_passes = WBG_MERGE_TILE_MAX_DEPTH,
        .seed = seed,
        .key_bit_size = 32,
        .n_segments = n_segments,
        .n_keys = n_keys,
        .n_warmup_runs = n_warmup_runs,
        .n_runs = n_runs,
    };

    memcpy(meta->bin_counts, bin_counts, sizeof(meta->bin_counts));
    memcpy(meta->bin_key_counts, bin_key_counts, sizeof(meta->bin_key_counts));
}

static uint32_t benchmark_segment_bucket(const uint32_t segment_len)
{
    const uint32_t v = (segment_len == 0u ? 1u : segment_len) - 1u;
    if (v == 0u) return 0u;
    const uint32_t b = 32u - (uint32_t)__builtin_clz(v);
    return b < 12u ? b : 12u;
}

static void benchmark_data_init(
    benchmark_data * const data,
    const size_t key_budget,
    const uint32_t max_key,
    hwstats_sampler * const bin_sampler,
    hwstats_sampler * const key_sampler
)
{
    size_t segments_len = 0;
    uint32_t * const segments = (uint32_t *)malloc(key_budget * 16 * sizeof(uint32_t));

    uint32_t bin_counts[13] = {0};
    uint32_t bin_key_counts[13] = {0};

    uint32_t keys_len = 0;
    while (keys_len < key_budget)
    {
        if (segments_len == key_budget)
        {
            fprintf(stderr, "segments overflow, probably an invalid bin sampler\n");
            abort();
        }
        uint32_t bin = sample_range(bin_sampler, 0, 12);

        // 0 -> [0,0]
        // 1 -> [1,1]
        // 2 -> [2,3]
        // 3 -> [4,7]
        // 4 -> [8,15]
        const uint32_t lo = bin <= 1 ? bin : (1u << (bin - 1u));
        const uint32_t hi = bin == 12 ?
            max_key :
            (bin <= 1 ? bin : (1u << bin) - 1u);
        uint32_t segment_len = sample_range(bin_sampler, lo, hi);

        if (keys_len + segment_len > key_budget)
        {
            segment_len = key_budget - keys_len;
            bin = benchmark_segment_bucket(segment_len);
        }

        bin_counts[bin]++;
        bin_key_counts[bin] += segment_len;
        keys_len += segment_len;
        segments[segments_len++] = keys_len;
    }

    uint32_t * const keys = (uint32_t *)malloc(keys_len * sizeof(uint32_t));

    for (uint32_t i = 0; i < keys_len; i++)
    {
        keys[i] = sample_range(key_sampler, 0, max_key);
    }

    *data = (benchmark_data){
        .keys_len = keys_len,
        .keys = keys,
        .segments_len = segments_len,
        .segments = segments,
    };

    memcpy(data->bin_counts, bin_counts, sizeof(bin_counts));
    memcpy(data->bin_key_counts, bin_key_counts, sizeof(bin_key_counts));
}

static void write_json_string_field(
    FILE * file,
    const bool last_field,
    const uint32_t indent,
    const char * const key,
    const char * const value
)
{
    const char * const comma = last_field ? "" : ",";
    for (uint32_t i = 0; i < indent; i++) fprintf(file, " ");
    fprintf(file, "\"%s\": \"%s\"%s\n", key, value, comma);
}

static void write_json_uint32_field(
    FILE * file,
    const bool last_field,
    const uint32_t indent,
    const char * const key,
    const uint32_t value
)
{
    const char * const comma = last_field ? "" : ",";
    for (uint32_t i = 0; i < indent; i++) fprintf(file, " ");
    fprintf(file, "\"%s\": %u%s\n", key, value, comma);
}

static void write_json_uint64_field(
    FILE * file,
    const bool last_field,
    const uint32_t indent,
    const char * const key,
    const uint64_t value
)
{
    static char STRING_VALUE[128];
    snprintf(STRING_VALUE, sizeof(STRING_VALUE), "%llu", value);
    write_json_string_field(file, last_field, indent, key, STRING_VALUE);
}

static void write_json_size_field(
    FILE * file,
    const bool last_field,
    const uint32_t indent,
    const char * const key,
    const size_t value
)
{
    const char * const comma = last_field ? "" : ",";
    for (uint32_t i = 0; i < indent; i++) fprintf(file, " ");
    fprintf(file, "\"%s\": %zu%s\n", key, value, comma);
}

static void write_json_bool_field(
    FILE * file,
    const bool last_field,
    const uint32_t indent,
    const char * const key,
    const bool value
)
{
    const char * const comma = last_field ? "" : ",";
    const char * const bool_name = value ? "true" : "false";
    for (uint32_t i = 0; i < indent; i++) fprintf(file, " ");
    fprintf(file, "\"%s\": %s%s\n", key, bool_name, comma);
}

static void write_benchmark_results(
    const char * const root_dir,
    const benchmark_meta * const meta,
    const size_t results_len,
    const benchmark_result * const results
)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    const uint64_t timestamp = (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;

    static char BENCHMARK_DIR[1024];
    snprintf(BENCHMARK_DIR, sizeof(BENCHMARK_DIR), "%s/benchmark_%llu", root_dir, timestamp);

    if (mkdir(BENCHMARK_DIR, 0755) != 0)
    {
        fprintf(stderr, "could not create benchmark dir: %s\n", BENCHMARK_DIR);
        abort();
    }

    static char BENCHMARK_FILE[1024];
    snprintf(BENCHMARK_FILE, sizeof(BENCHMARK_FILE), "%s/meta.json", BENCHMARK_DIR);

    FILE * mf = fopen(BENCHMARK_FILE, "w");
    fprintf(mf, "{\n");
        write_json_string_field(mf, false, 2, "git_commit", meta->git_commit);
        write_json_string_field(mf, false, 2, "os_string", meta->os_string);
        write_json_string_field(mf, false, 2, "cpu_string", meta->cpu_string);
        write_json_string_field(mf, false, 2, "gpu_vendor", meta->gpu_vendor);
        write_json_string_field(mf, false, 2, "gpu_architecture", meta->gpu_architecture);
        write_json_string_field(mf, false, 2, "gpu_device", meta->gpu_device);
        write_json_string_field(mf, false, 2, "gpu_description", meta->gpu_description);
        write_json_string_field(mf, false, 2, "gpu_backend_type", meta->gpu_backend_type);
        write_json_string_field(mf, false, 2, "gpu_adapter_type", meta->gpu_adapter_type);
        write_json_uint32_field(mf, false, 2, "gpu_vendor_id", meta->gpu_vendor_id);
        write_json_uint32_field(mf, false, 2, "gpu_device_id", meta->gpu_device_id);
        write_json_uint32_field(mf, false, 2, "subgroup_min_size", meta->subgroup_min_size);
        write_json_uint32_field(mf, false, 2, "subgroup_max_size", meta->subgroup_max_size);
        write_json_string_field(mf, false, 2, "wgpu_backend_name", meta->wgpu_backend_name);
        write_json_string_field(mf, false, 2, "wgpu_backend_version", meta->wgpu_backend_version);
        write_json_string_field(mf, false, 2, "wgpu_backend_release_type", meta->wgpu_backend_release_type);
        write_json_string_field(mf, false, 2, "cpu_release_type", meta->cpu_release_type);
        write_json_string_field(mf, false, 2, "bin_sampler", meta->bin_sampler);
        write_json_string_field(mf, false, 2, "key_sampler", meta->key_sampler);
        write_json_uint32_field(mf, false, 2, "max_key", meta->max_key);
        write_json_bool_field(mf, false, 2, "subgroups_enabled", meta->subgroups_enabled);
        write_json_uint32_field(mf, false, 2, "merge_wg", meta->merge_wg);
        write_json_uint32_field(mf, false, 2, "merge_tile_size", meta->merge_tile_size);
        write_json_uint32_field(mf, false, 2, "merge_input_tile_size", meta->merge_input_tile_size);
        write_json_uint32_field(mf, false, 2, "merge_max_passes", meta->merge_max_passes);
        write_json_uint64_field(mf, false, 2, "seed", meta->seed);
        write_json_size_field(mf, false, 2, "key_bit_size", meta->key_bit_size);
        write_json_size_field(mf, false, 2, "n_segments", meta->n_segments);
        write_json_size_field(mf, false, 2, "n_keys", meta->n_keys);
        write_json_size_field(mf, false, 2, "n_warmup_runs", meta->n_warmup_runs);
        write_json_size_field(mf, false, 2, "n_runs", meta->n_runs);

        fprintf(mf, "  \"bins\": [");
        for (int i = 0; i < 13; i++)
        {
            wbg_gpu_bin bin = meta->bins[i];
            if (i != 0) fprintf(mf, ", ");
            fprintf(mf, "{\n");
            if (i == 0)
            {
                write_json_uint32_field(mf, false, 4, "n_segments", meta->bin_counts[i]);
                write_json_uint32_field(mf, false, 4, "n_keys", meta->bin_key_counts[i]);
                write_json_bool_field(mf, true, 4, "is_empty", true);
            }
            else if (i > 0)
            {
                if ((bin.flags & wbg_bin_flag_is_variable) != 0)
                {
                    write_json_uint32_field(mf, false, 4, "n_segments", meta->bin_counts[i]);
                    write_json_uint32_field(mf, false, 4, "n_keys", meta->bin_key_counts[i]);
                    write_json_bool_field(mf, true, 4, "is_merge", true);
                }
                else
                {
                    write_json_uint32_field(mf, false, 4, "n_segments", meta->bin_counts[i]);
                    write_json_uint32_field(mf, false, 4, "n_keys", meta->bin_key_counts[i]);
                    write_json_uint32_field(mf, false, 4, "N", bin.n);
                    write_json_uint32_field(mf, false, 4, "M", bin.m);
                    write_json_uint32_field(mf, true, 4, "wg", bin.wg);
                }
            }
            fprintf(mf, "  }");
        }
        fprintf(mf, "]\n");
    fprintf(mf, "}\n");
}

int main(int argc, char ** argv)
{
    char * endptr;

    const char * const sampler_name = argv[1];
    const char * const key_sampler_name = argv[2];
    const uint64_t seed = strtoull(argv[3], &endptr, 10);
    const size_t key_budget = strtoull(argv[4], &endptr, 10);
    const size_t n_runs = strtoull(argv[5], &endptr, 10);
    const size_t n_warmup_runs = strtoull(argv[6], &endptr, 10);
    const uint32_t max_key = strtol(argv[7], &endptr, 10);
    const uint32_t subgroup_test = strtol(argv[8], &endptr, 10);
    
    const bool subgroups_enabled = subgroup_test == 1;

    hwgutil_wgpu_context context;
    wbg_pipeline pipeline;
    wbg_buffers buffers;
    wbg_bindings bindings;

    printf("Initializing WebGPU context...\n");

    size_t features_len;
    WGPUFeatureName features[2];
    if (subgroups_enabled)
    {
        features_len = 2;
        features[0] = WGPUFeatureName_Subgroups;
        features[1] = WGPUFeatureName_TimestampQuery;
    }
    else
    {
        features_len = 1;
        features[0] = WGPUFeatureName_TimestampQuery;
    }

    if (!hwgutil_wgpu_context_init(
        &context,
        1, (WGPUInstanceFeatureName[]){ WGPUInstanceFeatureName_TimedWaitAny },
        features_len, features
    )) abort();

    if (subgroups_enabled)
    {
        if (!wgpuAdapterHasFeature(context.adapter, WGPUFeatureName_Subgroups))
        {
            fprintf(stderr, "subgroups are not available with this adapter");
        }
    }

    wbg_pipeline_init(
        &pipeline,
        &(wbg_options){
            .sort_kernels_root_dir = "shaders/sort_kernels",
            .subgroups_enabled = subgroups_enabled,
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

    hwstats_sampler * const bin_sampler = create_sampler(
        sampler_name,
        seed
    );

    hwstats_sampler * const key_sampler = create_sampler(
        key_sampler_name,
        seed
    );

    benchmark_data data;
    benchmark_data_init(
        &data,
        key_budget,
        max_key,
        bin_sampler,
        key_sampler
    );

    benchmark_meta meta;
    benchmark_meta_init(
        &meta,
        data.bin_counts,
        data.bin_key_counts,
        sampler_name,
        key_sampler_name,
        seed,
        max_key,
        data.segments_len,
        data.keys_len,
        n_warmup_runs,
        n_runs,
        true,
        &pipeline,
        &context
    );

    benchmark_result * const results = (benchmark_result *)malloc(n_runs * sizeof(benchmark_result));

    benchmark(
        &pipeline,
        &buffers,
        &bindings,
        n_runs,
        n_warmup_runs,
        &data,
        results
    );

    write_benchmark_results(
        "output",
        &meta,
        n_runs,
        results
    );

    return 0;
}
