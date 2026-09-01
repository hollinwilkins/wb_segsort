#include <stdbool.h>
#include <sys/utsname.h>
#include <sys/sysctl.h>
#include <sys/stat.h>

#include <time.h>
#include <errno.h>
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

typedef enum benchmark_kind
{
    benchmark_wbc = 1,
    benchmark_wbg = 2,
    // add bb_segsort and friend here later
} benchmark_kind;

typedef struct benchmark_meta
{
    benchmark_kind kind;
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
    const char * store;
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

typedef struct benchmark_result_wbc
{
    uint64_t wall_start_ns;
    uint64_t wall_end_ns;
    uint64_t upload_start_ns;
    uint64_t upload_end_ns;
    uint64_t sort_start_ns;
    uint64_t sort_end_ns;
} benchmark_result_wbc;

typedef struct benchmark_result_wbg
{
    uint64_t timestamps[6];
    uint64_t wall_start_ns;
    uint64_t wall_end_ns;
    uint64_t upload_start_ns;
    uint64_t upload_end_ns;
    uint64_t sort_start_ns;
    uint64_t sort_end_ns;
} benchmark_result_wbg;

typedef struct benchmark_result_data
{
    benchmark_result_wbc wbc;
    benchmark_result_wbg wbg;
} benchmark_result_data;

typedef struct benchmark_result
{
    benchmark_kind kind;
    benchmark_result_data data;
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

typedef struct benchmark_config
{
    const char * kind_name;
    const char * results_dir;
    const char * sampler_name;
    const char * key_sampler_name;
    uint64_t seed;
    size_t key_budget;
    size_t n_runs;
    size_t n_warmup_runs;
    uint32_t max_key;
    bool subgroups_enabled;
    const char * store_name;
} benchmark_config;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

static void benchmark_work_done_cb(
    WGPUQueueWorkDoneStatus const status,
    WGPUStringView const message,
    void * const userdata1,
    void * const userdata2
)
{
    (void)status;
    (void)message;
    (void)userdata1;
    (void)userdata2;
}

static void benchmark_wait_idle(WGPUInstance const instance, WGPUQueue const queue) {
    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.callback = benchmark_work_done_cb;
    WGPUFuture f = wgpuQueueOnSubmittedWorkDone(queue, cb);

    WGPUFutureWaitInfo wait = WGPU_FUTURE_WAIT_INFO_INIT;
    wait.future = f;
    wgpuInstanceWaitAny(instance, 1, &wait, (uint64_t)5 * 1000000000);
}

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

static void benchmark_run_wbc_sample(
    const benchmark_data * const data,
    uint32_t * const keys,
    uint32_t * const value_indices,
    const size_t swap_len,
    void * const swap,
    benchmark_result * const result
)
{
    *result = (benchmark_result){
        .kind = benchmark_wbc,
    };

    const uint64_t start_ns = now_ns();
    memcpy(keys, data->keys, data->keys_len * sizeof(uint32_t));
    result->data.wbc.upload_start_ns = start_ns;
    result->data.wbc.upload_end_ns = now_ns();

    size_t required_size;
    if (!wbc_segsort(
        data->keys_len,
        keys,
        value_indices,
        data->segments_len,
        data->segments,
        swap_len,
        swap,
        &required_size
    )) PANIC("could not run CPU sort");

    const uint64_t end_ns = now_ns();

    result->data.wbc.wall_start_ns = start_ns;
    result->data.wbc.wall_end_ns = end_ns;
    result->data.wbc.sort_start_ns = result->data.wbc.upload_end_ns;
    result->data.wbc.sort_end_ns = end_ns;
}

#define QUERY_COUNT 8

static void benchmark_run_wbg_sample(
    const wbg_pipeline * const pipeline,
    const wbg_buffers * const buffers,
    const wbg_bindings * const bindings,
    const benchmark_data * const data,
    wbg_sort_timing * const timing,
    WGPUBuffer const query_buffer,
    benchmark_result * const result
)
{
    *result = (benchmark_result){
        .kind = benchmark_wbg,
    };

    wbg_gpu_config config = {0};

    const uint64_t start_ns = now_ns();
    wbg_prepare(
        pipeline->queue,
        buffers,
        data->segments_len, data->segments,
        data->keys_len, data->keys,
        &config
    );
    benchmark_wait_idle(pipeline->instance, pipeline->queue);
    const uint64_t end_upload_ns = now_ns();
    result->data.wbg.upload_start_ns = start_ns;
    result->data.wbg.upload_end_ns = end_upload_ns;

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(pipeline->device, &(WGPUCommandEncoderDescriptor){
        .label = (WGPUStringView){
            .data = "Merge Sort: Command Encoder",
            .length = WGPU_STRLEN,
        }
    });

    wbg_run_sort(
        pipeline,
        bindings,
        buffers,
        &config,
        encoder,
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

    const uint64_t start_sort_ns = now_ns();
    wgpuQueueSubmit(pipeline->queue, 1, &commands);
    benchmark_wait_idle(pipeline->instance, pipeline->queue);
    const uint64_t end_sort_ns = now_ns();
    result->data.wbg.sort_start_ns = start_sort_ns;
    result->data.wbg.sort_end_ns = end_sort_ns;
    result->data.wbg.wall_start_ns = start_ns;
    result->data.wbg.wall_end_ns = end_sort_ns;

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

    memcpy(result->data.wbg.timestamps, timestamps, sizeof(result->data.wbg.timestamps));
}

static void benchmark_validate(
    const wbg_pipeline * const pipeline,
    const wbg_buffers * const buffers,
    const benchmark_data * const data
)
{
    uint32_t * const expected_keys = (uint32_t *)malloc(data->keys_len * sizeof(uint32_t));
    uint32_t * const value_indices = (uint32_t *)malloc(data->keys_len * sizeof(uint32_t));

    memcpy(expected_keys, data->keys, data->keys_len * sizeof(uint32_t));

    wbc_segsort_alloc(
        data->keys_len,
        expected_keys,
        value_indices,
        data->segments_len,
        data->segments
    );

    uint32_t * keys;
    hwgutil_wgpu_read_buffer_alloc(
        pipeline->instance,
        pipeline->device,
        pipeline->queue,
        buffers->keys,
        &mems_system_allocator,
        (void **)&keys
    );

    for (size_t i = 0; i < data->keys_len; i++)
    {
        if (expected_keys[i] != keys[i])
        {
            fprintf(stderr, "invalid sort, doesn't match cpu\n");
            abort();
        }
    }

    free(expected_keys);
    mems_allocator_free(&mems_system_allocator, keys);
}

static void benchmark_run_wbc(
    const size_t n_runs,
    const size_t n_warmup_runs,
    const benchmark_data * const data,
    benchmark_result * const results
)
{
    size_t required_swap_size;

    if (wbc_segsort(
        data->keys_len,
        NULL,
        NULL,
        0,
        NULL,
        0,
        NULL,
        &required_swap_size
    )) PANIC("could not evaluate swap size");

    void * const keys = malloc(data->keys_len * sizeof(uint32_t));
    void * const swap = malloc(required_swap_size);
    uint32_t * const value_indices = (uint32_t *)malloc(data->keys_len * sizeof(uint32_t));

    benchmark_result warmup_result;
    for (size_t i = 0; i < n_warmup_runs; i++)
    {
        benchmark_run_wbc_sample(
            data,
            keys,
            value_indices,
            required_swap_size,
            swap,
            &warmup_result
        );
    }

    for (size_t i = 0; i < n_runs; i++)
    {
        benchmark_run_wbc_sample(
            data,
            keys,
            value_indices,
            required_swap_size,
            swap,
            results + i
        );
    }
}

static void benchmark_run_wbg(
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

    // validation run
    {
        timing.index = 0;

        benchmark_result validate_result;
        benchmark_run_wbg_sample(
            pipeline,
            buffers,
            bindings,
            data,
            &timing,
            query_buffer,
            &validate_result
        );

        benchmark_validate(pipeline, buffers, data);
    }

    benchmark_result warmup_result;
    for (size_t i = 0; i < n_warmup_runs; i++)
    {
        timing.index = 0;

        benchmark_run_wbg_sample(
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

        benchmark_run_wbg_sample(
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
    const benchmark_kind kind,
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
    WGPUAdapterInfo adapter_info = WGPU_ADAPTER_INFO_INIT;
    const char * gpu_vendor = "";
    const char * gpu_architecture = "";
    const char * gpu_device = "";
    const char * gpu_description = "";
    const char * gpu_backend_type = "";
    const char * gpu_adapter_type = "";
    uint32_t gpu_vendor_id = 0;
    uint32_t gpu_device_id = 0;
    uint32_t subgroup_min_size = 0;
    uint32_t subgroup_max_size = 0;
    if (context != NULL)
    {
        wgpuAdapterGetInfo(context->adapter, &adapter_info);
        gpu_vendor = copy_string_view(adapter_info.vendor);
        gpu_architecture = copy_string_view(adapter_info.architecture);
        gpu_device = copy_string_view(adapter_info.device);
        gpu_description = copy_string_view(adapter_info.description);
        gpu_backend_type = copy_string(benchmark_meta_gpu_backend_type_name(adapter_info.backendType));
        gpu_adapter_type = copy_string(benchmark_meta_gpu_adapter_type_name(adapter_info.adapterType));
        gpu_vendor_id = adapter_info.vendorID;
        gpu_device_id = adapter_info.deviceID;
        subgroup_min_size = adapter_info.subgroupMinSize;
        subgroup_max_size = adapter_info.subgroupMaxSize;
    }

    wbg_gpu_bin * bins = NULL;
    const char * store = NULL;
    if (pipeline != NULL)
    {
        bins = (wbg_gpu_bin *)malloc(sizeof(pipeline->bins));
        memcpy(bins, pipeline->bins, sizeof(pipeline->bins));

        switch (pipeline->options.store)
        {
            case wbg_store_block: store = "block";
            case wbg_store_striped: store = "striped";
            case wbg_store_adaptive: store = "adaptive";
            default: PANIC("invalid store kind");
        }
    }

    *meta = (benchmark_meta){
        .kind = kind,
        .git_commit = BENCH_GIT_COMMIT,
        .os_string = copy_string(benchmark_os_string()),
        .cpu_string = copy_string(benchmark_cpu_string()),
        .gpu_vendor = gpu_vendor,
        .gpu_architecture = gpu_architecture,
        .gpu_device = gpu_device,
        .gpu_description = gpu_description,
        .gpu_backend_type = gpu_backend_type,
        .gpu_adapter_type = gpu_adapter_type,
        .gpu_vendor_id = gpu_vendor_id,
        .gpu_device_id = gpu_device_id,
        .subgroup_min_size = subgroup_min_size,
        .subgroup_max_size = subgroup_max_size,
        .store = store,
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

static void write_benchmark_meta(
    FILE * mf,
    const benchmark_meta * const meta
)
{
    const char * kind_name;
    switch (meta->kind)
    {
        case benchmark_wbc: kind_name = "wbc"; break;
        case benchmark_wbg: kind_name = "wbg"; break;
        default: PANIC("unknown benchmark kind");
    }

    fprintf(mf, "{\n");
        write_json_string_field(mf, false, 2, "kind", kind_name);
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
        write_json_string_field(mf, false, 2, "store", meta->store);
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
            if (i != 0) fprintf(mf, ", ");
            fprintf(mf, "{\n");

            if (meta->bins != NULL)
            {
                wbg_gpu_bin bin = meta->bins[i];
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
            }
            else
            {
                write_json_uint32_field(mf, false, 4, "n_segments", meta->bin_counts[i]);
                write_json_uint32_field(mf, true, 4, "n_keys", meta->bin_key_counts[i]);
            }

            fprintf(mf, "  }");
        }
        fprintf(mf, "]\n");
    fprintf(mf, "}\n");
}

static bool mkdir_p(const char *path) {
    char buf[1024];
    size_t len = strlen(path);
    if (len >= sizeof(buf)) return false;
    memcpy(buf, path, len + 1);

    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(buf, 0755) != 0 && errno != EEXIST) return false;
            *p = '/';
        }
    }
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) return false;
    return true;
}

static void write_benchmark_results_wbc(
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

    if (!mkdir_p(BENCHMARK_DIR))
    {
        fprintf(stderr, "could not create benchmark dir: %s\n", BENCHMARK_DIR);
        abort();
    }

    static char BENCHMARK_FILE[1024];
    snprintf(BENCHMARK_FILE, sizeof(BENCHMARK_FILE), "%s/meta.json", BENCHMARK_DIR);

    FILE * mf = fopen(BENCHMARK_FILE, "w");
    write_benchmark_meta(mf, meta);
    fclose(mf);

    snprintf(BENCHMARK_FILE, sizeof(BENCHMARK_FILE), "%s/timing.csv", BENCHMARK_DIR);
    FILE * rf = fopen(BENCHMARK_FILE, "w");
    fprintf(rf, "wall_ms,wall_upload_ms,walL_sort_ms,wall_us,wall_upload_us,wall_sort_us,wall_ns,wall_upload_ns,wall_sort_ns\n");
    for (uint32_t i = 0; i < meta->n_runs; i++)
    {
        const uint64_t wall_start_ns = results[i].data.wbc.wall_start_ns;
        const uint64_t wall_end_ns = results[i].data.wbc.wall_end_ns;
        const uint64_t wall_upload_start_ns = results[i].data.wbc.upload_start_ns;
        const uint64_t wall_upload_end_ns = results[i].data.wbc.upload_end_ns;
        const uint64_t wall_sort_start_ns = results[i].data.wbc.sort_start_ns;
        const uint64_t wall_sort_end_ns = results[i].data.wbc.sort_end_ns;

        const uint64_t wall_ns = wall_end_ns - wall_start_ns;
        const uint64_t wall_upload_ns = wall_upload_end_ns - wall_upload_start_ns;
        const uint64_t wall_sort_ns = wall_sort_end_ns - wall_sort_start_ns;

        const uint64_t wall_ms = wall_ns / 1000000;
        const uint64_t wall_upload_ms = wall_upload_ns / 1000000;
        const uint64_t wall_sort_ms = wall_sort_ns / 1000000;

        const uint64_t wall_us = wall_ns / 1000;
        const uint64_t wall_upload_us = wall_upload_ns / 1000;
        const uint64_t wall_sort_us = wall_sort_ns / 1000;

        fprintf(rf, "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
            wall_ms, wall_upload_ms, wall_sort_ms,
            wall_us, wall_upload_us, wall_sort_us,
            wall_ns, wall_upload_ns, wall_sort_ns
        );
    }
    fclose(rf);
}

static void write_benchmark_results_wbg(
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

    if (!mkdir_p(BENCHMARK_DIR))
    {
        fprintf(stderr, "could not create benchmark dir: %s\n", BENCHMARK_DIR);
        abort();
    }

    static char BENCHMARK_FILE[1024];
    snprintf(BENCHMARK_FILE, sizeof(BENCHMARK_FILE), "%s/meta.json", BENCHMARK_DIR);

    FILE * mf = fopen(BENCHMARK_FILE, "w");
    write_benchmark_meta(mf, meta);
    fclose(mf);

    snprintf(BENCHMARK_FILE, sizeof(BENCHMARK_FILE), "%s/timing.csv", BENCHMARK_DIR);
    FILE * rf = fopen(BENCHMARK_FILE, "w");
    fprintf(rf,
        "wall_ms,wall_upload_ms,wall_sort_ms,wall_us,wall_upload_us,wall_sort_us,wall_ns,wall_upload_ns,wall_sort_ns,"
        "bin_ms,sort_ms,merge_ms,bin_us,sort_us,merge_us,bin_ns,sort_ns,merge_ns,"
        "bin_start_ns,bin_end_ns,sort_start_ns,sort_end_ns,merge_start_ns,merge_end_ns,"
        "wall_start_ns,wall_end_ns,wall_upload_start_ns,wall_upload_end_ns,wall_sort_start_ns,wall_sort_end_ns\n");
    for (uint32_t i = 0; i < meta->n_runs; i++)
    {
        const uint64_t bin_start_ns = results[i].data.wbg.timestamps[0];
        const uint64_t bin_end_ns = results[i].data.wbg.timestamps[1];
        const uint64_t sort_start_ns = results[i].data.wbg.timestamps[2];
        const uint64_t sort_end_ns = results[i].data.wbg.timestamps[3];
        const uint64_t merge_start_ns = results[i].data.wbg.timestamps[4];
        const uint64_t merge_end_ns = results[i].data.wbg.timestamps[5];

        const uint64_t bin_ns = bin_end_ns - bin_start_ns;
        const uint64_t sort_ns = sort_end_ns - sort_start_ns;
        const uint64_t merge_ns = merge_end_ns - merge_start_ns;

        const uint64_t bin_ms = bin_ns / 1000000;
        const uint64_t sort_ms = sort_ns / 1000000;
        const uint64_t merge_ms = merge_ns / 1000000;

        const uint64_t bin_us = bin_ns / 1000;
        const uint64_t sort_us = sort_ns / 1000;
        const uint64_t merge_us = merge_ns / 1000;

        const uint64_t wall_start_ns = results[i].data.wbg.wall_start_ns;
        const uint64_t wall_end_ns = results[i].data.wbg.wall_end_ns;
        const uint64_t wall_upload_start_ns = results[i].data.wbg.upload_start_ns;
        const uint64_t wall_upload_end_ns = results[i].data.wbg.upload_end_ns;
        const uint64_t wall_sort_start_ns = results[i].data.wbg.sort_start_ns;
        const uint64_t wall_sort_end_ns = results[i].data.wbg.sort_end_ns;

        const uint64_t wall_ns = wall_end_ns - wall_start_ns;
        const uint64_t wall_upload_ns = wall_upload_end_ns - wall_upload_start_ns;
        const uint64_t wall_sort_ns = wall_sort_end_ns - wall_sort_start_ns;

        const uint64_t wall_ms = wall_ns / 1000000;
        const uint64_t wall_upload_ms = wall_upload_ns / 1000000;
        const uint64_t wall_sort_ms = wall_sort_ns / 1000000;

        const uint64_t wall_us = wall_ns / 1000;
        const uint64_t wall_upload_us = wall_upload_ns / 1000;
        const uint64_t wall_sort_us = wall_sort_ns / 1000;

        fprintf(rf, "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
            wall_ms, wall_upload_ms, wall_sort_ms,
            wall_us, wall_upload_us, wall_sort_us,
            wall_ns, wall_upload_ns, wall_sort_ns,
            bin_ms, sort_ms, merge_ms,
            bin_us, sort_us, merge_us,
            bin_ns, sort_ns, merge_ns,
            bin_start_ns, bin_end_ns,
            sort_start_ns, sort_end_ns,
            merge_start_ns, merge_end_ns,
            wall_start_ns, wall_end_ns,
            wall_upload_start_ns, wall_upload_end_ns,
            wall_sort_start_ns, wall_sort_end_ns
        );
    }
    fclose(rf);
}

int benchmark_wbc_main(const benchmark_config * const config)
{
    hwstats_sampler * const bin_sampler = create_sampler(
        config->sampler_name,
        config->seed
    );

    hwstats_sampler * const key_sampler = create_sampler(
        config->key_sampler_name,
        config->seed
    );

    benchmark_data data;
    benchmark_data_init(
        &data,
        config->key_budget,
        config->max_key,
        bin_sampler,
        key_sampler
    );

    benchmark_meta meta;
    benchmark_meta_init(
        &meta,
        benchmark_wbc,
        data.bin_counts,
        data.bin_key_counts,
        config->sampler_name,
        config->key_sampler_name,
        config->seed,
        config->max_key,
        data.segments_len,
        data.keys_len,
        config->n_warmup_runs,
        config->n_runs,
        config->subgroups_enabled,
        NULL,
        NULL
    );

    benchmark_result * const results = (benchmark_result *)malloc(config->n_runs * sizeof(benchmark_result));

    benchmark_run_wbc(
        config->n_runs,
        config->n_warmup_runs,
        &data,
        results
    );

    write_benchmark_results_wbc(
        config->results_dir,
        &meta,
        config->n_runs,
        results
    );

    return 0;
}

int benchmark_wbg_main(const benchmark_config * const config)
{
    hwgutil_wgpu_context context;
    wbg_pipeline pipeline;
    wbg_buffers buffers;
    wbg_bindings bindings;

    printf("Initializing WebGPU context...\n");

    size_t features_len;
    WGPUFeatureName features[2];
    if (config->subgroups_enabled)
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

    if (config->subgroups_enabled)
    {
        if (!wgpuAdapterHasFeature(context.adapter, WGPUFeatureName_Subgroups))
        {
            fprintf(stderr, "subgroups are not available with this adapter");
        }
    }

    wbg_store store;
    if (strcmp("block", config->store_name) == 0) store = wbg_store_block;
    else if (strcmp("striped", config->store_name) == 0) store = wbg_store_striped;
    else if (strcmp("adaptive", config->store_name) == 0) store = wbg_store_adaptive;
    else PANIC("invalid store kind %s", config->store_name);

    wbg_pipeline_init(
        &pipeline,
        &(wbg_options){
            .sort_kernels_root_dir = "shaders/sort_kernels",
            .subgroups_enabled = config->subgroups_enabled,
            .store = store,
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

    hwstats_sampler * const bin_sampler = create_sampler(
        config->sampler_name,
        config->seed
    );

    hwstats_sampler * const key_sampler = create_sampler(
        config->key_sampler_name,
        config->seed
    );

    benchmark_data data;
    benchmark_data_init(
        &data,
        config->key_budget,
        config->max_key,
        bin_sampler,
        key_sampler
    );

    benchmark_meta meta;
    benchmark_meta_init(
        &meta,
        benchmark_wbg,
        data.bin_counts,
        data.bin_key_counts,
        config->sampler_name,
        config->key_sampler_name,
        config->seed,
        config->max_key,
        data.segments_len,
        data.keys_len,
        config->n_warmup_runs,
        config->n_runs,
        true,
        &pipeline,
        &context
    );

    benchmark_result * const results = (benchmark_result *)malloc(config->n_runs * sizeof(benchmark_result));

    benchmark_run_wbg(
        &pipeline,
        &buffers,
        &bindings,
        config->n_runs,
        config->n_warmup_runs,
        &data,
        results
    );

    write_benchmark_results_wbg(
        config->results_dir,
        &meta,
        config->n_runs,
        results
    );

    return 0;
}

int main(int argc, char ** argv)
{
    char * endptr;

    const char * const kind_name = argv[1];
    const char * const results_dir = argv[2];
    const char * const sampler_name = argv[3];
    const char * const key_sampler_name = argv[4];
    const uint64_t seed = strtoull(argv[5], &endptr, 10);
    const size_t key_budget = strtoull(argv[6], &endptr, 10);
    const size_t n_runs = strtoull(argv[7], &endptr, 10);
    const size_t n_warmup_runs = strtoull(argv[8], &endptr, 10);
    const uint32_t max_key = strtol(argv[9], &endptr, 10);
    const uint32_t subgroup_test = strtol(argv[10], &endptr, 10);
    const char * const store_name = argv[11];
    const bool subgroups_enabled = subgroup_test == 1;

    const benchmark_config config = (benchmark_config){
        .kind_name = kind_name,
        .results_dir = results_dir,
        .sampler_name = sampler_name,
        .key_sampler_name = key_sampler_name,
        .seed = seed,
        .key_budget = key_budget,
        .n_runs = n_runs,
        .n_warmup_runs = n_warmup_runs,
        .max_key = max_key,
        .subgroups_enabled = subgroups_enabled,
        .store_name = store_name,
    };

    if (strcmp("wbg", kind_name) == 0)
    {
        return benchmark_wbg_main(&config);
    }
    else if (strcmp("wbc", kind_name) == 0)
    {
        return benchmark_wbc_main(&config);
    }
    else
    {
        fprintf(stderr, "Unknown benchmark kind %s\n", kind_name);
    }
}
