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
#include <inttypes.h>
#include <webgpu/webgpu.h>

#define HWDS_MEMS_ENABLED
#define HWGUTIL_MEMS_ENABLED
#define HWGUTIL_WEBGPU_ENABLED

#define WB_SORT_CPU_IMPLEMENTATION
#define WB_SORT_GPU_IMPLEMENTATION
#define HWSTATS_IMPLEMENTATION
#define HWGUTIL_IMPLEMENTATION
#define HWDS_IMPLEMENTATION
#define HWARGS_IMPLEMENTATION
#define MEMS_IMPLEMENTATION

#include <webgpu/webgpu.h>

#include "hw_stats.h"
#include "hw_gutil.h"
#include "hw_args.h"
#include "cpu.h"
#include "gpu.h"

#define PANIC(...) { \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
    abort(); \
}

#define ENSURE_MSG(cond, ...) { \
    if (!(cond)) { \
        PANIC(__VA_ARGS__); \
    } \
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
    const char * memory;
    const char * store;
    bool has_subgroups_feature;
    const wbg_gpu_bin * bins;
    uint32_t bin_counts[13];
    uint32_t bin_key_counts[13];
    const char * wgpu_backend_name;
    const char * wgpu_backend_version;
    const char * wgpu_backend_release_type;
    const char * cpu_release_type;
    const char * bin_sampler;
    const char * key_sampler;
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
    size_t bin_iterations;
    size_t sort_iterations;
    size_t merge_iterations;
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
    const char * output_dir;
    const char * sampler_name;
    const char * key_sampler_name;
    uint64_t seed;
    size_t key_budget;
    size_t n_runs;
    size_t n_warmup_runs;
    const char * memory_name;
    const char * store_name;
    bool tune;
    size_t bin_iterations;
    size_t sort_iterations;
    size_t merge_iterations;
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

static hwstats_sampler * create_bin_sampler(const uint32_t bin)
{
    hwstats_const * const c = (hwstats_const *)malloc(sizeof(hwstats_const));
    hwstats_sampler * const sampler = (hwstats_sampler *)malloc(sizeof(hwstats_sampler));

    const double value = (1.0 / 11.0 * (double)(bin - 1)) + (1e-4);
    *c = (hwstats_const){ .value = value };

    hwstats_const_sampler_init(sampler, c);

    return sampler;
}

static hwstats_sampler * create_const_sampler(const double value)
{
    hwstats_const * const c = (hwstats_const *)malloc(sizeof(hwstats_const));
    hwstats_sampler * const sampler = (hwstats_sampler *)malloc(sizeof(hwstats_sampler));

    *c = (hwstats_const){ .value = value };

    hwstats_const_sampler_init(sampler, c);

    return sampler;
}

static hwstats_sampler * create_sampler(
    const char * const name,
    const uint64_t seed
)
{
    if (strcmp("uniform", name) == 0) return create_uniform_sampler(seed);
    else if (strncmp("bin", name, strlen("bin")) == 0)
    {
        uint32_t value;
        if (sscanf(name, "bin(%" SCNu32 ")", &value) != 1)
        {
            fprintf(stderr, "invalid bin sampler\n");
            abort();
        }
        return create_bin_sampler(value);
    }
    else if (strncmp("const", name, strlen("const")) == 0)
    {
        double value;
        if (sscanf(name, "const(%lf)", &value) != 1)
        {
            fprintf(stderr, "invalid const sampler\n");
            abort();
        }
        return create_const_sampler(value);
    }
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
    const size_t bin_iterations,
    const size_t sort_iterations,
    const size_t merge_iterations,
    benchmark_result * const result
)
{
    *result = (benchmark_result){
        .kind = benchmark_wbg,
    };

    wbg_gpu_config config = {0};

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(pipeline->device, &(WGPUCommandEncoderDescriptor){
        .label = (WGPUStringView){
            .data = "Merge Sort: Command Encoder",
            .length = WGPU_STRLEN,
        }
    });

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

    wbg_run_sort_iterations(
        pipeline,
        bindings,
        buffers,
        &config,
        bin_iterations,
        sort_iterations,
        merge_iterations,
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
    wbg_sort_timing * const timing,
    WGPUBuffer const query_buffer,
    const size_t bin_iterations,
    const size_t sort_iterations,
    const size_t merge_iterations,
    benchmark_result * const results
)
{
    // validation run
    {
        timing->index = 0;

        benchmark_result validate_result;
        benchmark_run_wbg_sample(
            pipeline,
            buffers,
            bindings,
            data,
            timing,
            query_buffer,
            1,
            1,
            1,
            &validate_result
        );

        benchmark_validate(pipeline, buffers, data);
    }

    benchmark_result warmup_result;
    for (size_t i = 0; i < n_warmup_runs; i++)
    {
        timing->index = 0;

        benchmark_run_wbg_sample(
            pipeline,
            buffers,
            bindings,
            data,
            timing,
            query_buffer,
            1,
            1,
            1,
            &warmup_result
        );
    }

    for (size_t i = 0; i < n_runs; i++)
    {
        timing->index = 0;

        benchmark_run_wbg_sample(
            pipeline,
            buffers,
            bindings,
            data,
            timing,
            query_buffer,
            bin_iterations,
            sort_iterations,
            merge_iterations,
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
    const size_t n_segments,
    const size_t n_keys,
    const size_t n_warmup_runs,
    const size_t n_runs,
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
    const char * memory = NULL;
    const char * store = NULL;
    bool has_subgroups_feature = false;
    if (pipeline != NULL)
    {
        bins = (wbg_gpu_bin *)malloc(sizeof(pipeline->bins));
        memcpy(bins, pipeline->bins, sizeof(pipeline->bins));

        switch (pipeline->options.memory)
        {
            case wbg_memory_register: memory = "register"; break;
            case wbg_memory_workgroup: memory = "workgroup"; break;
            case wbg_memory_adaptive: memory = "adaptive"; break;
            default: PANIC("invalid memory kind");
        }

        switch (pipeline->options.store)
        {
            case wbg_store_block: store = "block"; break;
            case wbg_store_striped: store = "striped"; break;
            case wbg_store_adaptive: store = "adaptive"; break;
            default: PANIC("invalid store kind");
        }

        has_subgroups_feature = pipeline->has_subgroups_feature;
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
        .memory = memory,
        .store = store,
        .has_subgroups_feature = has_subgroups_feature,
        .bins = bins,
        .wgpu_backend_name = BENCH_BACKEND_NAME,
        .wgpu_backend_version = BENCH_BACKEND_VERSION,
        .wgpu_backend_release_type = BENCH_BACKEND_RELEASE_TYPE,
        .cpu_release_type = BENCH_CPU_RELEASE_TYPE,
        .bin_sampler = copy_string(bin_sampler),
        .key_sampler = copy_string(key_sampler),
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

#define MAX_SEGMENT_LEN 4096

static void benchmark_data_init(
    benchmark_data * const data,
    const uint64_t seed,
    const size_t key_budget,
    hwstats_sampler * const bin_sampler,
    hwstats_sampler * const key_sampler
)
{
    size_t segments_len = 0;
    uint32_t * const segments = (uint32_t *)malloc(key_budget * 16 * sizeof(uint32_t));

    uint32_t bin_counts[13] = {0};
    uint32_t bin_key_counts[13] = {0};

    uint32_t bin_key_quota[13] = {0};
    for (size_t i = 0; i < key_budget; i++) {
        uint32_t bin = sample_range(bin_sampler, 1, 12);
        bin_key_quota[bin]++;
    }

    hwstats_randomizer r;
    hwstats_x256pp x256pp;
    hwstats_sampler uniform;
    hwstats_x256pp_init(&x256pp, seed);
    hwstats_x256pp_rand_init(&x256pp, &r);
    hwstats_uniform_sampler_init(&uniform, &r);

    uint32_t global_keys_len = 0;
    for (size_t i = 0; i < 13; i++)
    {
        uint32_t keys_len = 0;
        uint32_t quota = bin_key_quota[i];

        while (keys_len < quota)
        {
            uint32_t bin = i;

            // 0 -> [0,0]
            // 1 -> [1,1]
            // 2 -> [2,3]
            // 3 -> [4,7]
            // 4 -> [8,15]
            const uint32_t lo = bin <= 1 ? bin : (1u << (bin - 1u));
            const uint32_t hi = bin == 12 ?
                MAX_SEGMENT_LEN :
                (bin <= 1 ? bin : (1u << bin) - 1u);
            uint32_t segment_len = sample_range(key_sampler, lo, hi);

            if (keys_len + segment_len > quota)
            {
                segment_len = quota - keys_len;
                bin = benchmark_segment_bucket(segment_len);
            }

            bin_counts[bin]++;
            bin_key_counts[bin] += segment_len;
            keys_len += bin == 0 ? 1 : segment_len;
            global_keys_len += segment_len;
            segments[segments_len++] = segment_len;
        }
    }

    // shuffle the segments
    for (size_t i = segments_len - 1; i > 0; i--)
    {
        const uint32_t j = sample_range(&uniform, 0, i);
        const uint32_t tmp = segments[i];
        segments[i] = segments[j];
        segments[j] = tmp;
    }

    // prefix sum the segments
    uint32_t sum = 0;
    for (size_t i = 0; i < segments_len; i++)
    {
        const uint32_t segment_len = segments[i];
        sum += segment_len;
        segments[i] = sum;
    }

    if (segments[segments_len - 1] != key_budget)
    {
        printf("Expected %zu, found %u\n", key_budget, segments[segments_len - 1]);
        abort();
    }

    uint32_t * const keys = (uint32_t *)malloc(key_budget * sizeof(uint32_t));

    for (uint32_t i = 0; i < key_budget; i++)
    {
        keys[i] = sample_range(&uniform, 0, 1024 * 32);
    }

    *data = (benchmark_data){
        .keys_len = global_keys_len,
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
        write_json_string_field(mf, false, 2, "memory", meta->memory);
        write_json_string_field(mf, false, 2, "store", meta->store);
        write_json_bool_field(mf, false, 2, "has_subgroups_feature", meta->has_subgroups_feature);
        write_json_string_field(mf, false, 2, "bin_sampler", meta->bin_sampler);
        write_json_string_field(mf, false, 2, "key_sampler", meta->key_sampler);
        write_json_uint32_field(mf, false, 2, "merge_wg", meta->merge_wg);
        write_json_uint32_field(mf, false, 2, "merge_tile_size", meta->merge_tile_size);
        write_json_uint32_field(mf, false, 2, "merge_input_tile_size", meta->merge_input_tile_size);
        write_json_uint32_field(mf, false, 2, "merge_max_passes", meta->merge_max_passes);
        write_json_uint64_field(mf, false, 2, "seed", meta->seed);
        write_json_size_field(mf, false, 2, "key_bit_size", meta->key_bit_size);
        write_json_uint32_field(mf, false, 2, "max_segment_len", MAX_SEGMENT_LEN);
        write_json_size_field(mf, false, 2, "n_segments", meta->n_segments);
        write_json_size_field(mf, false, 2, "n_keys", meta->n_keys);
        write_json_size_field(mf, false, 2, "n_warmup_runs", meta->n_warmup_runs);
        write_json_size_field(mf, false, 2, "n_runs", meta->n_runs);
        write_json_uint32_field(mf, false, 2, "bin_iterations", meta->bin_iterations);
        write_json_uint32_field(mf, false, 2, "sort_iterations", meta->sort_iterations);
        write_json_uint32_field(mf, false, 2, "merge_iterations", meta->merge_iterations);

        fprintf(mf, "  \"bins\": [");
        for (int i = 0; i < 13; i++)
        {
            if (i != 0) fprintf(mf, ", ");
            fprintf(mf, "{\n");

            if (meta->bins != NULL)
            {
                wbg_gpu_bin bin = meta->bins[i];

                write_json_bool_field(mf, false, 4, "is_register", (bin.flags & wbg_bin_flag_is_register) != 0);
                write_json_bool_field(mf, false, 4, "is_striped", (bin.flags & wbg_bin_flag_is_striped) != 0);
                write_json_bool_field(mf, false, 4, "is_variable", (bin.flags & wbg_bin_flag_is_variable) != 0);

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
        config->seed,
        config->key_budget,
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
        data.segments_len,
        data.keys_len,
        config->n_warmup_runs,
        config->n_runs,
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
        config->output_dir,
        &meta,
        config->n_runs,
        results
    );

    return 0;
}

#define TUNE_THRESHOLD_NS (20000000) // 20ms

static void benchmark_wbg_tune_iterations(
    const wbg_pipeline * const pipeline,
    const wbg_buffers * const buffers,
    const wbg_bindings * const bindings,
    const benchmark_data * const data,
    wbg_sort_timing * const timing,
    WGPUBuffer const query_buffer,
    const bool tune_bin,
    const bool tune_sort,
    const bool tune_merge,
    size_t * const bin_iterations,
    size_t * const sort_iterations,
    size_t * const merge_iterations,
    uint64_t * const tuned_bin_ns,
    uint64_t * const tuned_sort_ns,
    uint64_t * const tuned_merge_ns
)
{
    wbg_gpu_config config = {0};

    wbg_prepare(
        pipeline->queue,
        buffers,
        data->segments_len, data->segments,
        data->keys_len, data->keys,
        &config
    );
    benchmark_wait_idle(pipeline->instance, pipeline->queue);

    uint64_t bin_threshold_ns = TUNE_THRESHOLD_NS;
    *tuned_bin_ns = 0;
    *bin_iterations = 1;

    // run at least once to prepare for sort stage
    while (*tuned_bin_ns < bin_threshold_ns)
    {
        timing->index = 0;

        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(pipeline->device, &(WGPUCommandEncoderDescriptor){
            .label = (WGPUStringView){
                .data = "WB Sort: Command Encoder for Tuning",
                .length = WGPU_STRLEN,
            }
        });

        WGPUComputePassDescriptor bin_pass_desc = (WGPUComputePassDescriptor){
            .label = (WGPUStringView){
                .data = "WB Sort: Bin Pass Tuner",
            },
        };
        WGPUPassTimestampWrites bin_ts = WGPU_PASS_TIMESTAMP_WRITES_INIT;
        if (timing != NULL)
        {
            bin_ts = (WGPUPassTimestampWrites){
                .querySet = timing->query,
                .beginningOfPassWriteIndex = timing->index++,
                .endOfPassWriteIndex = timing->index++,
            };
            bin_pass_desc.timestampWrites = &bin_ts;
        }

        WGPUComputePassEncoder bin_pass = wgpuCommandEncoderBeginComputePass(encoder, &bin_pass_desc);

        for (size_t i = 0; i < *bin_iterations; i++)
        {
            wbg_bin(
                pipeline,
                bindings,
                &config,
                bin_pass
            );
        }

        wgpuComputePassEncoderEnd(bin_pass);

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
        wgpuComputePassEncoderRelease(bin_pass);
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


        *tuned_bin_ns = timestamps[1] - timestamps[0];
        if (!tune_bin) break;
        if (*tuned_bin_ns < TUNE_THRESHOLD_NS) *bin_iterations *= 2;
    }

    *sort_iterations = 1;
    if (tune_sort)
    {
        *tuned_sort_ns = 0;
        while (*tuned_sort_ns < TUNE_THRESHOLD_NS)
        {
            timing->index = 0;

            WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(pipeline->device, &(WGPUCommandEncoderDescriptor){
                .label = (WGPUStringView){
                    .data = "WB Sort: Command Encoder for Tuning",
                    .length = WGPU_STRLEN,
                }
            });

            WGPUComputePassDescriptor sort_pass_desc = (WGPUComputePassDescriptor){
                .label = (WGPUStringView){
                    .data = "WB Sort: Sort Pass Tuner",
                },
            };
            WGPUPassTimestampWrites sort_ts = WGPU_PASS_TIMESTAMP_WRITES_INIT;
            if (timing != NULL)
            {
                sort_ts = (WGPUPassTimestampWrites){
                    .querySet = timing->query,
                    .beginningOfPassWriteIndex = timing->index++,
                    .endOfPassWriteIndex = timing->index++,
                };
                sort_pass_desc.timestampWrites = &sort_ts;
            }

            WGPUComputePassEncoder sort_pass = wgpuCommandEncoderBeginComputePass(encoder, &sort_pass_desc);

            for (size_t i = 0; i < *sort_iterations; i++)
            {
                wbg_segsort(
                    pipeline,
                    bindings,
                    buffers,
                    &config,
                    sort_pass
                );
            }

            wgpuComputePassEncoderEnd(sort_pass);

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
            wgpuComputePassEncoderRelease(sort_pass);
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

            *tuned_sort_ns = timestamps[1] - timestamps[0];

            if (*tuned_sort_ns < TUNE_THRESHOLD_NS) *sort_iterations *= 2;
        }
    }

    *merge_iterations = 1;
    if (tune_merge)
    {
        *tuned_merge_ns = 0;
        while (*tuned_merge_ns < TUNE_THRESHOLD_NS)
        {
            timing->index = 0;

            WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(pipeline->device, &(WGPUCommandEncoderDescriptor){
                .label = (WGPUStringView){
                    .data = "WB Sort: Command Encoder for Tuning",
                    .length = WGPU_STRLEN,
                }
            });

            WGPUComputePassDescriptor merge_pass_desc = (WGPUComputePassDescriptor){
                .label = (WGPUStringView){
                    .data = "WB Sort: Merge Pass Tuner",
                },
            };
            WGPUPassTimestampWrites merge_ts = WGPU_PASS_TIMESTAMP_WRITES_INIT;
            if (timing != NULL)
            {
                merge_ts = (WGPUPassTimestampWrites){
                    .querySet = timing->query,
                    .beginningOfPassWriteIndex = timing->index++,
                    .endOfPassWriteIndex = timing->index++,
                };
                merge_pass_desc.timestampWrites = &merge_ts;
            }

            WGPUComputePassEncoder merge_pass = wgpuCommandEncoderBeginComputePass(encoder, &merge_pass_desc);

            for (size_t i = 0; i < *merge_iterations; i++)
            {
                wbg_merge(
                    pipeline,
                    bindings,
                    buffers,
                    merge_pass
                );
            }

            wgpuComputePassEncoderEnd(merge_pass);

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
            wgpuComputePassEncoderRelease(merge_pass);
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

            *tuned_merge_ns = timestamps[1] - timestamps[0];

            if (*tuned_merge_ns < TUNE_THRESHOLD_NS) *merge_iterations *= 2;
        }
    }
}

int benchmark_wbg_main(const benchmark_config * const config)
{
    hwgutil_wgpu_context context;
    wbg_pipeline pipeline;
    wbg_buffers buffers;
    wbg_bindings bindings;

    printf("Initializing WebGPU context...\n");

    if (!hwgutil_wgpu_context_init(
        &context,
        1, (WGPUInstanceFeatureName[]){ WGPUInstanceFeatureName_TimedWaitAny },
        2, (WGPUFeatureName[]){ WGPUFeatureName_TimestampQuery, WGPUFeatureName_Subgroups }
    )) abort();

    wbg_memory memory;
    if (strcmp("register", config->memory_name) == 0) memory = wbg_memory_register;
    else if (strcmp("workgroup", config->memory_name) == 0) memory = wbg_memory_workgroup;
    else if (strcmp("adaptive", config->memory_name) == 0) memory = wbg_memory_adaptive;
    else PANIC("invalid memory kind %s", config->memory_name);

    wbg_store store;
    if (strcmp("block", config->store_name) == 0) store = wbg_store_block;
    else if (strcmp("striped", config->store_name) == 0) store = wbg_store_striped;
    else if (strcmp("adaptive", config->store_name) == 0) store = wbg_store_adaptive;
    else PANIC("invalid store kind %s", config->store_name);

    if (memory == wbg_memory_register)
    {
        if (!wgpuAdapterHasFeature(context.adapter, WGPUFeatureName_Subgroups))
        {
            PANIC("subgroups are not available with this adapter, but requested register memory, aborting");
        }
    }
    else if (memory == wbg_memory_adaptive)
    {
        if (!wgpuAdapterHasFeature(context.adapter, WGPUFeatureName_Subgroups))
        {
            fprintf(stderr, "subgroups are not available with this adapter, but requested adaptive memory\n");
        }
    }

    wbg_pipeline_init(
        &pipeline,
        &(wbg_options){
            .sort_kernels_root_dir = "shaders/sort_kernels",
            .memory = memory,
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
        config->seed,
        config->key_budget,
        bin_sampler,
        key_sampler
    );

    WGPUQuerySetDescriptor query_desc = WGPU_QUERY_SET_DESCRIPTOR_INIT;
    query_desc.label = (WGPUStringView){
        .data = "WB Sort: Timestamp Queries",
        .length = WGPU_STRLEN,
    };
    query_desc.type = WGPUQueryType_Timestamp;
    query_desc.count = QUERY_COUNT;

    WGPUQuerySet query = wgpuDeviceCreateQuerySet(pipeline.device, &query_desc);

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

    WGPUBuffer query_buffer = wgpuDeviceCreateBuffer(pipeline.device, &query_buffer_desc);

    if (config->tune)
    {
        size_t bin_iterations, sort_iterations, merge_iterations;
        uint64_t tuned_bin_ns, tuned_sort_ns, tuned_merge_ns;
        benchmark_wbg_tune_iterations(
            &pipeline,
            &buffers,
            &bindings,
            &data,
            &timing,
            query_buffer,
            true,
            true,
            true,
            &bin_iterations,
            &sort_iterations,
            &merge_iterations,
            &tuned_bin_ns,
            &tuned_sort_ns,
            &tuned_merge_ns
        );

        printf("Tuning params: --bin-iterations %zu --sort-iterations %zu --merge-iterations %zu\n",
            bin_iterations, sort_iterations, merge_iterations);

        printf("Tuned to: BinIterations(%llu), SortIterations(%llu), MergeIterations(%llu)\n",
            tuned_bin_ns, tuned_sort_ns, tuned_merge_ns);

        exit(0);
    }

    benchmark_meta meta;
    benchmark_meta_init(
        &meta,
        benchmark_wbg,
        data.bin_counts,
        data.bin_key_counts,
        config->sampler_name,
        config->key_sampler_name,
        config->seed,
        data.segments_len,
        data.keys_len,
        config->n_warmup_runs,
        config->n_runs,
        &pipeline,
        &context
    );
    meta.bin_iterations = config->bin_iterations;
    meta.sort_iterations = config->sort_iterations;
    meta.merge_iterations = config->merge_iterations;

    benchmark_result * const results = (benchmark_result *)malloc(config->n_runs * sizeof(benchmark_result));

    benchmark_run_wbg(
        &pipeline,
        &buffers,
        &bindings,
        config->n_runs,
        config->n_warmup_runs,
        &data,
        &timing,
        query_buffer,
        config->bin_iterations,
        config->sort_iterations,
        config->merge_iterations,
        results
    );

    write_benchmark_results_wbg(
        config->output_dir,
        &meta,
        config->n_runs,
        results
    );

    return 0;
}

static char ARGS_BUFFER[1024 * 4];

int main(int argc, const char ** argv)
{
    hwargs_size required_args_size;
    hwargs_parsed args;
    if (!hwargs_parse(
        &args,
        argc, argv,
        sizeof(ARGS_BUFFER),
        ARGS_BUFFER,
        &required_args_size
    )) PANIC("could not parse arguments");

    const hwargs_param * const kind_param = hwargs_get_param(&args, "kind");
    const hwargs_param * const output_param = hwargs_get_param(&args, "output");
    const hwargs_param * const sampler_name_param = hwargs_get_param(&args, "sampler");
    const hwargs_param * const key_sampler_name_param = hwargs_get_param(&args, "key-sampler");
    const hwargs_param * const seed_param = hwargs_get_param(&args, "seed");
    const hwargs_param * const key_budget_param = hwargs_get_param(&args, "keys");
    const hwargs_param * const runs_param = hwargs_get_param(&args, "runs");
    const hwargs_param * const warmup_runs_param = hwargs_get_param(&args, "warmup-runs");
    const hwargs_param * const memory_param = hwargs_get_param(&args, "memory");
    const hwargs_param * const store_param = hwargs_get_param(&args, "store");
    const hwargs_param * const bin_iterations_param = hwargs_get_param(&args, "bin-iterations");
    const hwargs_param * const sort_iterations_param = hwargs_get_param(&args, "sort-iterations");
    const hwargs_param * const merge_iterations_param = hwargs_get_param(&args, "merge-iterations");

    ENSURE_MSG(kind_param != NULL, "must provide --kind (wbc (CPU), wbg (WebGPU Segsort))");
    ENSURE_MSG(output_param != NULL, "must provide --output <dir>");
    ENSURE_MSG(sampler_name_param != NULL, "must provide --sampler <sampler> # examples: uniform, bin(4), const(0.3)");
    ENSURE_MSG(key_sampler_name_param != NULL, "must provide --key-sampler <sampler> # examples: uniform, bin(4), const(0.3)");
    ENSURE_MSG(seed_param != NULL, "must provide --seed <u64>");
    ENSURE_MSG(key_budget_param != NULL, "must provide --keys <u32> # number of keys to sort, distributed with --sampler across buckets");
    ENSURE_MSG(runs_param != NULL, "must provide --runs <u32>");
    ENSURE_MSG(store_param != NULL, "must provide --store <block|striped|adaptive>");

    char * endptr;
    const char * const kind_name = kind_param->value;
    const char * const output_dir = output_param->value;
    const char * const sampler_name = sampler_name_param->value;
    const char * const key_sampler_name = key_sampler_name_param->value;
    const uint64_t seed = strtoull(seed_param->value, &endptr, 10);
    const size_t key_budget = strtoull(key_budget_param->value, &endptr, 10);
    const size_t n_runs = strtoull(runs_param->value, &endptr, 10);
    const size_t n_warmup_runs = warmup_runs_param == NULL ? 10 : strtoull(warmup_runs_param->value, &endptr, 10);
    const char * const memory_name = memory_param == NULL ? "adaptive" : memory_param->value;
    const char * const store_name = store_param == NULL ? "adaptive" : store_param->value;
    const bool tune = hwargs_has_flag(&args, "tune");
    const size_t bin_iterations = bin_iterations_param == NULL ? 1 : strtol(bin_iterations_param->value, &endptr, 10);
    const size_t sort_iterations = sort_iterations_param == NULL ? 1 : strtol(sort_iterations_param->value, &endptr, 10);
    const size_t merge_iterations = merge_iterations_param == NULL ? 1 : strtol(merge_iterations_param->value, &endptr, 10);

    const benchmark_config config = (benchmark_config){
        .kind_name = kind_name,
        .output_dir = output_dir,
        .sampler_name = sampler_name,
        .key_sampler_name = key_sampler_name,
        .seed = seed,
        .key_budget = key_budget,
        .n_runs = n_runs,
        .n_warmup_runs = n_warmup_runs,
        .memory_name = memory_name,
        .store_name = store_name,
        .tune = tune,
        .bin_iterations = bin_iterations,
        .sort_iterations = sort_iterations,
        .merge_iterations = merge_iterations,
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
