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

typedef enum bench_memory_kind
{
    bench_memory_reg = 0,
    bench_memory_smem = 1,
    bench_memory_hybrid = 2,
    bench_memory_hybmerge = 3,
} bench_memory_kind;

typedef enum bench_store_kind
{
    bench_store_block = 0,
    bench_store_striped = 1,
} bench_store_kind;

typedef struct bench_smem
{
    const char * name;
    uint32_t bytes;
} bench_smem;

typedef struct bench_config
{
    const char * root;
    bench_memory_kind memory;
    bench_store_kind store;
    uint32_t seed;
    uint32_t runs;
    uint32_t N;
    uint32_t M;
    uint32_t R;
    uint32_t smem_bytes;
    uint32_t bin;
    uint32_t n_keys;
    uint32_t subgroups;
    uint32_t max_invocations;
    uint32_t max_smem_size;
    uint32_t target_wg_size;
    const char * sampler_name;
} bench_config;

typedef struct bench_buffers
{
    WGPUBuffer keys;
    WGPUBuffer keys_staging;
    WGPUBuffer value_indices;
    WGPUBuffer segments;
    WGPUBuffer bin_offsets;
    WGPUBuffer bin_indices;
} bench_buffers;

typedef struct bench_result
{
    uint64_t wall_start_ns;
    uint64_t wall_end_ns;
    uint64_t gpu_start_ns;
    uint64_t gpu_end_ns;
} bench_result;

static const bench_smem BENCH_SMEMS[2] = {
    { "16kb", 16 * 1024 },
    { "32kb", 32 * 1024 },
};

static const char * bench_memory_name(const bench_memory_kind kind)
{
    switch (kind)
    {
        case bench_memory_reg: return "reg";
        case bench_memory_smem: return "smem";
        case bench_memory_hybrid: return "hybrid";
        case bench_memory_hybmerge: return "hybmerge";
        default: PANIC("invalid memory kind");
    }
}

static const char * bench_store_name(const bench_store_kind kind)
{
    switch (kind)
    {
        case bench_store_block: return "block";
        case bench_store_striped: return "striped";
        default: PANIC("invalid store kind");
    }
}

static void make_parent_dirs(const char * const path)
{
    static char DIR_PATH[2048];
    snprintf(DIR_PATH, sizeof(DIR_PATH), "%s", path);

    for (char * p = DIR_PATH + 1; *p != '\0'; p++)
    {
        if (*p != '/') continue;

        *p = '\0';
        if (mkdir(DIR_PATH, 0755) != 0 && errno != EEXIST)
        {
            PANIC("could not create directory %s: %s", DIR_PATH, strerror(errno));
        }
        *p = '/';
    }
}

static void write_results_csv(
    const bench_config config,
    const uint32_t wg,
    const size_t segments_len,
    const size_t keys_len,
    const bench_result * const results
)
{
    static char CSV_PATH[2048];
    snprintf(CSV_PATH, sizeof(CSV_PATH), "%s.csv", config.root);

    make_parent_dirs(config.root);

    FILE * f = fopen(CSV_PATH, "w");
    if (f == NULL) PANIC("could not open csv file for writing: %s", CSV_PATH);

    fprintf(f,
        "root,memory,store,seed,runs,N,M,wpt,R,subgroups,smem_bytes,bin,n_keys,"
        "max_invocations,max_smem_size,target_wg_size,sampler,wg,segments,keys,"
        "run,wall_ns,gpu_ns,throughput_mkeys_s\n");

    const uint32_t wpt = config.N / config.M;
    const char * const memory = bench_memory_name(config.memory);
    const char * const store = bench_store_name(config.store);

    for (uint32_t i = 0; i < config.runs; i++)
    {
        const uint64_t wall_ns = results[i].wall_end_ns - results[i].wall_start_ns;
        const uint64_t gpu_ns = results[i].gpu_end_ns - results[i].gpu_start_ns;
        const double wall_s = (double)wall_ns / 1e9;
        const double throughput = wall_s > 0.0 ? ((double)keys_len / wall_s) / 1e6 : 0.0;

        fprintf(f,
            "%s,%s,%s,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%s,%u,%zu,%zu,"
            "%u,%" PRIu64 ",%" PRIu64 ",%.6f\n",
            config.root, memory, store, config.seed, config.runs, config.N, config.M,
            wpt, config.R, config.subgroups, config.smem_bytes, config.bin, config.n_keys,
            config.max_invocations, config.max_smem_size, config.target_wg_size,
            config.sampler_name, wg, segments_len, keys_len,
            i, wall_ns, gpu_ns, throughput);
    }

    fclose(f);
}

static bench_memory_kind bench_memory_for_name(const char * const name)
{
    if (strcmp("reg", name) == 0) return bench_memory_reg;
    else if (strcmp("smem", name) == 0) return bench_memory_smem;
    else if (strcmp("hybrid", name) == 0) return bench_memory_hybrid;
    else if (strcmp("hybmerge", name) == 0) return bench_memory_hybmerge;

    PANIC("invalid memory kind %s", name);
}

static bench_store_kind bench_store_for_name(const char * const name)
{
    if (strcmp("block", name) == 0) return bench_store_block;
    else if (strcmp("striped", name) == 0) return bench_store_striped;

    PANIC("invalid memory kind %s", name);
}

static void bench_buffers_init(
    bench_buffers * const buffers,
    WGPUDevice const device,
    const size_t max_keys,
    const size_t max_segments
)
{
    WGPUBufferDescriptor keys_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    keys_desc.label = (WGPUStringView){
        .data = "Benchmark Kernel: Keys",
        .length = WGPU_STRLEN,
    };
    keys_desc.size = max_keys * sizeof(uint32_t);
    keys_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    WGPUBufferDescriptor keys_staging_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    keys_staging_desc.label = (WGPUStringView){
        .data = "Benchmark Kernel: Keys Staging",
        .length = WGPU_STRLEN,
    };
    keys_staging_desc.size = max_keys * sizeof(uint32_t);
    keys_staging_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    WGPUBufferDescriptor value_indices_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    keys_desc.label = (WGPUStringView){
        .data = "Benchmark Kernel: Value Indices",
        .length = WGPU_STRLEN,
    };
    value_indices_desc.size = max_keys * sizeof(uint32_t);
    value_indices_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    WGPUBufferDescriptor segments_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    segments_desc.label = (WGPUStringView){
        .data = "Benchmark Kernel: Value Indices",
        .length = WGPU_STRLEN,
    };
    segments_desc.size = max_keys * sizeof(uint32_t);
    segments_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    WGPUBufferDescriptor bin_offsets_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bin_offsets_desc.label = (WGPUStringView){
        .data = "Benchmark Kernel: Bin Offsets",
        .length = WGPU_STRLEN,
    };
    bin_offsets_desc.size = 13 * sizeof(uint32_t);
    bin_offsets_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    WGPUBufferDescriptor bin_indices_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bin_indices_desc.label = (WGPUStringView){
        .data = "Benchmark Kernel: Bin Indices",
        .length = WGPU_STRLEN,
    };
    bin_indices_desc.size = max_segments * sizeof(uint32_t);
    bin_indices_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;

    *buffers = (bench_buffers){
        .keys = wgpuDeviceCreateBuffer(device, &keys_desc),
        .keys_staging = wgpuDeviceCreateBuffer(device, &keys_staging_desc),
        .value_indices = wgpuDeviceCreateBuffer(device, &value_indices_desc),
        .segments = wgpuDeviceCreateBuffer(device, &segments_desc),
        .bin_offsets = wgpuDeviceCreateBuffer(device, &bin_offsets_desc),
        .bin_indices = wgpuDeviceCreateBuffer(device, &bin_indices_desc),
    };
}

static char * read_file(
    const char * const path,
    const mems_allocator * const allocator,
    size_t * const buffer_len
)
{
    FILE * const f = fopen(path, "rb");
    if (f == NULL) abort();

    if (fseek(f, 0, SEEK_END) != 0) abort();

    long file_size = ftell(f);
    if (file_size < 0) abort();

    if (fseek(f, 0, SEEK_SET) != 0) abort();

    char * const buffer = (char *)mems_allocator_alloc(allocator, MEMS_ALIGN_DEFAULT, file_size + 1);
    size_t bytes_read = fread(buffer, 1, file_size, f);
    if (bytes_read < (size_t)file_size) {
        if (ferror(f)) {
            perror("Error reading file");
            free(buffer);
            fclose(f);
            abort();
        }
        file_size = bytes_read;
    }

    *buffer_len = file_size;
    buffer[file_size] = 0;

    fclose(f);

    return buffer;
}

static void bench_create_pipeline_layout(
    WGPUDevice const device,
    WGPUBindGroupLayout * const bind_layout,
    WGPUPipelineLayout * const pipeline_layout
)
{
    WGPUBindGroupLayoutDescriptor layout0_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    layout0_desc.label = (WGPUStringView){
        .data = "Benchark Kernel: Bindings",
        .length = WGPU_STRLEN,
    };
    layout0_desc.entryCount = 5;
    layout0_desc.entries = (WGPUBindGroupLayoutEntry[]){
        (WGPUBindGroupLayoutEntry){
            .binding = 0, // global_keys
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
            }
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 1, // global_value_indices
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
            }
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 2, // segments
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
            }
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 3, // bin_offsets
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
            },
        },
        (WGPUBindGroupLayoutEntry){
            .binding = 4, // bin_indices
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
            },
        },
    };

    WGPUBindGroupLayout layout0 = wgpuDeviceCreateBindGroupLayout(device, &layout0_desc);

    WGPUPipelineLayoutDescriptor pipeline_layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipeline_layout_desc.label = (WGPUStringView){
        .data = "WB Sort: Sort Pipeline Layout",
        .length = WGPU_STRLEN,
    };
    pipeline_layout_desc.bindGroupLayoutCount = 1;
    pipeline_layout_desc.bindGroupLayouts = (WGPUBindGroupLayout[]){
        layout0,
    };

    *bind_layout = layout0;
    *pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_desc);
}

static WGPUComputePipeline bench_create_pipeline(
    const char * const name,
    const char * const path,
    const uint32_t wg,
    WGPUDevice const device,
    WGPUPipelineLayout layout
)
{
    size_t source_len;
    char * const source = wbg__read_file(path, &mems_system_allocator, &source_len);

    WGPUShaderSourceWGSL shader_source_wgsl = (WGPUShaderSourceWGSL){
        .chain = (WGPUChainedStruct){
            .sType = WGPUSType_ShaderSourceWGSL
        },
        .code = (WGPUStringView){
            .data = source,
            .length = source_len,
        },
    };
    WGPUShaderModuleDescriptor shader_module_desc = (WGPUShaderModuleDescriptor){
        .label = (WGPUStringView){
            .data = name,
            .length = WGPU_STRLEN,
        },
        .nextInChain = (WGPUChainedStruct *)(&shader_source_wgsl),
    };
    WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(
        device,
        &shader_module_desc
    );

    static char PIPELINE_NAME[2048];
    snprintf(PIPELINE_NAME, 2048, "Sort %s", path);
    WGPUComputePipelineDescriptor sort_kernel_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    sort_kernel_desc.label = (WGPUStringView){
        .data = PIPELINE_NAME,
        .length = WGPU_STRLEN,
    };
    sort_kernel_desc.layout = layout;
    sort_kernel_desc.compute.entryPoint = (WGPUStringView){
        .data = name,
        .length = WGPU_STRLEN,
    };
    sort_kernel_desc.compute.module = shader_module;
    sort_kernel_desc.compute.constantCount = 1;
    sort_kernel_desc.compute.constants = (WGPUConstantEntry[]){
        (WGPUConstantEntry){
            .key = (WGPUStringView){
                .data = "WG",
                .length = WGPU_STRLEN
            },
            .value = (float)wg,
        },
    };

    return wgpuDeviceCreateComputePipeline(device, &sort_kernel_desc);
}

static WGPUBindGroup bench_create_bindings(
    const bench_buffers * const buffers,
    WGPUDevice const device,
    WGPUBindGroupLayout const layout
)
{
    WGPUBindGroupDescriptor sort_binding_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    sort_binding_desc.label = (WGPUStringView){
        .data = "Benchmark Kernel: Binding",
        .length = WGPU_STRLEN,
    };
    sort_binding_desc.layout = layout;
    sort_binding_desc.entryCount = 5;
    sort_binding_desc.entries = (WGPUBindGroupEntry[]){
        (WGPUBindGroupEntry){
            .binding = 0, // global_keys
            .buffer = buffers->keys,
            .size = WGPU_WHOLE_SIZE,
        },
        (WGPUBindGroupEntry){
            .binding = 1, // global_value_indices
            .buffer = buffers->value_indices,
            .size = WGPU_WHOLE_SIZE,
        },
        (WGPUBindGroupEntry){
            .binding = 2, // segments
            .buffer = buffers->segments,
            .size = WGPU_WHOLE_SIZE,
        },
        (WGPUBindGroupEntry){
            .binding = 3, // bin_offsets
            .buffer = buffers->bin_offsets,
            .size = WGPU_WHOLE_SIZE,
        },
        (WGPUBindGroupEntry){
            .binding = 4, // bin_indices
            .buffer = buffers->bin_indices,
            .size = WGPU_WHOLE_SIZE,
        },
    };

    return wgpuDeviceCreateBindGroup(device, &sort_binding_desc);
}

static hwstats_sampler * create_uniform_sampler(
    const uint64_t seed,
    const mems_allocator * const allocator
)
{
    hwstats_x256pp * const x256pp = (hwstats_x256pp *)mems_allocator_alloc(allocator, MEMS_ALIGNOF(hwstats_x256pp), sizeof(hwstats_x256pp));
    hwstats_randomizer * const r = (hwstats_randomizer *)mems_allocator_alloc(allocator, MEMS_ALIGNOF(hwstats_randomizer), sizeof(hwstats_randomizer));
    hwstats_sampler * const sampler = (hwstats_sampler *)mems_allocator_alloc(allocator, MEMS_ALIGNOF(hwstats_sampler), sizeof(hwstats_sampler));

    hwstats_x256pp_init(x256pp, seed);
    hwstats_x256pp_rand_init(x256pp, r);
    hwstats_uniform_sampler_init(sampler, r);

    return sampler;
}

static hwstats_sampler * create_const_sampler(
    const double value,
    const mems_allocator * const allocator
)
{
    hwstats_const * const c = (hwstats_const *)mems_allocator_alloc(allocator, MEMS_ALIGNOF(hwstats_const), sizeof(hwstats_const));
    hwstats_sampler * const sampler = (hwstats_sampler *)mems_allocator_alloc(allocator, MEMS_ALIGNOF(hwstats_sampler), sizeof(hwstats_sampler));

    *c = (hwstats_const){ .value = value };

    hwstats_const_sampler_init(sampler, c);

    return sampler;
}

static hwstats_sampler * create_sampler(
    const char * const name,
    const uint64_t seed,
    const mems_allocator * const allocator
)
{
    if (strcmp("uniform", name) == 0) return create_uniform_sampler(seed, allocator);
    else if (strncmp("const", name, strlen("const")) == 0)
    {
        double value;
        if (sscanf(name, "const(%lf)", &value) != 1)
        {
            fprintf(stderr, "invalid const sampler\n");
            abort();
        }
        return create_const_sampler(value, allocator);
    }
    return NULL;
}

static uint32_t bench_wg(
    const bench_memory_kind memory,
    const uint32_t N,
    const uint32_t M,
    const uint32_t subgroup_size,
    const uint32_t max_invocations,
    const uint32_t max_smem_size,
    const uint32_t target_wg_size
) {
    const uint32_t wpt = N / M;
    switch (memory) {
        case bench_memory_reg:
        {
            uint32_t wg = WB_MIN(target_wg_size, max_invocations);
            wg = (wg / subgroup_size) * subgroup_size;
            if (wg < subgroup_size) wg = subgroup_size;
            return wg;
        }
        case bench_memory_smem:
        case bench_memory_hybrid:
        {
            const uint32_t smem_cap = max_smem_size / (wpt * 2u * 4u);
            uint32_t wg = WB_MIN(target_wg_size, WB_MIN(max_invocations, smem_cap));
            wg = (wg / M) * M;
            if (wg < M) wg = M;
            return wg;
        }
        case bench_memory_hybmerge: return M;
        default: PANIC("invalid memory kind");
    }
}

static uint32_t segment_bucket(const uint32_t segment_len)
{
    if (segment_len <= 1u) return 0u;
    const uint32_t bucket = 32u - (uint32_t)__builtin_clz(segment_len - 1u);
    return bucket < 12u ? bucket : 12u;
}

static void bench_compute_bins(
    const uint32_t segments_len,
    const uint32_t * const segments,
    uint32_t * const bin_offsets,
    uint32_t * const bin_indices
)
{
    uint32_t histogram[13] = { 0 };
    for (uint32_t i = 0; i < segments_len; i++)
    {
        const uint32_t start = (i == 0u) ? 0u : segments[i - 1u];
        const uint32_t len = segments[i] - start;
        if (len > 0u) histogram[segment_bucket(len)]++;
    }

    uint32_t cursor[13];
    uint32_t sum = 0u;
    for (uint32_t b = 0; b < 13; b++)
    {
        cursor[b] = sum;
        sum += histogram[b];
        bin_offsets[b] = sum;
    }

    for (uint32_t i = 0; i < segments_len; i++)
    {
        const uint32_t start = (i == 0u) ? 0u : segments[i - 1u];
        const uint32_t len = segments[i] - start;
        if (len > 0u) bin_indices[cursor[segment_bucket(len)]++] = i;
    }
}

static uint64_t now_ns(void)
{
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

static void benchmark_wait_idle(WGPUInstance const instance, WGPUQueue const queue)
{
    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.callback = benchmark_work_done_cb;
    WGPUFuture f = wgpuQueueOnSubmittedWorkDone(queue, cb);

    WGPUFutureWaitInfo wait = WGPU_FUTURE_WAIT_INFO_INIT;
    wait.future = f;
    wgpuInstanceWaitAny(instance, 1, &wait, (uint64_t)5 * 1000000000);
}

static void run_benchmark(
    const bench_config config,
    WGPUPipelineLayout const pipeline_layout,
    const bench_buffers * const buffers,
    WGPUBindGroup const binding,
    const size_t segments_len, uint32_t * const segments,
    const size_t keys_len, uint32_t * const keys,
    WGPUQuerySet const query,
    WGPUBuffer const query_buffer,
    bench_result ** const results,
    WGPUInstance const instance,
    WGPUDevice const device,
    WGPUQueue const queue
)
{
    static char FILE_PATH[2048];
    static char KERNEL_NAME[2048];
    const char * store_name = bench_store_name(config.store);

    switch (config.memory)
    {
        case bench_memory_reg:
        {
            snprintf(KERNEL_NAME, sizeof(KERNEL_NAME), "segsort_reg_n%u_m%u_%s",
                config.N, config.M, store_name);
        } break;
        case bench_memory_smem:
        {
            snprintf(KERNEL_NAME, sizeof(KERNEL_NAME), "segsort_wg_n%u_m%u_%s",
                config.N, config.M, store_name);
        } break;
        case bench_memory_hybrid:
        {
            snprintf(KERNEL_NAME, sizeof(KERNEL_NAME), "segsort_hybrid_sg%u_n%u_m%u_%s",
                config.subgroups, config.N, config.M, store_name);
        } break;
        case bench_memory_hybmerge:
        {
            const uint32_t smem_kb = config.smem_bytes / 1024;
            snprintf(KERNEL_NAME, sizeof(KERNEL_NAME), "segsort_hybmerge_sg%u_smem%uk_n%u_m%u_%s",
                config.subgroups, smem_kb, config.N, config.M, store_name);
        } break;
    }

    const uint32_t wg = bench_wg(
        config.memory,
        config.N,
        config.M,
        config.subgroups,
        config.max_invocations,
        config.max_smem_size,
        config.target_wg_size
    );

    *results = (bench_result *)malloc(config.runs * sizeof(bench_result));
    memset(*results, 0, config.runs * sizeof(bench_result));

    snprintf(FILE_PATH, sizeof(FILE_PATH), "shaders/sort_kernels/%s.wgsl", KERNEL_NAME);
    WGPUComputePipeline pipeline = bench_create_pipeline(KERNEL_NAME, FILE_PATH, wg, device, pipeline_layout);

    wgpuQueueWriteBuffer(queue, buffers->keys, 0, keys, keys_len * sizeof(uint32_t));
    wgpuQueueWriteBuffer(queue, buffers->segments, 0, segments, segments_len * sizeof(uint32_t));

    const uint32_t segs_per_wg = wg / config.M;
    const wbg_dispatch_size base = { segs_per_wg, 1u, 1u };
    const wbg_dispatch_size grid = wbg_dispatch_size_for_len(&base, segments_len);

    {
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
        wgpuComputePassEncoderSetPipeline(pass, pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, binding, 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(pass, grid.x, grid.y, grid.z);
        wgpuComputePassEncoderEnd(pass);
        WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
        wgpuQueueSubmit(queue, 1, &commands);
        benchmark_wait_idle(instance, queue);
        wgpuCommandBufferRelease(commands);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandEncoderRelease(encoder);
    }

    for (uint32_t i = 0; i < config.runs; i++)
    {
        {
            WGPUCommandEncoder reset_encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
            wgpuCommandEncoderCopyBufferToBuffer(
                reset_encoder,
                buffers->keys_staging, 0,
                buffers->keys, 0,
                keys_len * sizeof(uint32_t)
            );
            WGPUCommandBuffer reset_commands = wgpuCommandEncoderFinish(reset_encoder, NULL);
            wgpuQueueSubmit(queue, 1, &reset_commands);
            benchmark_wait_idle(instance, queue);
            wgpuCommandBufferRelease(reset_commands);
            wgpuCommandEncoderRelease(reset_encoder);
        }

        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);

        WGPUComputePassDescriptor pass_desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
        WGPUPassTimestampWrites ts = WGPU_PASS_TIMESTAMP_WRITES_INIT;
        ts.querySet = query;
        ts.beginningOfPassWriteIndex = 0;
        ts.endOfPassWriteIndex = 1;
        pass_desc.timestampWrites = &ts;

        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
        wgpuComputePassEncoderSetPipeline(pass, pipeline);
        wgpuComputePassEncoderSetBindGroup(pass, 0, binding, 0, NULL);
        wgpuComputePassEncoderDispatchWorkgroups(pass, grid.x, grid.y, grid.z);
        wgpuComputePassEncoderEnd(pass);

        wgpuCommandEncoderResolveQuerySet(encoder, query, 0, 2, query_buffer, 0);

        WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);

        const uint64_t wall_start = now_ns();
        wgpuQueueSubmit(queue, 1, &commands);
        benchmark_wait_idle(instance, queue);
        const uint64_t wall_end = now_ns();

        wgpuCommandBufferRelease(commands);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandEncoderRelease(encoder);

        uint64_t timestamps[2] = { 0u, 0u };
        size_t required_size = 0u;
        hwgutil_wgpu_read_buffer(
            instance, device, queue, query_buffer,
            sizeof(timestamps), timestamps, &required_size
        );

        (*results)[i] = (bench_result){
            .wall_start_ns = wall_start,
            .wall_end_ns = wall_end,
            .gpu_start_ns = timestamps[0],
            .gpu_end_ns = timestamps[1],
        };
    }

    write_results_csv(config, wg, segments_len, keys_len, *results);

    wgpuComputePipelineRelease(pipeline);
}

int benchmark_main(
    const bench_config config,
    const hwgutil_wgpu_context context
)
{
    bench_buffers buffers;

    bench_buffers_init(
        &buffers,
        context.device,
        config.n_keys,
        config.n_keys
    );

    WGPUBindGroupLayout bind_layout;
    WGPUPipelineLayout pipeline_layout;

    bench_create_pipeline_layout(
        context.device,
        &bind_layout,
        &pipeline_layout
    );

    WGPUBindGroup binding = bench_create_bindings(
        &buffers,
        context.device,
        bind_layout
    );

    WGPUQuerySetDescriptor query_desc = WGPU_QUERY_SET_DESCRIPTOR_INIT;
    query_desc.label = (WGPUStringView){
        .data = "WB Sort: Timestamp Queries",
        .length = WGPU_STRLEN,
    };
    query_desc.type = WGPUQueryType_Timestamp;
    query_desc.count = 2;

    WGPUQuerySet query = wgpuDeviceCreateQuerySet(context.device, &query_desc);

    wbg_sort_timing timing = (wbg_sort_timing){
        .query = query,
    };

    WGPUBufferDescriptor query_buffer_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    query_buffer_desc.label = (WGPUStringView){
        .data = "WB Sort: Timestamp Queries",
        .length = WGPU_STRLEN,
    };
    query_buffer_desc.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    query_buffer_desc.size = 2 * sizeof(uint64_t);

    WGPUBuffer query_buffer = wgpuDeviceCreateBuffer(context.device, &query_buffer_desc);

    static uint8_t SAMPLER_BUFFER[1024 * 4];
    mems_bump bump;
    mems_allocator allocator;
    mems_bump_init(&bump, sizeof(SAMPLER_BUFFER), SAMPLER_BUFFER);
    mems_bump_allocator_init(&bump, &allocator);

    hwstats_sampler * const sampler = create_sampler(config.sampler_name, (uint64_t)config.seed, &allocator);

    hwstats_x256pp rx;
    hwstats_randomizer r;
    hwstats_x256pp_init(&rx, config.seed);
    hwstats_x256pp_rand_init(&rx, &r);

    uint32_t * const keys = (uint32_t *)malloc(config.n_keys * sizeof(uint32_t));
    uint32_t * const value_indices = (uint32_t *)malloc(config.n_keys * sizeof(uint32_t));
    uint32_t * const segments = (uint32_t *)malloc(config.n_keys * sizeof(uint32_t));
    uint32_t * const bin_indices = (uint32_t *)malloc(config.n_keys * sizeof(uint32_t));
    uint32_t * const bin_offsets = (uint32_t *)malloc(13 * sizeof(uint32_t));

    for (size_t i = 0; i < config.n_keys; i++)
    {
        keys[i] = (uint32_t)(hwstats_uniform(&r) * (double)UINT32_MAX);
    }

    const uint32_t bin = config.bin;
    const uint32_t lo = bin == 0u ? 1u : (1u << (bin - 1u)) + 1u;
    const uint32_t hi = 1u << bin;
    const uint32_t range = hi - lo;

    uint32_t segments_len = 0;
    uint32_t keys_len = 0;
    uint32_t key_budget = config.n_keys;

    while (keys_len < key_budget)
    {
        const uint32_t segment_len = lo + (uint32_t)(hwstats_sample(sampler) * (double)range);

        if (keys_len + segment_len > key_budget) break;

        keys_len += segment_len;
        segments[segments_len++] = keys_len;
    }

    bench_compute_bins(
        segments_len, segments,
        bin_offsets, bin_indices
    );

    wgpuQueueWriteBuffer(
        context.queue,
        buffers.bin_offsets,
        0,
        bin_offsets,
        13 * sizeof(uint32_t)
    );

    wgpuQueueWriteBuffer(
        context.queue,
        buffers.bin_indices,
        0,
        bin_indices,
        segments_len * sizeof(uint32_t)
    );

    wgpuQueueWriteBuffer(
        context.queue,
        buffers.keys_staging,
        0,
        keys,
        keys_len * sizeof(uint32_t)
    );

    bench_result * results;
    run_benchmark(
        config,
        pipeline_layout,
        &buffers,
        binding,
        segments_len, segments,
        keys_len, keys,
        query,
        query_buffer,
        &results,
        context.instance,
        context.device,
        context.queue
    );
        
    free(keys);
    free(value_indices);
    free(segments);

    return 0;
}

int main(const int argc, const char ** const argv)
{
    static char ARGS_BUFFER[1024 * 4];
    hwargs_size required_args_size;
    hwargs_parsed args;
    if (!hwargs_parse(
        &args,
        argc, argv,
        sizeof(ARGS_BUFFER),
        ARGS_BUFFER,
        &required_args_size
    )) PANIC("could not parse arguments");

    if (args.positionals_len < 2)
    {
        PANIC("must provide <output dir> as first poositional argument");
    }

    const char * const root = args.positionals[1];
    const hwargs_param * const memory_param = hwargs_get_param(&args, "memory");
    const hwargs_param * const store_param = hwargs_get_param(&args, "store");
    const hwargs_param * const seed_param = hwargs_get_param(&args, "seed");
    const hwargs_param * const bin_param = hwargs_get_param(&args, "bin");
    const hwargs_param * const n_keys_param = hwargs_get_param(&args, "keys");
    const hwargs_param * const runs_param = hwargs_get_param(&args, "runs");
    const hwargs_param * const N_param = hwargs_get_param(&args, "N");
    const hwargs_param * const M_param = hwargs_get_param(&args, "M");
    const hwargs_param * const R_param = hwargs_get_param(&args, "R");
    const hwargs_param * const smem_param = hwargs_get_param(&args, "smem");
    const hwargs_param * const subgroups_param = hwargs_get_param(&args, "subgroups");
    const hwargs_param * const sampler_name_param = hwargs_get_param(&args, "sampler");

    ENSURE_MSG(memory_param != NULL, "must provide --memory <reg|smem|hybrid|hybmerge>");
    ENSURE_MSG(store_param != NULL, "must provide --store <block|striped>");
    ENSURE_MSG(seed_param != NULL, "must provide --seed <u32>");
    ENSURE_MSG(bin_param != NULL, "must provide --bin <0..12>");
    ENSURE_MSG(runs_param != NULL, "must provide --keys <u32>");
    ENSURE_MSG(n_keys_param != NULL, "must provide --keys <u32>");
    ENSURE_MSG(N_param != NULL, "must provide --N <u32>");
    ENSURE_MSG(M_param != NULL, "must provide --M <u32>");
    ENSURE_MSG(smem_param != NULL, "must provide --smem <16|32>");
    ENSURE_MSG(subgroups_param != NULL, "must provide --subgroups <8|16|32|64|128>");
    ENSURE_MSG(sampler_name_param != NULL, "must provide --sampler <const([0.0,1.0]),uniform> # distributes segment sizes across bin range");

    char * endptr;
    const bench_memory_kind memory = bench_memory_for_name(memory_param->value);
    const bench_store_kind store = bench_store_for_name(store_param->value);
    const uint32_t seed = strtol(seed_param->value, &endptr, 10);
    const uint32_t N = strtol(N_param->value, &endptr, 10);
    const uint32_t M = strtol(M_param->value, &endptr, 10);
    uint32_t R = 0;
    const uint32_t smem_kb = strtol(smem_param->value, &endptr, 10);
    const uint32_t runs = strtol(runs_param->value, &endptr, 10);
    const uint32_t bin = strtol(bin_param->value, &endptr, 10);
    const uint32_t n_keys = strtol(n_keys_param->value, &endptr, 10);
    const uint32_t subgroups = strtol(subgroups_param->value, &endptr, 10);

    if (smem_kb != 16 && smem_kb != 32) PANIC("smem must be 16 or 32");
    if (bin > 11) PANIC("bin must be <= 12");

    switch (memory)
    {
        case bench_memory_hybrid:
        case bench_memory_hybmerge:
        {
            ENSURE_MSG(R_param != NULL, "must provide --R <u32>");
            R = strtol(R_param->value, &endptr, 10);
        } break;
        default: break;
    }

    hwgutil_wgpu_context context;

    WGPULimits required_limits = WGPU_LIMITS_INIT;
    required_limits.maxStorageBufferBindingSize = 1024llu * 1024 * 1024 * 2;
    required_limits.maxBufferSize = 1024llu * 1024 * 1024 * 2;

    if (!hwgutil_wgpu_context_init(
        &context,
        1, (WGPUInstanceFeatureName[]){ WGPUInstanceFeatureName_TimedWaitAny },
        2, (WGPUFeatureName[]){ WGPUFeatureName_TimestampQuery, WGPUFeatureName_Subgroups },
        &required_limits
    )) PANIC("could not initialize WebGPU context");

    WGPULimits limits;
    wgpuDeviceGetLimits(context.device, &limits);

    const bench_config config = (bench_config){
        .root = root,
        .memory = memory,
        .store = store,
        .seed = seed,
        .runs = runs,
        .N = N,
        .M = M,
        .R = R,
        .smem_bytes = smem_kb * 1024,
        .bin = bin,
        .n_keys = n_keys,
        .sampler_name = sampler_name_param->value,
        .subgroups = subgroups,
        .max_invocations = limits.maxComputeInvocationsPerWorkgroup,
        .max_smem_size = limits.maxComputeWorkgroupStorageSize,
        .target_wg_size = 256u,
    };

    return benchmark_main(config, context);
}
