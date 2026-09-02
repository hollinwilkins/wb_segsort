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
    bool validate;
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
    // Scratch for the GPU binning pass (wb_bin.wgsl).
    WGPUBuffer bin_config;      // uniform: Config { segments_len }
    WGPUBuffer bin_config_data; // storage: array<BinConfig> (unused by us, kept valid)
    WGPUBuffer bin_histogram;   // storage: array<atomic<u32>, 13>
    WGPUBuffer bin_dispatch;    // storage: array<DispatchSize, 13> (unused output)
} bench_buffers;

typedef struct bench_result
{
    uint64_t wall_start_ns;
    uint64_t wall_end_ns;
    uint64_t gpu_start_ns;
    uint64_t gpu_end_ns;
} bench_result;

// One row of the experiments CSV: a single kernel config to benchmark.
typedef struct bench_experiment
{
    char name[512];
    bench_memory_kind memory;
    bench_store_kind store;
    uint32_t N;
    uint32_t M;
    uint32_t R;
    uint32_t subgroups;
    uint32_t smem_kb;
    uint32_t bin;
} bench_experiment;

// Outcome of validating one kernel against the CPU reference sort.
typedef struct bench_validation
{
    bool valid;
    size_t keys_len;
    size_t segments_len;
    size_t fail_index;      // first mismatched global index (when !valid)
    const char * fail_kind; // "keys" | "values" | "" when valid
    uint32_t cpu_key;
    uint32_t gpu_key;
    uint32_t cpu_vi;
    uint32_t gpu_vi;
} bench_validation;

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

// Write one row per experiment describing the validation outcome, to
// "<experiments_path minus .csv>.validate.csv" (next to the experiments file).
static void write_validation_csv(
    const char * const experiments_path,
    const bench_experiment * const exps,
    const bench_validation * const results,
    const size_t count
)
{
    static char PATH[2048];
    const size_t len = strlen(experiments_path);
    if (len > 4 && strcmp(experiments_path + len - 4, ".csv") == 0)
    {
        snprintf(PATH, sizeof(PATH), "%.*s.validate.csv", (int)(len - 4), experiments_path);
    }
    else
    {
        snprintf(PATH, sizeof(PATH), "%s.validate.csv", experiments_path);
    }

    make_parent_dirs(PATH);

    FILE * const f = fopen(PATH, "w");
    if (f == NULL) PANIC("could not open validation csv for writing: %s", PATH);

    fprintf(f,
        "name,memory,store,N,M,R,subgroups,smem,bin,keys,segments,valid,"
        "fail_index,fail_kind,cpu_key,gpu_key,cpu_vi,gpu_vi\n");

    size_t passed = 0;
    size_t failed = 0;

    for (size_t e = 0; e < count; e++)
    {
        const bench_experiment * const x = &exps[e];
        const bench_validation * const v = &results[e];

        fprintf(f,
            "%s,%s,%s,%u,%u,%u,%u,%u,%u,%zu,%zu,%s,%zu,%s,%u,%u,%u,%u\n",
            x->name, bench_memory_name(x->memory), bench_store_name(x->store),
            x->N, x->M, x->R, x->subgroups, x->smem_kb, x->bin,
            v->keys_len, v->segments_len, v->valid ? "true" : "false",
            v->valid ? (size_t)0 : v->fail_index, v->valid ? "" : v->fail_kind,
            v->cpu_key, v->gpu_key, v->cpu_vi, v->gpu_vi);

        if (v->valid) passed++;
        else failed++;
    }

    fclose(f);

    fprintf(stdout, "\nvalidation: %zu passed, %zu failed -> %s\n", passed, failed, PATH);
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

// Parse the experiments CSV (header: name,memory,store,N,M,R,subgroups,smem,bin)
// into a malloc'd array. Caller frees. Sets *out_count.
static bench_experiment * bench_load_experiments(const char * const path, size_t * const out_count)
{
    FILE * const f = fopen(path, "r");
    if (f == NULL) PANIC("could not open experiments csv: %s", path);

    static char LINE[1024];

    if (fgets(LINE, sizeof(LINE), f) == NULL) PANIC("empty experiments csv: %s", path);
    const long data_start = ftell(f);

    size_t cap = 0;
    while (fgets(LINE, sizeof(LINE), f) != NULL) cap++;
    fseek(f, data_start, SEEK_SET);

    bench_experiment * const exps = (bench_experiment *)malloc((cap == 0 ? 1 : cap) * sizeof(bench_experiment));
    size_t n = 0;

    while (fgets(LINE, sizeof(LINE), f) != NULL)
    {
        if (LINE[0] == '\n' || LINE[0] == '\r' || LINE[0] == '\0') continue;

        char name[512];
        char memory[32];
        char store[32];
        uint32_t N, M, R, subgroups, smem_kb, bin;

        if (sscanf(LINE, "%511[^,],%31[^,],%31[^,],%u,%u,%u,%u,%u,%u",
                name, memory, store, &N, &M, &R, &subgroups, &smem_kb, &bin) != 9)
        {
            PANIC("malformed experiments row: %s", LINE);
        }

        bench_experiment * const e = &exps[n++];
        snprintf(e->name, sizeof(e->name), "%s", name);
        e->memory = bench_memory_for_name(memory);
        e->store = bench_store_for_name(store);
        e->N = N;
        e->M = M;
        e->R = R;
        e->subgroups = subgroups;
        e->smem_kb = smem_kb;
        e->bin = bin;
    }

    fclose(f);
    *out_count = n;
    return exps;
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

    WGPUBufferDescriptor bin_config_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bin_config_desc.label = (WGPUStringView){ .data = "Benchmark Kernel: Bin Config", .length = WGPU_STRLEN };
    bin_config_desc.size = 16; // Config { segments_len } padded to a uniform block
    bin_config_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;

    WGPUBufferDescriptor bin_config_data_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bin_config_data_desc.label = (WGPUStringView){ .data = "Benchmark Kernel: Bin Config Data", .length = WGPU_STRLEN };
    bin_config_data_desc.size = 13 * 16; // array<BinConfig{n,m,wg,flags}, 13>
    bin_config_data_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;

    WGPUBufferDescriptor bin_histogram_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bin_histogram_desc.label = (WGPUStringView){ .data = "Benchmark Kernel: Bin Histogram", .length = WGPU_STRLEN };
    bin_histogram_desc.size = 13 * sizeof(uint32_t);
    bin_histogram_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;

    WGPUBufferDescriptor bin_dispatch_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    bin_dispatch_desc.label = (WGPUStringView){ .data = "Benchmark Kernel: Bin Dispatch", .length = WGPU_STRLEN };
    bin_dispatch_desc.size = 13 * 16; // array<DispatchSize, 13> (over-allocated)
    bin_dispatch_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;

    *buffers = (bench_buffers){
        .keys = wgpuDeviceCreateBuffer(device, &keys_desc),
        .keys_staging = wgpuDeviceCreateBuffer(device, &keys_staging_desc),
        .value_indices = wgpuDeviceCreateBuffer(device, &value_indices_desc),
        .segments = wgpuDeviceCreateBuffer(device, &segments_desc),
        .bin_offsets = wgpuDeviceCreateBuffer(device, &bin_offsets_desc),
        .bin_indices = wgpuDeviceCreateBuffer(device, &bin_indices_desc),
        .bin_config = wgpuDeviceCreateBuffer(device, &bin_config_desc),
        .bin_config_data = wgpuDeviceCreateBuffer(device, &bin_config_data_desc),
        .bin_histogram = wgpuDeviceCreateBuffer(device, &bin_histogram_desc),
        .bin_dispatch = wgpuDeviceCreateBuffer(device, &bin_dispatch_desc),
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

// Generate the segment layout for a bin: segment sizes in (2^(bin-1), 2^bin],
// packed until the n_keys budget is exhausted. Deterministic in seed (a fresh
// sampler is created each call so a bin's layout is independent of iteration
// order). Fills `segments` (cumulative ends), returns segments_len, sets keys_len.
static uint32_t bench_generate_segments(
    const uint32_t bin,
    const uint32_t n_keys,
    const uint32_t seed,
    const char * const sampler_name,
    uint32_t * const segments,
    uint32_t * const out_keys_len
)
{
    uint8_t sampler_buffer[1024 * 4];
    mems_bump bump;
    mems_allocator allocator;
    mems_bump_init(&bump, sizeof(sampler_buffer), sampler_buffer);
    mems_bump_allocator_init(&bump, &allocator);

    hwstats_sampler * const sampler = create_sampler(sampler_name, (uint64_t)seed, &allocator);

    const uint32_t lo = bin == 0u ? 1u : (1u << (bin - 1u)) + 1u;
    const uint32_t hi = 1u << bin;
    const uint32_t range = hi - lo;

    uint32_t segments_len = 0;
    uint32_t keys_len = 0;

    while (keys_len < n_keys)
    {
        const uint32_t segment_len = lo + (uint32_t)(hwstats_sample(sampler) * (double)range);

        if (keys_len + segment_len > n_keys) break;

        keys_len += segment_len;
        segments[segments_len++] = keys_len;
    }

    *out_keys_len = keys_len;
    return segments_len;
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

// Compute bin_offsets/bin_indices on the GPU using the real binning kernels
// (shaders/wb_bin.wgsl), the same clear -> histogram -> schedule -> group passes
// the production pipeline uses. Writes directly into buffers->bin_offsets and
// buffers->bin_indices; segments must already be uploaded. This replaces the CPU
// prefix-sum, which is prohibitively slow when small N produces millions of
// segments. bin_config_data/bin_histogram/bin_dispatch are Dawn zero-initialized;
// only bin_dispatch's output (unused here) depends on bin_config_data.
static void bench_compute_bins_gpu(
    const bench_buffers * const buffers,
    const uint32_t segments_len,
    WGPUInstance const instance,
    WGPUDevice const device,
    WGPUQueue const queue
)
{
    size_t source_len;
    char * const source = wbg__read_file("shaders/wb_bin.wgsl", &mems_system_allocator, &source_len);

    WGPUShaderSourceWGSL source_wgsl = (WGPUShaderSourceWGSL){
        .chain = (WGPUChainedStruct){ .sType = WGPUSType_ShaderSourceWGSL },
        .code = (WGPUStringView){ .data = source, .length = source_len },
    };
    WGPUShaderModuleDescriptor module_desc = (WGPUShaderModuleDescriptor){
        .label = (WGPUStringView){ .data = "wb_bin", .length = WGPU_STRLEN },
        .nextInChain = (WGPUChainedStruct *)(&source_wgsl),
    };
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(device, &module_desc);

    WGPUBindGroupLayoutDescriptor layout_desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    layout_desc.entryCount = 7;
    layout_desc.entries = (WGPUBindGroupLayoutEntry[]){
        { .binding = 0, .visibility = WGPUShaderStage_Compute, .buffer = { .type = WGPUBufferBindingType_Uniform } },
        { .binding = 1, .visibility = WGPUShaderStage_Compute, .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
        { .binding = 2, .visibility = WGPUShaderStage_Compute, .buffer = { .type = WGPUBufferBindingType_ReadOnlyStorage } },
        { .binding = 3, .visibility = WGPUShaderStage_Compute, .buffer = { .type = WGPUBufferBindingType_Storage } },
        { .binding = 4, .visibility = WGPUShaderStage_Compute, .buffer = { .type = WGPUBufferBindingType_Storage } },
        { .binding = 5, .visibility = WGPUShaderStage_Compute, .buffer = { .type = WGPUBufferBindingType_Storage } },
        { .binding = 6, .visibility = WGPUShaderStage_Compute, .buffer = { .type = WGPUBufferBindingType_Storage } },
    };
    WGPUBindGroupLayout layout = wgpuDeviceCreateBindGroupLayout(device, &layout_desc);

    WGPUPipelineLayoutDescriptor pipeline_layout_desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipeline_layout_desc.bindGroupLayoutCount = 1;
    pipeline_layout_desc.bindGroupLayouts = (WGPUBindGroupLayout[]){ layout };
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &pipeline_layout_desc);

    // main_histogram / main_group are 16x16 workgroups (WORKGROUP_ITEMS = 256).
    WGPUConstantEntry wg_consts[2] = {
        { .key = (WGPUStringView){ .data = "WORKGROUP_SIZE_X", .length = WGPU_STRLEN }, .value = 16.0 },
        { .key = (WGPUStringView){ .data = "WORKGROUP_SIZE_Y", .length = WGPU_STRLEN }, .value = 16.0 },
    };

    WGPUComputePipelineDescriptor clear_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    clear_desc.layout = pipeline_layout;
    clear_desc.compute.module = module;
    clear_desc.compute.entryPoint = (WGPUStringView){ .data = "main_clear", .length = WGPU_STRLEN };
    WGPUComputePipeline clear_pipeline = wgpuDeviceCreateComputePipeline(device, &clear_desc);

    WGPUComputePipelineDescriptor histogram_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    histogram_desc.layout = pipeline_layout;
    histogram_desc.compute.module = module;
    histogram_desc.compute.entryPoint = (WGPUStringView){ .data = "main_histogram", .length = WGPU_STRLEN };
    histogram_desc.compute.constantCount = 2;
    histogram_desc.compute.constants = wg_consts;
    WGPUComputePipeline histogram_pipeline = wgpuDeviceCreateComputePipeline(device, &histogram_desc);

    WGPUComputePipelineDescriptor schedule_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    schedule_desc.layout = pipeline_layout;
    schedule_desc.compute.module = module;
    schedule_desc.compute.entryPoint = (WGPUStringView){ .data = "main_schedule", .length = WGPU_STRLEN };
    WGPUComputePipeline schedule_pipeline = wgpuDeviceCreateComputePipeline(device, &schedule_desc);

    WGPUComputePipelineDescriptor group_desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    group_desc.layout = pipeline_layout;
    group_desc.compute.module = module;
    group_desc.compute.entryPoint = (WGPUStringView){ .data = "main_group", .length = WGPU_STRLEN };
    group_desc.compute.constantCount = 2;
    group_desc.compute.constants = wg_consts;
    WGPUComputePipeline group_pipeline = wgpuDeviceCreateComputePipeline(device, &group_desc);

    WGPUBindGroupDescriptor bind_desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bind_desc.layout = layout;
    bind_desc.entryCount = 7;
    bind_desc.entries = (WGPUBindGroupEntry[]){
        { .binding = 0, .buffer = buffers->bin_config,      .size = WGPU_WHOLE_SIZE },
        { .binding = 1, .buffer = buffers->segments,        .size = WGPU_WHOLE_SIZE },
        { .binding = 2, .buffer = buffers->bin_config_data, .size = WGPU_WHOLE_SIZE },
        { .binding = 3, .buffer = buffers->bin_histogram,   .size = WGPU_WHOLE_SIZE },
        { .binding = 4, .buffer = buffers->bin_offsets,     .size = WGPU_WHOLE_SIZE },
        { .binding = 5, .buffer = buffers->bin_indices,     .size = WGPU_WHOLE_SIZE },
        { .binding = 6, .buffer = buffers->bin_dispatch,    .size = WGPU_WHOLE_SIZE },
    };
    WGPUBindGroup binding = wgpuDeviceCreateBindGroup(device, &bind_desc);

    wgpuQueueWriteBuffer(queue, buffers->bin_config, 0, &segments_len, sizeof(segments_len));

    // One workgroup (16x16=256 threads) per 256 segments; kernel flattens x/y.
    const wbg_dispatch_size bin_base = { 16u, 16u, 1u };
    const wbg_dispatch_size grid = wbg_dispatch_size_for_len(&bin_base, segments_len);

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    wgpuComputePassEncoderSetBindGroup(pass, 0, binding, 0, NULL);
    wgpuComputePassEncoderSetPipeline(pass, clear_pipeline);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    wgpuComputePassEncoderSetPipeline(pass, histogram_pipeline);
    wgpuComputePassEncoderDispatchWorkgroups(pass, grid.x, grid.y, grid.z);
    wgpuComputePassEncoderSetPipeline(pass, schedule_pipeline);
    wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    wgpuComputePassEncoderSetPipeline(pass, group_pipeline);
    wgpuComputePassEncoderDispatchWorkgroups(pass, grid.x, grid.y, grid.z);
    wgpuComputePassEncoderEnd(pass);

    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(queue, 1, &commands);
    benchmark_wait_idle(instance, queue);

    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(binding);
    wgpuComputePipelineRelease(clear_pipeline);
    wgpuComputePipelineRelease(histogram_pipeline);
    wgpuComputePipelineRelease(schedule_pipeline);
    wgpuComputePipelineRelease(group_pipeline);
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuBindGroupLayoutRelease(layout);
    wgpuShaderModuleRelease(module);
    mems_allocator_free(&mems_system_allocator, source);
}

// Dispatch the sort once and check the GPU output against a precomputed CPU
// reference sort (shared across every experiment in the same bin). Records the
// first mismatch (if any) and returns it; never aborts, so a sweep can validate
// every config and tabulate the results.
static bench_validation bench_validate(
    const char * const name,
    const bench_buffers * const buffers,
    WGPUComputePipeline const pipeline,
    WGPUBindGroup const binding,
    const wbg_dispatch_size grid,
    const size_t segments_len,
    const size_t keys_len,
    const uint32_t * const expected_keys,
    const uint32_t * const expected_value_indices,
    WGPUInstance const instance,
    WGPUDevice const device,
    WGPUQueue const queue
)
{
    // Single dispatch over the freshly-uploaded (unsorted) keys.
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

    // Read back the sorted keys and value indices the kernel wrote.
    uint32_t * gpu_keys;
    hwgutil_wgpu_read_buffer_alloc(
        instance, device, queue, buffers->keys, &mems_system_allocator, (void **)&gpu_keys);

    uint32_t * gpu_value_indices;
    hwgutil_wgpu_read_buffer_alloc(
        instance, device, queue, buffers->value_indices, &mems_system_allocator, (void **)&gpu_value_indices);

    bench_validation result = (bench_validation){
        .valid = true,
        .keys_len = keys_len,
        .segments_len = segments_len,
        .fail_kind = "",
    };

    for (size_t i = 0; i < keys_len; i++)
    {
        const bool key_bad = expected_keys[i] != gpu_keys[i];
        const bool vi_bad = expected_value_indices[i] != gpu_value_indices[i];

        if (key_bad || vi_bad)
        {
            result.valid = false;
            result.fail_index = i;
            result.fail_kind = key_bad ? "keys" : "values";
            result.cpu_key = expected_keys[i];
            result.gpu_key = gpu_keys[i];
            result.cpu_vi = expected_value_indices[i];
            result.gpu_vi = gpu_value_indices[i];
            break;
        }
    }

    if (result.valid)
    {
        fprintf(stdout, "VALIDATION PASSED %s (%zu keys, %zu segments)\n", name, keys_len, segments_len);
    }
    else
    {
        fprintf(stderr, "VALIDATION FAILED %s: %s[%zu] cpu_key=%u gpu_key=%u cpu_vi=%u gpu_vi=%u\n",
            name, result.fail_kind, result.fail_index,
            result.cpu_key, result.gpu_key, result.cpu_vi, result.gpu_vi);
    }

    mems_allocator_free(&mems_system_allocator, gpu_keys);
    mems_allocator_free(&mems_system_allocator, gpu_value_indices);

    return result;
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
    bench_validation * const out_validation,
    const uint32_t * const expected_keys,
    const uint32_t * const expected_value_indices,
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

    // segments + bin_offsets/bin_indices were uploaded/computed by the caller.
    wgpuQueueWriteBuffer(queue, buffers->keys, 0, keys, keys_len * sizeof(uint32_t));

    const uint32_t segs_per_wg = wg / config.M;
    const wbg_dispatch_size base = { segs_per_wg, 1u, 1u };
    const wbg_dispatch_size grid = wbg_dispatch_size_for_len(&base, segments_len);

    if (config.validate)
    {
        const bench_validation v = bench_validate(
            KERNEL_NAME, buffers, pipeline, binding, grid,
            segments_len, keys_len, expected_keys, expected_value_indices,
            instance, device, queue);
        if (out_validation != NULL) *out_validation = v;
        wgpuComputePipelineRelease(pipeline);
        return;
    }

    fprintf(stdout, "Warmup run...\n");
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

    fprintf(stdout,
        "Benchmarking %s:\n"
        "  root=%s memory=%s store=%s sampler=%s\n"
        "  N=%u M=%u wpt=%u R=%u subgroups=%u wg=%u bin=%u smem_bytes=%u\n"
        "  seed=%u runs=%u n_keys=%u segments=%zu keys=%zu\n"
        "  max_invocations=%u max_smem_size=%u target_wg_size=%u\n",
        KERNEL_NAME,
        config.root, bench_memory_name(config.memory), bench_store_name(config.store),
        config.sampler_name,
        config.N, config.M, config.N / config.M, config.R, config.subgroups, wg,
        config.bin, config.smem_bytes,
        config.seed, config.runs, config.n_keys, segments_len, keys_len,
        config.max_invocations, config.max_smem_size, config.target_wg_size);

    for (uint32_t i = 0; i < config.runs; i++)
    {
        fprintf(stdout, "Benchmark run (%u/%u)...\n", i, config.runs);
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


// ---- Batch runner: many experiments in one process -------------------------

// Constant per-device resources shared by every experiment that runs on a
// context. A distinct context (device) is needed per smem value because the
// workgroup-storage limit is fixed at device creation.
typedef struct bench_context_res
{
    hwgutil_wgpu_context context;
    bench_buffers buffers;
    WGPUBindGroupLayout bind_layout;
    WGPUPipelineLayout pipeline_layout;
    WGPUBindGroup binding;
    WGPUQuerySet query;
    WGPUBuffer query_buffer;
    uint32_t max_invocations;
    uint32_t max_smem_size;
} bench_context_res;

static void bench_context_res_init(
    bench_context_res * const res,
    const uint32_t smem_kb,
    const uint32_t n_keys
)
{
    WGPULimits required_limits = WGPU_LIMITS_INIT;
    required_limits.maxStorageBufferBindingSize = 1024llu * 1024 * 1024 * 2;
    required_limits.maxBufferSize = 1024llu * 1024 * 1024 * 2;
    required_limits.maxComputeWorkgroupStorageSize = smem_kb * 1024;

    fprintf(stdout, "Initializing WebGPU context (smem=%uk)...\n", smem_kb);

    if (!hwgutil_wgpu_context_init(
        &res->context,
        1, (WGPUInstanceFeatureName[]){ WGPUInstanceFeatureName_TimedWaitAny },
        2, (WGPUFeatureName[]){ WGPUFeatureName_TimestampQuery, WGPUFeatureName_Subgroups },
        &required_limits
    )) PANIC("could not initialize WebGPU context");

    WGPULimits limits = WGPU_LIMITS_INIT;
    wgpuDeviceGetLimits(res->context.device, &limits);
    res->max_invocations = limits.maxComputeInvocationsPerWorkgroup;
    res->max_smem_size = limits.maxComputeWorkgroupStorageSize;

    bench_buffers_init(&res->buffers, res->context.device, n_keys, n_keys);
    bench_create_pipeline_layout(res->context.device, &res->bind_layout, &res->pipeline_layout);
    res->binding = bench_create_bindings(&res->buffers, res->context.device, res->bind_layout);

    WGPUQuerySetDescriptor query_desc = WGPU_QUERY_SET_DESCRIPTOR_INIT;
    query_desc.label = (WGPUStringView){ .data = "WB Sort: Timestamp Queries", .length = WGPU_STRLEN };
    query_desc.type = WGPUQueryType_Timestamp;
    query_desc.count = 2;
    res->query = wgpuDeviceCreateQuerySet(res->context.device, &query_desc);

    WGPUBufferDescriptor query_buffer_desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    query_buffer_desc.label = (WGPUStringView){ .data = "WB Sort: Timestamp Queries", .length = WGPU_STRLEN };
    query_buffer_desc.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    query_buffer_desc.size = 2 * sizeof(uint64_t);
    res->query_buffer = wgpuDeviceCreateBuffer(res->context.device, &query_buffer_desc);
}

static void bench_context_res_release(bench_context_res * const res)
{
    wgpuBufferRelease(res->query_buffer);
    wgpuQuerySetRelease(res->query);
    wgpuBindGroupRelease(res->binding);
    wgpuPipelineLayoutRelease(res->pipeline_layout);
    wgpuBindGroupLayoutRelease(res->bind_layout);
    wgpuBufferRelease(res->buffers.keys);
    wgpuBufferRelease(res->buffers.keys_staging);
    wgpuBufferRelease(res->buffers.value_indices);
    wgpuBufferRelease(res->buffers.segments);
    wgpuBufferRelease(res->buffers.bin_offsets);
    wgpuBufferRelease(res->buffers.bin_indices);
    wgpuBufferRelease(res->buffers.bin_config);
    wgpuBufferRelease(res->buffers.bin_config_data);
    wgpuBufferRelease(res->buffers.bin_histogram);
    wgpuBufferRelease(res->buffers.bin_dispatch);
    hwgutil_wgpu_context_release(&res->context);
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
        PANIC("usage: %s <output_root_dir> --experiments <csv> --keys <n> --runs <n> "
              "[--seed <n>] [--sampler <s>] [-validate] [-skip-existing]", argv[0]);
    }

    const char * const root_dir = args.positionals[1];

    const hwargs_param * const experiments_param = hwargs_get_param(&args, "experiments");
    const hwargs_param * const keys_param = hwargs_get_param(&args, "keys");
    const hwargs_param * const runs_param = hwargs_get_param(&args, "runs");
    const hwargs_param * const seed_param = hwargs_get_param(&args, "seed");
    const hwargs_param * const sampler_param = hwargs_get_param(&args, "sampler");
    const bool validate = hwargs_has_flag(&args, "validate");
    const bool skip_existing = hwargs_has_flag(&args, "skip-existing");

    ENSURE_MSG(experiments_param != NULL, "must provide --experiments <csv path>");
    ENSURE_MSG(keys_param != NULL, "must provide --keys <u32>");
    ENSURE_MSG(runs_param != NULL, "must provide --runs <u32>");

    char * endptr;
    const uint32_t n_keys = strtol(keys_param->value, &endptr, 10);
    const uint32_t runs = strtol(runs_param->value, &endptr, 10);
    const uint32_t seed = seed_param != NULL ? (uint32_t)strtol(seed_param->value, &endptr, 10) : 1u;
    const char * const sampler_name = sampler_param != NULL ? sampler_param->value : "uniform";

    size_t exp_count = 0;
    bench_experiment * const exps = bench_load_experiments(experiments_param->value, &exp_count);
    fprintf(stdout, "Loaded %zu experiments from %s\n", exp_count, experiments_param->value);

    // In validate mode, one outcome per experiment; written to a single CSV at the end.
    bench_validation * const validations = validate
        ? (bench_validation *)calloc(exp_count == 0 ? 1 : exp_count, sizeof(bench_validation))
        : NULL;

    // Keys are independent of bin/smem: generate the whole set once.
    uint32_t * const keys = (uint32_t *)malloc((size_t)n_keys * sizeof(uint32_t));
    uint32_t * const segments = (uint32_t *)malloc((size_t)n_keys * sizeof(uint32_t));
    {
        hwstats_x256pp rx;
        hwstats_randomizer r;
        hwstats_x256pp_init(&rx, seed);
        hwstats_x256pp_rand_init(&rx, &r);
        fprintf(stdout, "Generating %u keys...\n", n_keys);
        for (size_t i = 0; i < n_keys; i++)
        {
            keys[i] = (uint32_t)(hwstats_uniform(&r) * (double)UINT32_MAX);
        }
    }

    size_t total = 0;

    // Iterate by smem: each distinct value needs its own device (the
    // workgroup-storage limit is fixed at device creation).
    for (uint32_t smem_kb = 1; smem_kb <= 64; smem_kb <<= 1)
    {
        bool smem_used = false;
        for (size_t e = 0; e < exp_count; e++)
        {
            if (exps[e].smem_kb == smem_kb) { smem_used = true; break; }
        }
        if (!smem_used) continue;

        bench_context_res res;
        bench_context_res_init(&res, smem_kb, n_keys);

        // Pristine keys uploaded once per device (reset source for every run).
        fprintf(stdout, "Uploading %u keys to device (smem=%uk)...\n", n_keys, smem_kb);
        wgpuQueueWriteBuffer(res.context.queue, res.buffers.keys_staging, 0, keys, (size_t)n_keys * sizeof(uint32_t));

        // Iterate by bin: the segment layout depends only on the bin.
        for (uint32_t bin = 0; bin <= 12; bin++)
        {
            bool bin_used = false;
            for (size_t e = 0; e < exp_count; e++)
            {
                if (exps[e].smem_kb == smem_kb && exps[e].bin == bin) { bin_used = true; break; }
            }
            if (!bin_used) continue;

            fprintf(stdout, "\n=== smem=%uk bin=%u ===\n", smem_kb, bin);

            fprintf(stdout, "  generating segment layout...\n");
            uint32_t keys_len = 0;
            const uint32_t segments_len = bench_generate_segments(bin, n_keys, seed, sampler_name, segments, &keys_len);
            fprintf(stdout, "  %u segments, %u keys\n", segments_len, keys_len);

            fprintf(stdout, "  uploading segments + binning (gpu)...\n");
            wgpuQueueWriteBuffer(res.context.queue, res.buffers.segments, 0, segments, segments_len * sizeof(uint32_t));
            bench_compute_bins_gpu(&res.buffers, segments_len, res.context.instance, res.context.device, res.context.queue);

            // CPU reference sort depends only on (keys, segments) == the bin, so
            // compute it once here and share it across every experiment in the bin
            // (avoids re-sorting ~1M keys per kernel, the validation bottleneck).
            uint32_t * expected_keys = NULL;
            uint32_t * expected_value_indices = NULL;
            if (validate)
            {
                fprintf(stdout, "  computing cpu reference sort (shared across the bin)...\n");
                expected_keys = (uint32_t *)malloc((size_t)keys_len * sizeof(uint32_t));
                expected_value_indices = (uint32_t *)malloc((size_t)keys_len * sizeof(uint32_t));
                memcpy(expected_keys, keys, (size_t)keys_len * sizeof(uint32_t));
                for (size_t i = 0; i < keys_len; i++) expected_value_indices[i] = (uint32_t)i;
                wbc_segsort_alloc(keys_len, expected_keys, expected_value_indices, segments_len, segments);
                fprintf(stdout, "  cpu reference ready\n");
            }

            fprintf(stdout, "  running %s...\n", validate ? "validation" : "benchmarks");
            for (size_t e = 0; e < exp_count; e++)
            {
                const bench_experiment * const exp = &exps[e];
                if (exp->smem_kb != smem_kb || exp->bin != bin) continue;

                total++;

                static char ROOT_PATH[2048];
                snprintf(ROOT_PATH, sizeof(ROOT_PATH), "%s/%s", root_dir, exp->name);

                if (skip_existing && !validate)
                {
                    static char CSV_PATH[2048];
                    snprintf(CSV_PATH, sizeof(CSV_PATH), "%s.csv", ROOT_PATH);
                    struct stat st;
                    if (stat(CSV_PATH, &st) == 0)
                    {
                        fprintf(stdout, "[%zu/%zu] skip (exists): %s\n", total, exp_count, exp->name);
                        continue;
                    }
                }

                const bench_config config = (bench_config){
                    .root = ROOT_PATH,
                    .memory = exp->memory,
                    .store = exp->store,
                    .seed = seed,
                    .runs = runs,
                    .N = exp->N,
                    .M = exp->M,
                    .R = exp->R,
                    .smem_bytes = exp->smem_kb * 1024,
                    .bin = exp->bin,
                    .n_keys = n_keys,
                    .subgroups = exp->subgroups,
                    .max_invocations = res.max_invocations,
                    .max_smem_size = res.max_smem_size,
                    .target_wg_size = 256u,
                    .validate = validate,
                    .sampler_name = sampler_name,
                };

                fprintf(stdout, "[%zu/%zu] %s\n", total, exp_count, exp->name);

                bench_result * results = NULL;
                run_benchmark(
                    config,
                    res.pipeline_layout,
                    &res.buffers,
                    res.binding,
                    segments_len, segments,
                    keys_len, keys,
                    res.query,
                    res.query_buffer,
                    &results,
                    validate ? &validations[e] : NULL,
                    expected_keys,
                    expected_value_indices,
                    res.context.instance,
                    res.context.device,
                    res.context.queue
                );

                free(results);
            }

            free(expected_keys);
            free(expected_value_indices);
        }

        bench_context_res_release(&res);
    }

    if (validate)
    {
        write_validation_csv(experiments_param->value, exps, validations, exp_count);
        free(validations);
    }

    free(keys);
    free(segments);
    free(exps);

    fprintf(stdout, "\ndone: %zu experiments\n", total);
    return 0;
}
