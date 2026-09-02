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
    bench_memory_hybrid_merge = 3,
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
    bench_memory_kind memory;
    bench_store_kind store;
    uint32_t N;
    uint32_t M;
    uint32_t R;
    uint32_t smem_bytes;
    uint32_t n_keys;
    const char * sampler_name;
} bench_config;

static const bench_smem BENCH_SMEMS[2] = {
    { "16kb", 16 * 1024 },
    { "32kb", 32 * 1024 },
};

static char ARGS_BUFFER[1024 * 4];

int main(const int argc, const char ** const argv)
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

    if (args.positionals_len < 2)
    {
        PANIC("must provide <output dir> as first poositional argument");
    }

    const char * const output_dir = args.positionals[1];
    const hwargs_param * const memory_param = hwargs_get_param(&args, "memory");
    const hwargs_param * const store_param = hwargs_get_param(&args, "store");
    const hwargs_param * const keys_param = hwargs_get_param(&args, "keys");
    const hwargs_param * const N_param = hwargs_get_param(&args, "N");
    const hwargs_param * const M_param = hwargs_get_param(&args, "M");
    const hwargs_param * const R_param = hwargs_get_param(&args, "R");
    const hwargs_param * const smem_param = hwargs_get_param(&args, "smem");
    const hwargs_param * const sampler_name_param = hwargs_get_param(&args, "sampler");

    printf("Hello, world!\n");
    return 0;
}
