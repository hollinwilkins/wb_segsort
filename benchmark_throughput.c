#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define HWGUTIL_MEMS_ENABLED
#define HWGUTIL_WEBGPU_ENABLED
#define MEMS_IMPLEMENTATION
#define HWGUTIL_IMPLEMENTATION

#include <webgpu/webgpu.h>
#include "hw_gutil.h"

#define WG 256u
#define WB_MAX_WORKGROUP_DIMENSION 65535u
#define QUERY_COUNT 2u
#define SHADER_PATH "shaders/throughput.wgsl"

typedef struct gpu_params {
    uint32_t n;
    uint32_t delta;
} gpu_params;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000u + (uint64_t)ts.tv_nsec;
}

static void work_done_cb(WGPUQueueWorkDoneStatus status, WGPUStringView message,
                         void *u1, void *u2) {
    (void)status; (void)message; (void)u1; (void)u2;
}

static void wait_idle(WGPUInstance instance, WGPUQueue queue) {
    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_WaitAnyOnly;
    cb.callback = work_done_cb;
    WGPUFuture f = wgpuQueueOnSubmittedWorkDone(queue, cb);
    WGPUFutureWaitInfo wait = WGPU_FUTURE_WAIT_INFO_INIT;
    wait.future = f;
    wgpuInstanceWaitAny(instance, 1, &wait, (uint64_t)30 * 1000000000);
}

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s (run from repo root)\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = (char *)malloc(size + 1);
    *len = fread(buf, 1, size, f);
    buf[*len] = 0;
    fclose(f);
    return buf;
}

// x capped at 65535, overflow into y (matches linear_id() in the shader).
static void dispatch_dims(uint32_t groups, uint32_t *x, uint32_t *y) {
    *x = groups; *y = 1u;
    if (groups > WB_MAX_WORKGROUP_DIMENSION) {
        *y = (groups + WB_MAX_WORKGROUP_DIMENSION - 1u) / WB_MAX_WORKGROUP_DIMENSION;
        *x = WB_MAX_WORKGROUP_DIMENSION;
    }
}

static WGPUComputePipeline make_pipeline(WGPUDevice dev, WGPUPipelineLayout layout,
                                         WGPUShaderModule mod, const char *entry) {
    WGPUComputePipelineDescriptor d = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    d.layout = layout;
    d.compute.module = mod;
    d.compute.entryPoint = (WGPUStringView){ .data = entry, .length = WGPU_STRLEN };
    return wgpuDeviceCreateComputePipeline(dev, &d);
}

static uint64_t bench(hwgutil_wgpu_context *ctx, WGPUComputePipeline pipe,
                      WGPUBindGroup bg, WGPUQuerySet qs, WGPUBuffer resolve_buf,
                      uint32_t gx, uint32_t gy, size_t iters, size_t warmup) {
    if (warmup > 0) {
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx->device, NULL);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
        wgpuComputePassEncoderSetPipeline(pass, pipe);
        wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
        for (size_t i = 0; i < warmup; i++)
            wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
        wgpuComputePassEncoderEnd(pass);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
        wgpuQueueSubmit(ctx->queue, 1, &cmd);
        wait_idle(ctx->instance, ctx->queue);
        wgpuCommandBufferRelease(cmd);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandEncoderRelease(enc);
    }

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx->device, NULL);

    WGPUComputePassDescriptor pass_desc = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    WGPUPassTimestampWrites tw = (WGPUPassTimestampWrites){
        .querySet = qs,
        .beginningOfPassWriteIndex = 0,
        .endOfPassWriteIndex = 1,
    };
    pass_desc.timestampWrites = &tw;

    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, &pass_desc);
    wgpuComputePassEncoderSetPipeline(pass, pipe);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    for (size_t i = 0; i < iters; i++)
        wgpuComputePassEncoderDispatchWorkgroups(pass, gx, gy, 1);
    wgpuComputePassEncoderEnd(pass);

    wgpuCommandEncoderResolveQuerySet(enc, qs, 0, QUERY_COUNT, resolve_buf, 0);

    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(ctx->queue, 1, &cmd);
    wait_idle(ctx->instance, ctx->queue);
    wgpuCommandBufferRelease(cmd);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(enc);

    uint64_t *ts;
    hwgutil_wgpu_read_buffer_alloc(ctx->instance, ctx->device, ctx->queue,
                                   resolve_buf, &mems_system_allocator, (void **)&ts);
    uint64_t gpu_ns = ts[1] - ts[0];
    mems_allocator_free(&mems_system_allocator, ts);
    return gpu_ns;
}

int main(int argc, char **argv) {
    size_t n = argc > 1 ? strtoull(argv[1], NULL, 10) : 268435456u;
    size_t iters = argc > 2 ? strtoull(argv[2], NULL, 10) : 50u;
    size_t warmup = argc > 3 ? strtoull(argv[3], NULL, 10) : 5u;

    printf("n = %zu u32 (%.2f GiB buffer), iters = %zu, warmup = %zu\n",
           n, (double)(n * 4) / (1024.0 * 1024.0 * 1024.0), iters, warmup);

    WGPULimits limits = WGPU_LIMITS_INIT;
    limits.maxStorageBufferBindingSize = (uint64_t)2 * 1024 * 1024 * 1024;
    limits.maxBufferSize = (uint64_t)2 * 1024 * 1024 * 1024;

    hwgutil_wgpu_context ctx;
    if (!hwgutil_wgpu_context_init(
            &ctx,
            1, (WGPUInstanceFeatureName[]){ WGPUInstanceFeatureName_TimedWaitAny },
            1, (WGPUFeatureName[]){ WGPUFeatureName_TimestampQuery },
            &limits)) {
        fprintf(stderr, "context init failed\n");
        return 1;
    }

    WGPUBuffer buf = wgpuDeviceCreateBuffer(ctx.device, &(WGPUBufferDescriptor){
        .size = (uint64_t)n * sizeof(uint32_t),
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
    });

    WGPUBuffer vals = wgpuDeviceCreateBuffer(ctx.device, &(WGPUBufferDescriptor){
        .size = (uint64_t)n * sizeof(uint32_t),
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst,
    });

    gpu_params params = { .n = (uint32_t)n, .delta = 0u };
    WGPUBuffer params_buf = wgpuDeviceCreateBuffer(ctx.device, &(WGPUBufferDescriptor){
        .size = sizeof(gpu_params),
        .usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst,
    });
    wgpuQueueWriteBuffer(ctx.queue, params_buf, 0, &params, sizeof(params));

    size_t src_len;
    char *src = read_file(SHADER_PATH, &src_len);
    WGPUShaderSourceWGSL wgsl = (WGPUShaderSourceWGSL){
        .chain = (WGPUChainedStruct){ .sType = WGPUSType_ShaderSourceWGSL },
        .code = (WGPUStringView){ .data = src, .length = src_len },
    };
    WGPUShaderModuleDescriptor mod_desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    mod_desc.nextInChain = &wgsl.chain;
    WGPUShaderModule mod = wgpuDeviceCreateShaderModule(ctx.device, &mod_desc);
    free(src);

    WGPUBindGroupLayoutEntry bgl_entries[3] = {
        (WGPUBindGroupLayoutEntry){ .binding = 0, .visibility = WGPUShaderStage_Compute,
            .buffer = { .type = WGPUBufferBindingType_Storage } },
        (WGPUBindGroupLayoutEntry){ .binding = 1, .visibility = WGPUShaderStage_Compute,
            .buffer = { .type = WGPUBufferBindingType_Uniform } },
        (WGPUBindGroupLayoutEntry){ .binding = 2, .visibility = WGPUShaderStage_Compute,
            .buffer = { .type = WGPUBufferBindingType_Storage } },
    };
    WGPUBindGroupLayout bgl = wgpuDeviceCreateBindGroupLayout(ctx.device,
        &(WGPUBindGroupLayoutDescriptor){ .entryCount = 3, .entries = bgl_entries });
    WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(ctx.device,
        &(WGPUPipelineLayoutDescriptor){ .bindGroupLayoutCount = 1, .bindGroupLayouts = &bgl });

    WGPUComputePipeline pipe_tp = make_pipeline(ctx.device, layout, mod, "main_throughput");
    WGPUComputePipeline pipe_add = make_pipeline(ctx.device, layout, mod, "main_throughput_add");
    WGPUComputePipeline pipe_sort2 = make_pipeline(ctx.device, layout, mod, "main_sort2");
    WGPUComputePipeline pipe_sort2v = make_pipeline(ctx.device, layout, mod, "main_sort2_vals");

    WGPUBindGroupEntry bg_entries[3] = {
        (WGPUBindGroupEntry){ .binding = 0, .buffer = buf, .offset = 0, .size = (uint64_t)n * sizeof(uint32_t) },
        (WGPUBindGroupEntry){ .binding = 1, .buffer = params_buf, .offset = 0, .size = sizeof(gpu_params) },
        (WGPUBindGroupEntry){ .binding = 2, .buffer = vals, .offset = 0, .size = (uint64_t)n * sizeof(uint32_t) },
    };
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(ctx.device,
        &(WGPUBindGroupDescriptor){ .layout = bgl, .entryCount = 3, .entries = bg_entries });

    WGPUQuerySet qs = wgpuDeviceCreateQuerySet(ctx.device, &(WGPUQuerySetDescriptor){
        .type = WGPUQueryType_Timestamp, .count = QUERY_COUNT });
    WGPUBuffer resolve_buf = wgpuDeviceCreateBuffer(ctx.device, &(WGPUBufferDescriptor){
        .size = QUERY_COUNT * sizeof(uint64_t),
        .usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
    });

    uint32_t groups = (uint32_t)((n + WG - 1) / WG);
    uint32_t gx, gy;
    dispatch_dims(groups, &gx, &gy);

    const double bytes_per_iter = (double)n * 8.0; // 4 byte read/write per key
    const double bytes_per_iter_val = (double)n * 12.0; // 4 byte read/write per key + 4 byte value index write

    uint64_t tp_ns = bench(&ctx, pipe_tp, bg, qs, resolve_buf, gx, gy, iters, warmup);
    uint64_t add_ns = bench(&ctx, pipe_add, bg, qs, resolve_buf, gx, gy, iters, warmup);
    uint64_t pipe_sort2_ns = bench(&ctx, pipe_sort2, bg, qs, resolve_buf, gx, gy, iters, warmup);
    uint64_t pipe_sort2v_ns = bench(&ctx, pipe_sort2v, bg, qs, resolve_buf, gx, gy, iters, warmup);

    double tp_gbps = bytes_per_iter * iters / ((double)tp_ns);
    double add_gbps = bytes_per_iter * iters / ((double)add_ns);
    double sort2_gbps = bytes_per_iter * iters / ((double)pipe_sort2_ns);
    double sort2v_gbps = bytes_per_iter_val * iters / ((double)pipe_sort2v_ns);

    printf("\n%-22s %12s %12s %12s\n", "kernel", "gpu_ms", "per-op ms", "GB/s (8B/e)");
    printf("%-22s %12.3f %12.4f %12.1f\n", "main_throughput",
           tp_ns / 1e6, tp_ns / 1e6 / iters, tp_gbps);
    printf("%-22s %12.3f %12.4f %12.1f\n", "main_throughput_add",
           add_ns / 1e6, add_ns / 1e6 / iters, add_gbps);
    printf("%-22s %12.3f %12.4f %12.1f\n", "main_sort2",
           pipe_sort2_ns / 1e6, pipe_sort2_ns / 1e6 / iters, sort2_gbps);
    printf("%-22s %12.3f %12.4f %12.1f\n", "main_sort2v",
           pipe_sort2v_ns / 1e6, pipe_sort2v_ns / 1e6 / iters, sort2v_gbps);

    return 0;
}
