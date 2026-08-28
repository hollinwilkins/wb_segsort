#include <stdlib.h>
#include <string.h>

#include "unity.h"
#include "unity_internals.h"

#define HWGUTIL_WEBGPU_ENABLED

#define MERGE_SORT_GPU_IMPLEMENTATION
#define HWGUTIL_IMPLEMENTATION

#include "gpu.h"
#include "hw_gutil.h"

hwgutil_wgpu_context context;
msg_pipeline pipeline;
msg_buffers buffers;
msg_bindings bindings;

void setUp(void)
{
}

void tearDown(void)
{
}

void test_bin_histogram(void)
{
    TEST_ASSERT_TRUE(true);
}

int main(void)
{
    if (!hwgutil_wgpu_context_init(&context)) abort();

    if (!msg_pipeline_init(
        &pipeline,
        context.instance,
        context.adapter,
        context.device, 
        context.queue
    )) abort();

    msg_buffers_init(
        &buffers,
        NULL,
        context.device
    );

    msg_bindings_init(
        &bindings,
        &pipeline,
        &buffers
    );

    UNITY_BEGIN();
    RUN_TEST(test_bin_histogram);
    return UNITY_END();
}
