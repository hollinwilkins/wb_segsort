#include "unity.h"
#include "unity_internals.h"

#define MERGE_SORT_CPU_IMPLEMENTATION

#include "cpu.h"

void setUp(void)
{
}

void tearDown(void)
{
}

void test_sort0()
{
    msc_sort(0, NULL);
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_sort0);
    return UNITY_END();
}
