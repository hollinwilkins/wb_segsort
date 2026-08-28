#include <stdlib.h>

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
    msc_sort_alloc(0, NULL);
}

void test_sort1()
{
    uint32_t arr[1] = { 42 };
    TEST_ASSERT_EQUAL(42, arr[0]);
    msc_sort_alloc(1, arr);

    TEST_ASSERT_EQUAL(42, arr[0]);
}

void test_sort2()
{
    uint32_t arr[2] = { 42, 13 };
    msc_sort_alloc(2, arr);

    TEST_ASSERT_EQUAL_UINT32_ARRAY(
        ((uint32_t[]){ 13, 42 }),
        arr,
        2
    );
}

void test_sort3()
{
    uint32_t arr[3] = { 7, 42, 13 };
    msc_sort_alloc(3, arr);

    TEST_ASSERT_EQUAL_UINT32_ARRAY(
        ((uint32_t[]){ 7, 13, 42 }),
        arr,
        3
    );
}

void test_sort4()
{
    uint32_t arr[4] = { 42, 13, 7, 2 };
    msc_sort_alloc(4, arr);

    TEST_ASSERT_EQUAL_UINT32_ARRAY(
        ((uint32_t[]){ 2, 7, 13, 42 }),
        arr,
        4
    );
}

static int qsort_compare_uint32(const void * const a, const void * const b)
{
    const uint32_t av = *((const uint32_t *)a);
    const uint32_t bv = *((const uint32_t *)b);

    if (av < bv) return -1;
    else if (av > bv) return 1;
    return 0;
}

void test_sort1024()
{
    srand(897312);

    uint32_t * const arr = (uint32_t *)malloc(1024 * sizeof(uint32_t));
    uint32_t * const expected = (uint32_t *)malloc(1024 * sizeof(uint32_t));

    for (int i = 0; i < 1024; i++)
    {
        arr[i] = rand() % 4096;
        expected[i] = arr[i];
    }

    qsort(expected, 1024, sizeof(uint32_t), qsort_compare_uint32);

    msc_sort_alloc(1024, arr);

    TEST_ASSERT_EQUAL_UINT32_ARRAY(
        expected,
        arr,
        1024
    );

    free(arr);
    free(expected);
}

void test_sort4033()
{
    srand(2213412);

    uint32_t * const arr = (uint32_t *)malloc(4033 * sizeof(uint32_t));
    uint32_t * const expected = (uint32_t *)malloc(4033 * sizeof(uint32_t));

    for (int i = 0; i < 4033; i++)
    {
        arr[i] = rand() % 4096;
        expected[i] = arr[i];
    }

    qsort(expected, 4033, sizeof(uint32_t), qsort_compare_uint32);

    msc_sort_alloc(4033, arr);

    TEST_ASSERT_EQUAL_UINT32_ARRAY(
        expected,
        arr,
        4033
    );

    free(arr);
    free(expected);
}

int main()
{
    UNITY_BEGIN();
    RUN_TEST(test_sort0);
    RUN_TEST(test_sort1);
    RUN_TEST(test_sort2);
    RUN_TEST(test_sort3);
    RUN_TEST(test_sort4);
    RUN_TEST(test_sort1024);
    RUN_TEST(test_sort4033);
    return UNITY_END();
}
