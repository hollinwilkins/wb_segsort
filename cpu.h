#ifndef MERGE_SORT_CPU_H
#define MERGE_SORT_CPU_H

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "common.h"

MERGE_EXPORT void msc_sort_alloc(size_t len, uint32_t * arr);

MERGE_EXPORT void msc_segsort_alloc(
    size_t len,
    uint32_t * arr,
    size_t segs_len,
    uint32_t * segs
);

#endif

#ifdef MERGE_SORT_CPU_IMPLEMENTATION
#ifndef MERGE_SORT_CPU_IMPLEMENTED
#define MERGE_SORT_CPU_IMPLEMENTED

static void msc__merge(
    const size_t lo,
    const size_t mid,
    const size_t hi,
    uint32_t * const dst,
    uint32_t * const src
)
{
    size_t a = lo;
    size_t b = mid + 1;

    for (size_t i = lo; i <= hi; i++)
    {
        if (a > mid) dst[i] = src[b++];
        else if (b > hi) dst[i] = src[a++];
        else if (src[a] < src[b]) dst[i] = src[a++];
        else dst[i] = src[b++];
    }
}

static void msc__sort_r(
    const size_t lo,
    const size_t hi,
    uint32_t * const dst,
    uint32_t * const src
)
{
    if (lo >= hi) return;
    const size_t mid = lo + (hi - lo) / 2;

    msc__sort_r(lo, mid , src, dst);
    msc__sort_r(mid + 1, hi, src, dst);

    msc__merge(lo, mid, hi, dst, src);
}

MERGE_EXPORT void msc_sort_alloc(const size_t len, uint32_t * const arr)
{
    if (len < 2) return;

    uint32_t * const swap = (uint32_t *)malloc(len * sizeof(uint32_t));
    memcpy(swap, arr, len * sizeof(uint32_t));
    msc__sort_r(0, len - 1, arr, swap);

    free(swap);
}

MERGE_EXPORT void msc_segsort_alloc(
    const size_t len,
    uint32_t * const arr,
    const size_t segs_len,
    uint32_t * const segs
)
{
    if (len < 2) return;

    uint32_t * const swap = (uint32_t *)malloc(len * sizeof(uint32_t));
    memcpy(swap, arr, len * sizeof(uint32_t));

    size_t seg_start = 0, seg_end = 0;
    for (size_t i = 0; i < segs_len; i++)
    {
        seg_end = segs[i];
        msc__sort_r(seg_start, seg_end, arr, swap);
        seg_start = seg_end + 1;
    }

    free(swap);
}

#endif
#endif
