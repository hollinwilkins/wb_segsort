#ifndef WB_SORT_CPU_H
#define WB_SORT_CPU_H

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include "common.h"

WB_EXPORT void wbc_sort_alloc(size_t len, uint32_t * arr, uint32_t * const value_indices);

WB_EXPORT bool wbc_segsort(
    size_t len,
    uint32_t * arr,
    uint32_t * value_indices,
    size_t segs_len,
    const uint32_t * segs,
    size_t buffer_len,
    void * buffer,
    size_t * required_size
);

WB_EXPORT void wbc_segsort_alloc(
    size_t len,
    uint32_t * arr,
    uint32_t * value_indices,
    size_t segs_len,
    const uint32_t * segs
);

#endif

#ifdef WB_SORT_CPU_IMPLEMENTATION
#ifndef WB_SORT_CPU_IMPLEMENTED
#define WB_SORT_CPU_IMPLEMENTED

static void wbc__merge(
    const size_t lo,
    const size_t mid,
    const size_t hi,
    uint32_t * const dst,
    uint32_t * const src,
    uint32_t * const dst_value_indices,
    uint32_t * const src_value_indices
)
{
    size_t a = lo;
    size_t b = mid + 1;

    for (size_t i = lo; i <= hi; i++)
    {
        if (a > mid)
        {
            dst[i] = src[b];
            dst_value_indices[i] = src_value_indices[b++];
        }
        else if (b > hi)
        {
            dst[i] = src[a];
            dst_value_indices[i] = src_value_indices[a++];
        }
        else if (src[a] < src[b])
        {
            dst[i] = src[a];
            dst_value_indices[i] = src_value_indices[a++];
        }
        else
        {
            dst[i] = src[b];
            dst_value_indices[i] = src_value_indices[b++];
        }
    }
}

static void wbc__sort_r(
    const size_t lo,
    const size_t hi,
    uint32_t * const dst,
    uint32_t * const src,
    uint32_t * const dst_value_indices,
    uint32_t * const src_value_indices
)
{
    if (lo >= hi) return;
    const size_t mid = lo + (hi - lo) / 2;

    wbc__sort_r(lo, mid , src, dst, src_value_indices, dst_value_indices);
    wbc__sort_r(mid + 1, hi, src, dst, src_value_indices, dst_value_indices);

    wbc__merge(lo, mid, hi, dst, src, dst_value_indices, src_value_indices);
}

WB_EXPORT void wbc_sort_alloc(const size_t len, uint32_t * const arr, uint32_t * const value_indices)
{
    if (len < 2) return;

    uint32_t * const swap = (uint32_t *)malloc(len * sizeof(uint32_t) * 2);
    memcpy(swap, arr, len * sizeof(uint32_t));

    for (uint32_t i = 0; i < (uint32_t)len; i++)
    {
        value_indices[i] = i;
    }
    memcpy(swap + len, value_indices, len * sizeof(uint32_t));

    wbc__sort_r(0, len - 1, arr, swap, value_indices, swap + len);

    free(swap);
}

WB_EXPORT bool wbc_segsort(
    const size_t len,
    uint32_t * const arr,
    uint32_t * const value_indices,
    const size_t segs_len,
    const uint32_t * const segs,
    const size_t buffer_len,
    void * const buffer,
    size_t * const required_size
)
{
    if (len < 2) return true;

    *required_size = len * sizeof(uint32_t) * 2;

    if (buffer_len < *required_size) return false;

    uint32_t * const swap = (uint32_t *)buffer;
    memcpy(swap, arr, len * sizeof(uint32_t));

    for (uint32_t i = 0; i < (uint32_t)len; i++)
    {
        value_indices[i] = i;
    }

    memcpy(swap + len, value_indices, len * sizeof(uint32_t));

    size_t seg_start = 0;
    for (size_t i = 0; i < segs_len; i++)
    {
        const size_t seg_end = segs[i];
        if (seg_end > seg_start)
        {
            wbc__sort_r(
                seg_start,
                seg_end - 1,
                arr,
                swap,
                value_indices,
                swap + len
            );
        }
        seg_start = seg_end;
    }

    return true;
}

WB_EXPORT void wbc_segsort_alloc(
    const size_t len,
    uint32_t * const arr,
    uint32_t * value_indices,
    const size_t segs_len,
    const uint32_t * const segs
)
{
    size_t required_size;
    if (!wbc_segsort(
        len,
        arr,
        value_indices,
        segs_len,
        segs,
        0,
        NULL,
        &required_size
    ))
    {
        void * const buffer = malloc(required_size);

        if (!wbc_segsort(
            len,
            arr,
            value_indices,
            segs_len,
            segs,
            required_size,
            buffer,
            &required_size
        )) abort();

        free(buffer);
    }
}

#endif
#endif
