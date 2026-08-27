#ifndef MERGE_SORT_CPU_H
#define MERGE_SORT_CPU_H

#include <stdint.h>
#include <stdlib.h>

#include "common.h"

MERGE_EXPORT void msc_sort(size_t len, uint32_t * arr);

#endif

#ifdef MERGE_SORT_CPU_IMPLEMENTATION
#ifndef MERGE_SORT_CPU_IMPLEMENTED
#define MERGE_SORT_CPU_IMPLEMENTED

static void msc__sort_r(
    const size_t len,
    const size_t lo,
    const size_t hi,
    uint32_t * const arr
)
{
    if (lo >= hi) return;
    const size_t mid = lo + (lo + hi) / 2;
}

MERGE_EXPORT void msc_sort(const size_t len, uint32_t * const arr)
{
    msc__sort_r(len, 0, len, arr);
}

#endif
#endif
