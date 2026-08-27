#ifndef MERGE_SORT_CPU_H
#define MERGE_SORT_CPU_H

#include "common.h"

MERGE_EXPORT void cpu_hello(void);

#endif

#ifdef MERGE_SORT_CPU_IMPLEMENTATION
#ifndef MERGE_SORT_CPU_IMPLEMENTED
#define MERGE_SORT_CPU_IMPLEMENTED

#include <stdio.h>

MERGE_EXPORT void cpu_hello(void)
{
    printf("Hello, world!\n");
}

#endif
#endif
