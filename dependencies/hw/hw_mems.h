// Copyright (C) 2026 Hollin Wilkins
// SPDX-License-Identifier: Zlib

#ifndef MEMS_H
#define MEMS_H

/*

hw_mems.h v0.1.0 - memory allocator abstraction and implementations
by Hollin Wilkins

SECURITY
    This library provides NO SECURITY GUARANTEES whatsoever.

WHAT DOES THIS LIBRARY DO?
    There are many strategies to allocate memory depending on the needs of an application
    or library. These include system memory allocation ala malloc, bump allocators on
    fixed buffers, arena allocation for memory tracking, memory pools for same-sized objects,
    and other more specialized memory allocation strategies. Some data structures do not care which
    allocation strategy is used to allocate memory as long as it meets the appropriate sizing requirements.
    In this case, it is useful to have an generic abstraction for a memory allocator. In addition to
    allocation memory buffers, data is also highly-dependent on its byte alignment in memory.

    This library implements specific allocation strategies and a generic allocator abstraction.
    It keeps in mind memory alignment requirements by requiring alignment with any function that allocates
    memory.

FEATURE OVERVIEW
    This library provides
    - A bump allocator implementation ala mems_bump
    - A system allocator (malloc by default) ala mems_system
    - A generic allocator abstraction ala mems_allocator

COMPILING & LINKING
    This library requires C99, though it may be refactored to support C89 in the future.

    In one C/C++ file that includes this file, do this:
        #define MEMS_IMPLEMENTATION
    before the include, that will create the implementation in that file.

    If you also do this:
        #define MEMS_STATIC
    then all of the functions will be declared as static.

    If you do this:
        #define MEMS_NO_CRT
    then we do not use the C standard library.
    In that case, if you want the system allocator implementation,
    you will need to provide your own defines for:
        MEMS_ALLOC
            defaults to malloc otherwise
        MEMS_REALLOC
            defaults to realloc otherwise
        MEMS_FREE
            defaults to free otherwise
    In other words, if you define MEMS_NO_CRT, and do not define the above 3 macros,
    you will not get the system allocator implementation.

EXAMPLES
    Bump allocation
        void * const buffer = malloc(1024);
        mems_bump bump;
        mems_bump_init(&bump, 1024, buffer);

        // allocate a buffer
        //  - alignment = use MEMS_ALIGN_DEFAULT for default alignment
        //  - size = 32 will allocate 32 bytes
        // NOTE: the total bytes used in the bump buffer will be more than 32 bytes
        //  because of header data used to track allocations
        void * const buf1 = mems_bump_alloc(&bump, MEMS_ALIGN_DEFAULT, 32);
        MEMS_ASSERT(buf1 != MEMS_NULL);

        // allocate another buffer
        void * const buf2 = mems_bump_alloc(&bump, MEMS_ALIGN_DEFAULT, 32);
        MEMS_ASSERT(buf2 != MEMS_NULL);

        // allocate a buffer that is too large
        // this will return NULL
        void * const buf3 = mems_bump_alloc(&bump, MEMS_ALIGN_DEFAULT, 1024);
        MEMS_ASSERT(buf3 == MEMS_NULL);

        // reallocate the first buffer to a larger size fails
        MEMS_ASSERT(mems_bump_realloc(&bump, buf1, MEMS_ALIGN_DEFAULT, 64) == MEMS_NULL);

        // reallocate the first buffer to a smaller size succeeds
        MEMS_ASSERT(mems_bump_realloc(&bump, buf1, MEMS_ALIGN_DEFAULT, 16) != MEMS_NULL);

        // reallocate the last buffer to a larger size succeeds
        MEMS_ASSERT(mems_bump_realloc(&bump, buf2, MEMS_ALIGN_DEFAULT, 64) != MEMS_NULL);

        // reallocate the last buffer to a smaller size succeeds
        MEMS_ASSERT(mems_bump_realloc(&bump, buf2, MEMS_ALIGN_DEFAULT, 16) != MEMS_NULL);

        // free the last buffer
        // this will free all memory that was previously
        // used by the buffer
        // memory is only freed when the last allocation is
        // passed to mems_bump_free
        const mems_size remaining_before = mems_bump_remaining(&bump, 1);
        mems_bump_free(&bump, buf2);
        MEMS_ASSERT(mems_bump_remaining(&bump, 1) > remaining_before);
    System allocation
        void * buf1 = mems_system_alloc(MEMS_ALIGN_DEFAULT, 32);
        MEMS_ASSERT(buf1 != MEMS_NULL);
        buf1 = mems_system_realloc(buf1, MEMS_ALIGN_DEFAULT, 64);
        MEMS_ASSERT(buf1 != MEMS_NULL);
        mems_system_free(buf1);
    Generic allocator
        void * const buffer = malloc(1024);
        mems_bump bump;
        mems_bump_init(&bump, 1024, buffer);
        mems_allocator allocator;
        mems_bump_allocator_init(&bump, &allocator);

        void * buf1 = mems_allocator_alloc(&allocator, MEMS_ALIGN_DEFAULT, 32);
        MEMS_ASSERT(buf1 != MEMS_NULL);
        buf1 = mems_allocator_realloc(&allocator, buf1, MEMS_ALIGN_DEFAULT, 64);
        MEMS_ASSERT(buf1 != MEMS_NULL);
        mems_allocator_free(&allocator, buf1);
*/

#ifndef mems_size
#   if defined(__SIZE_TYPE__)
#       define mems_size __SIZE_TYPE__
#   elif defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || \
    defined(__LP64__) || defined(_LP64) || (defined(__WORDSIZE) && __WORDSIZE == 64)
#       define mems_size unsigned long long
#   elif defined(__i386__) || defined(_M_IX86) || defined(__arm__) || \
    defined(__ILP32__) || (defined(__WORDSIZE) && __WORDSIZE == 32)
#       define mems_size unsigned int
#   else
#       define mems_size unsigned long
#   endif
#endif

#ifndef mems_ptr
#   if defined(__UINTPTR_TYPE__)
#       define mems_ptr __UINTPTR_TYPE__
#   elif defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || \
    defined(__LP64__) || defined(_LP64) || defined(_WIND64)
#       define mems_ptr unsigned long long
#   elif defined(__i386__) || defined(_M_IX86) || defined(__arm__) || \
    defined(__ILP32__) || defined(_WIN32)
#       define mems_ptr unsigned int
#   else
#       define mems_ptr unsigned long
#   endif
#endif

#ifndef mems_u8
#   define mems_u8 unsigned char
#endif
#ifndef mems_u32
#   define mems_u32 unsigned
#endif
#ifndef mems_bool
#   define mems_bool int
#endif
#ifndef mems_true
#   define mems_true 1
#endif
#ifndef mems_false
#   define mems_false 0
#endif
#ifndef MEMS_NULL
#   define MEMS_NULL 0
#endif

#ifndef MEMS_EXPORT
#   ifdef MEMS_STATIC
#       define MEMS_EXPORT static
#   else
#       ifdef __cplusplus
#           define MEMS_EXPORT extern "C"
#       else
#           define MEMS_EXPORT extern
#       endif
#   endif
#endif

#ifdef _MSC_VER
#   define MEMS_INLINE static __forceinline
#   define MEMS_NOINLINE static __declspec(noinline)
#   define MEMS_ALIGNOF __alignof
#else
#   ifdef __has_attribute
#       if __has_attribute(always_inline)
#           define MEMS_INLINE static inline __attribute__((always_inline))
#       endif
#       if __has_attribute(noinline)
#           define MEMS_NOINLINE static __attribute__((noinline))
#       endif
#   endif
#   define MEMS_ALIGNOF __alignof__
#endif

#ifndef MEMS_INLINE
#   define MEMS_INLINE static inline
#endif

#ifndef MEMS_USE
#   define MEMS_USE(x) (void)sizeof(x)
#endif

#ifndef MEMS_NO_CRT
#   ifndef MEMS_ABORT
#       include <stdlib.h>
#       define MEMS_ABORT() abort()
#   endif
#   ifndef MEMS_MEMMOVE
#       include <string.h>
#       define MEMS_MEMMOVE(dst, src, size) memmove(dst, src, size)
#   endif
#   ifndef MEMS_MALLOC
#       include <stdlib.h>
#       define MEMS_MALLOC(alignment, size) malloc(size)
#   endif
#   ifndef MEMS_REALLOC
#       include <stdlib.h>
#       define MEMS_REALLOC(ptr, size) realloc(ptr, size)
#   endif
#   ifndef MEMS_FREE
#       define MEMS_FREE(ptr) free(ptr)
#   endif
#   ifndef MEMS_ASSERT
#       ifndef MEMS_NO_ASSERT
#           include <assert.h>
#           define MEMS_ASSERT(conditon) assert(conditon)
#       else
#           define MEMS_ASSERT(conditon)
#       endif
#   endif
#else
#   ifndef MEMS_MEMMOVE
MEMS_INLINE void * mems__memmove(void * const dst, const void * const src, const mems_size size)
{
    unsigned char* d = (unsigned char*)dst;
    const unsigned char* s = (const unsigned char*)src;

    mems_ptr d_addr = (uintptr_t)d;
    mems_ptr s_addr = (uintptr_t)s;

    if (d_addr < s_addr || d_addr >= (s_addr + size)) {
        for (mems_size i = 0; i < size; i++) d[i] = s[i];
    } else if (d_addr > s_addr) {
        for (mems_size i = size; i > 0; i--) d[i - 1] = s[i - 1];
    }

    return dst;
}
#   define MEMS_MEMMOVE(dst, src, size) mems__memmove(dst, src, size)
#   endif
#   ifndef MEMS_ASSERT
#       define MEMS_ASSERT(condition) ((void)condition)
#   endif
#endif

#ifndef MEMS_NO_SYSTEM
#   define MEMS_SYSTEM
#endif

#if !defined(MEMS_MALLOC) || !defined(MEMS_REALLOC) || !defined(MEMS_FREE)
#   undef MEMS_SYSTEM
#endif

#ifndef MEMS_ALIGN_DEFAULT
#   define MEMS_ALIGN_DEFAULT 16
#endif

#ifndef MEMS_ALIGN_FORWARD
#   define MEMS_ALIGN_FORWARD(address, alignment) ((address + (alignment - 1)) & ~(alignment - 1))
#endif

#ifndef MEMS_ALIGN_BACKWARD
#   define MEMS_ALIGN_BACKWARD(address, alignment) (address & ~(alignment - 1))
#endif

#ifndef MEMS_IS_ALIGNED
#   define MEMS_IS_ALIGNED(address, alignment)  (((address) & ((alignment) - 1)) == 0)
#endif

#ifndef MEMS_ALIGN_IS_VALID
#   define MEMS_ALIGN_IS_VALID(x) ((x) != 0 && ((x) & ((x) - 1)) == 0)
#endif

#ifndef MEMS_MIN
#   define MEMS_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MEMS_MAX
#   define MEMS_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mems_allocator mems_allocator;

typedef void * (*mems_allocator_alloc_cb)(
    void * context,
    mems_size alignment,
    mems_size size
);
typedef void * (*mems_allocator_realloc_cb)(
    void * context,
    void * buffer,
    mems_size alignment,
    mems_size size
);
typedef void (*mems_allocator_free_cb)(
    void * context,
    void * buffer
);

typedef struct mems_allocator_vtable
{
    mems_allocator_alloc_cb alloc;
    mems_allocator_realloc_cb realloc;
    mems_allocator_free_cb free;
} mems_allocator_vtable;

struct mems_allocator
{
    void * context;
    const mems_allocator_vtable * vtable;
};

typedef struct mems_bump
{
    mems_size capacity;
    mems_size len;
    void * buffer;
    void * last_allocation;
} mems_bump;

typedef struct mems_pool_chunk mems_pool_chunk;
struct mems_pool_chunk
{
    mems_pool_chunk * next;
    // increments to end of chunk
    // then fall back to free_list on mems_pool
    // if no more objects are available
    mems_size index;
};

typedef struct mems_pool_item mems_pool_item;
struct  mems_pool_item {
    mems_pool_item * next;
};

typedef struct mems_pool
{
    mems_allocator parent;
    mems_size size;
    mems_size alignment;
    mems_size items_per_chunk;
    mems_pool_chunk * root;
    mems_pool_item * free_list;
} mems_pool;

typedef struct mems_chunks
{
    mems_size capacity;
    mems_size len;
    mems_size chunk_size;
    void ** chunks;
} mems_chunks;

#define MEMS_WHOLE_BUFFER ((mems_size)-1)

MEMS_EXPORT void * mems_allocator_alloc(const mems_allocator * allocator, mems_size alignment, mems_size size);
MEMS_EXPORT void * mems_allocator_realloc(const mems_allocator * allocator, void * buffer, mems_size alignment, mems_size size);
MEMS_EXPORT void mems_allocator_free(const mems_allocator * allocator, void * buffer);

#ifdef MEMS_SYSTEM

MEMS_EXPORT const mems_allocator mems_system_allocator;

MEMS_EXPORT void * mems_system_alloc(mems_size alignment, mems_size size);
MEMS_EXPORT void * mems_system_realloc(void * buffer, mems_size alignment, mems_size size);
MEMS_EXPORT void  mems_system_free(void * buffer);

#endif

MEMS_EXPORT void mems_bump_init(mems_bump * bump, mems_size buffer_len, void * buffer);
MEMS_EXPORT void * mems_bump_alloc(mems_bump * bump, mems_size alignment, mems_size size);
MEMS_EXPORT void * mems_bump_realloc(mems_bump * bump, void * buffer, mems_size alignment, mems_size size);
MEMS_EXPORT void mems_bump_free(mems_bump * bump, void * buffer);
MEMS_EXPORT void mems_bump_reset(mems_bump * bump);
MEMS_EXPORT void * mems_bump_rewind(mems_bump * bump, void * buffer, mems_size len);
MEMS_EXPORT mems_bool mems_bump_align(mems_bump * bump, mems_size alignment);

MEMS_EXPORT mems_size mems_bump_remaining(const mems_bump * bump, mems_size alignment);
MEMS_EXPORT void * mems_bump_remaining_buffer(mems_bump * bump, mems_size alignment, mems_size * size);
MEMS_EXPORT void mems_bump_allocator_init(mems_bump * bump, mems_allocator * allocator);

MEMS_EXPORT mems_size mems_pool_required_chunk_size(mems_size alignment, mems_size size, mems_size items_per_chunk);

MEMS_EXPORT void mems_pool_init(
    mems_pool * pool,
    const mems_allocator * parent,
    mems_size alignment,
    mems_size size,
    mems_size items_per_chunk
);

MEMS_EXPORT void mems_pool_init_buffer(
    mems_pool * pool,
    mems_size alignment,
    mems_size size,
    mems_size buffer_len,
    void * buffer
);

MEMS_EXPORT void mems_pool_deinit(mems_pool * pool);

MEMS_EXPORT mems_size mems_pool_get_free_len(const mems_pool * pool);
MEMS_EXPORT mems_size mems_pool_get_outstanding(const mems_pool * pool);

MEMS_EXPORT void * mems_pool_alloc(mems_pool * pool);
MEMS_EXPORT void mems_pool_free(mems_pool * pool, void * buffer);

MEMS_EXPORT mems_bool mems_chunks_init(
    mems_chunks * chunks,
    mems_size max_chunks,
    mems_size chunk_size,
    mems_size buffer_len,
    void * buffer,
    mems_size * required_size
);

MEMS_EXPORT mems_bool mems_chunks_init_alloc(
    mems_chunks * chunks,
    mems_size max_chunks,
    mems_size chunk_size,
    const mems_allocator * allocator
);

MEMS_EXPORT void mems_chunks_deinit_alloc(
    mems_chunks * chunks,
    const mems_allocator * allocator
);

MEMS_EXPORT mems_bool mems_chunks_append(mems_chunks * chunks, void * chunk);
MEMS_EXPORT void * mems_chunks_append_pool(mems_chunks * chunks, mems_pool * pool);
MEMS_EXPORT void * mems_chunks_pop(mems_chunks * chunks);
MEMS_EXPORT mems_bool mems_chunks_pop_pool(mems_chunks * chunks, mems_pool * pool);

#ifdef __cplusplus
}
#endif

#endif // MEMS

#ifdef MEMS_IMPLEMENTATION
#ifndef MEMS_IMPLEMENTED
#define MEMS_IMPLEMENTED

MEMS_EXPORT void * mems_allocator_alloc(
    const mems_allocator * const allocator,
    const mems_size alignment,
    const mems_size size
)
{
    MEMS_ASSERT(MEMS_ALIGN_IS_VALID(alignment));
    return allocator->vtable->alloc(allocator->context, alignment, size);
}

MEMS_EXPORT void * mems_allocator_realloc(
    const mems_allocator * const allocator,
    void * const buffer,
    const mems_size alignment,
    const mems_size size
)
{
    MEMS_ASSERT(MEMS_ALIGN_IS_VALID(alignment));
    return allocator->vtable->realloc(allocator->context, buffer, alignment, size);
}

MEMS_EXPORT void mems_allocator_free(const mems_allocator * const allocator, void * const buffer)
{
    allocator->vtable->free(allocator->context, buffer);
}

#ifdef MEMS_SYSTEM

static void * mems__system_allocator_alloc(void * const context, const mems_size alignment, const mems_size size)
{
    MEMS_ASSERT(MEMS_ALIGN_IS_VALID(alignment));
    MEMS_USE(context);
    return mems_system_alloc(alignment, size);
}

static void * mems__system_allocator_realloc(void * const context, void * const buffer, const mems_size alignment, const mems_size size)
{
    MEMS_ASSERT(MEMS_ALIGN_IS_VALID(alignment));
    MEMS_USE(context);
    return mems_system_realloc(buffer, alignment, size);
}

static void mems__system_allocator_free(void * const context, void * const buffer)
{
    MEMS_USE(context);
    mems_system_free(buffer);
}

static const mems_allocator_vtable mems__system_allocator_vtable = {
    .alloc = mems__system_allocator_alloc,
    .realloc = mems__system_allocator_realloc,
    .free = mems__system_allocator_free,
};
const mems_allocator mems_system_allocator = {
    .context = MEMS_NULL,
    .vtable = &mems__system_allocator_vtable,
};

MEMS_EXPORT void * mems_system_alloc(
    const mems_size alignment,
    const mems_size size
)
{
    MEMS_ASSERT(MEMS_ALIGN_IS_VALID(alignment));
    MEMS_ASSERT(alignment <= MEMS_ALIGN_DEFAULT);
    MEMS_USE(alignment);
    return MEMS_MALLOC(alignment, size);
}

MEMS_EXPORT void * mems_system_realloc(
    void * buffer,
    const mems_size alignment,
    const mems_size size
)
{
    MEMS_ASSERT(MEMS_ALIGN_IS_VALID(alignment));
    MEMS_ASSERT(alignment <= MEMS_ALIGN_DEFAULT);
    MEMS_USE(alignment);
    if (buffer == MEMS_NULL) return MEMS_MALLOC(alignment, size);
    return MEMS_REALLOC(buffer, size);
}

MEMS_EXPORT void  mems_system_free(
    void * buffer
)
{
    MEMS_FREE(buffer);
}
#endif // MEMS_SYSTEM

MEMS_EXPORT void mems_bump_init(mems_bump * bump, mems_size buffer_len, void * buffer)
{
    bump->capacity = buffer_len;
    bump->len = 0;
    bump->buffer = buffer;
    bump->last_allocation = MEMS_NULL;
}

MEMS_EXPORT void * mems_bump_alloc(mems_bump * const bump, const mems_size alignment, const mems_size size)
{
    MEMS_ASSERT(MEMS_ALIGN_IS_VALID(alignment));
    const mems_ptr buffer_ptr = MEMS_ALIGN_FORWARD((mems_ptr)bump->buffer + bump->len, alignment);
    if ((buffer_ptr - (mems_ptr)bump->buffer + size) > bump->capacity)
    {
        return MEMS_NULL;
    }

    bump->last_allocation = (void *)buffer_ptr;
    bump->len = (mems_size)(buffer_ptr - (mems_ptr)bump->buffer) + size;
    return (void *)buffer_ptr;
}

MEMS_EXPORT void * mems_bump_realloc(
    mems_bump * const bump,
    void * const buffer,
    const mems_size alignment,
    mems_size size
)
{
    MEMS_ASSERT(MEMS_ALIGN_IS_VALID(alignment));
    if (buffer == MEMS_NULL) return mems_bump_alloc(bump, alignment, size);

    if (buffer == bump->last_allocation)
    {
        const mems_ptr buffer_ptr = MEMS_ALIGN_FORWARD((mems_ptr)buffer, alignment);
        const mems_size prefix_size = buffer_ptr - (mems_ptr)bump->buffer;
        const mems_size original_size = bump->len - (mems_size)((mems_ptr)buffer - (mems_ptr)bump->buffer);
        size = size == MEMS_WHOLE_BUFFER ? bump->capacity - prefix_size : size;
        if (prefix_size + size > bump->capacity) return MEMS_NULL;
        bump->len = prefix_size + size;

        if (buffer_ptr != (mems_ptr)bump->last_allocation)
        {
            bump->last_allocation = MEMS_MEMMOVE((void *)buffer_ptr, bump->last_allocation, original_size);
            return bump->last_allocation;
        }

        return (void *)buffer_ptr;
    }

    return mems_bump_alloc(bump, alignment, size);
}

MEMS_EXPORT void mems_bump_free(mems_bump * bump, void * buffer)
{
    // only frees if the buffer is the last allocation in the bump
    if (buffer == bump->last_allocation)
    {
        bump->len = (mems_size)((mems_ptr)buffer - (mems_ptr)bump->buffer);
        bump->last_allocation = MEMS_NULL;
    }
}

MEMS_EXPORT void mems_bump_reset(mems_bump * bump)
{
    bump->len = 0;
    bump->last_allocation = MEMS_NULL;
}

MEMS_EXPORT void * mems_bump_rewind(mems_bump * const bump, void * const buffer, mems_size len)
{
    const mems_size check_len = len == MEMS_WHOLE_BUFFER ? 0 : len;
    if ((mems_ptr)bump->buffer > (mems_ptr)buffer) return MEMS_NULL;
    if (((mems_ptr)bump->buffer + bump->capacity) < ((mems_ptr)buffer + check_len)) return MEMS_NULL;

    bump->len = len == MEMS_WHOLE_BUFFER ?
        bump->capacity :
        (mems_ptr)buffer - (mems_ptr)bump->buffer + len;
    bump->last_allocation = buffer;

    return buffer;
}

MEMS_EXPORT mems_bool mems_bump_align(mems_bump * const bump, const mems_size alignment)
{
    MEMS_ASSERT(MEMS_ALIGN_IS_VALID(alignment));
    const mems_ptr aligned_buffer_ptr = MEMS_ALIGN_FORWARD((mems_ptr)bump->buffer + bump->len, alignment);
    const mems_size aligned_len = (mems_size)((mems_size)(aligned_buffer_ptr - (mems_ptr)bump->buffer));
    if (aligned_len >= bump->capacity) return mems_false;
    bump->len = aligned_len;

    return mems_true;
}

MEMS_EXPORT mems_size mems_bump_remaining(const mems_bump * const bump, const mems_size alignment)
{
    MEMS_ASSERT(MEMS_ALIGN_IS_VALID(alignment));
    const mems_ptr buffer_ptr = MEMS_ALIGN_FORWARD((mems_ptr)bump->buffer + bump->len, alignment);
    const mems_ptr end = (mems_ptr)bump->buffer + bump->capacity;
    if (buffer_ptr > end) return 0;
    return (mems_size)(end - buffer_ptr);
}

MEMS_EXPORT void * mems_bump_remaining_buffer(mems_bump * const bump, const mems_size alignment, mems_size * const size)
{
    MEMS_ASSERT(MEMS_ALIGN_IS_VALID(alignment));
    *size = mems_bump_remaining(bump, alignment);
    return mems_bump_alloc(bump, alignment, *size);
}

static void * mems__bump_allocator_alloc(void * const context, const mems_size alignment, const mems_size size)
{
    MEMS_ASSERT(MEMS_ALIGN_IS_VALID(alignment));
    return mems_bump_alloc((mems_bump *)context, alignment, size);
}

static void * mems__bump_allocator_realloc(void * const context, void * const buffer, const mems_size alignment, const mems_size size)
{
    MEMS_ASSERT(MEMS_ALIGN_IS_VALID(alignment));
    return mems_bump_realloc((mems_bump *)context, buffer, alignment, size);
}

static void mems__bump_allocator_free(void * const context, void * const buffer)
{
    mems_bump_free((mems_bump *)context, buffer);
}

static const mems_allocator_vtable mems__bump_allocator_vtable = {
    .alloc = mems__bump_allocator_alloc,
    .realloc = mems__bump_allocator_realloc,
    .free = mems__bump_allocator_free,
};
MEMS_EXPORT void mems_bump_allocator_init(mems_bump * const bump, mems_allocator * const allocator)
{
    allocator->context = bump;
    allocator->vtable = &mems__bump_allocator_vtable;
}

MEMS_EXPORT mems_size mems_pool_required_chunk_size(
    mems_size alignment,
    mems_size size,
    const mems_size items_per_chunk
)
{
    MEMS_ASSERT(items_per_chunk > 0);

    alignment = MEMS_MAX(alignment, MEMS_ALIGNOF(mems_pool_item));
    size = MEMS_ALIGN_FORWARD(MEMS_MAX(size, sizeof(mems_pool_item)), alignment);

    const mems_size chunk_alignment = MEMS_MAX(MEMS_ALIGNOF(mems_pool_chunk), alignment);
    mems_size chunk_size = MEMS_ALIGN_FORWARD(sizeof(mems_pool_chunk), alignment);
    chunk_size += items_per_chunk * size;

    return chunk_size;
}

MEMS_EXPORT void mems_pool_init(
    mems_pool * const pool,
    const mems_allocator * const parent,
    mems_size alignment,
    mems_size size,
    const mems_size items_per_chunk
)
{
    MEMS_ASSERT(items_per_chunk > 0);

    alignment = MEMS_MAX(alignment, MEMS_ALIGNOF(mems_pool_item));
    size = MEMS_ALIGN_FORWARD(MEMS_MAX(size, sizeof(mems_pool_item)), alignment);

    *pool = (mems_pool){
        .parent = *parent,
        .alignment = alignment,
        .size = size,
        .items_per_chunk = items_per_chunk,
    };
}

MEMS_EXPORT void mems_pool_init_buffer(
    mems_pool * const pool,
    mems_size alignment,
    mems_size size,
    const mems_size buffer_len,
    void * const buffer
)
{
    alignment = MEMS_MAX(alignment, MEMS_ALIGNOF(mems_pool_item));
    size = MEMS_ALIGN_FORWARD(MEMS_MAX(size, sizeof(mems_pool_item)), alignment);

    const mems_size chunk_alignment = MEMS_MAX(MEMS_ALIGNOF(mems_pool_chunk), alignment);
    mems_size chunk_size = MEMS_ALIGN_FORWARD(sizeof(mems_pool_chunk), alignment);

    const mems_size items_per_chunk = (buffer_len - chunk_size) / size;
    chunk_size += items_per_chunk * size;

    mems_pool_chunk * const root = (mems_pool_chunk *)buffer;
    *root = (mems_pool_chunk){0};

    *pool = (mems_pool){
        .alignment = alignment,
        .size = size,
        .items_per_chunk = items_per_chunk,
        .root = root,
    };
}

MEMS_EXPORT void mems_pool_deinit(mems_pool * const pool)
{
    mems_pool_chunk * cur = pool->root;
    while (cur != MEMS_NULL)
    {
        mems_pool_chunk * const next = cur->next;
        mems_allocator_free(&pool->parent, (void *)cur);
        cur = next;
    }

    pool->free_list = MEMS_NULL;
    pool->root = MEMS_NULL;
}

MEMS_EXPORT mems_size mems_pool_get_free_len(const mems_pool * const pool)
{
    size_t free_len = 0;
    for (mems_pool_item * item = pool->free_list; item != MEMS_NULL; item = item->next) free_len++;
    return free_len;
}

MEMS_EXPORT mems_size mems_pool_get_outstanding(const mems_pool * const pool)
{
    const mems_size free_len = mems_pool_get_free_len(pool);
    MEMS_ASSERT(pool->root->index >= free_len);

    return pool->root->index - free_len;
}

MEMS_EXPORT void * mems_pool_alloc(mems_pool * const pool)
{
    if (pool->free_list != MEMS_NULL)
    {
        void * const item = (void *)pool->free_list;
        pool->free_list = pool->free_list->next;
        return item;
    }

    if (
        pool->root == MEMS_NULL ||
        (pool->root->index >= pool->items_per_chunk)
    )
    {
        if (pool->parent.vtable == MEMS_NULL) return MEMS_NULL;

        const mems_size chunk_alignment = MEMS_MAX(MEMS_ALIGNOF(mems_pool_chunk), pool->alignment);
        mems_size chunk_size = MEMS_ALIGN_FORWARD(sizeof(mems_pool_chunk), pool->alignment);
        chunk_size += pool->items_per_chunk * pool->size;
        mems_pool_chunk * const chunk = (mems_pool_chunk *)mems_allocator_alloc(&pool->parent, chunk_alignment, chunk_size);
        if (chunk == MEMS_NULL) return MEMS_NULL;

        *chunk = (mems_pool_chunk){
            .next = pool->root,
        };
        pool->root = chunk;
    }

    return (void *)(MEMS_ALIGN_FORWARD(
        (mems_ptr)(pool->root + 1),
        pool->alignment
    ) + (pool->root->index++ * pool->size));
}

MEMS_EXPORT void mems_pool_free(mems_pool * const pool, void * const buffer)
{
    if (buffer == MEMS_NULL) return;

    mems_pool_item * const item = (mems_pool_item *)buffer;
    *item = (mems_pool_item){
        .next = pool->free_list,
    };
    pool->free_list = item;
}

MEMS_EXPORT mems_bool mems_chunks_init(
    mems_chunks * const chunks,
    const mems_size max_chunks,
    const mems_size chunk_size,
    const mems_size buffer_len,
    void * const buffer,
    mems_size * const required_size
)
{
    *required_size = 0;
    *required_size += max_chunks * sizeof(void *);

    if (buffer_len < *required_size) return mems_false;

    *chunks = (mems_chunks){
        .capacity = max_chunks,
        .chunk_size = chunk_size,
        .chunks = (void **)buffer,
    };

    return mems_true;
}

MEMS_EXPORT mems_bool mems_chunks_init_alloc(
    mems_chunks * const chunks,
    const mems_size max_chunks,
    const mems_size chunk_size,
    const mems_allocator * const allocator
)
{
    mems_size required_size;

    if (mems_chunks_init(
        chunks,
        max_chunks,
        chunk_size,
        0,
        MEMS_NULL,
        &required_size
    )) MEMS_ABORT();

    void * const buffer = mems_allocator_alloc(allocator, MEMS_ALIGNOF(void *), required_size);
    if (buffer == MEMS_NULL) return mems_false;

    if (!mems_chunks_init(
        chunks,
        max_chunks,
        chunk_size,
        required_size,
        buffer,
        &required_size
    )) MEMS_ABORT();

    return mems_true;
}

MEMS_EXPORT void mems_chunks_deinit_alloc(
    mems_chunks * const chunks,
    const mems_allocator * const allocator
)
{
    mems_allocator_free(allocator, chunks->chunks);
}

MEMS_EXPORT mems_bool mems_chunks_append(mems_chunks * const chunks, void * const chunk)
{
    if (chunks->len >= chunks->capacity) return mems_false;

    chunks->chunks[chunks->len++] = chunk;

    return mems_true;
}

MEMS_EXPORT void * mems_chunks_append_pool(mems_chunks * const chunks, mems_pool * const pool)
{
    MEMS_ASSERT(pool->size == chunks->chunk_size);
    if (chunks->len >= chunks->capacity) return MEMS_NULL;

    void * const chunk = mems_pool_alloc(pool);
    if (chunk == MEMS_NULL) return MEMS_NULL;

    chunks->chunks[chunks->len++] = chunk;

    return chunk;
}

MEMS_EXPORT void * mems_chunks_pop(mems_chunks * const chunks)
{
    if (chunks->len <= 0) return MEMS_NULL;

    return chunks->chunks[--chunks->len];
}

MEMS_EXPORT mems_bool mems_chunks_pop_pool(mems_chunks * const chunks, mems_pool * const pool)
{
    void * const chunk = mems_chunks_pop(chunks);
    if (chunk == MEMS_NULL) return mems_false;

    mems_pool_free(pool, chunk);

    return mems_true;
}


#endif
#endif

#ifndef MEMS_LICENSE
#define MEMS_LICENSE

/*
Zlib License

Copyright (c) 2026 Hollin Wilkins

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#endif // MEMS_LICENSE
