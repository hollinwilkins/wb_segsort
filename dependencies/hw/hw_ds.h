// Copyright (C) 2026 Hollin Wilkins
// SPDX-License-Identifier: Zlib

#ifndef HWDS_H
#define HWDS_H

/*

hw_ds.h v1.0.0 - useful data structures
by Hollin Wilkins

SECURITY
    This library provides NO SECURITY GUARANTEES whatsoever.

WHAT DOES THIS LIBRARY DO?
    There are very common data structures that are required to write programs,
    such as array lists and hash maps. C does not provide these in the standard
    library, and so programmers are required to provide their own implementations.
    There are also some data structures that are reusable and useful in certain
    circumstances, for example the chunk list, for when you need a list that
    does not ever move memory.

    This library provides common data structures that are useful to a wide array
    of programming tasks. In addition, there are some data structures that are
    helpful for programming using a single buffer allocation.

    For each data type you want to create, you include this header twice: once
    for the declarations, and once for the actual implementation. Before
    including the header, you configure the data type using macro definitions.

FEATURE OVERVIEW
    - no malloc - users of this library pass buffers into library functions
        to manage memory
    - hash map - a hash map
    - list - an array list of items
    - chunk list - an list of chunks of items, the chunk capacity grows
        exponentially

COMPILING & LINKING
    This library requires C99, though it may be refactored to support C89 in the future.

    In one C/C++ file that includes this file, do this:
        #define HWDS_IMPLEMENTATION
    before the include, that will create the implementation in that file.

    If you also do this:
        #define HWDS_STATIC
    then all of the functions will be declared as static.

EXAMPLES
    Hash map
        uint32_t string_hash(const char * const key);
        hwds_bool string_equal(const char * const a, const char * const b);

        // create a hash map of string -> string
        // include this in your declarations
        #define HWDS_HM_DECLARATION
        #define HWDS_NAME string_map
        #define HWDS_KEY const char *
        #define HWDS_VALUE char *
        #define HWDS_HASH string_hash
        #define HWDS_EQUAL string_equal
        #include "hw_ds.h"

        // include again for the implementation
        #define HWDS_HM_IMPLEMENTATION
        #define HWDS_NAME string_map
        #define HWDS_KEY const char *
        #define HWDS_VALUE char *
        #define HWDS_HASH string_hash
        #define HWDS_EQUAL string_equal
        #include "hw_ds.h"

        hwds_size capacity = 1024;
        hwds_size required_size;
        string_map map;

        // first init to determine required buffer size
        hwds_result result = string_map_init(
            HWDS_NULL,
            capacity * 2,
            capacity,
            0,
            HWDS_NULL,
            &required_size
        );
        HWDS_ASSERT(result == hwds_alloc);

        // allocate buffer and run init again
        // with the same parameters
        void * const buffer = malloc(required_size);
        result = string_map_init(
            &map,
            capacity * 2,
            capacity,
            required_size,
            buffer,
            &required_size
        );
        HWDS_ASSERT(result == hwds_ok);

        // get or put (gop) a key into the hash map
        string_map_gop_result gop;
        result = string_map_get_or_put(&map, "key1", &gop);
        HWDS_ASSERT(result == hwds_ok);
        HWDS_ASSERT(gop.found_existing == hwds_false);
        *gop.value = "value1";

        // run again, and it will already have the value
        result = string_map_get_or_put(&map, "key1", &gop);
        HWDS_ASSERT(gop.found_existing == hwds_true);
    List
        // create an array list of chars
        // include this in your declarations
        #define HWDS_LIST_DECLARATION
        #define HWDS_NAME char_list
        #define HWDS_ITEM char
        #include "hw_ds.h" // HWDS_* definitions are cleared automatically

        // include again for the implementation
        #define HWDS_LIST_IMPLEMENTATION
        #define HWDS_NAME char_list
        #define HWDS_ITEM char
        #include "hw_ds.h" // HWDS_* definitions are cleared automatically

        hwds_size capacity = 1024;
        char_list list;
        hwds_size required_size;

        // run init to determine required buffer size
        hwds_result result = char_list_init(
            HWDS_NULL,
            capacity,
            0,
            HWDS_NULL,
            &required_size
        );
        HWDS_ASSERT(result == hwds_alloc);

        // allocate buffer and run init again
        // with the same parameters
        void * buffer = malloc(required_size);
        result = char_list_init(
            &list,
            capacity,
            required_size,
            buffer,
            &required_size
        );
        HWDS_ASSERT(result == hwds_ok);

        // append a single value
        char * const add = char_list_append(&list);
        HWDS_ASSERT(add != HWDS_NULL);
        *add = 42;

        // append 10 values
        char * const addmany = char_list_append_many(&list, 10);
        for (char i = 0; i < 10; i++) addmany[i] = i + 100;

        // move list to a new buffer
        capacity *= 2;
        result = char_list_init(
            HWDS_NULL,
            capacity,
            0,
            HWDS_NULL,
            &required_size
        );
        HWDS_ASSERT(result == hwds_alloc);
        buffer = malloc(required_size);
        hwds_bool success = char_list_update_buffer(&list, required_size, buffer);
        HWDS_ASSERT(success);

    Chunk list
        // create a chunk list of chars
        // include this in your declarations
        #define HWDS_SEGMENT_LIST_DECLARATION
        #define HWDS_NAME char_list
        #define HWDS_ITEM char
        // customize number of chunks to skip before allocating chunks
        #define HWDS_SKIP_SEGMENTS 10
        // customize number of chunks
        #define HWDS_SEGMENTS 10
        #include "hw_ds.h" // HWDS_* definitions are cleared automatically

        // include again for the implementation
        #define HWDS_SEGMENT_LIST_IMPLEMENTATION
        #define HWDS_NAME char_list
        #define HWDS_ITEM char
        // make sure these match declaration
        #define HWDS_SKIP_SEGMENTS 10
        #define HWDS_SEGMENTS 10
        #include "hw_ds.h" // HWDS_* definitions are cleared automatically

        hwds_size capacity = 1024;
        char_list list;
        hwds_size required_size;
        hwds_bool success;

        char_list_init(&list);

        // ensure list has a certain capacity
        // determine additional buffer size required for capacity
        if (!char_list_ensure_capacity(&list, capacity, &required_size))
        {
            void * const add_buffer = malloc(required_size);
            hwds_u8 added_count = char_list_add_buffer(&list, required_size, add_buffer);
            HWDS_ASSERT(added_count > 0);
            success = char_list_ensure_capacity(&list, capacity, &required_size);
            HWDS_ASSERT(success == hwds_true);
        }

        // append a single value
        char * const add = char_list_append(&list);
        HWDS_ASSERT(add != HWDS_NULL);
        *add = 42;

        // append 10 values
        // need to use an iterator, because the values
        // may span across multiple chunks
        char_list_iter addmany;
        char * item;
        success = char_list_append_many(&list, 10, &addmany);
        HWDS_ASSERT(success == hwds_true);

        int i = 0;
        while (char_list_iter_next(&addmany, &item))
        {
            *item = i + 100;
            i++;
        }

*/

#ifndef HWDS_NO_CRT
#   include <stddef.h>
#   include <stdint.h>
#   include <stdbool.h>
#   include <string.h>
#   include <math.h>
#   include <assert.h>
#endif

#ifndef HWDS_NO_CRT
#   ifndef hwds_size
#       define hwds_size size_t
#   endif
#   ifndef hwds_ptr
#       define hwds_ptr uintptr_t
#   endif
#   ifndef hwds_u8
#       define hwds_u8 uint8_t
#   endif
#   ifndef hwds_u32
#       define hwds_u32 uint32_t
#   endif
#   ifndef hwds_u64
#       define hwds_u64 uint64_t
#   endif
#   ifndef hwds_bool
#       define hwds_bool bool
#   endif
#   ifndef hwds_true
#       define hwds_true true
#   endif
#   ifndef hwds_false
#       define hwds_false false
#   endif
#   ifndef HWDS_NULL
#       define HWDS_NULL NULL
#   endif
#else
#ifndef hwds_size
#   if defined(__SIZE_TYPE__)
#       define hwds_size __SIZE_TYPE__
#   elif defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || \
    defined(__LP64__) || defined(_LP64) || (defined(__WORDSIZE) && __WORDSIZE == 64)
#       define hwds_size unsigned long long
#   elif defined(__i386__) || defined(_M_IX86) || defined(__arm__) || \
    defined(__ILP32__) || (defined(__WORDSIZE) && __WORDSIZE == 32)
#       define hwds_size unsigned int
#   else
#       define hwds_size unsigned long
#   endif
#endif
#   ifndef hwds_ptr
#       if defined(__UINTPTR_TYPE__)
#           define hwds_ptr __UINTPTR_TYPE__
#       elif defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || \
        defined(__LP64__) || defined(_LP64) || defined(_WIND64)
#           define hwds_ptr unsigned long long
#       elif defined(__i386__) || defined(_M_IX86) || defined(__arm__) || \
        defined(__ILP32__) || defined(_WIN32)
#           define hwds_ptr unsigned int
#       else
#           define hwds_ptr unsigned long
#       endif
#   endif
#   ifndef hwds_u8
#      define hwds_u8 unsigned char
#   endif
#   ifndef hwds_u32
#      define hwds_u32 unsigned
#   endif
#   ifndef hwds_u64
#      define hwds_u64 unsigned long long
#   endif
#   ifndef hwds_bool
#       define hwds_bool int
#   endif
#   ifndef hwds_true
#       define hwds_true 1
#   endif
#   ifndef hwds_false
#       define hwds_false 0
#   endif
#   ifndef HWDS_NULL
#      define HWDS_NULL 0
#   endif
#endif

#ifndef HWDS_EXPORT
#   ifdef HWDS_STATIC
#       define HWDS_EXPORT static
#   else
#       ifdef __cplusplus
#           define HWDS_EXPORT extern "C"
#       else
#           define HWDS_EXPORT extern
#       endif
#   endif
#endif

#ifdef _MSC_VER
#   define HWDS_INLINE static __forceinline
#   define HWDS_NOINLINE static __declspec(noinline)
#   define HWDS_ALIGNOF __alignof
#else
#   ifdef __has_attribute
#       if __has_attribute(always_inline)
#           define HWDS_INLINE static inline __attribute__((always_inline))
#       endif
#       if __has_attribute(noinline)
#           define HWDS_NOINLINE static __attribute__((noinline))
#       endif
#   endif
#   define HWDS_ALIGNOF __alignof__
#endif

#ifndef HWDS_INLINE
#   define HWDS_INLINE static inline
#endif

#ifndef HWDS_MAX
#   define HWDS_MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#ifndef HWDS_MIN
#   define HWDS_MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef HWDS_NO_CRT
#   ifndef HWDS_MEMSET
#       define HWDS_MEMSET(ptr, value, size) memset(ptr, value, size)
#   endif
#   ifndef HWDS_LOG2F
#       define HWDS_LOG2F(v) log2f(v)
#   endif
#   ifndef HWDS_ASSERT
#       if defined(HWDS_NO_ASSERT) || defined(NDEBUG)
#           define HWDS_ASSERT(condition) ((void)(condition))
#       else
#           define HWDS_ASSERT(condition) assert(condition)
#       endif
#   endif
#   ifndef HWDS_ABORT
#       define HWDS_ABORT() abort()
#   endif
#else
#   ifndef HWDS_MEMSET
#       define HWDS_MEMSET(ptr, value, size) do { \
    for (hwds_size hwds__memset_i = 0; hwds__memset_i < size; hwds__memset_i++) \
        { ((int *)ptr)[hwds__memset_i] = value; } \
} while (0)
#   endif
#   ifndef HWDS_ASSERT
#       define HWDS_ASSERT(condition) ((void)condition)
#   endif
#endif

#ifndef HWDS_ENSURE
#   define HWDS_ENSURE(condition) if (!(condition)) { \
    HWDS_ABORT(); \
}
#endif

#ifndef HWDS_ALIGN_FORWARD
#   define HWDS_ALIGN_FORWARD(address, alignment) ((address + (alignment - 1)) & ~(alignment - 1))
#endif

#if defined(MEMS_H) && !defined(HWDS_MEMS_DISABLED)
#   define HWDS_MEMS_ENABLED
#endif

#ifdef HWDS_MEMS_ENABLED

#include "hw_mems.h"

#endif // HWDS_MEMS_ENABLED

#if defined(__GNUC__) || defined(__clang__)
#define HWDS_UNUSED __attribute__((unused))
#else
#define HWDS_UNUSED
#endif

#define HWDS_CONCAT_RAW(a, b) a##_##b
#define HWDS_CONCAT(a, b) HWDS_CONCAT_RAW(a, b)
#define HWDS_NAME_(b) HWDS_CONCAT(HWDS_NAME, b)

#define HWDS_HEAP_PARENT(i) ((i - 1) / 2)
#define HWDS_HEAP_LEFT(i) (2 * i + 1)
#define HWDS_HEAP_RIGHT(i) (2 * i + 2)

typedef hwds_u32 hwds_result;
typedef enum hwds_result_enum
{
    hwds_ok = 0x0000000,
    hwds_error = 0x0000001,
    hwds_alloc = 0x0000002,
} hwds_result_enum;

HWDS_EXPORT hwds_u32 hwds_jenkins32(const void * key, hwds_size length);
HWDS_EXPORT void hwds_jenkins32_update(hwds_u32 * hash, const void * key, hwds_size length);

typedef hwds_u32 hwds__hm_slot_state;

#endif // HWDS_H

#ifdef HWDS_IMPLEMENTATION
#ifndef HWDS_IMPLEMENTED
#define HWDS_IMPLEMENTED

static hwds__hm_slot_state hwds__hm_get_state(const hwds_u32 * sets, hwds_u32 index);
static void hwds__hm_set_state(hwds_u32 * sets, hwds_u32 index, hwds__hm_slot_state state);

static hwds_size hwds__hm_get_keys_offset(const hwds_size capacity, const hwds_size key_alignment);
static hwds_size hwds__hm_get_values_offset(
    const hwds_size capacity,
    const hwds_size key_alignment,
    const hwds_size key_size,
    const hwds_size value_alignment
);
static hwds_size hwds__hm_get_required_size(
    const hwds_size capacity,
    const hwds_size key_alignment,
    const hwds_size key_size,
    const hwds_size value_alignment,
    const hwds_size value_size
);

typedef enum hwds__hm_slot_state_enum
{
    hwds__hm_slot_state_empty = 0x00000000,
    hwds__hm_slot_state_occupied = 0x00000001,
    hwds__hm_slot_state_tombstone = 0x00000002,
} hwds__hm_slot_state_enum;

HWDS_EXPORT hwds_u32 hwds_jenkins32(const void * const key, const hwds_size length) {
    hwds_u32 hash = 0;
    hwds_jenkins32_update(&hash, key, length);
    return hash;
}

HWDS_EXPORT void hwds_jenkins32_update(hwds_u32 * const hash, const void * const key, const hwds_size length)
{
    const hwds_u8 * const bytes = (hwds_u8 *)key;
    hwds_size i = 0;
    while (i != length) {
        *hash += bytes[i++];
        *hash += *hash << 10;
        *hash ^= *hash >> 6;
    }

    *hash += *hash << 3;
    *hash ^= *hash >> 11;
    *hash += *hash << 15;
}

static hwds__hm_slot_state hwds__hm_get_state(const hwds_u32 * const sets, const hwds_u32 index)
{
    return (hwds__hm_slot_state)((sets[index >> 4] >> ((index & 15) << 1)) & 0x3u);
}

static void hwds__hm_set_state(hwds_u32 * const sets, const hwds_u32 index, const hwds__hm_slot_state state)
{
    const hwds_u32 shift = (index & 15) << 1;
    sets[index >> 4] = (sets[index >> 4] & ~(0x3u << shift)) | ((hwds_u32)state << shift);
}

static hwds_size hwds__hm_get_keys_offset(const hwds_size capacity, const hwds_size key_alignment)
{
    return HWDS_ALIGN_FORWARD((((capacity - 1) >> 4) + 1) * sizeof(hwds_u32), key_alignment);
}

static hwds_size hwds__hm_get_values_offset(
    const hwds_size capacity,
    const hwds_size key_alignment,
    const hwds_size key_size,
    const hwds_size value_alignment
)
{
    return HWDS_ALIGN_FORWARD(hwds__hm_get_keys_offset(capacity, key_alignment) + (capacity * key_size), value_alignment);
}

static hwds_size hwds__hm_get_required_size(
    const hwds_size capacity,
    const hwds_size key_alignment,
    const hwds_size key_size,
    const hwds_size value_alignment,
    const hwds_size value_size)
{
    return HWDS_ALIGN_FORWARD(hwds__hm_get_keys_offset(capacity, key_alignment) +
        (capacity * key_size), value_alignment) +
        (capacity * value_size);
}

#endif // HWDS_IMPLEMENTED
#endif // HWDS_IMPLEMENTATION

#ifdef HWDS_HM_DECLARATION

typedef struct HWDS_NAME
{
    hwds_u32 capacity;
    hwds_u32 allowed;
    hwds_u32 len;
    void * buffer;
} HWDS_NAME;

typedef struct HWDS_NAME_(entry)
{
    HWDS_KEY key;
    HWDS_VALUE value;
} HWDS_NAME_(entry);

typedef struct HWDS_NAME_(entry_ptr)
{
    HWDS_KEY * key;
    HWDS_VALUE * value;
} HWDS_NAME_(entry_ptr);

typedef struct HWDS_NAME_(gop_result)
{
    HWDS_KEY * key;
    HWDS_VALUE * value;
    hwds_bool found_existing;
} HWDS_NAME_(gop_result);

typedef struct HWDS_NAME_(iter)
{
    const HWDS_NAME * map;
    hwds_u32 index;
} HWDS_NAME_(iter);

HWDS_EXPORT hwds_result HWDS_NAME_(init)(
    HWDS_NAME * const hm,
    hwds_size capacity,
    hwds_size allowed,
    hwds_size buffer_len,
    void * buffer,
    hwds_size * required_size
);

HWDS_EXPORT hwds_result HWDS_NAME_(put)(
    HWDS_NAME * hm,
    HWDS_KEY key,
    HWDS_VALUE value,
    HWDS_NAME_(entry) ** old_entry
);

HWDS_EXPORT hwds_result HWDS_NAME_(get_or_put)(
    HWDS_NAME * hm,
    HWDS_KEY key,
    HWDS_NAME_(gop_result) * gop
);

HWDS_EXPORT HWDS_VALUE * HWDS_NAME_(get)(
    const HWDS_NAME * hm,
    HWDS_KEY key
);

HWDS_EXPORT void HWDS_NAME_(remove)(
    HWDS_NAME * hm,
    HWDS_KEY key,
    HWDS_NAME_(entry) ** old_entry
);

HWDS_EXPORT void HWDS_NAME_(reset)(HWDS_NAME * hm);

HWDS_EXPORT hwds_bool HWDS_NAME_(has_key)(
    HWDS_NAME * hm,
    HWDS_KEY key
);

HWDS_EXPORT void HWDS_NAME_(iter_init)(
    HWDS_NAME_(iter) * iter,
    const HWDS_NAME * map
);

HWDS_EXPORT hwds_bool HWDS_NAME_(iter_next)(
    HWDS_NAME_(iter) * iter,
    HWDS_NAME_(entry_ptr) * entry
);

#ifdef HWDS_MEMS_ENABLED

HWDS_EXPORT hwds_bool HWDS_NAME_(init_alloc)(
    HWDS_NAME * const hm,
    const mems_allocator * allocator,
    hwds_size capacity,
    hwds_size allowed
);

#endif // HWDS_MEMS_ENABLED

#endif // HWDS_HM_DECLARATION


#ifdef HWDS_HM_IMPLEMENTATION

static hwds_u32 * HWDS_NAME_(get_states)(const HWDS_NAME * const hm)
{
    return (hwds_u32 *)(hm->buffer);
}

static HWDS_KEY * HWDS_NAME_(get_keys)(const HWDS_NAME * const hm)
{
    return (HWDS_KEY *)((hwds_u8 *)hm->buffer + hwds__hm_get_keys_offset(hm->capacity, HWDS_ALIGNOF(HWDS_KEY)));
}

static HWDS_VALUE * HWDS_NAME_(get_values)(const HWDS_NAME * const hm)
{
    return (HWDS_VALUE *)((hwds_u8 *)hm->buffer + hwds__hm_get_values_offset(hm->capacity, HWDS_ALIGNOF(HWDS_KEY), sizeof(HWDS_KEY), HWDS_ALIGNOF(HWDS_VALUE)));
}

static hwds_size hwds__round_pow2(hwds_size a) {
    if (a <= 1) return 1;
    a--;
    a |= a >> 1; a |= a >> 2; a |= a >> 4;
    a |= a >> 8; a |= a >> 16; a |= a >> 32;
    return a + 1;
}

HWDS_EXPORT hwds_result HWDS_NAME_(init)(
    HWDS_NAME * const hm,
    hwds_size capacity,
    const hwds_size allowed,
    const hwds_size buffer_len,
    void * const buffer,
    hwds_size * const required_size
)
{
    HWDS_ASSERT(capacity > 0);
    capacity = hwds__round_pow2(capacity);

    *required_size = hwds__hm_get_required_size(
        capacity,
        HWDS_ALIGNOF(HWDS_KEY),
        sizeof(HWDS_KEY),
        HWDS_ALIGNOF(HWDS_VALUE),
        sizeof(HWDS_VALUE)
    );
    if (buffer_len < *required_size) return hwds_alloc;

    hm->capacity = capacity;
    hm->allowed = allowed;
    hm->len = 0;
    hm->buffer = buffer;

    memset(HWDS_NAME_(get_states)(hm), 0, (((capacity - 1) >> 4) + 1) * sizeof(hwds_u32));

    return hwds_ok;
}

HWDS_EXPORT hwds_result HWDS_NAME_(put)(
    HWDS_NAME * const hm,
    const HWDS_KEY key,
    const HWDS_VALUE value,
    HWDS_NAME_(entry) ** const old_entry
)
{
    if (hm->len >= hm->allowed) return hwds_alloc;

    hwds_u32 hash = HWDS_HASH(key);
    hwds_u32 start_slot = hash & (hm->capacity - 1);
    hwds_u32 end_slot = start_slot + hm->capacity;
    hwds_u32 * states = HWDS_NAME_(get_states)(hm);
    HWDS_KEY * keys = HWDS_NAME_(get_keys)(hm);
    HWDS_VALUE * values = HWDS_NAME_(get_values)(hm);
    hwds_u32 tombstone = ((hwds_u32)0 - 1);

    for (hwds_u32 i = start_slot; i < end_slot; i++)
    {
        hwds_u32 slot = i & (hm->capacity - 1);

        switch (hwds__hm_get_state(states, slot))
        {
            case hwds__hm_slot_state_tombstone:
            {
                if (tombstone == ((hwds_u32)0 - 1)) tombstone = slot;
            } continue;
            case hwds__hm_slot_state_empty:
            {
                if (tombstone != ((hwds_u32)0 - 1)) slot = tombstone;

                hm->len++;
                hwds__hm_set_state(states, slot, hwds__hm_slot_state_occupied);
                if (old_entry != HWDS_NULL) *old_entry = HWDS_NULL;
                keys[slot] = key;
                values[slot] = value;
                return hwds_ok;
            } break;
            case hwds__hm_slot_state_occupied:
            {
                if (HWDS_EQUAL(key, keys[slot]))
                {
                    if (old_entry != HWDS_NULL)
                    {
                        (*old_entry)->key = keys[slot];
                        (*old_entry)->value = values[slot];
                    }
                    keys[slot] = key;
                    values[slot] = value;
                    return hwds_ok;
                }
            } break;
        }
    }

    if (tombstone != ((hwds_u32)0 - 1))
    {
        hm->len++;
        hwds__hm_set_state(states, tombstone, hwds__hm_slot_state_occupied);
        *old_entry = HWDS_NULL;
        keys[tombstone] = key;
        values[tombstone] = value;
        return hwds_ok;
    }

    return hwds_alloc;
}

HWDS_EXPORT hwds_result HWDS_NAME_(get_or_put)(
    HWDS_NAME * const hm,
    const HWDS_KEY key,
    HWDS_NAME_(gop_result) * gop
)
{
    hwds_u32 hash = HWDS_HASH(key);
    hwds_u32 start_slot = hash & (hm->capacity - 1);
    hwds_u32 end_slot = start_slot + hm->capacity;
    hwds_u32 * states = HWDS_NAME_(get_states)(hm);
    HWDS_KEY * keys = HWDS_NAME_(get_keys)(hm);
    hwds_u32 tombstone = ((hwds_u32)0 - 1);

    for (hwds_u32 i = start_slot; i < end_slot; i++)
    {
        hwds_u32 slot = i & (hm->capacity - 1);

        switch (hwds__hm_get_state(states, slot))
        {
            case hwds__hm_slot_state_tombstone:
            {
                if (tombstone == ((hwds_u32)0 - 1)) tombstone = slot;
            } continue;
            case hwds__hm_slot_state_empty:
            {
                if (tombstone != ((hwds_u32)0 - 1)) slot = tombstone;
                HWDS_VALUE * values = HWDS_NAME_(get_values)(hm);
                hwds__hm_set_state(states, slot, hwds__hm_slot_state_occupied);
                hm->len++;
                keys[slot] = key;
                gop->key = &(keys[slot]);
                gop->value = &(values[slot]);
                gop->found_existing = hwds_false;
                return hwds_ok;
            } break;
            case hwds__hm_slot_state_occupied:
            {
                if (HWDS_EQUAL(key, keys[slot]))
                {
                    HWDS_VALUE * values = HWDS_NAME_(get_values)(hm);
                    gop->key = &(keys[slot]);
                    gop->value = &(values[slot]);
                    gop->found_existing = hwds_true;
                    return hwds_ok;
                }
            } break;
        }
    }

    if (tombstone != ((hwds_u32)0 - 1))
    {
        HWDS_VALUE * values = HWDS_NAME_(get_values)(hm);
        hwds__hm_set_state(states, tombstone, hwds__hm_slot_state_occupied);
        hm->len++;
        keys[tombstone] = key;
        gop->key = &(keys[tombstone]);
        gop->value = &(values[tombstone]);
        gop->found_existing = hwds_false;
        return hwds_ok;
    }

    return hwds_alloc;
}

HWDS_EXPORT HWDS_VALUE * HWDS_NAME_(get)(
    const HWDS_NAME * const hm,
    const HWDS_KEY key
)
{
    hwds_u32 hash = HWDS_HASH(key);
    hwds_u32 start_slot = hash & (hm->capacity - 1);
    hwds_u32 end_slot = start_slot + hm->capacity;
    hwds_u32 * states = HWDS_NAME_(get_states)(hm);
    const HWDS_KEY * const keys = HWDS_NAME_(get_keys)(hm);

    for (hwds_u32 i = start_slot; i < end_slot; i++)
    {
        const hwds_u32 slot = i & (hm->capacity - 1);

        switch (hwds__hm_get_state(states, slot))
        {
            case hwds__hm_slot_state_empty:
            {
                return HWDS_NULL;
            } break;
            case hwds__hm_slot_state_tombstone: continue;
            case hwds__hm_slot_state_occupied:
            {
                if (HWDS_EQUAL(key, keys[slot]))
                {
                    HWDS_VALUE * const values = HWDS_NAME_(get_values)(hm);
                    return &(values[slot]);
                }
            } break;
        }
    }

    return HWDS_NULL;
}

HWDS_EXPORT void HWDS_NAME_(remove)(
    HWDS_NAME * const hm,
    const HWDS_KEY key,
    HWDS_NAME_(entry) ** const old_entry
)
{
    hwds_u32 hash = HWDS_HASH(key);
    hwds_u32 start_slot = hash & (hm->capacity - 1);
    hwds_u32 end_slot = start_slot + hm->capacity;
    hwds_u32 * states = HWDS_NAME_(get_states)(hm);
    const HWDS_KEY * const keys = HWDS_NAME_(get_keys)(hm);

    for (hwds_u32 i = start_slot; i < end_slot; i++)
    {
        const hwds_u32 slot = i & (hm->capacity - 1);

        switch (hwds__hm_get_state(states, slot))
        {
            case hwds__hm_slot_state_empty:
            {
                if (old_entry != HWDS_NULL)
                {
                    *old_entry = HWDS_NULL;
                }

                return;
            } break;
            case hwds__hm_slot_state_tombstone: continue;
            case hwds__hm_slot_state_occupied:
            {
                if (HWDS_EQUAL(key, keys[slot]))
                {
                    hm->len -= 1;
                    const HWDS_VALUE * const values = HWDS_NAME_(get_values)(hm);
                    if (old_entry != HWDS_NULL)
                    {
                        (*old_entry)->key = keys[slot];
                        (*old_entry)->value = values[slot];
                    }
                    hwds__hm_set_state(states, slot, hwds__hm_slot_state_tombstone);
                    return;
                }
            } break;
        }
    }
}

HWDS_EXPORT void HWDS_NAME_(reset)(HWDS_NAME * const hm)
{
    const hwds_size buffer_len = hwds__hm_get_required_size(
        hm->capacity,
        HWDS_ALIGNOF(HWDS_KEY),
        sizeof(HWDS_KEY),
        HWDS_ALIGNOF(HWDS_VALUE),
        sizeof(HWDS_VALUE)
    );

    HWDS_MEMSET(hm->buffer, 0, buffer_len);
    hm->len = 0;
}

HWDS_EXPORT hwds_bool HWDS_NAME_(has_key)(
    HWDS_NAME * const hm,
    const HWDS_KEY key
)
{
    hwds_u32 hash = HWDS_HASH(key);
    hwds_u32 start_slot = hash & (hm->capacity - 1);
    hwds_u32 end_slot = start_slot + hm->capacity;
    hwds_u32 * states = HWDS_NAME_(get_states)(hm);
    HWDS_KEY * keys = HWDS_NAME_(get_keys)(hm);

    for (hwds_u32 i = start_slot; i < end_slot; i++)
    {
        const hwds_u32 slot = i & (hm->capacity - 1);

        switch (hwds__hm_get_state(states, slot))
        {
            case hwds__hm_slot_state_empty: return hwds_false;
            case hwds__hm_slot_state_tombstone: continue;
            case hwds__hm_slot_state_occupied:
            {
                if (HWDS_EQUAL(key, keys[slot])) return hwds_true;
            } break;
        }
    }

    return hwds_false;
}

HWDS_EXPORT void HWDS_NAME_(iter_init)(
    HWDS_NAME_(iter) * const iter,
    const HWDS_NAME * const map
)
{
    *iter = (HWDS_NAME_(iter)){
        .index = 0,
        .map = map,
    };
}

HWDS_EXPORT hwds_bool HWDS_NAME_(iter_next)(
    HWDS_NAME_(iter) * const iter,
    HWDS_NAME_(entry_ptr) * const entry
)
{
    const hwds_u32 * const states = HWDS_NAME_(get_states)(iter->map);
    const hwds_u32 capacity = iter->map->capacity;
    for (hwds_u32 i = iter->index; i < capacity; i++)
    {
        const hwds__hm_slot_state state = hwds__hm_get_state(states, i);

        if (state == hwds__hm_slot_state_occupied)
        {
            *entry = (HWDS_NAME_(entry_ptr)){
                .key = HWDS_NAME_(get_keys)(iter->map) + i,
                .value = HWDS_NAME_(get_values)(iter->map) + i,
            };
            iter->index = i + 1;
            return hwds_true;
        }
    }
    iter->index = capacity;

    return hwds_false;
}

#ifdef HWDS_MEMS_ENABLED

HWDS_EXPORT hwds_bool HWDS_NAME_(init_alloc)(
    HWDS_NAME * const hm,
    const mems_allocator * const allocator,
    hwds_size capacity,
    hwds_size allowed
)
{
    hwds_size required_size;
    hwds_result result = HWDS_NAME_(init)(
        hm,
        capacity,
        allowed,
        0,
        HWDS_NULL,
        &required_size
    );
    HWDS_ASSERT(result == hwds_alloc);

    void * const buffer = mems_allocator_alloc(allocator, HWDS_ALIGNOF(hwds_size), required_size);
    if (buffer == HWDS_NULL) return hwds_false;

    result = HWDS_NAME_(init)(
        hm,
        capacity,
        allowed,
        required_size,
        buffer,
        &required_size
    );
    HWDS_ASSERT(result == hwds_ok);

    return hwds_true;
}

#endif // HWDS_MEMS_ENABLED

#endif // HWDS_HM_IMPLEMENTATION

#ifdef HWDS_LIST_DECLARATION

typedef struct HWDS_NAME
{
    hwds_size capacity;
    hwds_size len;
    HWDS_ITEM * items;
} HWDS_NAME;

HWDS_EXPORT hwds_result HWDS_NAME_(init)(
    HWDS_NAME * list,
    hwds_size items_capacity,
    hwds_size buffer_len,
    void * buffer,
    hwds_size * required_size
);

HWDS_EXPORT void HWDS_NAME_(init_buffer)(
    HWDS_NAME * list,
    hwds_size buffer_len,
    void * buffer
);

HWDS_EXPORT hwds_bool HWDS_NAME_(update_buffer)(
    HWDS_NAME * list,
    hwds_size buffer_len,
    void * buffer
);

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append)(HWDS_NAME * list);
HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_many)(HWDS_NAME * list, hwds_size n);
HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(pop)(HWDS_NAME * list);

#ifdef HWDS_MEMS_ENABLED

HWDS_EXPORT hwds_bool HWDS_NAME_(init_alloc)(
    HWDS_NAME * list,
    const mems_allocator * allocator,
    hwds_size items_capacity
);

HWDS_EXPORT hwds_bool HWDS_NAME_(init_align_alloc)(
    HWDS_NAME * list,
    const mems_allocator * allocator,
    hwds_size alignment,
    hwds_size items_capacity
);

HWDS_EXPORT void HWDS_NAME_(deinit_alloc)(
    HWDS_NAME * list,
    const mems_allocator * allocator
);

HWDS_EXPORT hwds_bool HWDS_NAME_(ensure_capacity_alloc)(
    HWDS_NAME * list,
    const mems_allocator * allocator,
    hwds_size items_capacity
);

HWDS_EXPORT hwds_bool HWDS_NAME_(ensure_capacity_align_alloc)(
    HWDS_NAME * list,
    const mems_allocator * allocator,
    hwds_size alignment,
    hwds_size items_capacity
);

HWDS_EXPORT void HWDS_NAME_(expand_to_capacity)(HWDS_NAME * list);

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_align_alloc)(HWDS_NAME * list, const mems_allocator * allocator, hwds_size alignment);
HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_alloc)(HWDS_NAME * list, const mems_allocator * allocator);
HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_many_align_alloc)(HWDS_NAME * list, const mems_allocator * allocator, hwds_size alignment, hwds_size n);
HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_many_alloc)(HWDS_NAME * list, const mems_allocator * allocator, hwds_size n);

#endif // HWDS_MEMS_ENABLED

HWDS_EXPORT void HWDS_NAME_(reverse)(HWDS_NAME * list);
HWDS_EXPORT void HWDS_NAME_(reset)(HWDS_NAME * list);
HWDS_EXPORT hwds_size HWDS_NAME_(grow_capacity)(const HWDS_NAME * list, hwds_size capacity);
HWDS_EXPORT hwds_size HWDS_NAME_(grow_additional_capacity)(const HWDS_NAME * list, hwds_size n);

#endif // HWDS_LIST_DECLARATION

#ifdef HWDS_LIST_IMPLEMENTATION

HWDS_EXPORT hwds_result HWDS_NAME_(init)(
    HWDS_NAME * const list,
    const hwds_size items_capacity,
    const hwds_size buffer_len,
    void * const buffer,
    hwds_size * const required_size
)
{
    *required_size = items_capacity * sizeof(HWDS_ITEM);
    if (buffer_len < *required_size) return hwds_alloc;

    list->capacity = items_capacity;
    list->len = 0;
    list->items = (HWDS_ITEM *)buffer;

    return hwds_ok;
}

HWDS_EXPORT void HWDS_NAME_(init_buffer)(
    HWDS_NAME * const list,
    const hwds_size buffer_len,
    void * const buffer
)
{
    const hwds_ptr aligned_buffer = HWDS_ALIGN_FORWARD((hwds_ptr)buffer, HWDS_ALIGNOF(HWDS_ITEM));
    const hwds_size items_capacity = (hwds_size)(((hwds_ptr)buffer + buffer_len) - aligned_buffer) / sizeof(HWDS_ITEM);

    list->capacity = items_capacity;
    list->len = 0;
    list->items = (HWDS_ITEM *)aligned_buffer;
}

HWDS_EXPORT hwds_bool HWDS_NAME_(update_buffer)(
    HWDS_NAME * const list,
    const hwds_size buffer_len,
    void * buffer
)
{
    const hwds_size items_capacity = buffer_len / sizeof(HWDS_ITEM);
    if (items_capacity < list->len) return hwds_false;

    list->capacity = items_capacity;
    list->items = (HWDS_ITEM *)buffer;

    return hwds_true;
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append)(HWDS_NAME * const list)
{
    if (list->len >= list->capacity) return HWDS_NULL;
    return list->items + list->len++;
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_many)(HWDS_NAME * const list, const hwds_size n)
{
    if (list->len + n > list->capacity) return HWDS_NULL;
    list->len += n;
    return list->items + (list->len - n);
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(pop)(HWDS_NAME * const list)
{
    if (list->len == 0) return HWDS_NULL;
    return list->items + (--list->len);
}

#ifdef HWDS_MEMS_ENABLED

HWDS_EXPORT hwds_bool HWDS_NAME_(init_alloc)(
    HWDS_NAME * const list,
    const mems_allocator * const allocator,
    const hwds_size items_capacity
)
{
    return HWDS_NAME_(init_align_alloc)(
        list,
        allocator,
        HWDS_ALIGNOF(HWDS_ITEM),
        items_capacity
    );
}

HWDS_EXPORT hwds_bool HWDS_NAME_(init_align_alloc)(
    HWDS_NAME * const list,
    const mems_allocator * allocator,
    const hwds_size alignment,
    const hwds_size items_capacity
)
{
    hwds_result result;
    hwds_size required_size;

    result = HWDS_NAME_(init)(
        list,
        items_capacity,
        0,
        HWDS_NULL,
        &required_size
    );
    HWDS_ASSERT(result == hwds_alloc);

    void * const buffer = mems_allocator_alloc(
        allocator,
        alignment,
        required_size
    );
    if (buffer == HWDS_NULL) return hwds_false;

    result = HWDS_NAME_(init)(
        list,
        items_capacity,
        required_size,
        buffer,
        &required_size
    );
    HWDS_ASSERT(result == hwds_ok);

    return hwds_true;
}

HWDS_EXPORT void HWDS_NAME_(deinit_alloc)(
    HWDS_NAME * const list,
    const mems_allocator * const allocator
)
{
    mems_allocator_free(allocator, list->items);
}

HWDS_EXPORT hwds_bool HWDS_NAME_(ensure_capacity_alloc)(
    HWDS_NAME * const list,
    const mems_allocator * const allocator,
    const hwds_size items_capacity
)
{
    return HWDS_NAME_(ensure_capacity_align_alloc)(
        list,
        allocator,
        HWDS_ALIGNOF(HWDS_ITEM),
        items_capacity
    );
}

HWDS_EXPORT hwds_bool HWDS_NAME_(ensure_capacity_align_alloc)(
    HWDS_NAME * const list,
    const mems_allocator * const allocator,
    const hwds_size alignment,
    const hwds_size items_capacity
)
{
    if (list->capacity >= items_capacity) return hwds_true;
    const hwds_size new_capacity = HWDS_NAME_(grow_capacity)(list, items_capacity);
    HWDS_ITEM * const new_items = (HWDS_ITEM *)mems_allocator_realloc(
        allocator,
        (void *)list->items,
        alignment,
        new_capacity * sizeof(HWDS_ITEM)
    );
    if (new_items == HWDS_NULL) return hwds_false;

    list->capacity = new_capacity;
    list->items = new_items;

    return hwds_true;
}

HWDS_EXPORT void HWDS_NAME_(expand_to_capacity)(HWDS_NAME * const list)
{
    list->len = list->capacity;
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_align_alloc)(HWDS_NAME * const list, const mems_allocator * const allocator, const hwds_size alignment)
{
    HWDS_ITEM * item = HWDS_NAME_(append)(list);

    if (item == HWDS_NULL)
    {
        const hwds_size new_capacity = HWDS_NAME_(grow_additional_capacity)(list, 1) * sizeof(HWDS_ITEM);
        void * const new_buffer = mems_allocator_realloc(
            allocator,
            (void *)list->items,
            alignment,
            new_capacity
        );
        if (new_buffer == HWDS_NULL) return HWDS_NULL;

        HWDS_NAME_(update_buffer)(list, new_capacity, new_buffer);
        item = HWDS_NAME_(append)(list);
    }

    return item;
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_alloc)(HWDS_NAME * const list, const mems_allocator * const allocator)
{
    return HWDS_NAME_(append_align_alloc)(list, allocator, HWDS_ALIGNOF(HWDS_ITEM));
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_many_align_alloc)(HWDS_NAME * const list, const mems_allocator * const allocator, const hwds_size alignment, const hwds_size n)
{
    HWDS_ITEM * items = HWDS_NAME_(append_many)(list, n);

    if (items == HWDS_NULL)
    {
        const hwds_size new_capacity = HWDS_NAME_(grow_additional_capacity)(list, n) * sizeof(HWDS_ITEM);
        void * const new_buffer = mems_allocator_realloc(
            allocator,
            (void *)list->items,
            alignment,
            new_capacity
        );
        if (new_buffer == HWDS_NULL) return HWDS_NULL;

        HWDS_NAME_(update_buffer)(list, new_capacity, new_buffer);
        items = HWDS_NAME_(append_many)(list, n);
    }

    return items;
}


HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_many_alloc)(HWDS_NAME * const list, const mems_allocator * const allocator, const hwds_size n)
{
    return HWDS_NAME_(append_many_align_alloc)(list, allocator, HWDS_ALIGNOF(HWDS_ITEM), n);
}

#endif // HWDS_MEMS_ENABLED

HWDS_EXPORT void HWDS_NAME_(reverse)(HWDS_NAME * list)
{
    for (hwds_size i = 0; i < list->len / 2; i++)
    {
        HWDS_ITEM tmp = list->items[i];
        list->items[i] = list->items[list->len - i - 1];
        list->items[list->len - i - 1] = tmp;
    }
}

HWDS_EXPORT void HWDS_NAME_(reset)(HWDS_NAME * list)
{
    list->len = 0;
}

HWDS_EXPORT hwds_size HWDS_NAME_(grow_capacity)(const HWDS_NAME * const list, const hwds_size capacity)
{
    const hwds_size minimum = HWDS_MAX(list->len, capacity);
    const hwds_size init_capacity = HWDS_MAX(1, 128 / sizeof(HWDS_ITEM));
    return minimum + (minimum / 2 + init_capacity);
}

HWDS_EXPORT hwds_size HWDS_NAME_(grow_additional_capacity)(const HWDS_NAME * const list, const hwds_size n)
{
    const hwds_size minimum = list->len + n;
    const hwds_size init_capacity = HWDS_MAX(1, 128 / sizeof(HWDS_ITEM));
    return minimum + (minimum / 2 + init_capacity);
}

#endif // HWDS_LIST_IMPLEMENTATION

#ifdef HWDS_SEGMENT_LIST_DECLARATION

#ifndef HWDS_SKIP_SEGMENTS
#   define HWDS_SKIP_SEGMENTS 6
#endif // HWDS_SKIP_SEGMENTS

#ifndef HWDS_SEGMENTS
#   define HWDS_SEGMENTS 26
#endif // HWDS_SEGMENTS

typedef struct HWDS_NAME
{
    hwds_size len;
    HWDS_ITEM * segments[HWDS_SEGMENTS];
    hwds_u8 used_segments;
} HWDS_NAME;

typedef struct HWDS_NAME_(iter)
{
    const HWDS_NAME * list;
    hwds_size index;
    hwds_size segment_capacity;
    hwds_size segment;
    hwds_size slot;
} HWDS_NAME_(iter);

HWDS_INLINE hwds_u8 HWDS_NAME_(segment_for_slot)(const hwds_size slot)
{
    return (hwds_u8)HWDS_LOG2F((float)((slot >> HWDS_SKIP_SEGMENTS) + 1));
}

HWDS_INLINE hwds_size HWDS_NAME_(capacity_for_segments)(const hwds_size segments)
{
    return (((hwds_size)1 << HWDS_SKIP_SEGMENTS) << segments) - ((hwds_size)1 << HWDS_SKIP_SEGMENTS);
}

HWDS_INLINE hwds_size HWDS_NAME_(capacity_for_segment)(const hwds_size segment)
{
    return ((hwds_size)1 << HWDS_SKIP_SEGMENTS) << segment;
}

HWDS_EXPORT void HWDS_NAME_(init)(HWDS_NAME * list);
HWDS_EXPORT void HWDS_NAME_(reset)(HWDS_NAME * list);

HWDS_EXPORT hwds_bool HWDS_NAME_(ensure_capacity)(
    HWDS_NAME * list,
    hwds_size items_capacity,
    hwds_size * required_size
);
HWDS_EXPORT hwds_bool HWDS_NAME_(ensure_capacity_align)(
    HWDS_NAME * list,
    hwds_size alignment,
    hwds_size items_capacity,
    hwds_size * required_size
);

HWDS_EXPORT hwds_u8 HWDS_NAME_(add_buffer)(
    HWDS_NAME * const list,
    hwds_size buffer_len,
    void * buffer
);
HWDS_EXPORT hwds_u8 HWDS_NAME_(add_buffer_align)(
    HWDS_NAME * list,
    hwds_size alignment,
    hwds_size buffer_len,
    void * buffer
);

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append)(HWDS_NAME * list);
HWDS_EXPORT hwds_bool HWDS_NAME_(append_many)(HWDS_NAME * list, hwds_size n, HWDS_NAME_(iter) * iter);
HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(get)(const HWDS_NAME * list, hwds_size index);
HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(pop)(HWDS_NAME * list);

HWDS_EXPORT void HWDS_NAME_(iter_init)(HWDS_NAME_(iter) * iter, const HWDS_NAME * list);
HWDS_EXPORT void HWDS_NAME_(iter_init_end)(HWDS_NAME_(iter) * iter, const HWDS_NAME * list);
HWDS_EXPORT void HWDS_NAME_(iter_init_at)(HWDS_NAME_(iter) * iter, const HWDS_NAME * list, hwds_size index);
HWDS_EXPORT hwds_bool HWDS_NAME_(iter_next)(HWDS_NAME_(iter) * iter, HWDS_ITEM ** item);

#ifdef HWDS_MEMS_ENABLED

HWDS_EXPORT hwds_bool HWDS_NAME_(ensure_capacity_alloc)(
    HWDS_NAME * list,
    const mems_allocator * allocator,
    hwds_size items_capacity
);

HWDS_EXPORT hwds_bool HWDS_NAME_(ensure_capacity_align_alloc)(
    HWDS_NAME * list,
    const mems_allocator * allocator,
    hwds_size alignment,
    hwds_size items_capacity
);

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_align_alloc)(HWDS_NAME * list, const mems_allocator * allocator, hwds_size alignment);
HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_alloc)(HWDS_NAME * list, const mems_allocator * allocator);
HWDS_EXPORT hwds_bool HWDS_NAME_(append_many_align_alloc)(HWDS_NAME * list, const mems_allocator * allocator, hwds_size alignment, hwds_size n, HWDS_NAME_(iter) * iter);
HWDS_EXPORT hwds_bool HWDS_NAME_(append_many_alloc)(HWDS_NAME * list, const mems_allocator * allocator, hwds_size n, HWDS_NAME_(iter) * iter);

#endif // HWDS_MEMS_ENABLED

#endif // HWDS_SEGMENT_LIST_DECLARATION

#ifdef HWDS_SEGMENT_LIST_IMPLEMENTATION

#ifndef HWDS_SKIP_SEGMENTS
#   define HWDS_SKIP_SEGMENTS 6
#endif // HWDS_SKIP_SEGMENTS

#ifndef HWDS_SEGMENTS
#   define HWDS_SEGMENTS 26
#endif // HWDS_SEGMENTS

HWDS_EXPORT void HWDS_NAME_(init)(HWDS_NAME * const list)
{
    *list = (HWDS_NAME){0};
}

HWDS_EXPORT void HWDS_NAME_(reset)(HWDS_NAME * const list)
{
    list->len = 0;
}

HWDS_EXPORT hwds_bool HWDS_NAME_(ensure_capacity)(
    HWDS_NAME * const list,
    const hwds_size items_capacity,
    hwds_size * const required_size
)
{
    return HWDS_NAME_(ensure_capacity_align)(
        list,
        HWDS_ALIGNOF(HWDS_ITEM),
        items_capacity,
        required_size
    );
}

HWDS_EXPORT hwds_bool HWDS_NAME_(ensure_capacity_align)(
    HWDS_NAME * const list,
    const hwds_size alignment,
    const hwds_size items_capacity,
    hwds_size * const required_size
)
{
    HWDS_ASSERT(alignment >= HWDS_ALIGNOF(HWDS_ITEM));
    if (items_capacity == 0) return hwds_true;

    const hwds_u8 segment = HWDS_NAME_(segment_for_slot)(items_capacity - 1);

    if (list->used_segments < segment + 1)
    {
        *required_size = 0;

        for (hwds_u8 i = list->used_segments; i < segment + 1; i ++)
        {
            *required_size = HWDS_ALIGN_FORWARD(*required_size, alignment);
            *required_size += HWDS_NAME_(capacity_for_segment)(i) * sizeof(HWDS_ITEM);
        }

        return hwds_false;
    }

    return hwds_true;
}

HWDS_EXPORT hwds_u8 HWDS_NAME_(add_buffer)(
    HWDS_NAME * const list,
    const hwds_size buffer_len,
    void * buffer
)
{
    return HWDS_NAME_(add_buffer_align)(
        list,
        HWDS_ALIGNOF(HWDS_ITEM),
        buffer_len,
        buffer
    );
}

HWDS_EXPORT hwds_u8 HWDS_NAME_(add_buffer_align)(
    HWDS_NAME * const list,
    const hwds_size alignment,
    const hwds_size buffer_len,
    void * buffer
)
{
    HWDS_ASSERT(alignment >= HWDS_ALIGNOF(HWDS_ITEM));

    hwds_u8 added_buffers = 0;
    hwds_size offset = 0;
    hwds_u8 * bytes = (hwds_u8 *)buffer;

    for (hwds_u8 i = list->used_segments; i < HWDS_SEGMENTS; i++)
    {
        offset = HWDS_ALIGN_FORWARD(offset, alignment);
        const hwds_size segment_bytes = HWDS_NAME_(capacity_for_segment)(i) * sizeof(HWDS_ITEM);

        if (offset + segment_bytes > buffer_len) break;

        list->segments[i] = (HWDS_ITEM *)(bytes + offset);
        list->used_segments++;
        offset += segment_bytes;
        added_buffers++;
    }

    return added_buffers;
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append)(HWDS_NAME * const list)
{
    const hwds_u8 segment = HWDS_NAME_(segment_for_slot)(list->len);
    const hwds_size slot = list->len - HWDS_NAME_(capacity_for_segments)(segment);

    if (list->used_segments < segment + 1)
    {
        return HWDS_NULL;
    }

    list->len++;
    return &(list->segments[segment][slot]);
}

HWDS_EXPORT hwds_bool HWDS_NAME_(append_many)(
    HWDS_NAME * const list,
    const hwds_size n,
    HWDS_NAME_(iter) * const iter
)
{
    const hwds_u8 end_segment = HWDS_NAME_(segment_for_slot)(list->len + n - 1);

    if (list->used_segments <= end_segment)
    {
        return hwds_false;
    }

    HWDS_NAME_(iter_init_at)(iter, list, list->len);
    list->len += n;

    return hwds_true;
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(get)(const HWDS_NAME * const list, const hwds_size index)
{
    if (index >= list->len) return HWDS_NULL;

    const hwds_u8 segment = HWDS_NAME_(segment_for_slot)(index);
    const hwds_size slot = index - HWDS_NAME_(capacity_for_segments)(segment);

    return &(list->segments[segment][slot]);
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(pop)(HWDS_NAME * const list)
{
    if (list->len == 0) return HWDS_NULL;

    return HWDS_NAME_(get)(list, --list->len);
}

HWDS_EXPORT void HWDS_NAME_(iter_init)(HWDS_NAME_(iter) * const iter, const HWDS_NAME * const list)
{
    iter->list = list;
    iter->index = 0;
    iter->segment_capacity = HWDS_NAME_(capacity_for_segments)(1);
    iter->segment = 0;
    iter->slot = 0;
}

HWDS_EXPORT void HWDS_NAME_(iter_init_end)(HWDS_NAME_(iter) * const iter, const HWDS_NAME * const list)
{
    HWDS_NAME_(iter_init_at)(iter, list, list->len);
}

HWDS_EXPORT void HWDS_NAME_(iter_init_at)(HWDS_NAME_(iter) * const iter, const HWDS_NAME * const list, hwds_size index)
{
    index = HWDS_MIN(index, list->len);
    const hwds_u8 segment = HWDS_NAME_(segment_for_slot)(index);
    const hwds_size slot = index - HWDS_NAME_(capacity_for_segments)(segment);

    iter->list = list;
    iter->index = index;
    iter->segment_capacity = HWDS_NAME_(capacity_for_segment)(segment);
    iter->segment = segment;
    iter->slot = slot;
}

HWDS_EXPORT hwds_bool HWDS_NAME_(iter_next)(HWDS_NAME_(iter) * const iter, HWDS_ITEM ** const item)
{
    while (iter->index < iter->list->len)
    {
        while (iter->slot < iter->segment_capacity)
        {
            iter->index++;
            *item = &(iter->list->segments[iter->segment][iter->slot++]);
            return hwds_true;
        }

        iter->slot = 0;
        iter->segment++;
        iter->segment_capacity = HWDS_NAME_(capacity_for_segment)(iter->segment);
    }

    return hwds_false;
}

HWDS_EXPORT hwds_bool HWDS_NAME_(iter_previous)(HWDS_NAME_(iter) * const iter, HWDS_ITEM ** const item)
{
    while (iter->index > 0)
    {
        while (iter->slot > 0)
        {
            iter->index--;
            *item = &(iter->list->segments[iter->segment][--iter->slot]);
            return hwds_true;
        }

        iter->segment--;
        iter->segment_capacity = HWDS_NAME_(capacity_for_segment)(iter->segment);
        iter->slot = iter->segment_capacity;
    }

    return hwds_false;
}

#ifdef HWDS_MEMS_ENABLED

HWDS_EXPORT hwds_bool HWDS_NAME_(ensure_capacity_alloc)(
    HWDS_NAME * const list,
    const mems_allocator * const allocator,
    const hwds_size items_capacity
)
{
    return HWDS_NAME_(ensure_capacity_align_alloc)(
        list,
        allocator,
        HWDS_ALIGNOF(HWDS_ITEM),
        items_capacity
    );
}

HWDS_EXPORT hwds_bool HWDS_NAME_(ensure_capacity_align_alloc)(
    HWDS_NAME * const list,
    const mems_allocator * const allocator,
    const hwds_size alignment,
    const hwds_size items_capacity
)
{
    hwds_size required_size;

    if (!HWDS_NAME_(ensure_capacity_align)(
        list,
        alignment,
        items_capacity,
        &required_size
    ))
    {
        void * const buffer = mems_allocator_alloc(allocator, alignment, required_size);
        if (buffer == HWDS_NULL) return hwds_false;
        const hwds_u8 added_buffers = HWDS_NAME_(add_buffer_align)(
            list,
            alignment,
            required_size,
            buffer
        );
        HWDS_ASSERT(added_buffers > 0);
    }

    return hwds_true;
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_align_alloc)(
    HWDS_NAME * const list,
    const mems_allocator * const allocator,
    const hwds_size alignment
)
{
    if (!HWDS_NAME_(ensure_capacity_align_alloc)(
        list,
        allocator,
        alignment,
        list->len + 1
    ))
    {
        return HWDS_NULL;
    }

    list->len++;

    return HWDS_NAME_(get)(list, list->len - 1);
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append_alloc)(HWDS_NAME * const list, const mems_allocator * const allocator)
{
    return HWDS_NAME_(append_align_alloc)(list, allocator, HWDS_ALIGNOF(HWDS_ITEM));
}

HWDS_EXPORT hwds_bool HWDS_NAME_(append_many_align_alloc)(
    HWDS_NAME * const list,
    const mems_allocator * const allocator,
    const hwds_size alignment,
    const hwds_size n,
    HWDS_NAME_(iter) * const iter
)
{
    if (!HWDS_NAME_(ensure_capacity_align_alloc)(
        list,
        allocator,
        alignment,
        list->len + n
    ))
    {
        return hwds_false;
    }

    return HWDS_NAME_(append_many)(list, n, iter);
}

HWDS_EXPORT hwds_bool HWDS_NAME_(append_many_alloc)(
    HWDS_NAME * const list,
    const mems_allocator * const allocator,
    const hwds_size n,
    HWDS_NAME_(iter) * const iter
)
{
    return HWDS_NAME_(append_many_align_alloc)(
        list,
        allocator,
        HWDS_ALIGNOF(HWDS_ITEM),
        n,
        iter
    );
}

#endif // HWDS_MEMS_ENABLED

#endif // HWDS_SEGMENT_LIST_IMPLEMENTATION

#ifdef HWDS_SLOT_MAP_DECLARATION

typedef struct HWDS_NAME
{
    hwds_size capacity;
    hwds_size len;
    hwds_size free_len;
    hwds_u32 * generations;
    hwds_u32 * free_indices;
    HWDS_ITEM * items;
} HWDS_NAME;

HWDS_EXPORT hwds_result HWDS_NAME_(init)(
    HWDS_NAME * list,
    hwds_u32 capacity,
    hwds_size buffer_len,
    void * buffer,
    hwds_size * required_size
);

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append)(HWDS_NAME * list, hwds_u64 * id);

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(get)(HWDS_NAME * list, hwds_u64 id);
HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(get_index)(HWDS_NAME * list, hwds_u32 index);

HWDS_EXPORT void HWDS_NAME_(remove)(HWDS_NAME * list, hwds_u64 id);
HWDS_EXPORT void HWDS_NAME_(remove_index)(HWDS_NAME * list, hwds_u32 index);

#ifdef HWDS_MEMS_ENABLED

HWDS_EXPORT hwds_result HWDS_NAME_(init_alloc)(
    HWDS_NAME * list,
    hwds_u32 capacity,
    const mems_allocator * allocator
);

HWDS_EXPORT void HWDS_NAME_(deinit_alloc)(
    HWDS_NAME * list,
    const mems_allocator * allocator
);

#endif // HWDS_MEMS_ENABLED

#endif // HWDS_SLOT_MAP_DECLARATION

#ifdef HWDS_SLOT_MAP_IMPLEMENTATION

HWDS_EXPORT hwds_result HWDS_NAME_(init)(
    HWDS_NAME * const list,
    hwds_u32 capacity,
    const hwds_size buffer_len,
    void * const buffer,
    hwds_size * const required_size
)
{
    *required_size = 0;
    *required_size += capacity * sizeof(hwds_u32);
    *required_size += capacity * sizeof(hwds_u32);
    *required_size = MEMS_ALIGN_FORWARD(*required_size, MEMS_ALIGNOF(HWDS_ITEM));
    *required_size += capacity * sizeof(HWDS_ITEM);

    if (buffer_len < *required_size) return hwds_alloc;

    hwds_size buffer_offset = 0;

    *list = (HWDS_NAME){
        .capacity = capacity,
    };

    list->generations = (hwds_u32 *)((hwds_ptr)buffer + buffer_offset);

    buffer_offset += capacity * sizeof(hwds_u32);

    list->free_indices = (hwds_u32 *)((hwds_ptr)buffer + buffer_offset);

    buffer_offset += capacity * sizeof(hwds_u32);
    buffer_offset = MEMS_ALIGN_FORWARD(buffer_offset, MEMS_ALIGNOF(HWDS_ITEM));

    list->items = (HWDS_ITEM *)((hwds_ptr)buffer + buffer_offset);

    buffer_offset += capacity * sizeof(HWDS_ITEM);
    HWDS_ASSERT(buffer_offset == *required_size);

    return hwds_ok;
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(append)(HWDS_NAME * const list, hwds_u64 * const id)
{
    hwds_u32 * const free_index = list->free_len > 0 ?
        list->free_indices + --list->free_len :
        HWDS_NULL;
    
    hwds_u32 index = free_index != HWDS_NULL ?
        *free_index :
        list->len;

    if (index >= list->capacity) return HWDS_NULL;
    HWDS_ITEM * const item = list->items + index;

    if (free_index == HWDS_NULL)
    {
        list->len++;

        // start at generation 1 so that zero initialization is a dead handle
        list->generations[index] = 1;
    }

    *id = (hwds_u64)index | ((hwds_u64)list->generations[index] << 32);
    return item;
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(get)(HWDS_NAME * const list, const hwds_u64 id)
{
    const hwds_u32 generation = (hwds_u32)(id >> 32);
    const hwds_u32 index = (hwds_u32)(id & 0xFFFFFFFFu);
    if (generation != list->generations[index]) return HWDS_NULL;

    return list->items + index;
}

HWDS_EXPORT HWDS_ITEM * HWDS_NAME_(get_index)(HWDS_NAME * const list, const hwds_u32 index)
{
    return list->items + index;
}

HWDS_EXPORT void HWDS_NAME_(remove)(HWDS_NAME * const list, const hwds_u64 id)
{
    const hwds_u32 generation = (hwds_u32)(id >> 32);
    const hwds_u32 index = (hwds_u32)(id & 0xFFFFFFFFu);
    if (generation != list->generations[index]) return;

    HWDS_NAME_(remove_index)(list, index);
}

HWDS_EXPORT void HWDS_NAME_(remove_index)(HWDS_NAME * const list, const hwds_u32 index)
{
    list->generations[index]++;
    list->free_indices[list->free_len++] = index;
}

#ifdef HWDS_MEMS_ENABLED

HWDS_EXPORT hwds_result HWDS_NAME_(init_alloc)(
    HWDS_NAME * const list,
    const hwds_u32 capacity,
    const mems_allocator * const allocator
)
{
    hwds_size required_size;

    HWDS_ENSURE(HWDS_NAME_(init)(
        HWDS_NULL,
        capacity,
        0,
        HWDS_NULL,
        &required_size
    ) == hwds_alloc);

    void * const buffer = mems_allocator_alloc(allocator, MEMS_ALIGNOF(HWDS_ITEM), required_size);
    if (buffer == HWDS_NULL) return hwds_alloc;

    HWDS_ENSURE(HWDS_NAME_(init)(
        list,
        capacity,
        required_size,
        buffer,
        &required_size
    ) == hwds_ok)

    return hwds_ok;
}

HWDS_EXPORT void HWDS_NAME_(deinit_alloc)(
    HWDS_NAME * const list,
    const mems_allocator * const allocator
)
{
    mems_allocator_free(allocator, list->generations);
}

#endif // HWDS_MEMS_ENABLED

#endif // HWDS_SLOT_MAP_IMPLEMENTATION

#ifdef HWDS_HEAP_DECLARATION

typedef struct HWDS_NAME
{
    hwds_size len;
    hwds_size capacity;
    HWDS_ITEM * items;
} HWDS_NAME;

HWDS_EXPORT hwds_result HWDS_NAME_(init)(
    HWDS_NAME * heap,
    hwds_size capacity,
    hwds_size buffer_len,
    void * buffer,
    hwds_size * required_size
);

#ifdef HWDS_MEMS_ENABLED

HWDS_EXPORT hwds_result HWDS_NAME_(init_alloc)(
    HWDS_NAME * heap,
    hwds_size capacity,
    const mems_allocator * allocator
);

HWDS_EXPORT void HWDS_NAME_(deinit_alloc)(
    HWDS_NAME * heap,
    const mems_allocator * allocator
);

HWDS_EXPORT hwds_result HWDS_NAME_(push)(
    HWDS_NAME * heap,
    const HWDS_ITEM * item
);

HWDS_EXPORT const HWDS_ITEM * HWDS_NAME_(peek)(HWDS_NAME * heap);

HWDS_EXPORT hwds_bool HWDS_NAME_(pop)(HWDS_NAME * heap, HWDS_ITEM * item);

#endif // HWDS_MEMS_ENABLED

#endif // HWDS_HEAP_DECLARATION

#ifdef HWDS_HEAP_IMPLEMENTATION

HWDS_EXPORT hwds_result HWDS_NAME_(init)(
    HWDS_NAME * const heap,
    const hwds_size capacity,
    const hwds_size buffer_len,
    void * const buffer,
    hwds_size * const required_size
)
{
    // must be power of 2
    HWDS_ASSERT((capacity > 0) && capacity == (capacity & ~(capacity - 1)));

    *required_size = 0;
    *required_size += capacity * sizeof(HWDS_ITEM);

    if (buffer_len < *required_size) return hwds_alloc;

    *heap = (HWDS_NAME){
        .capacity = capacity,
        .items = (HWDS_ITEM *)buffer,
    };

    return hwds_ok;
}

HWDS_EXPORT hwds_result HWDS_NAME_(push)(
    HWDS_NAME * const heap,
    const HWDS_ITEM * const item
)
{
    if (heap->len >= heap->capacity) return hwds_alloc;

    heap->len++;
    heap->items[heap->len - 1] = *item;

    hwds_size current = heap->len - 1;
    while (
        current > 0 &&
        HWDS_COMPARE(&heap->items[HWDS_HEAP_PARENT(current)], item) > 0
    )
    {
        const HWDS_ITEM tmp = heap->items[HWDS_HEAP_PARENT(current)];
        heap->items[HWDS_HEAP_PARENT(current)] = heap->items[current];
        heap->items[current] = tmp;
        current = HWDS_HEAP_PARENT(current);
    }

    return hwds_ok;
}

HWDS_EXPORT const HWDS_ITEM * HWDS_NAME_(peek)(HWDS_NAME * const heap)
{
    if (heap->len == 0) return HWDS_NULL;
    return &heap->items[0];
}

static void HWDS_NAME_(heapify)(HWDS_NAME * const heap, const hwds_size index)
{
    if (heap->len <= 1) return;

    const hwds_size left = HWDS_HEAP_LEFT(index);
    const hwds_size right = HWDS_HEAP_RIGHT(index);
    hwds_size smallest = index;

    if (
        left < heap->len &&
        HWDS_COMPARE(&heap->items[left], &heap->items[index]) < 0
    )
    {
        smallest = left;
    }

    if (
        right < heap->len &&
        HWDS_COMPARE(&heap->items[right], &heap->items[smallest]) < 0
    )
    {
        smallest = right;
    }

    if (smallest != index)
    {
        const HWDS_ITEM tmp = heap->items[index];
        heap->items[index] = heap->items[smallest];
        heap->items[smallest] = tmp;
        HWDS_NAME_(heapify)(heap, smallest);
    }
}

HWDS_EXPORT hwds_bool HWDS_NAME_(pop)(HWDS_NAME * const heap, HWDS_ITEM * const item)
{
    if (heap->len == 0) return hwds_false;

    const HWDS_ITEM last_element = heap->items[heap->len - 1];
    *item = heap->items[0];
    heap->items[0] = last_element;

    heap->len--;
    
    HWDS_NAME_(heapify)(heap, 0);

    return hwds_true;
}

#ifdef HWDS_MEMS_ENABLED

HWDS_EXPORT hwds_result HWDS_NAME_(init_alloc)(
    HWDS_NAME * const heap,
    const hwds_size capacity,
    const mems_allocator * const allocator
)
{
    hwds_size required_size;

    HWDS_ENSURE(HWDS_NAME_(init)(
        HWDS_NULL,
        capacity,
        0,
        HWDS_NULL,
        &required_size
    ) == hwds_alloc);

    void * const buffer = mems_allocator_alloc(allocator, MEMS_ALIGNOF(HWDS_ITEM), required_size);
    if (buffer == HWDS_NULL) return hwds_alloc;

    HWDS_ENSURE(HWDS_NAME_(init)(
        heap,
        capacity,
        required_size,
        buffer,
        &required_size
    ) == hwds_ok);

    return hwds_ok;
}

HWDS_EXPORT void HWDS_NAME_(deinit_alloc)(
    HWDS_NAME * const heap,
    const mems_allocator * const allocator
)
{
    mems_allocator_free(allocator, heap->items);
}

#endif // HWDS_MEMS_ENABLED

#endif // HWDS_HEAP_IMPLEMENTATION

#undef HWDS_HM_DECLARATION
#undef HWDS_HM_IMPLEMENTATION

#undef HWDS_HEAP_DECLARATION
#undef HWDS_HEAP_IMPLEMENTATION

#undef HWDS_LIST_DECLARATION
#undef HWDS_LIST_IMPLEMENTATION

#undef HWDS_SEGMENT_LIST_DECLARATION
#undef HWDS_SEGMENT_LIST_IMPLEMENTATION

#undef HWDS_SLOT_MAP_DECLARATION
#undef HWDS_SLOT_MAP_IMPLEMENTATION


#undef HWDS_NAME
#undef HWDS_ITEM
#undef HWDS_KEY
#undef HWDS_VALUE
#undef HWDS_HASH
#undef HWDS_EQUAL
#undef HWDS_COMPARE
#undef HWDS_SEGMENTS
#undef HWDS_SKIP_SEGMENTS

#ifndef HWDS_LICENSE
#define HWDS_LICENSE

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

#endif // HWDS_LICENSE
