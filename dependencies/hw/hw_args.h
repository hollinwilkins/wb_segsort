// Copyright (C) 2026 Hollin Wilkins
// SPDX-License-Identifier: Zlib

#ifndef HWARGS_H
#define HWARGS_H

/*

hw_stats.h v0.1.0 - stats functions for random numbers and distribution sampling
by Hollin Wilkins

SECURITY
    This library provides NO SECURITY GUARANTEES whatsoever.
*/

#ifndef HWARGS_NO_CRT
#   include <stddef.h>
#   include <stdint.h>
#   include <stdbool.h>
#   include <string.h>
#   include <math.h>
#   include <assert.h>
#endif

#ifndef HWARGS_NO_CRT
#   ifndef hwargs_size
#       define hwargs_size size_t
#   endif
#   ifndef hwargs_ptr
#       define hwargs_ptr uintptr_t
#   endif
#   ifndef hwargs_u8
#       define hwargs_u8 uint8_t
#   endif
#   ifndef hwargs_u32
#       define hwargs_u32 uint32_t
#   endif
#   ifndef hwargs_u64
#       define hwargs_u64 uint64_t
#   endif
#   ifndef hwargs_bool
#       define hwargs_bool bool
#   endif
#   ifndef hwargs_true
#       define hwargs_true true
#   endif
#   ifndef hwargs_false
#       define hwargs_false false
#   endif
#   ifndef HWARGS_NULL
#       define HWARGS_NULL NULL
#   endif
#else
#ifndef hwargs_size
#   if defined(__SIZE_TYPE__)
#       define hwargs_size __SIZE_TYPE__
#   elif defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || \
    defined(__LP64__) || defined(_LP64) || (defined(__WORDSIZE) && __WORDSIZE == 64)
#       define hwargs_size unsigned long long
#   elif defined(__i386__) || defined(_M_IX86) || defined(__arm__) || \
    defined(__ILP32__) || (defined(__WORDSIZE) && __WORDSIZE == 32)
#       define hwargs_size unsigned int
#   else
#       define hwargs_size unsigned long
#   endif
#endif
#   ifndef hwargs_ptr
#       if defined(__UINTPTR_TYPE__)
#           define hwargs_ptr __UINTPTR_TYPE__
#       elif defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || \
        defined(__LP64__) || defined(_LP64) || defined(_WIND64)
#           define hwargs_ptr unsigned long long
#       elif defined(__i386__) || defined(_M_IX86) || defined(__arm__) || \
        defined(__ILP32__) || defined(_WIN32)
#           define hwargs_ptr unsigned int
#       else
#           define hwargs_ptr unsigned long
#       endif
#   endif
#   ifndef hwargs_u8
#      define hwargs_u8 unsigned char
#   endif
#   ifndef hwargs_u32
#      define hwargs_u32 unsigned
#   endif
#   ifndef hwargs_u64
#      define hwargs_u64 unsigned long long
#   endif
#   ifndef hwargs_bool
#       define hwargs_bool int
#   endif
#   ifndef hwargs_true
#       define hwargs_true 1
#   endif
#   ifndef hwargs_false
#       define hwargs_false 0
#   endif
#   ifndef HWARGS_NULL
#      define HWARGS_NULL 0
#   endif
#endif

#ifndef HWARGS_EXPORT
#   ifdef HWARGS_STATIC
#       define HWARGS_EXPORT static
#   else
#       ifdef __cplusplus
#           define HWARGS_EXPORT extern "C"
#       else
#           define HWARGS_EXPORT extern
#       endif
#   endif
#endif

#ifdef _MSC_VER
#   define HWARGS_INLINE static __forceinline
#   define HWARGS_NOINLINE static __declspec(noinline)
#   define HWARGS_ALIGNOF __alignof
#else
#   ifdef __has_attribute
#       if __has_attribute(always_inline)
#           define HWARGS_INLINE static inline __attribute__((always_inline))
#       endif
#       if __has_attribute(noinline)
#           define HWARGS_NOINLINE static __attribute__((noinline))
#       endif
#   endif
#   define HWARGS_ALIGNOF __alignof__
#endif

typedef struct hwargs_param
{
    const char * name;
    const char * value;
} hwargs_param;

typedef struct hwargs_parsed
{
    const char * process;
    hwargs_size flags_len;
    const char ** flags;
    hwargs_size params_len;
    hwargs_param * params;
    hwargs_size positionals_len;
    const char ** positionals;
} hwargs_parsed;

HWARGS_EXPORT hwargs_bool hwargs_parse(
    hwargs_parsed * parsed,
    int argc, const char ** argv,
    hwargs_size buffer_len,
    void * buffer,
    hwargs_size * required_size
);

HWARGS_EXPORT hwargs_bool hwargs_has_flag(const hwargs_parsed * parsed, const char * name);
HWARGS_EXPORT const hwargs_param * hwargs_get_param(const hwargs_parsed * parsed, const char * name);

#endif // HWARGS_H

#ifdef HWARGS_IMPLEMENTATION
#ifndef HWARGS_IMPLEMENTATED
#define HWARGS_IMPLEMENTATED

typedef enum hwargs__state_kind
{
    hwargs__state_running = 0,
    hwargs__state_param = 1,
    hwargs__state_done = 2,
} hwargs__state_kind;

typedef union hwargs__state_data
{
    const char * param;
} hwargs__state_data;

typedef struct hwargs__state
{
    hwargs__state_kind kind;
    hwargs__state_data data;
} hwargs__state;

static hwargs_size hwargs__required_size(
    const int argc,
    const char ** const argv,
    hwargs_size * const flags_len,
    hwargs_size * const params_len,
    hwargs_size * const positionals_len,
    hwargs_size * const string_len
)
{
    *flags_len = 0;
    *params_len = 0;
    *positionals_len = 0;
    *string_len = 0;

    hwargs__state state = {0};

    for (int i = 0; i < argc; i++)
    {
        const char * const arg = argv[i];
        switch (state.kind)
        {
            case hwargs__state_running:
            {
                if (strncmp("--", arg, 2) == 0)
                {
                    (*params_len)++;
                    const char * const name = arg + 2;
                    *string_len += strlen(name) + 1;
                    state.kind = hwargs__state_param;
                    state.data.param = name;
                }
                else if (strncmp("-", arg, 1) == 0)
                {
                    (*flags_len)++;
                    const char * const name = arg + 1;
                    *string_len += strlen(name) + 1;
                }
                else
                {
                    (*positionals_len)++;
                    *string_len += strlen(arg) + 1;
                }
            } break;
            case hwargs__state_param:
            {
                *string_len += strlen(arg) + 1;
                state.kind = hwargs__state_running;
            } break;
            default: return 0;
        }
    }

    return (*flags_len) * sizeof(const char *) +
        (*positionals_len) * sizeof(const char *) +
        (*params_len) * sizeof(hwargs_param) +
        *string_len;
}

HWARGS_EXPORT hwargs_bool hwargs_parse(
    hwargs_parsed * const parsed,
    const int argc, const char ** const argv,
    const hwargs_size buffer_len,
    void * const buffer,
    hwargs_size * const required_size
)
{
    hwargs_size flags_len,
        positionals_len,
        params_len,
        required_string_size;

    *required_size = hwargs__required_size(
        argc, argv,
        &flags_len,
        &params_len,
        &positionals_len,
        &required_string_size
    );

    if (buffer_len < *required_size) return hwargs_false;

    *parsed = (hwargs_parsed){
        .flags_len = flags_len,
        .params_len = params_len,
        .positionals_len = positionals_len
    };

    hwargs_size buffer_offset = 0;

    parsed->flags = (const char **)((hwargs_ptr)buffer + buffer_offset);

    buffer_offset += flags_len * sizeof(const char *);

    parsed->positionals = (const char **)((hwargs_ptr)buffer + buffer_offset);

    buffer_offset += positionals_len * sizeof(const char *);

    parsed->params = (hwargs_param *)((hwargs_ptr)buffer + buffer_offset);

    buffer_offset += params_len * sizeof(hwargs_param);

    hwargs__state state = {0};

    hwargs_size flag_index = 0,
        positional_index = 0,
        param_index = 0;

    for (int i = 0; i < argc; i++)
    {
        const char * const arg = argv[i];
        switch (state.kind)
        {
            case hwargs__state_running:
            {
                if (strncmp("--", arg, 2) == 0)
                {
                    const char * const name = arg + 2;
                    state.kind = hwargs__state_param;
                    state.data.param = name;
                }
                else if (strncmp("-", arg, 1) == 0)
                {
                    const char * const name = arg + 1;
                    const hwargs_size name_len = strlen(name);
                    char * const copy_name = (char *)((hwargs_ptr)buffer + buffer_offset);
                    memcpy(copy_name, name, name_len);
                    copy_name[name_len] = 0;
                    parsed->flags[flag_index++] = copy_name;
                    buffer_offset += name_len + 1;
                }
                else
                {
                    const char * const name = arg;
                    const hwargs_size name_len = strlen(name);
                    char * const copy_name = (char *)((hwargs_ptr)buffer + buffer_offset);
                    memcpy(copy_name, name, name_len);
                    copy_name[name_len] = 0;
                    parsed->positionals[positional_index++] = copy_name;
                    buffer_offset += name_len + 1;
                }
            } break;
            case hwargs__state_param:
            {
                const char * const name = state.data.param;
                const char * const value = arg;
                const hwargs_size name_len = strlen(name);
                const hwargs_size value_len = strlen(value);
                char * const copy_name = (char *)((hwargs_ptr)buffer + buffer_offset);
                buffer_offset += name_len + 1;
                char * const copy_value = (char *)((hwargs_ptr)buffer + buffer_offset);
                buffer_offset += value_len + 1;
                memcpy(copy_name, name, name_len);
                memcpy(copy_value, value, value_len);
                copy_name[name_len] = 0;
                copy_value[value_len] = 0;
                parsed->params[param_index++] = (hwargs_param){ name, value };
                state.kind = hwargs__state_running;
            } break;
            default: return 0;
        }
    }

    if (state.kind != hwargs__state_running) return hwargs_false;

    return hwargs_true;
}


HWARGS_EXPORT hwargs_bool hwargs_has_flag(const hwargs_parsed * const parsed, const char * const name)
{
    for (hwargs_size i = 0; i < parsed->flags_len; i++)
    {
        if (strcmp(parsed->flags[i], name) == 0) return hwargs_true;
    }

    return hwargs_false;
}

HWARGS_EXPORT const hwargs_param * hwargs_get_param(const hwargs_parsed * const parsed, const char * const name)
{
    for (hwargs_size i = 0; i < parsed->params_len; i++)
    {
        if (strcmp(parsed->params[i].name, name) == 0) return parsed->params + i;
    }

    return HWARGS_NULL;
}

#endif // HWARGS_IMPLEMENTATION
#endif // HWARGS_IMPLEMENTED

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
