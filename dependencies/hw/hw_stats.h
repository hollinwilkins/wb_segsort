// Copyright (C) 2026 Hollin Wilkins
// SPDX-License-Identifier: Zlib

#ifndef HWSTATS_H
#define HWSTATS_H

/*

hw_stats.h v0.1.0 - stats functions for random numbers and distribution sampling
by Hollin Wilkins

SECURITY
    This library provides NO SECURITY GUARANTEES whatsoever.
*/

#ifndef HWSTATS_NO_CRT
#   include <stddef.h>
#   include <stdint.h>
#   include <stdbool.h>
#   include <string.h>
#   include <math.h>
#   include <assert.h>
#endif

#ifndef HWSTATS_NO_CRT
#   ifndef hwstats_size
#       define hwstats_size size_t
#   endif
#   ifndef hwstats_ptr
#       define hwstats_ptr uintptr_t
#   endif
#   ifndef hwstats_u8
#       define hwstats_u8 uint8_t
#   endif
#   ifndef hwstats_u32
#       define hwstats_u32 uint32_t
#   endif
#   ifndef hwstats_u64
#       define hwstats_u64 uint64_t
#   endif
#   ifndef hwstats_bool
#       define hwstats_bool bool
#   endif
#   ifndef hwstats_true
#       define hwstats_true true
#   endif
#   ifndef hwstats_false
#       define hwstats_false false
#   endif
#   ifndef HWSTATS_NULL
#       define HWSTATS_NULL NULL
#   endif
#else
#ifndef hwstats_size
#   if defined(__SIZE_TYPE__)
#       define hwstats_size __SIZE_TYPE__
#   elif defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || \
    defined(__LP64__) || defined(_LP64) || (defined(__WORDSIZE) && __WORDSIZE == 64)
#       define hwstats_size unsigned long long
#   elif defined(__i386__) || defined(_M_IX86) || defined(__arm__) || \
    defined(__ILP32__) || (defined(__WORDSIZE) && __WORDSIZE == 32)
#       define hwstats_size unsigned int
#   else
#       define hwstats_size unsigned long
#   endif
#endif
#   ifndef hwstats_ptr
#       if defined(__UINTPTR_TYPE__)
#           define hwstats_ptr __UINTPTR_TYPE__
#       elif defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || \
        defined(__LP64__) || defined(_LP64) || defined(_WIND64)
#           define hwstats_ptr unsigned long long
#       elif defined(__i386__) || defined(_M_IX86) || defined(__arm__) || \
        defined(__ILP32__) || defined(_WIN32)
#           define hwstats_ptr unsigned int
#       else
#           define hwstats_ptr unsigned long
#       endif
#   endif
#   ifndef hwstats_u8
#      define hwstats_u8 unsigned char
#   endif
#   ifndef hwstats_u32
#      define hwstats_u32 unsigned
#   endif
#   ifndef hwstats_u64
#      define hwstats_u64 unsigned long long
#   endif
#   ifndef hwstats_bool
#       define hwstats_bool int
#   endif
#   ifndef hwstats_true
#       define hwstats_true 1
#   endif
#   ifndef hwstats_false
#       define hwstats_false 0
#   endif
#   ifndef HWSTATS_NULL
#      define HWSTATS_NULL 0
#   endif
#endif

#ifndef HWSTATS_EXPORT
#   ifdef HWSTATS_STATIC
#       define HWSTATS_EXPORT static
#   else
#       ifdef __cplusplus
#           define HWSTATS_EXPORT extern "C"
#       else
#           define HWSTATS_EXPORT extern
#       endif
#   endif
#endif

#ifdef _MSC_VER
#   define HWSTATS_INLINE static __forceinline
#   define HWSTATS_NOINLINE static __declspec(noinline)
#   define HWSTATS_ALIGNOF __alignof
#else
#   ifdef __has_attribute
#       if __has_attribute(always_inline)
#           define HWSTATS_INLINE static inline __attribute__((always_inline))
#       endif
#       if __has_attribute(noinline)
#           define HWSTATS_NOINLINE static __attribute__((noinline))
#       endif
#   endif
#   define HWSTATS_ALIGNOF __alignof__
#endif

typedef hwstats_u64 (*hwstats_rand_cb)(void * state);
typedef double (*hwstats_sample_cb)(void * state);

typedef struct hwstats_randomizer
{
    void * state;
    hwstats_rand_cb rand;
} hwstats_randomizer;

typedef struct hwstats_sampler
{
    void * state;
    hwstats_sample_cb sample;
} hwstats_sampler;

typedef struct hwstats_x256pp
{
    hwstats_u64 c[4];
} hwstats_x256pp;

typedef struct hwstats_normal
{
    hwstats_randomizer * r;
    hwstats_bool has_spare;
    double spare;
    double mean;
    double stddev;
} hwstats_normal;

typedef struct hwstats_beta
{
    hwstats_normal normal;
    double alpha;
    double beta;
} hwstats_beta;

typedef struct hwstats_mix2
{
    hwstats_randomizer * r;
    hwstats_sampler * sampler1;
    hwstats_sampler * sampler2;
    double weight;
} hwstats_mix2;

HWSTATS_EXPORT hwstats_u64 hwstats_rand(hwstats_randomizer * r);
HWSTATS_EXPORT double hwstats_uniform(hwstats_randomizer * r);

HWSTATS_EXPORT double hwstats_sample(hwstats_sampler * sampler);

HWSTATS_EXPORT void hwstats_x256pp_init(
    hwstats_x256pp * state,
    hwstats_u64 seed
);

HWSTATS_EXPORT void hwstats_x256pp_rand_init(
    hwstats_x256pp * state,
    hwstats_randomizer * r
);

HWSTATS_EXPORT void hwstats_normal_init_standard(hwstats_normal * state, hwstats_randomizer * r);
HWSTATS_EXPORT void hwstats_normal_init(hwstats_normal * state, hwstats_randomizer * r, double mean, double stddev);
HWSTATS_EXPORT double hwstats_normal_sample(hwstats_normal * state);
HWSTATS_EXPORT void hwstats_normal_sampler_init(
    hwstats_sampler * sampler,
    hwstats_normal * normal
);

HWSTATS_EXPORT void hwstats_beta_init(
    hwstats_beta * state,
    hwstats_randomizer * r,
    double alpha,
    double beta
);
HWSTATS_EXPORT double hwstats_beta_sample(hwstats_beta * state);
HWSTATS_EXPORT void hwstats_beta_sampler_init(
    hwstats_sampler * sampler,
    hwstats_beta * state
);

HWSTATS_EXPORT void hwstats_mix2_init(
    hwstats_mix2 * state,
    hwstats_randomizer * r,
    double weight,
    hwstats_sampler * sampler1,
    hwstats_sampler * sampler2
);
HWSTATS_EXPORT double hwstats_mix2_sample(hwstats_mix2 * state);
HWSTATS_EXPORT void hwstats_mix2_sampler_init(
    hwstats_sampler * sampler,
    hwstats_mix2 * state
);

#endif

#ifdef HWSTATS_IMPLEMENTATION
#ifndef HWSTATS_IMPLEMENTED
#define HWSTATS_IMPLEMENTED

HWSTATS_EXPORT hwstats_u64 hwstats_rand(hwstats_randomizer * const r)
{
    return r->rand(r->state);
}

HWSTATS_EXPORT double hwstats_uniform(hwstats_randomizer * const r)
{
    const hwstats_u64 x = hwstats_rand(r) >> 11;
    return ((double)x + 0.5) * (1.0 / 9007199254740992.0);
}

HWSTATS_EXPORT double hwstats_sample(hwstats_sampler * const sampler)
{
    return sampler->sample(sampler->state);
}

static hwstats_u64 hwstats__splitmix64(hwstats_u64 * const seed)
{
    hwstats_u64 z = (*seed += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

HWSTATS_EXPORT void hwstats_x256pp_init(
    hwstats_x256pp * const state,
    const hwstats_u64 seed
)
{
    hwstats_u64 _seed = seed;
    for (int i = 0; i < 4; i++)
    {
        state->c[i] = hwstats__splitmix64(&_seed);
    }
}

HWSTATS_INLINE hwstats_u64 hwstats__rotl(const hwstats_u64 x, const int k)
{
    return (x << k) | (x >> (64 - k));
}

static hwstats_u64 hwstats__x256pp_rand(void * const state)
{
    hwstats_x256pp * const _state = (hwstats_x256pp *)state;
    hwstats_u64 * const c = _state->c;

    const uint64_t result = hwstats__rotl(c[0] + c[3], 23) + c[0];
    const uint64_t t = c[1] << 17;
    c[2] ^= c[0];
    c[3] ^= c[1];
    c[1] ^= c[2];
    c[0] ^= c[3];
    c[2] ^= t;
    c[3] = hwstats__rotl(c[3], 45);
    return result;
}

HWSTATS_EXPORT void hwstats_x256pp_rand_init(
    hwstats_x256pp * const state,
    hwstats_randomizer * const r
)
{
    *r = (hwstats_randomizer){
        .state = (void *)state,
        .rand = hwstats__x256pp_rand,
    };
}

HWSTATS_EXPORT void hwstats_normal_init_standard(hwstats_normal * const state, hwstats_randomizer * const r)
{
    return hwstats_normal_init(state, r, 0.0, 1.0);
}


HWSTATS_EXPORT void hwstats_normal_init(
    hwstats_normal * const state,
    hwstats_randomizer * const r,
    const double mean,
    const double stddev
)
{
    *state = (hwstats_normal){
        .r = r,
        .mean = mean,
        .stddev = stddev,
    };
}

HWSTATS_EXPORT double hwstats_normal_sample(hwstats_normal * const state)
{
    if (state->has_spare)
    {
        state->has_spare = hwstats_false;
        return state->mean + state->stddev * state->spare;
    }

    double u, v, s;
    do {
        u = 2.0 * hwstats_uniform(state->r) - 1.0;
        v = 2.0 * hwstats_uniform(state->r) - 1.0;
        s = u * u + v * v;
    } while (s >= 1.0 || s == 0.0);

    const double factor = sqrt(-2.0 * log(s) / s);
    state->spare = v * factor;
    state->has_spare = hwstats_true;
    return state->mean + state->stddev * (u * factor);
}

static double hwstats__normal_sample(void * const state)
{
    return hwstats_normal_sample((hwstats_normal *)state);
}

HWSTATS_EXPORT void hwstats_normal_sampler_init(
    hwstats_sampler * const sampler,
    hwstats_normal * const normal
)
{
    *sampler = (hwstats_sampler){
        .sample = hwstats__normal_sample,
        .state = (void *)normal,
    };
}

HWSTATS_EXPORT void hwstats_beta_init(
    hwstats_beta * const state,
    hwstats_randomizer * const r,
    const double alpha,
    const double beta
)
{
    hwstats_normal normal;
    hwstats_normal_init_standard(&normal, r);

    *state = (hwstats_beta){
        .normal = normal,
        .alpha = alpha,
        .beta = beta,
    };
}

static double hwstats__gamma_sample(hwstats_normal * const n, const double k)
{
    if (k < 1.0)
    {
        const double u = hwstats_uniform(n->r);
        return hwstats__gamma_sample(n, k + 1.0) * pow(u, 1.0 / k);
    }

    const double d = k - 1.0 / 3.0;
    const double c = 1.0 / sqrt(9.0 * d);

    for (;;)
    {
        double x, v;
        do {
            x = hwstats_normal_sample(n);
            v = 1.0 + c * x;
        } while (v <= 0.0);

        v = v * v * v;
        const double u = hwstats_uniform(n->r);
        const double x2 = x * x;

        if (u < 1.0 - 0.0331 * x2 * x2) return d * v;
        if (log(u) < 0.5 * x2 + d * (1.0 - v + log(v))) return d * v;
    }
}

HWSTATS_EXPORT double hwstats_beta_sample(hwstats_beta * const state)
{
    const double ga = hwstats__gamma_sample(&state->normal, state->alpha);
    const double gb = hwstats__gamma_sample(&state->normal, state->beta);
    return ga / (ga + gb);
}

static double hwstats__beta_sample(void * const state)
{
    return hwstats_beta_sample((hwstats_beta *)state);
}

HWSTATS_EXPORT void hwstats_beta_sampler_init(
    hwstats_sampler * const sampler,
    hwstats_beta * const state
)
{
    *sampler = (hwstats_sampler){
        .sample = hwstats__beta_sample,
        .state = (void *)state,
    };
}

HWSTATS_EXPORT void hwstats_mix2_init(
    hwstats_mix2 * const state,
    hwstats_randomizer * const r,
    const double weight,
    hwstats_sampler * const sampler1,
    hwstats_sampler * const sampler2
)
{
    *state = (hwstats_mix2){
        .r = r,
        .weight = weight,
        .sampler1 = sampler1,
        .sampler2 = sampler2,
    };
}

HWSTATS_EXPORT double hwstats_mix2_sample(hwstats_mix2 * const state)
{
    hwstats_mix2 * const m = (hwstats_mix2 *)state;
    return hwstats_uniform(m->r) < m->weight ?
        hwstats_sample(m->sampler1) :
        hwstats_sample(m->sampler2);
}

static double hwstats__mix2_sample(void * const state)
{
    return hwstats_mix2_sample((hwstats_mix2 *)state);
}


HWSTATS_EXPORT void hwstats_mix2_sampler_init(
    hwstats_sampler * const sampler,
    hwstats_mix2 * const state
)
{
    *sampler = (hwstats_sampler){
        .sample = hwstats__mix2_sample,
        .state = state,
    };
}

#endif // HWSTATS_IMPLEMENTATION
#endif // HWSTATS_IMPLEMENTED

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
