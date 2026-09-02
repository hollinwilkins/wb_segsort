#!/bin/sh

BK=./build-debug/wb_benchmark_kernel
COMMON="--seed 1 --runs 50 --keys 1048576 --sampler uniform"

# reg  — pure register + subgroup shuffle. M must be <= device subgroup (32 here). R unused.
$BK output/kernels/reg_n256_m32_block    --memory reg      --store block   --bin 8 --N 256 --M 32  --smem 16 --subgroups 32 $COMMON

# smem — pure shared-memory sort. WG packs multiple segments.
$BK output/kernels/smem_n256_m64_block   --memory smem     --store block   --bin 8 --N 256 --M 64  --smem 16 --subgroups 32 $COMMON

# hybrid — shuffle within subgroup, smem beyond. sg{R} tag => --subgroups and --R = 32.
$BK output/kernels/hybrid_n256_m64_block --memory hybrid   --store block   --bin 8 --N 256 --M 64  --smem 16 --subgroups 32 --R 32 $COMMON

# hybmerge — bb_segsort merge-path, one segment/workgroup. smem16k tag => --smem 16.
$BK output/kernels/hybmerge_n256_m128_block --memory hybmerge --store block --bin 8 --N 256 --M 128 --smem 16 --subgroups 32 --R 32 $COMMON
