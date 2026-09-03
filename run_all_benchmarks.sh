#!/bin/sh

./run_kernel_benchmarks.sh output/kernels/keys_12400000 \
  --experiments experiments/kernel_experiments.csv \
  --keys 12400000 --runs 50 \
  --sampler "const(1.0)" \
  -skip-existing

./run_kernel_benchmarks.sh output/kernels/keys_102400000 \
  --experiments experiments/kernel_experiments.csv \
  --keys 102400000 --runs 50 \
  --sampler "const(1.0)" \
  -skip-existing

./run_kernel_benchmarks.sh output/kernels/keys_268435456 \
  --experiments experiments/kernel_experiments.csv \
  --keys 268435456 --runs 50 \
  --sampler "const(1.0)" \
  -skip-existing
