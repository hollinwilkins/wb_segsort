# wb_segsort

wb_segsort aims to generate different GPU sorting algorithms in WebGPU and provide a benchmark framework for comparing them.
The sorting algorithms primarily come from these two sources.

1. Hou et. al: [Fast segmented sort on GPUs](https://dl.acm.org/doi/abs/10.1145/3079079.3079105). See [bb_segsort](https://github.com/vtsynergy/bb_segsort) reference implementation.
2. b0nes164's [GPUSorting](https://github.com/b0nes164/GPUSorting/).
3. dondragmer's [CuteSort.hlsl](https://gist.github.com/dondragmer/0c0b3eed0f7c30f7391deb11121a5aa1).

This project is very early, and while some of the kernels are quite fast, there is still a lot of work to be done to create an optimal segmented sort.

All notebooks, data collected, and code used in this project is available in this repository (no warranty or guarantees provided whatsoever).

Notebooks: [notebooks](./notebooks/)
Data (collected on MacBook M2): [output](./output/)
Kernel Generator: [kernel_generator.py](./kernel_generator.py)
Sorting Kernels: [shaders/sort_kernels](./shaders/sort_kernels/)

## Early Results

### Throughput

### Register vs Shared Memory

### CuteSort + merge vs Register Bitonic Sort + merge

## Summary of Kernels

Every kernel has a block and striped write mode. Striped mode coalesces writes to global memory at the expense of shuffling data in registers or shared memory.

| Family    | N Kernels | Description |
|-----------|-----------|-------------|
| reg       | 50        | register bitonic sort (subgroups) |
| smem      | 114       | shared-memory bitonic sort (workgroups) |
| hybrid    | 132       | register + shared-memory bitonic sort |
| hybmerge  | 102       | register + block merge sort |
| cute      | 10        | register radix sort |
| cutemerge | 24        | register radix sort + block merge sort |

All valid kernels are generated regardless of their expected efficiency for segment lengths up to 2048. Only the `hybmerge` and `cutemerge` kernels are generated for segment length 4096, as they have higher shared-memory requirements at that size.

### Kernel Generation

All generated kernels are checked in to source control here: [shaders/sort_kernels](./shaders/sort_kernels/). They are generated using a single Python script that can be found here: [kernel_generator.py](./kernel_generator.py).

- Kernels sort 32-bit keys and write 32-bit sorted value indices. This implies a memory bandwidth requirement of 12 bytes / key (1 key read, 1 key write, 1 value index write).
- All valid kernels are generated regardless of their expected efficiency for segment lengths __(N)__ up to 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, and 2048.
- Only the `hybmerge` and `cutemerge` kernels are generated for N=4096, as they have higher shared-memory requirements at that size.
- Kernels that require the subgroups feature are compiles for subgroups 8, 16, 32, 64 and 128. At runtime, a kernel with `subgroups <= device subgroups` can be selected.
- Number of work items per thread __(WPT)__ is used to select number of threads per segment __(M)__. WPT is targeted for 2, 4, 8, 16, and 32 work items.

## Roadmap

These are things that I want to implement for this project to make it feel more complete.

- Benchmarking for binning algorithm, which is probably too reliant on atomics in its current state.
- Benchmarking for the block merge spill bin, which is currently only lightly tested, and not benchmarked anywhere.
- Benchmarking on real-world data, which needs furuther explortion. Maybe a biology data set, NLP dataset, and a vector graphics dataset.
- Benchmarking against the state of the art, which would pretty much be incorporating wb_segsort into the [faster-segmented-sort-on-gpus](https://gitlab.rlp.net/pararch/faster-segmented-sort-on-gpus) benchmark suite. This will require benchmarking on different hardware on a Linux machine to support the CUDA algorithms.
- [Faster segmented sort on GPUs](https://link.springer.com/chapter/10.1007/978-3-031-39698-4_45) offers further refinements of the segmented sort presented by Hou et. al. We should incorporate these into wb_segsort.

## Interesting Future Work

These are some possible avenues of application and exploration that I find interesting.

- [GPUSorting](https://github.com/b0nes164/GPUSorting/) makes extensive use of radix sort to outperform both the low-level register sorting kernels and the block merges of Hou et. al, we should implement two new sorting kernels: `cuteradix` for fixed-size sorting kernels and `radixmerge` to compare against the multi-pass merge sort.
- Rust kernel driver for segmented sort.
- 16, 64, and 128 bit key/value sizes.
- Key-only sorting kernels.
- Automated script for determining best sorting kernels to use on a given hardware.
- Implementation for hardware, using either [Radiance](https://github.com/ucb-bar/radiance/) or [Vortex](https://github.com/vortexgpgpu/vortex/) as a platform.
- Implementing a vector graphics pipeline using [Sparse Strips](https://docs.google.com/document/d/1gEqf7ehTzd89Djf_VpkL0B_Fb15e0w5fuv_UzyacAPU/edit?tab=t.0#heading=h.27frrspcevv4) and Li et. al's [scanline rasterizer](http://kunzhou.net/zjugaps/pathrendering/) as a basis for the pipeline.
