# Merge Sort

Merge sort and segmented merge sort are important sorting algorithms for multiple uses in computing: tile-based sorting for 2d renderers, ray tracing, (TODO: 1 more example).
Implementing these algorithms on CPU for a single thread of execution is straightforward. You can even optimize the algorithm with SIMD to make it run significantly faster.
Accelerating merge sort using SIMT hardware is a significantly different story however, and requires many hundreds of lines of code to do efficiently.


