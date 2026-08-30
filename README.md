# Segmented Sort

Segmented sort are important sorting algorithms for multiple uses in computing: tile-based sorting for 2d renderers, ray tracing, (TODO: 1 more example).

Implementing these algorithms on CPU for a single thread of execution is straightforward. There is a reference implementation at [cpu.h](./cpu.h) implemented in ~100 lines of code.

Accelerating merge sort using SIMT hardware is a significantly different story however, and requires many hundreds of lines of code to do efficiently.
