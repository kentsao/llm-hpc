# ch03 · GPU Architecture & CUDA Basics — *syllabus stub*

> **Week 4 · Hardware: Colab T4.** Full chapter (examples, exercises, tests,
> notebook, challenges) is generated when you reach it — after finishing ch02.

## What you'll learn

- The GPU execution model: SMs, warps (32 threads in lockstep), thread blocks,
  grids — and how it maps onto ch02's cores/SIMD hierarchy.
- The GPU memory hierarchy: registers, shared memory, L2, HBM/GDDR — and why
  **coalesced** global-memory access is the first rule of kernel performance.
- Writing first kernels two ways: CUDA C++ (compiled in Colab with `nvcc`) and
  `numba.cuda` (same concepts, Python syntax).
- Launch configuration, occupancy, and timing GPU code correctly (ch00 discipline
  + CUDA events).

## Planned exercises (preview)

1. Vector add + SAXPY: grid-stride loops, bounds checks.
2. Coalescing lab: strided vs contiguous access, measure the bandwidth cliff.
3. Parallel reduction: naive atomic → shared-memory tree reduction (the GPU
   version of ch02's histogram fix).

## Get a head start

- PMPP (Kirk & Hwu) ch. 1–3
- [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- GPU MODE lecture 1–2
