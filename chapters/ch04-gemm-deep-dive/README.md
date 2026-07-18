# ch04 · GEMM Deep Dive — *syllabus stub*

> **Week 5 · Hardware: Colab T4.** Generated when you reach it.

## What you'll learn

- Why matmul is *the* kernel of deep learning (>90% of transformer FLOPs).
- The optimization ladder, each rung measured against cuBLAS:
  naive (1 thread = 1 output) → tiled with shared memory (ch02's blocking, now
  explicit) → register tiling / more work per thread → vectorized loads.
- Arithmetic-intensity analysis of each rung (ch01's roofline, applied for real).
- What cuBLAS/CUTLASS add beyond your best kernel; a first look at tensor cores.

## Planned exercises (preview)

1. Fill in the tiled shared-memory GEMM (the load-sync-compute-sync loop).
2. Add register tiling: each thread computes a 4×4 output patch.
3. Benchmark ladder + roofline placement; find your % of cuBLAS.

## Get a head start

- PMPP ch. 4–6 (tiling); [How to Optimize a CUDA Matmul Kernel](https://siboehm.com/articles/22/CUDA-MMM) (Boehm) — the exact ladder we climb
