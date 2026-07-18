# ch05 · Triton Kernels — *syllabus stub*

> **Week 6 · Hardware: Colab T4.** Generated when you reach it.

## What you'll learn

- The Triton programming model: block-level programs, `tl.load`/`tl.store` with
  masks, autotuning — CUDA's performance with 10× less code, and what the
  compiler now does for you (coalescing, shared memory) vs what you still own
  (tiling choices, masking, numerics).
- Writing the kernels PyTorch fuses badly: row softmax (online, numerically
  stable), LayerNorm (Welford), fused bias+GeLU.
- **Kernel fusion** as roofline surgery: three memory-bound ops → one pass over
  the data → ~3× fewer bytes → ~3× faster. You predicted this in ch01; now build it.

## Planned exercises (preview)

1. Fused softmax: fill in the load-mask-max-exp-sum-store pipeline.
2. LayerNorm forward (and a stretch: backward).
3. Fused bias+GeLU vs unfused PyTorch: measure the 
   bytes-moved ratio and confirm the speedup matches it.

## Get a head start

- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) 01–03 — we build on these directly
- GPU MODE lectures on Triton
