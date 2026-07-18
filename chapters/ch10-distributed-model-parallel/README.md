# ch10 · Distributed Training II: Model Parallelism — *syllabus stub*

> **Week 11 · Hardware: local (gloo).** Generated when you reach it.

## What you'll learn

- **Tensor parallelism** (Megatron-style): column-parallel and row-parallel
  linear layers; why the MLP needs exactly one all-reduce forward and one
  backward; implement both over gloo and verify numerics vs a single-process run.
- **Pipeline parallelism**: split layers across stages, micro-batching, the
  bubble; derive bubble fraction = (p-1)/(m+p-1) and measure it in a simulation.
- Putting it together: 3D parallelism (DP × TP × PP) and how real systems
  (Megatron-LM, DeepSpeed) choose the split; communication-volume accounting.

## Planned exercises (preview)

1. Column/row-parallel `Linear` modules with correct autograd over gloo.
2. A 2-way tensor-parallel MLP block matching the serial reference bit-for-bit
   (well — float-for-float).
3. Pipeline schedule simulator: GPipe vs 1F1B, visualize the bubble.

## Get a head start

- [Megatron-LM paper](https://arxiv.org/abs/1909.08053) — §3 is the whole trick, and it's readable
- [GPipe](https://arxiv.org/abs/1811.06965), [PipeDream/1F1B](https://arxiv.org/abs/1806.03377)
