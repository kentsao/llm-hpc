# ch07 · Attention Optimization — *syllabus stub*

> **Week 8 · Hardware: Colab T4 (+ concepts locally).** Generated when you reach it.

## What you'll learn

- The quadratic problem: attention materializes an S×S matrix — compute the
  bytes (you did, in ch01 ex02!) and see why it's memory-bound.
- **Online softmax**: computing softmax in one pass with running max/sum — the
  algorithmic key that makes tiling attention possible.
- **FlashAttention**: tile Q/K/V (ch02's blocking + ch05's fusion), never write
  S to memory; derive the rescaling algebra and implement it in Triton.
- KV-cache mathematics for inference (bridge to ch11).

## Planned exercises (preview)

1. Online softmax in NumPy (runs locally — the algebra without the GPU).
2. Fill in a Triton FlashAttention forward (simplified: no dropout, causal).
3. Benchmark vs naive attention and `F.scaled_dot_product_attention` across
   sequence lengths; find the crossover.

## Get a head start

- [FlashAttention paper](https://arxiv.org/abs/2205.14135); [Online softmax](https://arxiv.org/abs/1805.02867)
- ELI5 walkthroughs of FlashAttention tiling (many good blog posts — collect your favorite)
