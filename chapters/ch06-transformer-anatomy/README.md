# ch06 · Transformer Anatomy & Profiling — *syllabus stub*

> **Week 7 · Hardware: local (MPS).** Generated when you reach it.

## What you'll learn

- Build a GPT from scratch (nanoGPT-scale: ~10M params) — attention, MLP,
  residuals, LayerNorm — from partial code, and train it on tiny Shakespeare
  on your Mac's GPU (MPS).
- `torch.profiler`: op-level traces, finding the real bottlenecks; compare your
  measured per-op time budget against your ch01 ★★★ analytic predictions.
- Where memory goes during training: parameters, gradients, optimizer states,
  activations — the accounting that motivates ch08–ch10.
- DataLoader tuning in anger (your ch02 C2.3 knowledge, applied).

## Planned exercises (preview)

1. Fill in causal self-attention (the einsum/matmul path, masking, scaling).
2. Fill in the training step; make loss go down on Shakespeare.
3. Profile: produce a trace, name the top-3 ops, compare vs analytic FLOPs.
4. Memory audit: measure and explain peak memory vs the formula.

## Get a head start

- Karpathy, ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) (video) & [nanoGPT](https://github.com/karpathy/nanoGPT)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
