# ch08 · Training Efficiency — *syllabus stub*

> **Week 9 · Hardware: local + Colab.** Generated when you reach it.

## What you'll learn

- Mixed precision: fp16/bf16 formats, loss scaling, what `autocast` actually does;
  measure step-time and memory on your ch06 GPT.
- Activation/gradient checkpointing: trade FLOPs for memory; derive and verify
  the memory formula.
- `torch.compile`: graph capture, kernel fusion — inspect the generated Triton
  (you can now *read* it, thanks to ch05).
- Gradient accumulation and the effective-batch-size algebra.

## Planned exercises (preview)

1. Add AMP to the ch06 training loop; verify convergence parity + measure gains.
2. Checkpoint the transformer blocks; plot memory vs seq-length before/after.
3. `torch.compile` the model; profile which ops fused; explain one fusion.

## Get a head start

- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- PyTorch docs: `torch.amp`, `torch.utils.checkpoint`, `torch.compile` tutorial
