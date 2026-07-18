# ch09 · Distributed Training I: Data Parallelism — *syllabus stub*

> **Week 10 · Hardware: local (multi-process gloo — really!).** Generated when you reach it.

Distributed training concepts don't need a GPU cluster: `torch.distributed` with
the `gloo` backend runs real collectives across processes on your Mac. Same code,
same bugs, same mental model as 1000 GPUs.

## What you'll learn

- The collectives: broadcast, all-reduce, reduce-scatter, all-gather — run each,
  then *implement ring all-reduce yourself* and verify its 2(N-1)/N bandwidth math.
- Data parallelism: replicate model, shard data, all-reduce gradients — build a
  minimal DDP by hand (ch02's split→map→reduce, now with gradients), then compare
  with `torch.nn.parallel.DistributedDataParallel`.
- ZeRO stages 1–3 / FSDP: shard optimizer states, gradients, parameters; the
  memory arithmetic per stage.
- Communication/computation overlap: why DDP buckets gradients.

## Planned exercises (preview)

1. Ring all-reduce from send/recv primitives; verify against `dist.all_reduce`.
2. Hand-rolled DDP training of the ch06 GPT on 4 CPU processes; loss-curve parity
   with single-process training at the same effective batch.
3. Memory accounting: implement the ZeRO-stage memory formula and check it.

## Get a head start

- [ZeRO paper](https://arxiv.org/abs/1910.02054); PyTorch DDP + FSDP docs
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook), data-parallel sections
