# ch12 · Capstone: Optimize a Small LLM End-to-End — *syllabus stub*

> **Weeks 13+ · Hardware: both.** Scoped together when you get here, based on
> which chapters you found most interesting — this is the portfolio piece.

## The shape of it

Take the ch06 GPT as the baseline and apply the full toolkit; every claim backed
by a measurement:

1. **Training track**: fused kernels (ch05), FlashAttention (ch07), AMP +
   checkpointing + compile (ch08), multi-process DDP (ch09). Target: a single
   table — tokens/s and memory, baseline → each optimization cumulatively.
2. **Inference track**: KV cache, quantization, batching, speculative decoding
   (ch11). Target: tokens/s and tokens/s/GB vs the roofline ceiling.
3. **Report**: a short paper-style write-up (abstract, method, results tables,
   roofline plots, honest limitations). Written to be readable by someone who
   has never seen this repo.

## Bar for "done"

- Reproducible: one script per experiment, seeds fixed, hardware documented.
- Honest: every speedup has an error bar and a "why" grounded in ch01's models.
- Readable: the report stands alone; plots have units; tables have baselines.
