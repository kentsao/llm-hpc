# ch11 · Inference Optimization — *syllabus stub*

> **Week 12 · Hardware: local (M1 shines here).** Generated when you reach it.

## What you'll learn

- The two phases of LLM inference: prefill (compute-bound) vs decode
  (memory-bound — roofline says a token can't be generated faster than the
  weights stream from memory; compute the tokens/s ceiling for your M1).
- **KV cache**: implement it in the ch06 GPT; measure the O(S²)→O(S) win.
- **Quantization**: int8/int4 weight quantization — implement absmax/group-wise
  quantization in NumPy, then run llama.cpp with Metal on your Mac and benchmark
  q8/q4 models against the bandwidth roofline.
- **Batching & PagedAttention** (concepts + simulator): continuous batching,
  KV-cache fragmentation, why vLLM wins.
- **Speculative decoding**: draft/verify, acceptance-rate math, expected speedup.

## Planned exercises (preview)

1. Add a KV cache to the ch06 GPT's `generate()`; verify identical outputs.
2. Group-wise int4 quantize/dequantize with max-error bounds; perplexity check.
3. Continuous-batching simulator: throughput vs naive static batching.
4. Speculative decoding on the ch06 GPT (small draft model), measure acceptance.

## Get a head start

- [vLLM/PagedAttention](https://arxiv.org/abs/2309.06180); [Speculative sampling](https://arxiv.org/abs/2302.01318)
- llama.cpp — build it on your Mac now for fun; quantization docs
