# llm-hpc — High-Performance Computing for Large Language Models

A hands-on, 12-week self-study curriculum covering the systems side of modern LLMs:
from single-core performance fundamentals to GPU kernels, attention optimization,
distributed training, and inference serving.

Everything here is designed to run on either:

- **Local machine** — Apple Silicon Mac (CPU / MPS / `gloo` for distributed exercises), or
- **Free cloud GPUs** — Google Colab / Kaggle (T4) for chapters that need CUDA. Those
  chapters ship a `colab/` notebook.

## How this curriculum works

Each chapter is a directory under [`chapters/`](chapters/) containing:

| File / dir      | What it is |
|-----------------|------------|
| `README.md`     | Concepts, worked explanations, example usage, progressive hints, self-assessment checklist |
| `examples/`     | Complete, runnable reference code |
| `exercises/`    | Partial code with `# TODO(you)` markers for you to fill in |
| `tests/`        | `pytest` tests that verify your exercise solutions (they *skip* until you implement, then pass/fail) |
| `CHALLENGES.md` | 3–5 graded challenges: ★ warm-up → ★★★ research-flavored |
| `colab/`        | (GPU chapters only) notebook designed for a free Colab T4 |

**Workflow per chapter:**

1. Open the chapter's GitHub Issue and create a branch, e.g. `ch01-performance-fundamentals`.
2. Read `README.md`, run everything in `examples/`.
3. Fill in `exercises/` until `uv run pytest chapters/chNN-*/tests` passes.
4. Attempt the challenges; commit results (plots, notes, numbers) into the chapter's `results/` dir.
5. Tick boxes in [`PROGRESS.md`](PROGRESS.md), open a PR, let CI run, merge.

**Solutions policy:** reference solutions live on the `solutions` branch. Honor system —
only look after a real attempt. Struggling productively is the point.

## Roadmap

| Week | Chapter | Topic | Hardware |
|------|---------|-------|----------|
| 1  | [ch00](chapters/ch00-setup/) | Setup, tooling & benchmarking methodology | local |
| 2  | [ch01](chapters/ch01-performance-fundamentals/) | Performance fundamentals: roofline, Amdahl, vectorization, profiling | local |
| 3  | [ch02](chapters/ch02-parallelism-cpu/) | CPU parallelism: processes, threads, GIL, SIMD, cache blocking | local |
| 4  | [ch03](chapters/ch03-gpu-cuda-basics/) | GPU architecture & CUDA basics | Colab |
| 5  | [ch04](chapters/ch04-gemm-deep-dive/) | GEMM deep dive: naive → tiled → cuBLAS | Colab |
| 6  | [ch05](chapters/ch05-triton-kernels/) | Triton kernels: softmax, LayerNorm, fusion | Colab |
| 7  | [ch06](chapters/ch06-transformer-anatomy/) | Build & profile a GPT from scratch | local (MPS) |
| 8  | [ch07](chapters/ch07-attention-optimization/) | Attention optimization: online softmax, FlashAttention | Colab |
| 9  | [ch08](chapters/ch08-training-efficiency/) | Mixed precision, checkpointing, `torch.compile` | local + Colab |
| 10 | [ch09](chapters/ch09-distributed-data-parallel/) | Distributed training I: collectives, DDP, ZeRO/FSDP | local (gloo) |
| 11 | [ch10](chapters/ch10-distributed-model-parallel/) | Distributed training II: tensor & pipeline parallelism | local (gloo) |
| 12 | [ch11](chapters/ch11-inference-optimization/) | Inference: KV cache, quantization, batching, speculative decoding | local |
| 13+ | [ch12](chapters/ch12-capstone/) | Capstone: optimize a small LLM end-to-end + benchmark report | both |

Chapters 03–12 start as syllabus stubs and are fleshed out as the curriculum progresses,
so later material can build on measured results from earlier chapters.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Then:

```bash
uv sync            # creates .venv with Python 3.11 + all dependencies
uv run pytest      # run all tests (exercise tests skip until implemented)
```

Quick sanity check that the benchmarking harness works:

```bash
uv run python chapters/ch00-setup/examples/hello_bench.py
```

## Shared infrastructure

[`common/`](common/) holds the benchmarking harness used throughout:

- `common.bench.benchmark(fn)` — robust timing: warm-up, repeats, median/IQR, GB/s & GFLOP/s helpers
- `common.testing.run_exercise(fn, ...)` — calls exercise code, converts `NotImplementedError` into a pytest skip

## References (curriculum-wide)

- [Programming Massively Parallel Processors](https://www.elsevier.com/books/programming-massively-parallel-processors/kirk/978-0-323-91231-0) (Kirk & Hwu) — the GPU textbook
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) (He) — the mental model for ML perf
- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (Hugging Face) — distributed training
- [FlashAttention](https://arxiv.org/abs/2205.14135), [Megatron-LM](https://arxiv.org/abs/1909.08053), [ZeRO](https://arxiv.org/abs/1910.02054), [vLLM/PagedAttention](https://arxiv.org/abs/2309.06180) — the papers behind ch07–ch11
- [GPU MODE lectures](https://www.youtube.com/@GPUMODE) — kernel programming community
