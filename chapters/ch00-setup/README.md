# ch00 · Setup, Tooling & Benchmarking Methodology

> **Week 1 · Hardware: local.** Goal: a working environment, a working GPU-cloud
> workflow, and — most importantly — the ability to produce *trustworthy numbers*.
> Every later chapter says "measure it"; this chapter is where you earn the right
> to believe your measurements.

## Concepts

### 1. Why benchmarking is hard

Run the same code twice and you'll get two different times. Sources of noise:

- **One-time costs**: memory allocation, `numba`/`torch.compile` JIT compilation,
  file-system caches, GPU context creation. → *Warm up before timing.*
- **Right-skewed noise**: OS scheduling, interrupts, thermal throttling make some runs
  *slower*, never faster. The mean is dragged up by outliers; the **median** is robust.
  → *Repeat ≥10×, report median + IQR.*
- **Asynchronous execution**: GPU (and Apple MPS) calls *enqueue* work and return
  immediately. `t1 - t0` around an unsynchronized kernel call times the *enqueue*,
  not the work — often 1000× off. → *Synchronize before stopping the clock.*
- **Dead-code elimination**: optimizers (and lazy frameworks) may skip work whose
  result is never used. → *Consume the output* (e.g., reduce it to a scalar).

The harness in [`common/bench.py`](../../common/bench.py) encodes all four rules. Read it —
it's ~100 lines and you'll use it for three months.

### 2. Two numbers that describe every kernel

For any piece of code, ask:

- **How many bytes** does it move between memory and the compute units?
- **How many FLOPs** (floating-point operations) does it perform?

Divide time into each and you get **GB/s** and **GFLOP/s** — the two axes of the
roofline model (ch01). Your M1 has roughly:

| Resource | M1 (8-core) approx. peak |
|---|---|
| Memory bandwidth | ~68 GB/s (unified memory) |
| CPU FP32 compute | ~100+ GFLOP/s (all cores, NEON) |
| GPU FP32 compute | ~2.6 TFLOP/s |

You will *measure* how close you can get to these — peak numbers on spec sheets are
marketing; *achievable* fractions of peak are engineering.

### 3. The GPU-cloud workflow (Colab)

Chapters 03–05, 07 need NVIDIA GPUs. The workflow, which you'll rehearse in this
chapter's ★★ challenge:

1. Open the chapter's `colab/*.ipynb` in [Google Colab](https://colab.research.google.com)
   (Runtime → Change runtime type → **T4 GPU**).
2. First cell clones this repo: `!git clone https://github.com/<your-user>/llm-hpc && %cd llm-hpc`.
3. Work in the notebook; copy final code/results back into the repo and commit locally.
   Treat Colab as a disposable execution venue, the repo as the source of truth.

## Example usage

```bash
uv sync                                            # one-time environment setup
uv run python chapters/ch00-setup/examples/hello_bench.py
uv run python chapters/ch00-setup/examples/mps_check.py
```

`hello_bench.py` demonstrates the harness on NumPy matmul; `mps_check.py` verifies
your Apple-GPU (MPS) backend works and shows why synchronization matters —
look at the "WRONG (no sync)" line in its output.

## Exercises

Fill in the `# TODO(you)` blocks in [`exercises/ex00_bandwidth.py`](exercises/ex00_bandwidth.py), then:

```bash
uv run pytest chapters/ch00-setup/tests -v
```

You'll implement the classic **STREAM triad** (`a = b + s*c`) and the bytes
accounting for it — your first real bandwidth measurement.

## Hints

<details><summary>Hint 1 — how many bytes does the triad move?</summary>

For each of the *n* elements: read `b[i]`, read `c[i]`, write `a[i]`.
That's 3 array-traversals of 8-byte float64s → `3 * n * 8` bytes.
(Real STREAM counts it exactly this way.)
</details>

<details><summary>Hint 2 — my bandwidth number looks too good</summary>

Is your array bigger than the CPU caches? An array that fits in the M1's
large caches measures *cache* bandwidth, not *memory* bandwidth. Use
arrays of at least ~100 MB.
</details>

<details><summary>Hint 3 — avoiding temporaries</summary>

`a[:] = b + s * c` allocates a temporary for `s * c` and another for the sum —
extra traffic! `np.multiply(c, s, out=a); a += b` does it with no temporaries.
Measure both; the difference *is* the lesson.
</details>

## Self-assessment checklist

- [ ] I can explain why we warm up, use medians, and synchronize.
- [ ] I know my machine's approximate peak memory bandwidth and measured % of it.
- [ ] I ran code on a free Colab T4 and got results back into this repo.
- [ ] `uv run pytest chapters/ch00-setup/tests` passes.

## References

- [STREAM benchmark](https://www.cs.virginia.edu/stream/) — McCalpin's original memory-bandwidth benchmark
- [Producing Wrong Data Without Doing Anything Obviously Wrong](https://dl.acm.org/doi/10.1145/1508244.1508275) — why measurement bias is real
- `torch.utils.benchmark` docs — PyTorch's own robust-timing utilities (we build a minimal version)
