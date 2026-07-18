# ch01 · Performance Fundamentals

> **Week 2 · Hardware: local.** Goal: the three mental models that explain ~90% of
> all ML-systems performance: **Amdahl's law**, the **roofline model**, and
> **"vectorize or die"**. Plus the profiling skills to find where time actually goes.

## Concepts

### 1. Latency vs. throughput

- **Latency**: time for one item (one token, one request, one kernel). Seconds.
- **Throughput**: items per unit time (tokens/s, samples/s, GFLOP/s).

They trade off: batching raises throughput *and* per-item latency. LLM training
optimizes throughput; interactive LLM inference optimizes latency (per token).
Always say which one you mean.

### 2. Amdahl's law — the ceiling on parallel speedup

If a fraction *p* of a program's runtime can be parallelized (or otherwise sped up)
by a factor *s*, total speedup is:

```
speedup(p, s) = 1 / ((1 - p) + p / s)
```

The brutal consequence: with p = 90% and *infinite* parallel hardware, maximum
speedup is 10×. The serial 10% becomes the whole runtime. This is why "we bought
more GPUs but training barely got faster" happens: data loading, logging, Python
overhead — the serial parts — take over. You'll implement and *invert* this law
in exercise 1 (given a measured speedup, infer the serial fraction).

### 3. The roofline model — is your kernel memory-bound or compute-bound?

For a kernel, define **arithmetic intensity (AI)** = FLOPs performed / bytes moved
(units: FLOP/byte). Your machine has a peak compute rate `P` (GFLOP/s) and a peak
memory bandwidth `B` (GB/s). The best achievable performance is:

```
perf(AI) = min(P, AI × B)      ← the "roofline" (a slanted roof and a flat roof)
```

- AI < P/B (the *ridge point*): **memory-bound** — the compute units starve;
  optimizing math is pointless, reduce bytes (fusion, better dtypes, reuse).
- AI > P/B: **compute-bound** — worth optimizing the math (better algorithms,
  tensor cores, lower precision).

Why LLM people care: elementwise ops (bias, GeLU, residual add) have AI ≈ 0.1;
big matmuls have AI in the hundreds. Attention without FlashAttention is
memory-bound; FlashAttention's whole contribution (ch07) is *raising AI by not
writing the attention matrix to memory*. This one model organizes the entire
curriculum.

### 4. Vectorization — why Python loops lose 100×

A Python `for` loop pays interpreter dispatch (~50–100 ns) per element; NumPy
dispatches once per *array* and runs a compiled SIMD loop. The M1's NEON units
process 4 float32 per instruction per lane. Rule: *the inner loop must live in
compiled code* — NumPy, `numba`, or (later) a GPU kernel.

### 5. Profiling — measure before optimizing

- `cProfile` + `pstats`: function-level, low effort, always start here.
- `py-spy` (`uvx py-spy top -- python script.py`): sampling profiler, no code changes, flame graphs.
- Later: `torch.profiler` (ch06) for op-level GPU/MPS traces.

The discipline: profile → find the top item → predict what fixing it is worth
(Amdahl!) → fix → re-measure. Never optimize what you haven't measured.

## Example usage

```bash
uv run python chapters/ch01-performance-fundamentals/examples/vectorization.py
uv run python chapters/ch01-performance-fundamentals/examples/roofline.py     # writes results/roofline.png
uv run python chapters/ch01-performance-fundamentals/examples/profile_demo.py
```

## Exercises

```bash
uv run pytest chapters/ch01-performance-fundamentals/tests -v
```

1. [`ex01_amdahl.py`](exercises/ex01_amdahl.py) — implement Amdahl's law forward
   and *inverse* (infer serial fraction from a measurement).
2. [`ex02_roofline.py`](exercises/ex02_roofline.py) — compute FLOPs, bytes, and AI
   for the core LLM ops (elementwise, dot, matmul, attention scores) and classify
   each as memory- or compute-bound on given hardware.
3. [`ex03_vectorize.py`](exercises/ex03_vectorize.py) — three slow loopy functions
   (row softmax, pairwise distances, moving average); rewrite each in pure NumPy.
   Tests check correctness; a perf test checks you're ≥20× faster than the loop.

## Hints

<details><summary>Hint 1 — inverse Amdahl</summary>

Solve `S = 1 / ((1-p) + p/s)` for `p`:  `p = (1 - 1/S) / (1 - 1/s)`.
Sanity-check: measured speedup S=4 on s=8 workers → p ≈ 0.857.
</details>

<details><summary>Hint 2 — matmul FLOPs and bytes</summary>

`(M,K) @ (K,N)`: FLOPs = `2*M*K*N`. Minimum bytes (each matrix touched once) =
`(M*K + K*N + M*N) * dtype_size`. AI grows with size — that's why *big* matmuls
are compute-bound but tiny ones aren't.
</details>

<details><summary>Hint 3 — row softmax without loops</summary>

`x - x.max(axis=-1, keepdims=True)`, then `np.exp`, then divide by
`sum(axis=-1, keepdims=True)`. `keepdims=True` makes broadcasting line up.
</details>

<details><summary>Hint 4 — pairwise distances</summary>

`||xi - yj||² = ||xi||² + ||yj||² - 2·xi·yj`. The cross term is one matmul:
`x @ y.T`. Clip tiny negatives before `sqrt` (floating-point!).
</details>

<details><summary>Hint 5 — moving average</summary>

`np.cumsum` then subtract the shifted cumsum: window sums in O(n).
Or `np.convolve(x, np.ones(w)/w, mode="valid")`.
</details>

## Self-assessment checklist

- [ ] Given (FLOPs, bytes, hardware peaks) I can predict a kernel's max GFLOP/s and say which resource limits it.
- [ ] I can explain why fusing elementwise ops helps but "fusing" two big matmuls wouldn't.
- [ ] I can read a cProfile/py-spy output and name the top bottleneck of a program.
- [ ] All ch01 tests pass, including the ×20 vectorization perf bar (locally).

## References

- [Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785) (Williams et al.)
- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) — compute/memory/overhead trichotomy, LLM-flavored
- Hennessy & Patterson, *Computer Architecture*, Ch. 1 (quantitative principles)
