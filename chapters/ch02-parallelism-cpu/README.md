# ch02 · CPU Parallelism: Processes, Threads, SIMD & Caches

> **Week 3 · Hardware: local.** Goal: exploit all levels of parallelism a CPU
> offers — instruction-level (SIMD), core-level (threads/processes) — and meet
> the memory hierarchy, which decides whether parallel code is actually fast.
> Every concept here reappears on GPUs in ch03 at 100× the scale.

## Concepts

### 1. The parallelism hierarchy (one core is already parallel)

| Level | Mechanism | Typical win | GPU analogue (ch03) |
|---|---|---|---|
| Instruction | SIMD (NEON/AVX): one instruction, 4–16 elements | 4–16× | a warp's 32 lanes |
| Core | threads / processes across 4–8 P-cores | ~cores× | SMs |
| Node | multiple machines (ch09+) | ~machines× | multi-GPU |

The multiplicative view: a "serial" NumPy op may already use SIMD; parallel code
that abandons SIMD (e.g., naive Python threads) can *lose* to good serial code.

### 2. Threads vs processes, and the GIL

- **Threads** share one address space — cheap communication, but in CPython only
  one thread executes *Python bytecode* at a time (the **GIL**).
- **Processes** each have their own interpreter and memory — true parallelism for
  Python code, but arguments/results are *pickled* across process boundaries
  (expensive for big arrays!).

When threads still help: the GIL is *released* during blocking I/O and inside
most NumPy/PyTorch C kernels. So: **I/O-bound or NumPy-heavy → threads fine;
pure-Python CPU-bound → processes (or numba, which sidesteps the GIL entirely).**
`examples/gil_demo.py` demonstrates all four quadrants.

This matters directly for LLM work: PyTorch `DataLoader(num_workers=N)` uses
*processes* for exactly these reasons — and its overheads are these pickling costs.

### 3. numba — compiled, SIMD-vectorized, GIL-free Python

`@numba.njit` compiles a Python function to machine code (first call = compile,
remember your ch00 warm-up discipline!). `@njit(parallel=True)` + `numba.prange`
parallelizes loops across cores. This is our stand-in for OpenMP, and the closest
CPU experience to writing CUDA kernels: you write the *per-element* logic and
think about what each parallel worker does.

**Race conditions appear here for the first time**: two `prange` iterations
writing the same accumulator produce silently wrong results. The fix — per-thread
partial results, then reduce — is *the* pattern of parallel programming (and of
ch09's all-reduce).

### 4. The memory hierarchy & cache blocking

| Level (M1) | Size | Latency |
|---|---|---|
| L1 | 128 KB/P-core | ~3 cycles |
| L2 | 12 MB shared | ~18 cycles |
| DRAM | 8–16 GB | ~100+ ns |

Naive triple-loop matmul streams the same data from DRAM O(n) times. **Blocking
(tiling)** processes sub-matrices that fit in cache, so each block loaded from
DRAM is reused ~block_size times. This exact idea — copy a tile into fast memory,
compute on it a lot, write back — *is* the shared-memory GEMM of ch04 and the
core trick of FlashAttention (ch07). Learn it on the CPU where you can debug it.

## Example usage

```bash
uv run python chapters/ch02-parallelism-cpu/examples/gil_demo.py        # threads vs processes, 4 quadrants
uv run python chapters/ch02-parallelism-cpu/examples/scaling.py         # speedup curve vs worker count
uv run python chapters/ch02-parallelism-cpu/examples/numba_simd.py      # njit + prange + the race-condition trap
uv run python chapters/ch02-parallelism-cpu/examples/blocked_matmul.py  # cache blocking, block-size sweep
```

## Exercises

```bash
uv run pytest chapters/ch02-parallelism-cpu/tests -v
```

1. [`ex01_parallel_map.py`](exercises/ex01_parallel_map.py) — implement `chunked_parallel_map`
   (split work into per-worker chunks, process pool, reassemble in order) and a
   parallel Monte-Carlo π estimator on top of it.
2. [`ex02_blocked_matmul.py`](exercises/ex02_blocked_matmul.py) — fill in the tiled
   triple loop of a blocked matmul (numba), then find the best block size empirically.
3. [`ex03_numba_hist.py`](exercises/ex03_numba_hist.py) — fix a racy parallel
   histogram using per-thread partials + reduction.

## Hints

<details><summary>Hint 1 — chunking: why not one task per element?</summary>

Each task submitted to a pool pays pickling + scheduling overhead (~µs–ms). With
n tiny tasks the overhead dominates. Split the input into ~n_workers chunks so
each worker gets one big task. `np.array_split` handles the arithmetic.
</details>

<details><summary>Hint 2 — blocked matmul loop structure</summary>

Three *block* loops (ii, kk, jj in steps of B) around three *element* loops
(bounded by `min(ii+B, n)`). Accumulate into C in place: `C[i,j] += A[i,k]*B[k,j]`.
Verify with a tiny non-square case against `A @ B` first.
</details>

<details><summary>Hint 3 — which loop to prange?</summary>

Parallelize the outermost *i*-block loop: different i-blocks write disjoint rows
of C, so there's no race. Parallelizing the k-loop WOULD race (all k-blocks add
into the same C entries).
</details>

<details><summary>Hint 4 — racy histogram fix</summary>

Allocate `partials = np.zeros((n_threads, n_bins))`; each prange chunk writes only
`partials[thread_chunk_id]`; return `partials.sum(axis=0)`. In numba, get the
chunk id by iterating over chunks in prange, not elements.
</details>

## Self-assessment checklist

- [ ] I can predict whether threads or processes will speed up a given Python workload, before running it.
- [ ] I can explain my measured π-estimator speedup with Amdahl (serial fraction = ?).
- [ ] My blocked matmul beats the naive numba loop and I can explain the best block size via L1/L2 sizes.
- [ ] I found and fixed a race condition and can explain why partial+reduce is correct.

## References

- [Programming Massively Parallel Processors, ch. 4–5](https://www.sciencedirect.com/book/9780323912310/) — tiling (read now, reused in ch04)
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) (Drepper) — §3–4 on caches
- [numba parallel docs](https://numba.readthedocs.io/en/stable/user/parallel.html) — `prange`, race conditions, reductions
