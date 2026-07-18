# ch02 · Challenges

Commit evidence to `chapters/ch02-parallelism-cpu/results/`.

### ★ C2.1 — Scaling study of your π estimator
Run `parallel_pi` with 1, 2, 4, 8 workers (fixed total samples). Plot speedup and
efficiency; use ch01's `parallel_fraction` to infer the serial fraction. Then
*reduce* it (bigger chunks? fewer pickles?) and show the improved curve.
*Examined: strong scaling methodology + acting on an Amdahl diagnosis.*

### ★★ C2.2 — Block-size sweep, explained by cache sizes
Sweep block sizes {8..256} for your `blocked_matmul` at n=512 and plot GFLOP/s vs
block size. Compute the working-set size (3 tiles of bs² float64s) for each bs and
overlay your machine's L1/L2 sizes on the plot. Does the optimum sit where theory
says it should? Write 3–5 sentences.
*Examined: connecting a measured optimum to the memory hierarchy.*

### ★★ C2.3 — DataLoader anatomy
Build a toy input pipeline: `load_sample(i)` that does 5 ms of fake decode work
(`time.sleep` + a small NumPy op), consumed by a training loop needing a batch of
32 every 50 ms. Compare: serial loading, `ThreadPoolExecutor`, `ProcessPoolExecutor`,
and a background *prefetching* thread with a `queue.Queue`. Report steady-state
step time for each and explain which quadrant of the GIL demo each lands in.
*Examined: exactly why `DataLoader(num_workers=N)` exists — you'll meet it again in ch06.*

### ★★★ C2.4 — Parallel prefix sum (scan)
Implement an O(n) *parallel* inclusive prefix-sum with numba: chunk the array,
scan each chunk in prange, scan the per-chunk totals serially, then add offsets
in a second prange pass. Verify against `np.cumsum`, benchmark vs it, and plot
scaling. Bonus: where does a scan show up in LLM inference? (Hint: cumulative
probabilities in sampling; also state-space models.)
*Examined: designing a two-phase parallel algorithm with a sequential dependency — the hardest pattern in this chapter.*
