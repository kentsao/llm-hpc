# Result

I got three results. Actual byte move I think is 3 * n * 8 (float64) -> n is the
number of elements, in this exercise set to 50,000,000. One movement is read b,
read c (8 bytes each) and write a (8 bytes).

## Naive
STREAM triad bandwidth: 21.8 GB/s -> about 1/3 of SPEC

## Numpy Multiply
STREAM triad bandwidth: 30.0 GB/s -> about 1/2 of SPEC

## numba njit with naive
STREAM triad bandwidth: 47.0 GB/s -> about 2/3 of SPEC

# Why temporary-free version is faster
Temporary-free version is faster because that no extra data movement and can
reduce the overall time of computation.

# Challenges II & III
I think I know the bottleneck and context switch delay due to heavy load. Also,
the colab operation is fine for me. Therefore I skip these tutorials.

---

# Review feedback (corrections)

Verdict: `triad_bytes` = 3·n·8 ✓, harness usage ✓, all 4 tests pass ✓.
Two corrections below — the first one is the whole point of the exercise, so
read it carefully.

## 1. "numba njit with naive" is mislabeled — it is not naive

`@njit(parallel=True)` runs numba's *parfors* pass, which **fuses**
`a[:] = b + scalar * c` into a single parallel loop with **no temporaries**.
The compiler transformed your naive algorithm into the ideal one. The three
variants therefore move different amounts of *actual* memory per element:

| variant | actual traffic/elt | why | STREAM GB/s | real GB/s = STREAM × actual/24 |
|---|---|---|---|---|
| NumPy naive | 56 B | `scalar*c`→tmp1 (16), `b+tmp1`→tmp2 (24), copy→a (16) | 21.8 | **50.9** |
| NumPy temp-free | 40 B | `multiply(c,s,out=a)` (16) + `a += b` (24) | 30.0 | **50.0** |
| numba fused ∥ | 24 B | read b, read c, write a — one pass | 47.0 | **47.0** |

All three sustain ≈ 47–51 GB/s of *real* traffic (≈ 70–75% of the M1's 68 GB/s
spec — normal for STREAM). "No extra data movement" was the right instinct; the
correction is that this is the *entire* effect: the speedups come from moving
fewer bytes, not from computing faster. Memory-bound code speeds up only by
reducing bytes — that is the roofline lesson, and it is why kernel fusion
(ch05) and FlashAttention (ch07) exist.

The exercise file now keeps all three variants side by side
(`triad`, `triad_nontemp`, `triad_numba`); rerun the comparison with:

    uv run python chapters/ch00-setup/exercises/ex00_bandwidth.py

## 2. The committed Colab notebook was empty

`colab_env_for_ex00.ipynb` had no cells (probably committed before saving).
It now contains a minimal GPU-triad starter so the artifact matches its name.
Skipping C0.3/C0.4 is your call and fine — the Colab muscle gets mandatory in
ch03, so nothing is lost yet.

## Grading note

Your three-variant comparison **is** challenge C0.2's deliverable (variants ✓,
GB/s + % of spec ✓); the missing piece was the per-variant byte accounting,
which is supplied above — marked complete in PROGRESS.md.
