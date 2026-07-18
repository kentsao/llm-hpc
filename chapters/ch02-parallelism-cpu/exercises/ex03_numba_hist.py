"""Exercise 3: fix a real race condition.

`histogram_racy` below is genuinely wrong: multiple prange iterations increment
the same bin concurrently and increments get lost. (Unlike `total += x`, numba
can NOT rescue an indexed `bins[b] += 1` — it isn't a recognizable reduction.)

Your job: the classic fix — per-thread partial histograms, then a reduction.
This split-accumulate-reduce pattern is exactly what gradient all-reduce does
across GPUs in ch09.
"""

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True)
def histogram_racy(x: np.ndarray, n_bins: int) -> np.ndarray:
    """BROKEN reference — do not fix this one; keep it as the exhibit."""
    bins = np.zeros(n_bins, dtype=np.int64)
    for i in prange(x.size):
        b = int(x[i] * n_bins)
        if b == n_bins:  # x == 1.0 edge case
            b -= 1
        bins[b] += 1  # RACE: lost updates under parallel execution
    return bins


@njit(parallel=True, cache=True)
def histogram_fixed(x: np.ndarray, n_bins: int, n_chunks: int = 8) -> np.ndarray:
    """Correct parallel histogram for x in [0, 1].

    Plan:
      1. partials = zeros((n_chunks, n_bins), int64)
      2. for c in prange(n_chunks):  process x[start:end] for that chunk,
         writing ONLY into partials[c]  (disjoint rows -> no race)
      3. return partials.sum(axis=0)   (the reduction)
    Chunk boundaries: split x.size as evenly as you can; make sure every element
    is counted exactly once (check start/end arithmetic for the last chunk).
    """
    # TODO(you)
    raise NotImplementedError


if __name__ == "__main__":
    x = np.random.rand(5_000_000)
    expected = np.bincount((x * 16).astype(np.int64).clip(0, 15), minlength=16)
    racy = histogram_racy(x, 16)
    fixed = histogram_fixed(x, 16)
    print(f"true total {x.size},  racy total {racy.sum()}  (lost {x.size - racy.sum()} updates!)")
    print(f"fixed matches: {np.array_equal(fixed, expected)}")
