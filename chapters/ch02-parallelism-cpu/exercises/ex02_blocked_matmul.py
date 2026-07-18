"""Exercise 2: write the blocked (tiled) matmul yourself, then parallelize it.

You saw the finished version in examples/blocked_matmul.py. Here you write it
from the loop structure up — because in ch04 you'll write the same algorithm in
CUDA where nobody can help you debug it, and in ch07 FlashAttention IS this
trick applied to attention.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def blocked_matmul(A: np.ndarray, B: np.ndarray, bs: int) -> np.ndarray:
    """Tiled matmul: C = A @ B, float64, any (compatible, possibly non-square,
    not-multiple-of-bs) shapes.

    Structure: three BLOCK loops (ii, pp, jj stepping by bs) enclosing three
    ELEMENT loops bounded with min(.. + bs, dim). Accumulate into C in place.
    """
    n, k = A.shape
    k2, m = B.shape
    assert k == k2
    C = np.zeros((n, m))
    for ii in range(0, n, bs):
        for pp in range(0, k, bs):
            for jj in range(0, m, bs):
                for i in range(ii, min(ii + bs, n)):
                    for p in range(pp, min(pp + bs, k)):
                        a = A[i, p]
                        for j in range(jj, min(jj + bs, m)):
                            C[i, j] += a * B[p, j]
    return C


@njit(parallel=True, cache=True)
def blocked_matmul_parallel(A: np.ndarray, B: np.ndarray, bs: int) -> np.ndarray:
    """Same, but the OUTER i-block loop runs in prange.

    Think first: why is parallelizing ii safe, while parallelizing pp would be
    a race? Write your answer as a comment here.
    numba note: prange needs a plain range form — use prange(n_blocks) and
    compute ii = block_index * bs.
    """
    n, k = A.shape
    k2, m = B.shape
    assert k == k2
    C = np.zeros((n, m))
    # Parallelizing ii is safe: each i-block writes a disjoint set of C rows.
    # Parallelizing pp would race: every p-block accumulates into the SAME C[i, j].
    n_iblocks = (n + bs - 1) // bs
    for bi in prange(n_iblocks):
        ii = bi * bs
        for pp in range(0, k, bs):
            for jj in range(0, m, bs):
                for i in range(ii, min(ii + bs, n)):
                    for p in range(pp, min(pp + bs, k)):
                        a = A[i, p]
                        for j in range(jj, min(jj + bs, m)):
                            C[i, j] += a * B[p, j]
    return C


def best_block_size(n: int = 384, candidates: tuple[int, ...] = (8, 16, 32, 64, 128)) -> int:
    """Time blocked_matmul on an (n,n) problem for each candidate block size and
    return the fastest. Remember ch00: warm up (JIT!) and take a median of >=3.
    """
    import statistics
    import time

    rng = np.random.default_rng(0)
    A = rng.random((n, n))
    B = rng.random((n, n))
    best_bs, best_t = candidates[0], float("inf")
    for bs in candidates:
        blocked_matmul(A, B, bs)  # warm-up (includes JIT on first call)
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            blocked_matmul(A, B, bs)
            times.append(time.perf_counter() - t0)
        t = statistics.median(times)
        if t < best_t:
            best_bs, best_t = bs, t
    return best_bs
