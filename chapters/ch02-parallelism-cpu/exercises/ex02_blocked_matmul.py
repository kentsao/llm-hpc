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
    # TODO(you): six loops. Get a 3x5 @ 5x2 case right before going big.
    raise NotImplementedError
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
    # TODO(you)
    raise NotImplementedError
    return C


def best_block_size(n: int = 384, candidates: tuple[int, ...] = (8, 16, 32, 64, 128)) -> int:
    """Time blocked_matmul on an (n,n) problem for each candidate block size and
    return the fastest. Remember ch00: warm up (JIT!) and take a median of >=3.
    """
    # TODO(you)
    raise NotImplementedError
