"""Exercise 1: chunked parallel map + a parallel Monte-Carlo pi estimator.

The pattern (split -> map over a pool -> reassemble) is the CPU miniature of
data-parallel training: ch09's DDP is this exercise with gradients.
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Sequence

import numpy as np


def chunked_parallel_map(
    fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    n_workers: int = 4,
) -> np.ndarray:
    """Apply `fn` (an array -> array function that preserves length/order) to x
    in parallel and return results in the ORIGINAL order.

    Requirements:
    - Split x into exactly n_workers contiguous chunks (np.array_split handles
      uneven sizes) — one task per worker, not one per element (see Hint 1).
    - Use ProcessPoolExecutor; pool.map preserves order.
    - Concatenate and return.
    """
    # TODO(you)
    raise NotImplementedError


def _pi_chunk(args: tuple[int, int]) -> int:
    """Helper for one worker: (n_samples, seed) -> number of hits inside the
    quarter unit circle. Implement with a seeded np.random.default_rng —
    DIFFERENT seeds per worker, or all workers sample identical points!
    """
    # TODO(you)
    raise NotImplementedError


def parallel_pi(n_samples: int, n_workers: int = 4) -> float:
    """Estimate pi with n_samples total Monte-Carlo samples split across workers.

    pi ≈ 4 * (hits inside quarter circle) / n_samples.
    Distribute samples across workers (sum of per-worker samples must equal
    n_samples exactly), give each a distinct seed, sum the hits.
    """
    # TODO(you): build the (samples, seed) args list, map _pi_chunk over a pool.
    raise NotImplementedError


if __name__ == "__main__":
    import time

    for w in (1, 2, 4, 8):
        t0 = time.perf_counter()
        est = parallel_pi(20_000_000, n_workers=w)
        dt = time.perf_counter() - t0
        print(f"workers={w}: pi≈{est:.5f}  {dt:.2f}s")
    print("Now: use ch01's parallel_fraction() on these timings. How serial is this program?")
