"""Exercise 00: measure your machine's memory bandwidth with the STREAM triad.

The STREAM triad is the canonical memory-bandwidth benchmark:

    a[i] = b[i] + scalar * c[i]        for all i

It's *memory-bound*: 2 FLOPs per element but 24 bytes of traffic (read b, read c,
write a — 8 bytes each for float64). No CPU can compute slower than memory
delivers here, so its throughput ≈ your memory bandwidth.

Fill in the TODOs, then run:
    uv run pytest chapters/ch00-setup/tests -v
Then run it as a script to see your numbers:
    uv run python chapters/ch00-setup/exercises/ex00_bandwidth.py
"""

import numpy as np

from common import benchmark


def triad(a: np.ndarray, b: np.ndarray, c: np.ndarray, scalar: float) -> None:
    """Compute a[i] = b[i] + scalar * c[i] IN PLACE (write into `a`).

    Constraints:
    - Vectorized NumPy only (no Python-level loop over elements).
    - Do not allocate a new output array; write into `a`.
      (Temporaries from `b + scalar * c` are acceptable for the basic version —
      the ★★ challenge asks you to eliminate them.)
    """
    a[:] = b + scalar * c


def triad_bytes(n: int, dtype: np.dtype) -> int:
    """Return the number of bytes of memory traffic one triad over n elements
    performs, using STREAM's counting convention (each array traversed once).
    """
    return 3 * n * dtype.itemsize  # read b, read c, write a


def measure_bandwidth(n: int = 50_000_000) -> float:
    """Run the triad on arrays of n float64s and return achieved GB/s.

    Steps:
      1. Allocate a, b, c as float64 arrays of length n (b, c random or ones).
      2. Use common.benchmark(...) to time `lambda: triad(a, b, c, 2.5)`,
         passing bytes_moved=triad_bytes(...).
      3. Return the result's .gb_per_s.
    """
    a = np.zeros(n)
    b = np.ones(n)
    c = np.ones(n)
    r = benchmark(
        lambda: triad(a, b, c, 2.5),
        name="stream triad",
        bytes_moved=triad_bytes(n, a.dtype),
    )
    return r.gb_per_s


if __name__ == "__main__":
    gbs = measure_bandwidth()
    print(f"STREAM triad bandwidth: {gbs:.1f} GB/s")
    print("Compare against your machine's spec-sheet bandwidth. What fraction did you achieve?")
