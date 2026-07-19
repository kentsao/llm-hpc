"""Exercise 00: measure your machine's memory bandwidth with the STREAM triad.

The STREAM triad is the canonical memory-bandwidth benchmark:

    a[i] = b[i] + scalar * c[i]        for all i

It's *memory-bound*: 2 FLOPs per element but 24 bytes of traffic (read b, read c,
write a — 8 bytes each for float64). No CPU can compute slower than memory
delivers here, so its throughput ≈ your memory bandwidth.

SOLVED (ch00 complete). Three variants are kept side by side because they move
different amounts of *actual* memory per element — see results/parallel_result.md
for the accounting. `triad` (the exercise contract, tested) is the naive one.

Run the comparison:
    uv run python chapters/ch00-setup/exercises/ex00_bandwidth.py
"""

import numpy as np
from numba import njit

from common import benchmark


def triad(a: np.ndarray, b: np.ndarray, c: np.ndarray, scalar: float) -> None:
    """Naive NumPy triad, in place. Allocates temporaries: scalar*c makes one,
    b + tmp makes another, then the copy into a — 56 bytes/element of real
    traffic instead of STREAM's ideal 24."""
    a[:] = b + scalar * c


def triad_nontemp(a: np.ndarray, b: np.ndarray, c: np.ndarray, scalar: float) -> None:
    """Temporary-free NumPy: two passes, no allocations.
    multiply: read c + write a (16 B/elt); +=: read a, read b, write a (24 B/elt)
    -> 40 B/elt of real traffic."""
    np.multiply(c, scalar, out=a)
    a += b


@njit(parallel=True, cache=True)
def triad_numba(a: np.ndarray, b: np.ndarray, c: np.ndarray, scalar: float) -> None:
    """numba parfors FUSES the whole expression into one parallel loop:
    read b, read c, write a — the ideal 24 B/elt — and uses all cores.
    (Same source line as `triad`; the compiler, not the algorithm, changed.)"""
    a[:] = b + scalar * c


def triad_bytes(n: int, dtype: np.dtype) -> int:
    """Bytes of memory traffic per triad, STREAM convention (each array
    traversed once): read b, read c, write a."""
    return 3 * n * dtype.itemsize


def measure_bandwidth(n: int = 50_000_000, impl=triad) -> float:
    """Run a triad implementation on n float64s and return achieved GB/s
    (computed against STREAM-convention bytes, like the official benchmark)."""
    a = np.zeros(n)
    b = np.random.random(n)
    c = np.random.random(n)
    result = benchmark(
        lambda: impl(a, b, c, 2.5),
        name=getattr(impl, "__name__", "triad"),
        bytes_moved=triad_bytes(n, np.dtype(np.float64)),
    )
    return result.gb_per_s


if __name__ == "__main__":
    # actual traffic per element for each variant (see docstrings)
    actual_bytes = {"triad": 56, "triad_nontemp": 40, "triad_numba": 24}
    print(f"{'variant':<16} {'STREAM GB/s':>12} {'real GB/s':>10}   (real = STREAM x actual/24)")
    for impl in (triad, triad_nontemp, triad_numba):
        gbs = measure_bandwidth(impl=impl)
        real = gbs * actual_bytes[impl.__name__] / 24
        print(f"{impl.__name__:<16} {gbs:12.1f} {real:10.1f}")
    print("\nAll variants sustain roughly the same REAL bandwidth — the speedup")
    print("in the STREAM number comes from moving fewer bytes, not computing faster.")
