"""numba: compiled Python, parallel loops, and the race-condition trap.

Run:  uv run python chapters/ch02-parallelism-cpu/examples/numba_simd.py
"""

import numpy as np
from numba import njit, prange

from common import benchmark


def gelu_python(x):
    c = 0.7978845608028654  # sqrt(2/pi)
    out = np.empty_like(x)
    for i in range(x.size):
        v = x[i]
        out[i] = 0.5 * v * (1 + np.tanh(c * (v + 0.044715 * v**3)))
    return out


@njit(cache=True)
def gelu_njit(x):
    c = 0.7978845608028654
    out = np.empty_like(x)
    for i in range(x.size):  # same loop — but compiled + SIMD-vectorized
        v = x[i]
        out[i] = 0.5 * v * (1 + np.tanh(c * (v + 0.044715 * v**3)))
    return out


@njit(parallel=True, cache=True)
def gelu_parallel(x):
    c = 0.7978845608028654
    out = np.empty_like(x)
    for i in prange(x.size):  # parallel across cores, no GIL
        v = x[i]
        out[i] = 0.5 * v * (1 + np.tanh(c * (v + 0.044715 * v**3)))
    return out


@njit(parallel=True, cache=True)
def racy_sum_of_squares(x):
    """WRONG on purpose: prange iterations race on `total`... sometimes.

    (numba actually recognizes simple `total += ...` as a reduction and fixes it;
    the exercise shows a histogram, where it CANNOT, and the race is real.)
    """
    total = 0.0
    for i in prange(x.size):
        total += x[i] * x[i]
    return total


def main() -> None:
    x = np.random.randn(2_000_000).astype(np.float64)

    # First calls include JIT compilation — the harness's warmup absorbs it.
    r_py = benchmark(lambda: gelu_python(x[:50_000]), name="python loop (50k elts!)", repeats=3)
    r_jit = benchmark(lambda: gelu_njit(x), name="njit (2M elts)")
    r_par = benchmark(lambda: gelu_parallel(x), name="njit parallel (2M elts)")
    print(r_py)
    print(r_jit)
    print(r_par)

    per_elt_py = r_py.median_s / 50_000
    per_elt_jit = r_jit.median_s / 2_000_000
    print(f"\nper-element: python {per_elt_py * 1e9:.0f} ns  vs  njit {per_elt_jit * 1e9:.2f} ns "
          f"({per_elt_py / per_elt_jit:.0f}x)")
    print("parallel gains over njit are modest here — gelu at 2M elements is close")
    print("to memory-bound, so extra cores mostly wait on the same DRAM (roofline!).")


if __name__ == "__main__":
    main()
