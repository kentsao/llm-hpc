"""Python loop vs NumPy vectorization — feel the 100x.

Run:  uv run python chapters/ch01-performance-fundamentals/examples/vectorization.py
"""

import numpy as np

from common import benchmark


def gelu_loop(x: np.ndarray) -> np.ndarray:
    """GeLU (tanh approx) with a Python-level loop. Painfully slow on purpose."""
    out = np.empty_like(x)
    c = np.sqrt(2 / np.pi)
    for i in range(x.size):
        v = x.flat[i]
        out.flat[i] = 0.5 * v * (1 + np.tanh(c * (v + 0.044715 * v**3)))
    return out


def gelu_numpy(x: np.ndarray) -> np.ndarray:
    """Same math, one interpreter dispatch, compiled SIMD inner loop."""
    c = np.sqrt(2 / np.pi)
    return 0.5 * x * (1 + np.tanh(c * (x + 0.044715 * x**3)))


def main() -> None:
    x = np.random.randn(200_000).astype(np.float32)

    np.testing.assert_allclose(gelu_loop(x[:100]), gelu_numpy(x[:100]), rtol=1e-5)

    r_loop = benchmark(lambda: gelu_loop(x), name="gelu python loop", repeats=3)
    r_np = benchmark(lambda: gelu_numpy(x), name="gelu numpy", repeats=10)
    print(r_loop)
    print(r_np)
    print(f"\nspeedup: {r_loop.median_s / r_np.median_s:.0f}x")
    print("Same FLOPs. The loop pays ~100ns of interpreter overhead per element;")
    print("NumPy pays it once per array. The inner loop must live in compiled code.")


if __name__ == "__main__":
    main()
