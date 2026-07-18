"""Demonstrates the benchmarking harness on NumPy matmul.

Run:  uv run python chapters/ch00-setup/examples/hello_bench.py
"""

import numpy as np

from common import benchmark


def main() -> None:
    n = 1024
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)

    # An n^3 matmul does 2*n^3 FLOPs (n multiplies + n adds per output element).
    flops = 2 * n**3

    print(f"matmul {n}x{n} (float32), {flops / 1e9:.2f} GFLOP per call\n")

    r = benchmark(lambda: a @ b, name=f"numpy matmul {n}", flops=flops)
    print(r)

    # Same computation without the BLAS fast path — smaller size, or we'd wait all day:
    m = 512
    a2, b2 = a[:m, :m], b[:m, :m]
    r2 = benchmark(
        lambda: np.einsum("ik,kj->ij", a2, b2),
        name=f"einsum {m} (no BLAS path)",
        flops=2 * m**3,
        warmup=1,
        repeats=3,
    )
    print(r2)

    print(
        "\nTakeaway: identical math, wildly different throughput. The *algorithm's*"
        "\nFLOPs are fixed; the *implementation* decides how fast the hardware runs them."
    )


if __name__ == "__main__":
    main()
