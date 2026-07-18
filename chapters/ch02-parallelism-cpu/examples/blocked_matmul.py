"""Cache blocking: why the SAME FLOPs run faster when you tile.

Run:  uv run python chapters/ch02-parallelism-cpu/examples/blocked_matmul.py
"""

import numpy as np
from numba import njit

from common import benchmark


@njit(cache=True)
def matmul_naive(A, B, C):
    n, k, m = A.shape[0], A.shape[1], B.shape[1]
    for i in range(n):
        for j in range(m):
            acc = 0.0
            for p in range(k):
                acc += A[i, p] * B[p, j]
            C[i, j] = acc


@njit(cache=True)
def matmul_blocked(A, B, C, bs):
    n, k, m = A.shape[0], A.shape[1], B.shape[1]
    C[:] = 0.0
    for ii in range(0, n, bs):
        for pp in range(0, k, bs):
            for jj in range(0, m, bs):
                for i in range(ii, min(ii + bs, n)):
                    for p in range(pp, min(pp + bs, k)):
                        a = A[i, p]
                        for j in range(jj, min(jj + bs, m)):
                            C[i, j] += a * B[p, j]


def main() -> None:
    n = 512
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    C = np.empty((n, n))
    flops = 2 * n**3

    # correctness first, always
    matmul_naive(A, B, C)
    np.testing.assert_allclose(C, A @ B, rtol=1e-9)
    matmul_blocked(A, B, C, 64)
    np.testing.assert_allclose(C, A @ B, rtol=1e-9)

    print(benchmark(lambda: matmul_naive(A, B, C), name="naive triple loop", flops=flops, repeats=3))
    for bs in (16, 32, 64, 128, 256):
        print(benchmark(lambda: matmul_blocked(A, B, C, bs), name=f"blocked bs={bs}", flops=flops, repeats=3))
    print(benchmark(lambda: A @ B, name="numpy (BLAS)", flops=flops))

    print(
        "\nSame 2n^3 FLOPs in every row. Blocking reuses each loaded tile ~bs times"
        "\nbefore evicting it, cutting DRAM traffic by ~bs. BLAS adds SIMD micro-"
        "\nkernels, packing, and multithreading — the remaining gap. In ch04 you"
        "\nbuild exactly this ladder again on a GPU with shared memory as the cache."
    )


if __name__ == "__main__":
    main()
