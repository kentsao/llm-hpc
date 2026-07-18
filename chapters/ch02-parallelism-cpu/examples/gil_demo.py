"""Threads vs processes on CPU-bound vs GIL-releasing work — the four quadrants.

Run:  uv run python chapters/ch02-parallelism-cpu/examples/gil_demo.py
"""

import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np

N_WORKERS = 4


def pure_python_work(n: int = 2_000_000) -> int:
    """CPU-bound Python bytecode: holds the GIL the whole time."""
    total = 0
    for i in range(n):
        total += i * i
    return total


def numpy_work(n: int = 1200) -> float:
    """CPU-bound, but NumPy RELEASES the GIL inside its C kernels."""
    a = np.random.rand(n, n)
    for _ in range(6):
        a = a @ a
        a /= np.abs(a).max()
    return float(a.sum())


def run(pool_cls, fn, label: str) -> float:
    t0 = time.perf_counter()
    with pool_cls(N_WORKERS) as pool:
        list(pool.map(fn, [None] * N_WORKERS))
    return time.perf_counter() - t0


def serial(fn) -> float:
    t0 = time.perf_counter()
    for _ in range(N_WORKERS):
        fn(None)
    return time.perf_counter() - t0


def main() -> None:
    py = lambda _: pure_python_work()
    np_ = lambda _: numpy_work()

    for name, fn in [("pure-Python loop (holds GIL)", py), ("NumPy matmul (releases GIL)", np_)]:
        t_ser = serial(fn)
        t_thr = run(ThreadPoolExecutor, fn, "threads")
        t_prc = run(ProcessPoolExecutor, fn, "processes")
        print(f"\n{name}: {N_WORKERS} tasks")
        print(f"  serial     {t_ser:6.2f} s   (1.0x)")
        print(f"  threads    {t_thr:6.2f} s   ({t_ser / t_thr:4.1f}x)")
        print(f"  processes  {t_prc:6.2f} s   ({t_ser / t_prc:4.1f}x)")

    print(
        "\nExpected pattern: threads ~1x on pure Python (GIL serializes them),"
        "\nbut real speedup on NumPy work; processes speed up both, at the cost of"
        "\nprocess startup + pickling. Choose the tool by *where the GIL is held*."
    )


if __name__ == "__main__":
    main()
