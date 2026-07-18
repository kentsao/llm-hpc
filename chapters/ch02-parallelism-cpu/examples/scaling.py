"""Strong-scaling curve: speedup vs worker count, and the Amdahl fit.

Run:  uv run python chapters/ch02-parallelism-cpu/examples/scaling.py
Writes: chapters/ch02-parallelism-cpu/results/scaling.png
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).parent.parent / "results"


def work_chunk(seed: int) -> float:
    """A fixed slice of a Monte-Carlo integral (compute-bound, no shared state)."""
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(20):
        x = rng.random(200_000)
        total += float(np.sqrt(1 - x * x).sum())
    return total


def timed(n_workers: int, n_chunks: int = 8) -> float:
    t0 = time.perf_counter()
    if n_workers == 1:
        for i in range(n_chunks):
            work_chunk(i)
    else:
        with ProcessPoolExecutor(n_workers) as pool:
            list(pool.map(work_chunk, range(n_chunks)))
    return time.perf_counter() - t0


def main() -> None:
    n_cores = os.cpu_count() or 8
    workers = [1, 2, 4, min(8, n_cores)]
    t1 = timed(1)
    print(f"{'workers':>8s} {'time':>8s} {'speedup':>8s} {'efficiency':>10s}")
    speedups = []
    for w in workers:
        t = timed(w)
        s = t1 / t
        speedups.append(s)
        print(f"{w:8d} {t:7.2f}s {s:7.2f}x {s / w:9.0%}")

    plt.figure(figsize=(6, 4))
    plt.plot(workers, speedups, "o-", label="measured")
    plt.plot(workers, workers, "k--", alpha=0.4, label="ideal")
    plt.xlabel("workers")
    plt.ylabel("speedup")
    plt.title("Strong scaling (Monte-Carlo chunks)")
    plt.legend()
    plt.grid(alpha=0.3)
    RESULTS.mkdir(exist_ok=True)
    plt.savefig(RESULTS / "scaling.png", dpi=120, bbox_inches="tight")
    print(f"\nwrote {RESULTS / 'scaling.png'}")
    print("Efficiency falls below 100%: pool startup, pickling, and (on M1) the")
    print("4 efficiency cores being slower than the 4 performance cores.")
    print("Exercise: use ch01's parallel_fraction() to infer the serial fraction here.")


if __name__ == "__main__":
    main()
