"""Measure real kernels and place them on a roofline plot for THIS machine.

Run:  uv run python chapters/ch01-performance-fundamentals/examples/roofline.py
Writes: chapters/ch01-performance-fundamentals/results/roofline.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import benchmark

RESULTS = Path(__file__).parent.parent / "results"


def main() -> None:
    n = 4096
    x = np.random.rand(n * n).astype(np.float32)
    y = np.random.rand(n * n).astype(np.float32)
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)

    kernels = []

    # 1. Elementwise add: 1 FLOP per element, 12 bytes (read x, read y, write out).
    r = benchmark(lambda: x + y, name="add", flops=x.size, bytes_moved=12 * x.size)
    kernels.append(r)

    # 2. Dot product: 2 FLOPs per element, 8 bytes (read x, read y; scalar out).
    r = benchmark(lambda: x @ y, name="dot", flops=2 * x.size, bytes_moved=8 * x.size)
    kernels.append(r)

    # 3. tanh: ~10s of FLOPs per element (impl-dependent; call it 10), 8 bytes.
    r = benchmark(lambda: np.tanh(x), name="tanh(~10 flop/elt)", flops=10 * x.size, bytes_moved=8 * x.size)
    kernels.append(r)

    # 4. Big matmul: 2n^3 FLOPs, minimum traffic 3*n^2*4 bytes. AI in the hundreds.
    r = benchmark(lambda: a @ b, name=f"matmul {n}", flops=2 * n**3, bytes_moved=3 * n * n * 4, repeats=5)
    kernels.append(r)

    print(f"{'kernel':<20s} {'AI (FLOP/B)':>12s} {'GFLOP/s':>10s} {'GB/s':>8s}")
    for r in kernels:
        ai = r.flops / r.bytes_moved
        print(f"{r.name:<20s} {ai:12.2f} {r.gflop_per_s:10.1f} {r.gb_per_s:8.1f}")

    # Estimate this machine's two ceilings from the extremes we just measured:
    bw = max(r.gb_per_s for r in kernels)        # best achieved bandwidth
    peak = max(r.gflop_per_s for r in kernels)   # best achieved compute
    ridge = peak / bw
    print(f"\nachieved bandwidth ceiling ≈ {bw:.0f} GB/s, compute ceiling ≈ {peak:.0f} GFLOP/s")
    print(f"ridge point ≈ {ridge:.1f} FLOP/byte — kernels left of this are memory-bound")

    ais = np.logspace(-2, 3, 200)
    plt.figure(figsize=(7, 5))
    plt.loglog(ais, np.minimum(peak, ais * bw), "k-", lw=2, label="roofline (achieved ceilings)")
    for r in kernels:
        plt.loglog(r.flops / r.bytes_moved, r.gflop_per_s, "o", ms=9, label=r.name)
    plt.xlabel("arithmetic intensity (FLOP/byte)")
    plt.ylabel("GFLOP/s")
    plt.title("Measured roofline")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "roofline.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
