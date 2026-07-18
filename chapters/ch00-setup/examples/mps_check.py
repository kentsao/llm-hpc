"""Verify the Apple-GPU (MPS) backend and demonstrate WHY synchronization matters.

Run:  uv run python chapters/ch00-setup/examples/mps_check.py
"""

import time

import torch

from common import benchmark


def main() -> None:
    if not torch.backends.mps.is_available():
        print("MPS not available — running CPU only. (On Apple Silicon this should not happen.)")
        return

    n = 2048
    flops = 2 * n**3
    a_cpu = torch.rand(n, n)
    b_cpu = torch.rand(n, n)
    a_mps = a_cpu.to("mps")
    b_mps = b_cpu.to("mps")

    print(f"matmul {n}x{n} float32, {flops / 1e9:.1f} GFLOP per call\n")

    r_cpu = benchmark(lambda: a_cpu @ b_cpu, name="cpu matmul", flops=flops)
    print(r_cpu)

    # --- The trap: timing an async device without synchronizing -------------
    for _ in range(3):
        a_mps @ b_mps  # warm-up
    torch.mps.synchronize()
    t0 = time.perf_counter()
    a_mps @ b_mps  # returns as soon as the kernel is ENQUEUED
    t_wrong = time.perf_counter() - t0
    torch.mps.synchronize()
    print(f"{'WRONG (no sync)':<32s} {t_wrong * 1e3:9.3f} ms  {flops / t_wrong / 1e12:8.1f} 'TFLOP/s' (nonsense!)")

    # --- Correct: the harness syncs the device inside the timed region ------
    r_mps = benchmark(lambda: a_mps @ b_mps, name="mps matmul (synced)", flops=flops, device="mps")
    print(r_mps)

    print(
        f"\nGPU speedup over CPU: {r_cpu.median_s / r_mps.median_s:.1f}x"
        "\nTakeaway: the un-synced number above 'measured' only the kernel launch."
        "\nEvery GPU timing you ever report must bracket a synchronize."
    )


if __name__ == "__main__":
    main()
