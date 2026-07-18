"""Benchmarking harness used throughout the curriculum.

Design principles (these are taught in ch00, and every chapter relies on them):

1.  **Warm up** before timing: first calls pay one-time costs (allocation,
    JIT/compile, cache population, GPU kernel launch setup).
2.  **Repeat** and report the **median**, not the mean: timing distributions are
    right-skewed (interrupts, thermal throttling), so the median is the robust
    estimate of "typical" cost and the IQR shows stability.
3.  **Synchronize accelerators**: GPU/MPS work is asynchronous — a kernel call
    returns before the work finishes. Timing without a sync measures nothing.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable


def sync(device: str | None = None) -> None:
    """Block until pending accelerator work is done. No-op for CPU/None."""
    if device is None or device == "cpu":
        return
    import torch

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


@dataclass
class BenchResult:
    """Timing statistics for one benchmarked callable."""

    name: str
    times_s: list[float] = field(repr=False, default_factory=list)
    bytes_moved: int | None = None  # set to compute GB/s
    flops: int | None = None        # set to compute GFLOP/s

    @property
    def median_s(self) -> float:
        return statistics.median(self.times_s)

    @property
    def iqr_s(self) -> float:
        q = statistics.quantiles(self.times_s, n=4)
        return q[2] - q[0]

    @property
    def gb_per_s(self) -> float | None:
        if self.bytes_moved is None:
            return None
        return self.bytes_moved / self.median_s / 1e9

    @property
    def gflop_per_s(self) -> float | None:
        if self.flops is None:
            return None
        return self.flops / self.median_s / 1e9

    def __str__(self) -> str:
        parts = [f"{self.name:<32s} {self.median_s * 1e3:9.3f} ms  (IQR {self.iqr_s * 1e3:.3f} ms)"]
        if self.gb_per_s is not None:
            parts.append(f"{self.gb_per_s:8.1f} GB/s")
        if self.gflop_per_s is not None:
            parts.append(f"{self.gflop_per_s:8.1f} GFLOP/s")
        return "  ".join(parts)


def benchmark(
    fn: Callable[[], Any],
    *,
    name: str | None = None,
    warmup: int = 3,
    repeats: int = 10,
    device: str | None = None,
    bytes_moved: int | None = None,
    flops: int | None = None,
) -> BenchResult:
    """Time ``fn()`` robustly and return a :class:`BenchResult`.

    Example
    -------
    >>> import numpy as np
    >>> a = np.random.rand(1000, 1000); b = np.random.rand(1000, 1000)
    >>> r = benchmark(lambda: a @ b, name="matmul 1000^3", flops=2 * 1000**3)
    >>> print(r)  # doctest: +SKIP
    matmul 1000^3    3.1 ms (IQR 0.2 ms)  645.2 GFLOP/s
    """
    label = name or getattr(fn, "__name__", "anonymous")
    for _ in range(warmup):
        fn()
    sync(device)

    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)

    return BenchResult(name=label, times_s=times, bytes_moved=bytes_moved, flops=flops)
