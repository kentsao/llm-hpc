"""Exercise 1: Amdahl's law, forward and inverse.

speedup = 1 / ((1 - p) + p / s)
  p: parallelizable fraction of runtime (0..1)
  s: speedup applied to that fraction (e.g., number of workers)
"""


def amdahl_speedup(p: float, s: float) -> float:
    """Return the overall speedup when fraction p is accelerated by factor s."""
    return 1.0 / ((1.0 - p) + p / s)


def max_speedup(p: float) -> float:
    """Return the speedup ceiling with INFINITE parallel hardware (s -> inf).

    (Take the limit analytically; don't pass a huge number.)
    """
    if p >= 1.0:
        return float("inf")
    return 1.0 / (1.0 - p)


def parallel_fraction(measured_speedup: float, s: float) -> float:
    """Inverse problem: you measured `measured_speedup` using s workers.
    Return the implied parallelizable fraction p.

    This is what you'll actually do in practice: run on 1 and on 8 cores,
    then infer how serial your program is.
    """
    return (1.0 - 1.0 / measured_speedup) / (1.0 - 1.0 / s)


def workers_needed(p: float, target_speedup: float) -> int:
    """Smallest integer worker count s achieving target_speedup, or raise
    ValueError if the target exceeds max_speedup(p).
    """
    import math

    denom = 1.0 / target_speedup - (1.0 - p)
    if denom <= 0:
        raise ValueError(
            f"target {target_speedup}x exceeds Amdahl ceiling {max_speedup(p):.2f}x for p={p}"
        )
    # tiny epsilon guards float error when the solution is an exact integer
    return math.ceil(p / denom - 1e-9)
