"""Exercise 1: Amdahl's law, forward and inverse.

speedup = 1 / ((1 - p) + p / s)
  p: parallelizable fraction of runtime (0..1)
  s: speedup applied to that fraction (e.g., number of workers)
"""

import math


def amdahl_speedup(p: float, s: float) -> float:
    """Return the overall speedup when fraction p is accelerated by factor s."""
    # TODO(you): one line.
    # raise NotImplementedError
    return 1 / ((1-p) + p / s)


def max_speedup(p: float) -> float:
    """Return the speedup ceiling with INFINITE parallel hardware (s -> inf).

    (Take the limit analytically; don't pass a huge number.)
    """
    # TODO(you)
    #raise NotImplementedError
    return 1 / (1 - p)


def parallel_fraction(measured_speedup: float, s: float) -> float:
    """Inverse problem: you measured `measured_speedup` using s workers.
    Return the implied parallelizable fraction p.

    This is what you'll actually do in practice: run on 1 and on 8 cores,
    then infer how serial your program is.
    """
    # TODO(you): solve the Amdahl equation for p.
    # raise NotImplementedError
    # 1 / ((1 - p) + p / s) = measured speed up = x
    # x(1 - p) + xp / s = 1
    # xs(1 - p) + xp = s
    # xs - xsp + xp = s
    # s(x - 1) = p(xs - x)
    # p = s(x-1)/x(s-1)
    return (s*(measured_speedup-1)) / (measured_speedup*(s-1))
    


def workers_needed(p: float, target_speedup: float) -> int:
    """Smallest integer worker count s achieving target_speedup, or raise
    ValueError if the target exceeds max_speedup(p).
    """
    # xs - xsp + xp = s
    # s - xs + xsp = xp
    # s = xp / (1-x+xp)
    # Review fix 1: infeasibility must be checked on the DENOMINATOR, not by
    # comparing against max_speedup(p) — fp rounding makes 1/(1-0.9) come out
    # as 10.000000000000002, so `target >= max` misses target=10.0 exactly and
    # the division below blows up. (The ceiling is a supremum anyway: no finite
    # worker count attains it.)
    denom = 1 - target_speedup + target_speedup * p
    if denom <= 1e-12:
        raise ValueError("Target Speedup is larger than the max speed up.")
    s_continuous = (target_speedup * p) / denom
    # Review fix 2: the spec asks for the smallest INTEGER worker count, so
    # ceil the continuous solution (8.14 workers doesn't exist — you need 9).
    # The 1e-9 guards against fp residue pushing an exact answer up one worker.
    return math.ceil(s_continuous - 1e-9)
