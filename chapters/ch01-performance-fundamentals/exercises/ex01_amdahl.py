"""Exercise 1: Amdahl's law, forward and inverse.

speedup = 1 / ((1 - p) + p / s)
  p: parallelizable fraction of runtime (0..1)
  s: speedup applied to that fraction (e.g., number of workers)
"""


def amdahl_speedup(p: float, s: float) -> float:
    """Return the overall speedup when fraction p is accelerated by factor s."""
    # TODO(you): one line.
    raise NotImplementedError


def max_speedup(p: float) -> float:
    """Return the speedup ceiling with INFINITE parallel hardware (s -> inf).

    (Take the limit analytically; don't pass a huge number.)
    """
    # TODO(you)
    raise NotImplementedError


def parallel_fraction(measured_speedup: float, s: float) -> float:
    """Inverse problem: you measured `measured_speedup` using s workers.
    Return the implied parallelizable fraction p.

    This is what you'll actually do in practice: run on 1 and on 8 cores,
    then infer how serial your program is.
    """
    # TODO(you): solve the Amdahl equation for p.
    raise NotImplementedError


def workers_needed(p: float, target_speedup: float) -> int:
    """Smallest integer worker count s achieving target_speedup, or raise
    ValueError if the target exceeds max_speedup(p).
    """
    # TODO(you): solve for s, then ceil. Watch the infeasible case.
    raise NotImplementedError
