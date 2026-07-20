import math

import pytest

from chapters.importer import load_exercise
from common.testing import run_exercise

ex = load_exercise("ch01-performance-fundamentals", "ex01_amdahl")


def test_speedup_basic():
    assert run_exercise(ex.amdahl_speedup, 0.0, 8) == pytest.approx(1.0)
    assert run_exercise(ex.amdahl_speedup, 1.0, 8) == pytest.approx(8.0)
    assert run_exercise(ex.amdahl_speedup, 0.9, 10) == pytest.approx(1 / (0.1 + 0.09))


def test_max_speedup():
    assert run_exercise(ex.max_speedup, 0.9) == pytest.approx(10.0)
    assert run_exercise(ex.max_speedup, 0.5) == pytest.approx(2.0)


def test_inverse_roundtrip():
    p, s = 0.85, 8
    speedup = 1 / ((1 - p) + p / s)
    assert run_exercise(ex.parallel_fraction, speedup, s) == pytest.approx(p)


def test_workers_needed():
    # p=0.9, target 5x: solve gives s=9
    assert run_exercise(ex.workers_needed, 0.9, 5.0) == 9
    # ceiling check: p=0.9 can never reach 10x (sup, not attained) or beyond
    with pytest.raises(ValueError):
        run_exercise(ex.workers_needed, 0.9, 12.0)


def test_workers_needed_is_minimal():
    s = run_exercise(ex.workers_needed, 0.95, 6.0)
    assert isinstance(s, int), "worker counts are integers — ceil the continuous solution"
    assert s == 9  # continuous solution is 8.14…, so 9 is the smallest feasible count
    assert 1 / (0.05 + 0.95 / s) >= 6.0 - 1e-9
    assert 1 / (0.05 + 0.95 / (s - 1)) < 6.0


def test_workers_needed_infeasible_at_exact_ceiling():
    # max_speedup(0.9) == 10 is a supremum: no finite worker count attains it.
    with pytest.raises(ValueError):
        run_exercise(ex.workers_needed, 0.9, 10.0)
