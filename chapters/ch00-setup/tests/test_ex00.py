import numpy as np
import pytest

from chapters.importer import load_exercise
from common.testing import run_exercise

ex = load_exercise("ch00-setup", "ex00_bandwidth")


def test_triad_correct():
    rng = np.random.default_rng(0)
    b = rng.random(1000)
    c = rng.random(1000)
    a = np.zeros(1000)
    run_exercise(ex.triad, a, b, c, 2.5)
    np.testing.assert_allclose(a, b + 2.5 * c)


def test_triad_in_place():
    b = np.ones(10)
    c = np.ones(10)
    a = np.zeros(10)
    before = a.__array_interface__["data"][0]
    run_exercise(ex.triad, a, b, c, 1.0)
    assert a.__array_interface__["data"][0] == before, "triad must write into `a`, not rebind it"
    np.testing.assert_allclose(a, 2.0)


def test_triad_bytes():
    n, dt = 1000, np.dtype(np.float64)
    assert run_exercise(ex.triad_bytes, n, dt) == 3 * n * 8


@pytest.mark.perf
def test_measured_bandwidth_plausible():
    gbs = run_exercise(ex.measure_bandwidth, 20_000_000)
    # Any modern machine achieves well over 5 GB/s; below that means a bug
    # (e.g., timing allocation, or a Python-level loop).
    assert gbs > 5.0, f"measured {gbs:.1f} GB/s — implausibly low, check your implementation"
