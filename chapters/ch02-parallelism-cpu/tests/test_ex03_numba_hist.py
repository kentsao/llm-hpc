import numpy as np

from chapters.importer import load_exercise
from common.testing import run_exercise

ex = load_exercise("ch02-parallelism-cpu", "ex03_numba_hist")


def _expected(x: np.ndarray, n_bins: int) -> np.ndarray:
    b = (x * n_bins).astype(np.int64)
    b[b == n_bins] -= 1
    return np.bincount(b, minlength=n_bins)


def test_fixed_histogram_correct():
    rng = np.random.default_rng(0)
    x = rng.random(1_000_000)
    out = run_exercise(ex.histogram_fixed, x, 32)
    np.testing.assert_array_equal(out, _expected(x, 32))


def test_fixed_histogram_counts_every_element():
    rng = np.random.default_rng(1)
    # size chosen to not divide evenly by typical chunk counts
    x = rng.random(999_983)
    out = run_exercise(ex.histogram_fixed, x, 16)
    assert out.sum() == x.size, "every element must be counted exactly once"


def test_fixed_histogram_edge_value_one():
    x = np.array([0.0, 0.5, 1.0, 1.0])
    out = run_exercise(ex.histogram_fixed, x, 2)
    np.testing.assert_array_equal(out, [1, 3])
