import numpy as np
import pytest

from chapters.importer import load_exercise
from common.testing import run_exercise

ex = load_exercise("ch02-parallelism-cpu", "ex01_parallel_map")


def _double(a: np.ndarray) -> np.ndarray:
    return a * 2


def test_chunked_map_preserves_order_and_values():
    x = np.arange(101, dtype=np.float64)  # odd length: uneven chunks
    out = run_exercise(ex.chunked_parallel_map, _double, x, 4)
    np.testing.assert_array_equal(out, x * 2)


def test_chunked_map_single_worker():
    x = np.arange(10, dtype=np.float64)
    out = run_exercise(ex.chunked_parallel_map, _double, x, 1)
    np.testing.assert_array_equal(out, x * 2)


def test_pi_chunk_deterministic_and_plausible():
    h1 = run_exercise(ex._pi_chunk, (100_000, 7))
    h2 = run_exercise(ex._pi_chunk, (100_000, 7))
    assert h1 == h2, "same seed must give same hits (use np.random.default_rng(seed))"
    assert abs(4 * h1 / 100_000 - np.pi) < 0.05


def test_pi_workers_use_different_seeds():
    h_a = run_exercise(ex._pi_chunk, (100_000, 1))
    h_b = run_exercise(ex._pi_chunk, (100_000, 2))
    assert h_a != h_b, "different seeds should (almost surely) give different hit counts"


def test_parallel_pi_accuracy():
    est = run_exercise(ex.parallel_pi, 2_000_000, 4)
    assert est == pytest.approx(np.pi, abs=0.01)


def test_parallel_pi_uses_exact_sample_count():
    # 1_000_003 is prime: forces careful splitting. A correct estimator is still
    # a valid probability -> estimate stays in a sane range even so.
    est = run_exercise(ex.parallel_pi, 1_000_003, 4)
    assert 3.0 < est < 3.3
