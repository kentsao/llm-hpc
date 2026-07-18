import os
import time

import numpy as np
import pytest

from chapters.importer import load_exercise
from common.testing import run_exercise

ex = load_exercise("ch01-performance-fundamentals", "ex03_vectorize")
rng = np.random.default_rng(42)


def test_softmax_rows_matches_reference():
    x = rng.standard_normal((50, 40))
    out = run_exercise(ex.softmax_rows, x)
    np.testing.assert_allclose(out, ex.softmax_rows_loop(x), rtol=1e-6)


def test_softmax_rows_stable_with_huge_logits():
    x = np.array([[1000.0, 1000.0, -1000.0]])
    out = run_exercise(ex.softmax_rows, x)
    assert np.all(np.isfinite(out)), "overflow — did you subtract the row max?"
    np.testing.assert_allclose(out[0], [0.5, 0.5, 0.0], atol=1e-9)


def test_pairwise_sq_dists_matches_reference():
    x = rng.standard_normal((30, 8))
    y = rng.standard_normal((20, 8))
    out = run_exercise(ex.pairwise_sq_dists, x, y)
    np.testing.assert_allclose(out, ex.pairwise_sq_dists_loop(x, y), rtol=1e-5, atol=1e-8)
    assert np.all(out >= 0), "negative distances — clip floating-point residue"


def test_moving_average_matches_reference():
    x = rng.standard_normal(500)
    out = run_exercise(ex.moving_average, x, 7)
    np.testing.assert_allclose(out, ex.moving_average_loop(x, 7), rtol=1e-9, atol=1e-9)


@pytest.mark.perf
@pytest.mark.skipif(os.environ.get("CI") == "true", reason="perf bar is for local runs")
def test_vectorized_is_20x_faster():
    x = rng.standard_normal((400, 400))

    def clock(fn, reps=3):
        fn()  # warm-up
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    t_loop = clock(lambda: ex.softmax_rows_loop(x))
    t_vec = clock(lambda: run_exercise(ex.softmax_rows, x))
    assert t_loop / t_vec > 20, f"only {t_loop / t_vec:.1f}x — is there still a Python loop?"
