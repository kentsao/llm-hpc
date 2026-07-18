import numpy as np

from chapters.importer import load_exercise
from common.testing import run_exercise

ex = load_exercise("ch02-parallelism-cpu", "ex02_blocked_matmul")


def test_blocked_small_nonsquare():
    rng = np.random.default_rng(0)
    A = rng.random((3, 5))
    B = rng.random((5, 2))
    C = run_exercise(ex.blocked_matmul, A, B, 2)
    np.testing.assert_allclose(C, A @ B, rtol=1e-12)


def test_blocked_not_multiple_of_block():
    rng = np.random.default_rng(1)
    A = rng.random((97, 53))
    B = rng.random((53, 71))
    C = run_exercise(ex.blocked_matmul, A, B, 16)
    np.testing.assert_allclose(C, A @ B, rtol=1e-10)


def test_blocked_block_bigger_than_matrix():
    rng = np.random.default_rng(2)
    A = rng.random((10, 10))
    B = rng.random((10, 10))
    C = run_exercise(ex.blocked_matmul, A, B, 64)
    np.testing.assert_allclose(C, A @ B, rtol=1e-12)


def test_parallel_matches_serial():
    rng = np.random.default_rng(3)
    A = rng.random((150, 80))
    B = rng.random((80, 120))
    C = run_exercise(ex.blocked_matmul_parallel, A, B, 32)
    np.testing.assert_allclose(C, A @ B, rtol=1e-10)


def test_best_block_size_returns_candidate():
    bs = run_exercise(ex.best_block_size, 192, (16, 32, 64))
    assert bs in (16, 32, 64)
