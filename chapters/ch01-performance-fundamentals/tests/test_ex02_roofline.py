import pytest

from chapters.importer import load_exercise
from common.testing import run_exercise

ex = load_exercise("ch01-performance-fundamentals", "ex02_roofline")


def test_elementwise_cost():
    n = 1000
    f, b = run_exercise(ex.elementwise_cost, n, 1, 2)
    assert f == n
    assert b == 3 * n * 4  # 2 inputs + 1 output, float32


def test_matmul_cost():
    f, b = run_exercise(ex.matmul_cost, 64, 128, 32)
    assert f == 2 * 64 * 128 * 32
    assert b == (64 * 128 + 128 * 32 + 64 * 32) * 4


def test_attention_scores_cost():
    bsz, h, s, d = 2, 12, 512, 64
    f, b = run_exercise(ex.attention_scores_cost, bsz, h, s, d)
    assert f == 2 * bsz * h * s * s * d
    # Q + K read, S written:
    assert b == (2 * bsz * h * s * d + bsz * h * s * s) * 4


def test_classify_and_predict():
    hw = ex.Hardware(peak_gflops=100.0, peak_gbs=50.0)  # ridge = 2 FLOP/byte
    assert run_exercise(ex.classify, 100, 100, hw) == "memory-bound"      # AI=1
    assert run_exercise(ex.classify, 1000, 100, hw) == "compute-bound"    # AI=10
    # memory-bound kernel: perf = AI * BW = 1 * 50 = 50 GFLOP/s
    assert run_exercise(ex.predicted_gflops, 100, 100, hw) == pytest.approx(50.0)
    # compute-bound kernel: flat roof
    assert run_exercise(ex.predicted_gflops, 1000, 100, hw) == pytest.approx(100.0)


def test_big_matmul_is_compute_bound_everywhere():
    f, b = run_exercise(ex.matmul_cost, 4096, 4096, 4096)
    assert run_exercise(ex.classify, f, b, ex.M1_CPU) == "compute-bound"
    assert run_exercise(ex.classify, f, b, ex.T4_GPU) == "compute-bound"


def test_elementwise_is_memory_bound_everywhere():
    f, b = run_exercise(ex.elementwise_cost, 1_000_000, 8, 1)
    assert run_exercise(ex.classify, f, b, ex.M1_CPU) == "memory-bound"
    assert run_exercise(ex.classify, f, b, ex.T4_GPU) == "memory-bound"
