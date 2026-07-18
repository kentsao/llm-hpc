"""Exercise 3: vectorize three loopy functions.

The `*_loop` reference implementations are correct but slow. Rewrite each in
pure NumPy (no Python-level loop over elements/rows). Tests check you match the
reference; the perf test checks >=20x speedup locally.
"""

import numpy as np


# --- 1. Row-wise softmax (the workhorse of attention) -----------------------

def softmax_rows_loop(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        row = x[i]
        m = row.max()                      # subtract max: numerical stability
        e = np.exp(row - m)
        out[i] = e / e.sum()
    return out


def softmax_rows(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis, no Python loop."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# --- 2. Pairwise squared distances (kNN, clustering, RoPE-ish geometry) -----

def pairwise_sq_dists_loop(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n, m = x.shape[0], y.shape[0]
    out = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            d = x[i] - y[j]
            out[i, j] = d @ d
    return out


def pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """||x_i - y_j||^2 for all pairs, via the expansion trick (one matmul).

    Clip tiny negative values (floating-point) to 0 before returning.
    """
    x2 = (x * x).sum(axis=1)[:, None]
    y2 = (y * y).sum(axis=1)[None, :]
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


# --- 3. Moving average (signal smoothing; think loss curves) ----------------

def moving_average_loop(x: np.ndarray, w: int) -> np.ndarray:
    out = np.empty(x.size - w + 1)
    for i in range(out.size):
        out[i] = x[i : i + w].mean()
    return out


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """Window-w moving average in O(n), no Python loop."""
    c = np.concatenate(([0.0], np.cumsum(x)))
    return (c[w:] - c[:-w]) / w
