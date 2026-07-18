"""Profile-then-fix workflow on a deliberately slow 'training step'.

Run:  uv run python chapters/ch01-performance-fundamentals/examples/profile_demo.py

Also try the sampling profiler (no code changes needed):
    uvx py-spy record -o profile.svg -- \
        .venv/bin/python chapters/ch01-performance-fundamentals/examples/profile_demo.py
"""

import cProfile
import pstats
import io

import numpy as np


def load_batch(n=512, d=256):
    # Slow on purpose: builds Python lists element by element.
    return np.array([[float(i * j % 97) / 97 for j in range(d)] for i in range(n)])


def forward(x, w):
    return np.maximum(x @ w, 0.0)


def loss(y):
    return float((y**2).mean())


def train_steps(steps=5):
    w = np.random.randn(256, 256) * 0.01
    total = 0.0
    for _ in range(steps):
        x = load_batch()          # <- where does the time actually go?
        y = forward(x, w)
        total += loss(y)
    return total


def main() -> None:
    prof = cProfile.Profile()
    prof.enable()
    train_steps()
    prof.disable()

    s = io.StringIO()
    pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(8)
    print(s.getvalue())

    print("Read the table: 'data loading' (a nested Python listcomp) dwarfs the matmul.")
    print("Amdahl says: even an infinitely fast GPU would barely speed this program up.")
    print("Real-world moral: profile the WHOLE step — input pipelines are classic serial bottlenecks.")


if __name__ == "__main__":
    main()
