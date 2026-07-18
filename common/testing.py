"""Helpers that connect exercises to pytest.

Exercise functions start life raising ``NotImplementedError``. Tests call them
through :func:`run_exercise`, so an untouched exercise shows up as a *skip*
("not implemented yet") rather than a failure — and the moment you implement
it, the same test verifies your work for real.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest


def run_exercise(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call an exercise function; skip the test if it isn't implemented yet."""
    try:
        return fn(*args, **kwargs)
    except NotImplementedError:
        pytest.skip(f"exercise '{fn.__name__}' not implemented yet — fill in the TODOs")
