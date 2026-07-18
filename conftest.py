"""Repo-root conftest: makes the repo root importable so tests can do
``import common`` and ``from chapters... import`` regardless of invocation dir."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
