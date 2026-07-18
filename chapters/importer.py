"""Import exercise/example modules from chapter directories.

Chapter directory names (ch00-setup) aren't valid Python package names, so we
put the module's own directory on sys.path and import it by its plain name.
This keeps modules importable in *spawned worker processes* too (macOS
multiprocessing pickles functions by module name, and spawn children inherit
sys.path) — which the parallel exercises rely on.

Consequence: exercise/example module filenames must be unique repo-wide
(e.g. ex01_parallel_map.py, not utils.py).
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType

CHAPTERS_DIR = Path(__file__).parent


def _load(directory: Path, module: str) -> ModuleType:
    assert directory.is_dir(), f"no such directory: {directory}"
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    return importlib.import_module(module)


def load_exercise(chapter: str, module: str) -> ModuleType:
    """load_exercise("ch00-setup", "ex00_bandwidth") -> module object"""
    return _load(CHAPTERS_DIR / chapter / "exercises", module)


def load_example(chapter: str, module: str) -> ModuleType:
    return _load(CHAPTERS_DIR / chapter / "examples", module)
