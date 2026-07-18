"""Utility to import exercise/example modules from chapter directories,
whose names (ch00-setup) aren't valid Python package names."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

CHAPTERS_DIR = Path(__file__).parent


def _load(path: Path, qualname: str) -> ModuleType:
    if qualname in sys.modules:
        return sys.modules[qualname]
    spec = importlib.util.spec_from_file_location(qualname, path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = mod
    spec.loader.exec_module(mod)
    return mod


def load_exercise(chapter: str, module: str) -> ModuleType:
    """load_exercise("ch00-setup", "ex00_bandwidth") -> module object"""
    return _load(CHAPTERS_DIR / chapter / "exercises" / f"{module}.py", f"{chapter}.exercises.{module}")


def load_example(chapter: str, module: str) -> ModuleType:
    return _load(CHAPTERS_DIR / chapter / "examples" / f"{module}.py", f"{chapter}.examples.{module}")
