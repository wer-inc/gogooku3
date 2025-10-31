"""Shared helpers to configure sys.path for standalone scripts."""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_import_paths() -> None:
    """Add the package src directory and script roots to ``sys.path``."""

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    src_root = project_root / "src"

    for path in (src_root, project_root, script_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
