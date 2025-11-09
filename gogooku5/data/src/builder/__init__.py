"""Dataset builder package for gogooku5.

Modules are organized by concern (API clients, features, pipelines, config, utils).
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    _PROJECT_ROOT = Path(__file__).resolve().parents[4]
except IndexError:  # pragma: no cover - defensive for unusual layouts
    _PROJECT_ROOT = Path(__file__).resolve().parent

_SRC_ROOT = _PROJECT_ROOT / "src"
if _SRC_ROOT.exists():
    src_str = str(_SRC_ROOT)
    if src_str not in sys.path:
        sys.path.append(src_str)

__all__ = [
    "api",
    "features",
    "pipelines",
    "config",
    "utils",
    "chunks",
]
