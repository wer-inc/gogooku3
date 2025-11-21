"""Hash utilities for dataset artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Stream a file and return its SHA256 digest as hex string."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def schema_hash(pairs: Iterable[tuple[str, str]]) -> str:
    """Build a stable hash from (column_name, dtype) pairs."""

    digest = hashlib.sha256()
    for name, dtype in pairs:
        digest.update(f"{name}:{dtype}".encode("utf-8"))
    return digest.hexdigest()
