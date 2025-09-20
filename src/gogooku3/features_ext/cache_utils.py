from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import polars as pl


def _hash_key(name: str, kwargs: dict[str, Any]) -> str:
    payload = json.dumps({"name": name, "kwargs": kwargs}, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def cache_parquet(
    cache_dir: str | Path,
    *,
    name: str,
    kwargs: dict[str, Any],
    builder: Callable[[], pl.DataFrame],
) -> pl.DataFrame:
    """Cache a Polars DataFrame to Parquet keyed by function name and kwargs.

    This is a lightweight utility for deterministic, side-effect-free feature
    builders. It does not handle partial periods or appends; callers should
    encode the time range inside `kwargs`.
    """
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    key = _hash_key(name, kwargs)
    path = p / f"{name}-{key}.parquet"
    if path.exists():
        return pl.read_parquet(str(path))
    df = builder()
    df.write_parquet(str(path))
    return df


__all__ = ["cache_parquet"]

