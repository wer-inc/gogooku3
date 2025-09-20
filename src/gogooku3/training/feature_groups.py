from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import yaml


def resolve_groups_from_prefixes(
    feature_cols: Sequence[str],
    config_path: str | Path,
) -> Dict[str, list[int]]:
    """Resolve group prefixes to column indices based on the provided feature columns.

    YAML schema:
    version: 1
    groups:
      FLOW:
        prefixes: ["flow_"]
      MARGIN:
        prefixes: ["margin_"]
      ...
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    groups: Dict[str, list[int]] = {}
    for name, spec in (cfg.get("groups") or {}).items():
        prefixes = list(spec.get("prefixes") or [])
        idxs: list[int] = []
        for i, c in enumerate(feature_cols):
            if any(c.startswith(p) for p in prefixes):
                idxs.append(i)
        if idxs:
            groups[name] = idxs
    return groups


__all__ = ["resolve_groups_from_prefixes"]

