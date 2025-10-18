from __future__ import annotations

"""Minimal data lineage tracker (JSON Lines)."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DataLineageTracker:
    output_dir: Path
    filename: str = "lineage.jsonl"

    def _path(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir / self.filename

    def track_transformation(
        self,
        inputs: list[str],
        output: str,
        transformation: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "transformation": transformation,
            "inputs": inputs,
            "output": output,
            "metadata": metadata or {},
        }
        with open(self._path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
