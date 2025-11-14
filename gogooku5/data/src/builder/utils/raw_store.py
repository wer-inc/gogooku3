"""Raw data access utilities backed by the raw manifest."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import polars as pl


@dataclass
class RawDataSlice:
    source: str
    chunk_id: str
    start: str
    end: str
    path: Path


class RawDataStore:
    """Load raw Parquet/Arrow files listed in raw_manifest.json."""

    def __init__(self, manifest_path: Path) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Raw manifest not found: {manifest_path}")
        self.manifest_path = manifest_path
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_root = Path(payload.get("raw_root") or manifest_path.parent)
        self.raw_root = raw_root
        sources = payload.get("sources", {})
        self.entries: dict[str, list[RawDataSlice]] = {}
        for source, items in sources.items():
            slices: list[RawDataSlice] = []
            for item in items:
                file_rel = Path(item["file"])
                slices.append(
                    RawDataSlice(
                        source=source,
                        chunk_id=item["chunk_id"],
                        start=item["start"],
                        end=item["end"],
                        path=raw_root / file_rel,
                    )
                )
            self.entries[source] = slices

    def load_range(self, *, source: str, start: str, end: str) -> pl.DataFrame:
        """Load raw data for the requested [start, end] range (inclusive)."""

        slices = self.entries.get(source, [])
        if not slices:
            raise FileNotFoundError(f"No raw data available for source={source}")

        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)

        matching: list[RawDataSlice] = []
        for slice_ in slices:
            try:
                slice_start = datetime.fromisoformat(slice_.start)
                slice_end = datetime.fromisoformat(slice_.end)
            except ValueError:
                continue
            if slice_.path.exists() and slice_start <= end_dt and slice_end >= start_dt:
                matching.append(slice_)

        if not matching:
            raise FileNotFoundError(
                f"No raw data covering {start}→{end} for source={source}"
            )

        frames = []
        for slice_ in matching:
            if slice_.path.suffix.lower() == ".arrow":
                frames.append(pl.read_ipc(slice_.path))
            else:
                frames.append(pl.read_parquet(slice_.path))
        return pl.concat(frames, how="vertical")
