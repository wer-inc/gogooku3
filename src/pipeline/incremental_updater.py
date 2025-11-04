from __future__ import annotations

"""Incremental dataset updater utilities (Polars-based)."""

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl


@dataclass
class IncrementalConfig:
    update_mode: str = "full"  # "full" | "incremental"
    since_date: str | None = None  # YYYY-MM-DD


class IncrementalDatasetUpdater:
    def __init__(
        self, output_dir: Path, cfg: IncrementalConfig | None = None
    ) -> None:
        self.output_dir = output_dir
        self.cfg = cfg or IncrementalConfig()

    def latest_artifacts(self) -> tuple[Path | None, Path | None]:
        pq = self.output_dir / "ml_dataset_latest_full.parquet"
        meta = self.output_dir / "ml_dataset_latest_full_metadata.json"
        return (pq if pq.exists() else None, meta if meta.exists() else None)

    def read_last_date_range(self) -> tuple[str | None, str | None]:
        _, meta = self.latest_artifacts()
        if not meta:
            return (None, None)
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            dr = data.get("date_range", {})
            return (dr.get("start"), dr.get("end"))
        except Exception:
            return (None, None)

    def compute_since_date(self, fallback_end: str | None = None) -> str | None:
        if self.cfg.since_date:
            return self.cfg.since_date
        _, last_end = self.read_last_date_range()
        last = last_end or fallback_end
        if not last:
            return None
        try:
            dt = datetime.strptime(last, "%Y-%m-%d")
            return (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        except Exception:
            return None

    def merge_enriched(
        self,
        old_path: Path,
        inc_path: Path,
        key_date: str = "Date",
        key_code: str = "Code",
    ) -> pl.DataFrame:
        """Merge old and incremental enriched datasets, preferring incremental rows on key collisions."""
        old_df = pl.read_parquet(old_path)
        inc_df = pl.read_parquet(inc_path)
        # Concat then drop duplicates keeping last (incremental wins if concatenated last)
        df = pl.concat([old_df, inc_df], how="vertical_relaxed")
        subset = [c for c in (key_date, key_code) if c in df.columns]
        if subset:
            df = df.unique(subset=subset, keep="last")
            if key_date in df.columns:
                df = df.sort(
                    by=[key_date, key_code] if key_code in df.columns else [key_date]
                )
        return df
