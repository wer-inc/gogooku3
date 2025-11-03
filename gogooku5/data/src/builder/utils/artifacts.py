"""Dataset artifact management utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

from ..config import DatasetBuilderSettings, get_settings
from .logger import get_logger

LOGGER = get_logger("artifacts")


@dataclass(slots=True)
class DatasetArtifact:
    """Information about a persisted dataset artifact."""

    parquet_path: Path
    metadata_path: Path
    latest_symlink: Path
    tagged_symlink: Path
    created_at: datetime


class DatasetArtifactWriter:
    """Handle parquet + metadata writes, symlinks, and retention."""

    def __init__(self, *, settings: DatasetBuilderSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.output_dir = self.settings.data_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tag = self.settings.dataset_tag
        self.retention_keep = max(1, self.settings.dataset_retention_keep)

    def write(
        self,
        df: pl.DataFrame,
        *,
        start_date: str | None,
        end_date: str | None,
    ) -> DatasetArtifact:
        """Persist dataframe and create metadata + symlinks."""

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        range_token = f"{start_date.replace('-', '')}_{end_date.replace('-', '')}" if start_date and end_date else None
        base_name = (
            f"ml_dataset_{range_token}_{timestamp}_{self.tag}" if range_token else f"ml_dataset_{timestamp}_{self.tag}"
        )

        parquet_path = self.output_dir / f"{base_name}.parquet"
        metadata_path = self.output_dir / f"{base_name}_metadata.json"

        # Phase 1-4 Fix: Validate dataset is not empty
        if df.height == 0:
            error_msg = (
                f"Cannot persist empty dataset (0 rows). "
                f"Dataset should have actual data before writing to {parquet_path}. "
                f"Columns: {df.width}, Start: {start_date}, End: {end_date}"
            )
            LOGGER.error(error_msg)
            raise ValueError(error_msg)

        # Phase 1-4 Fix: Warn if dataset is suspiciously small
        if df.height < 100:
            LOGGER.warning(
                "Dataset has only %d rows (expected thousands). "
                "This might indicate a data fetching issue. Columns: %d",
                df.height,
                df.width,
            )

        df.write_parquet(parquet_path, compression=self.settings.dataset_parquet_compression)
        metadata = self._build_metadata(df)
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

        latest_symlink = self._update_symlink(
            link_name=self.settings.latest_dataset_symlink,
            target=parquet_path,
        )
        tagged_symlink = self._update_symlink(
            link_name=f"ml_dataset_latest_{self.tag}.parquet",
            target=parquet_path,
        )
        self._update_symlink(
            link_name=self.settings.latest_metadata_symlink,
            target=metadata_path,
        )
        self._update_symlink(
            link_name=f"ml_dataset_latest_{self.tag}_metadata.json",
            target=metadata_path,
        )

        self._prune_history(pattern=f"ml_dataset_*_{self.tag}.parquet")
        self._prune_history(pattern=f"ml_dataset_*_{self.tag}_metadata.json")

        return DatasetArtifact(
            parquet_path=parquet_path,
            metadata_path=metadata_path,
            latest_symlink=latest_symlink,
            tagged_symlink=tagged_symlink,
            created_at=datetime.utcnow(),
        )

    def _build_metadata(self, df: pl.DataFrame) -> dict:
        """Reuse gogooku3 metadata builder for schema parity."""
        try:
            from src.gogooku3.pipeline.builder import MLDatasetBuilder
        except Exception:
            import sys
            from pathlib import Path

            root = Path(__file__).resolve().parents[5]
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            try:
                from src.gogooku3.pipeline.builder import (
                    MLDatasetBuilder,  # type: ignore
                )
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.warning("Failed to import legacy MLDatasetBuilder for metadata: %s", exc)
                return {
                    "rows": df.height,
                    "cols": len(df.columns),
                    "features": {"count": len(df.columns)},
                }

        metadata_builder = MLDatasetBuilder(output_dir=self.output_dir)
        prepared = self._prepare_for_metadata(df)
        return metadata_builder.create_metadata(prepared)

    @staticmethod
    def _prepare_for_metadata(df: pl.DataFrame) -> pl.DataFrame:
        """Ensure metadata inspector sees legacy column names."""
        rename_map = {}
        if "code" in df.columns and "Code" not in df.columns:
            rename_map["code"] = "Code"
        if "date" in df.columns and "Date" not in df.columns:
            rename_map["date"] = "Date"
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns and col.capitalize() not in df.columns:
                rename_map[col] = col.capitalize()
        if rename_map:
            return df.rename(rename_map)
        return df

    def _update_symlink(self, *, link_name: str, target: Path) -> Path:
        link_path = self.output_dir / link_name
        try:
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to remove existing symlink %s: %s", link_path, exc)
        try:
            link_path.symlink_to(target.name)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to create symlink %s -> %s: %s", link_path, target, exc)
        return link_path

    def _prune_history(self, *, pattern: str) -> None:
        keep = self.retention_keep
        try:
            candidates = sorted(
                [
                    path
                    for path in self.output_dir.glob(pattern)
                    if path.is_file() and not path.is_symlink() and "latest" not in path.name
                ],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Failed to enumerate dataset history for %s: %s", pattern, exc)
            return
        for obsolete in candidates[keep:]:
            try:
                obsolete.unlink()
                LOGGER.info("🧹 Pruned dataset artifact: %s", obsolete.name)
            except FileNotFoundError:
                continue
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.debug("Failed to prune %s: %s", obsolete, exc)


def resolve_latest_dataset(*, settings: Optional[DatasetBuilderSettings] = None) -> Optional[Path]:
    """Return the resolved path to the latest dataset symlink, if present."""

    current_settings = settings or get_settings()
    candidate = current_settings.data_output_dir / current_settings.latest_dataset_symlink
    if candidate.exists():
        try:
            return candidate.resolve()
        except Exception:  # pragma: no cover - defensive
            return candidate
    return None
