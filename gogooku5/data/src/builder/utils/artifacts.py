"""Dataset artifact management utilities."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl

from ..config import DatasetBuilderSettings, get_settings
from .hash_utils import file_sha256, schema_hash
from .lazy_io import save_with_cache
from .logger import get_logger

LOGGER = get_logger("artifacts")


@dataclass(slots=True)
class DatasetArtifact:
    """Information about a persisted dataset artifact."""

    parquet_path: Path
    metadata_path: Path
    latest_symlink: Path
    tagged_symlink: Path
    feature_index_path: Path
    latest_feature_index_symlink: Path
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
        extra_metadata: dict | None = None,
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

        # Save with IPC cache for 3-5x faster reads
        parquet_kwargs = {"compression": self.settings.dataset_parquet_compression}
        _, ipc_path = save_with_cache(df, parquet_path, create_ipc=True, parquet_kwargs=parquet_kwargs)
        if ipc_path:
            LOGGER.debug("Created IPC cache: %s (3-5x faster reads)", ipc_path)

        feature_index_payload = self._build_feature_index_payload(
            df=df,
            dataset_path=parquet_path,
        )
        feature_index_path = self._write_feature_index(
            payload=feature_index_payload,
            base_name=base_name,
        )

        metadata = self._build_metadata(df)
        feature_index_summary = {
            "path": str(feature_index_path),
            "column_hash": feature_index_payload["column_hash"],
            "feature_hash": feature_index_payload.get("feature_hash"),
            "target_hash": feature_index_payload.get("target_hash"),
            "schema_hash": feature_index_payload.get("schema_hash"),
            "total_columns": len(feature_index_payload["columns"]),
            "feature_columns": len(feature_index_payload["feature_columns"]),
            "target_columns": len(feature_index_payload["target_columns"]),
            "metadata_columns": len(feature_index_payload["metadata_columns"]),
        }
        metadata = self._merge_metadata(metadata, {"feature_index": feature_index_summary})
        dataset_hash = file_sha256(parquet_path)
        metadata = self._merge_metadata(
            metadata,
            {
                "dataset_hash": dataset_hash,
                "dataset_hash_algorithm": "sha256",
                "feature_schema_version": feature_index_payload.get("schema_hash"),
            },
        )
        if extra_metadata:
            metadata = self._merge_metadata(metadata, extra_metadata)
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

        # BATCH-2B Safety: Check if dataset should update 'latest' symlinks
        should_update = self._should_update_latest(df, metadata)

        if should_update:
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
            latest_feature_index_symlink = self._update_symlink(
                link_name=self.settings.latest_feature_index_symlink,
                target=feature_index_path,
            )
            LOGGER.info(
                "✅ Updated 'latest' symlinks to %s (%d rows, %d cols)",
                parquet_path.name,
                df.height,
                df.width,
            )
        else:
            # Return paths to non-existent symlinks (not updated)
            latest_symlink = self.output_dir / self.settings.latest_dataset_symlink
            tagged_symlink = self.output_dir / f"ml_dataset_latest_{self.tag}.parquet"
            latest_feature_index_symlink = self.output_dir / self.settings.latest_feature_index_symlink
            LOGGER.info(
                "⚠️  Skipped 'latest' symlink update (safety gate). " "Dataset saved to %s (%d rows, %d cols)",
                parquet_path.name,
                df.height,
                df.width,
            )

        self._prune_history(pattern=f"ml_dataset_*_{self.tag}.parquet")
        self._prune_history(pattern=f"ml_dataset_*_{self.tag}_metadata.json")

        return DatasetArtifact(
            parquet_path=parquet_path,
            metadata_path=metadata_path,
            latest_symlink=latest_symlink,
            tagged_symlink=tagged_symlink,
            feature_index_path=feature_index_path,
            latest_feature_index_symlink=latest_feature_index_symlink,
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
    def _build_feature_index_payload(*, df: pl.DataFrame, dataset_path: Path) -> dict:
        """Construct feature index payload capturing deterministic schema details."""

        columns = list(df.columns)
        schema = df.schema
        numeric_types = {
            pl.Float64,
            pl.Float32,
            pl.Int64,
            pl.Int32,
            pl.Int16,
            pl.Int8,
            pl.UInt64,
            pl.UInt32,
            pl.UInt16,
            pl.UInt8,
            pl.Boolean,
        }
        metadata_name_set = {
            "code",
            "date",
            "sectorcode",
            "marketcode",
            "section",
            "section_norm",
            "row_idx",
            "sectionid",
            "section_name",
            "sector17_code",
            "sector17_name",
            "sector17_id",
            "sector33_code",
            "sector33_name",
            "sector33_id",
            "shares_outstanding",
        }
        feature_columns: list[str] = []
        target_columns: list[str] = []
        metadata_columns: list[str] = []
        column_details: list[dict[str, object]] = []
        cs_normalized_columns: list[str] = []

        for name in columns:
            dtype = schema[name]
            dtype_str = str(dtype)
            role = DatasetArtifactWriter._infer_column_role(
                name=name,
                dtype=dtype,
                metadata_name_set=metadata_name_set,
            )
            normalization = DatasetArtifactWriter._infer_normalization_tag(name)
            dtype_category = "numeric" if dtype in numeric_types else "categorical"

            if role == "feature" and dtype not in numeric_types:
                # Promote unexpected non-numeric features to metadata to avoid loader crashes.
                role = "metadata"

            if role == "feature":
                feature_columns.append(name)
                if normalization == "cs_z":
                    cs_normalized_columns.append(name)
            elif role == "target":
                target_columns.append(name)
            else:
                metadata_columns.append(name)

            column_details.append(
                {
                    "name": name,
                    "dtype": dtype_str,
                    "role": role,
                    "dtype_category": dtype_category,
                    "normalization": normalization,
                }
            )

        column_hash = hashlib.sha256("||".join(columns).encode("utf-8")).hexdigest()
        feature_hash = hashlib.sha1("||".join(feature_columns).encode("utf-8")).hexdigest() if feature_columns else None
        target_hash = hashlib.sha1("||".join(target_columns).encode("utf-8")).hexdigest() if target_columns else None
        schema_pairs = [(name, str(schema[name])) for name in columns]
        schema_fingerprint = schema_hash(schema_pairs)

        payload = {
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
            "generator": "DatasetArtifactWriter",
            "dataset": {
                "path": str(dataset_path.name),
            },
            "columns": columns,
            "column_details": column_details,
            "feature_columns": feature_columns,
            "target_columns": target_columns,
            "metadata_columns": metadata_columns,
            "cs_normalized_columns": cs_normalized_columns,
            "column_hash": column_hash,
            "feature_hash": feature_hash,
            "target_hash": target_hash,
            "schema_hash": schema_fingerprint,
            "strict": True,
        }
        return payload

    def _write_feature_index(self, *, payload: dict, base_name: str) -> Path:
        """Persist feature index payload to JSON."""

        feature_index_path = self.output_dir / f"{base_name}_feature_index.json"
        feature_index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return feature_index_path

    @staticmethod
    def _infer_column_role(
        *,
        name: str,
        dtype: pl.DataType,
        metadata_name_set: set[str],
    ) -> str:
        """Infer semantic role for a column."""

        lowered = name.lower()
        if lowered in metadata_name_set:
            return "metadata"
        target_prefixes = ("target_", "ret_fwd_", "label_", "feat_ret_", "future_", "y_")
        if any(lowered.startswith(prefix) for prefix in target_prefixes):
            return "target"
        if lowered.endswith("_binary") and lowered.startswith("target"):
            return "target"
        return "feature"

    @staticmethod
    def _infer_normalization_tag(name: str) -> str | None:
        """Infer normalization tag from column name."""

        lowered = name.lower()
        if lowered.endswith("_cs_z"):
            return "cs_z"
        if lowered.endswith("_cs_rank"):
            return "cs_rank"
        if lowered.endswith("_zscore"):
            return "zscore"
        if lowered.endswith("_z") and not lowered.endswith("_cs_z"):
            return "zscore"
        if lowered.endswith("_pct") or lowered.endswith("_ratio") or lowered.endswith("_rate"):
            return "relative"
        return None

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
        # Provide canonical adjusted columns as legacy aliases for metadata tooling.
        adjusted_pairs = {
            "adjustmentclose": "Close",
            "adjustmentopen": "Open",
            "adjustmenthigh": "High",
            "adjustmentlow": "Low",
            "adjustmentvolume": "Volume",
        }
        for src, dst in adjusted_pairs.items():
            if src in df.columns and dst not in rename_map and dst not in df.columns:
                rename_map[src] = dst
        if rename_map:
            return df.rename(rename_map)
        return df

    @staticmethod
    def _merge_metadata(base: dict, extra: dict) -> dict:
        """Shallow merge `extra` metadata into metadata builder output."""
        merged = json.loads(json.dumps(base, default=str))
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged

    def _should_update_latest(self, df: pl.DataFrame, metadata: dict) -> bool:
        """Determine if dataset should update 'latest' symlinks.

        Safety gates:
        1. NO_LATEST_SYMLINK=1: Test mode (prevents test data from overwriting production)
        2. Minimum row threshold: Prevents tiny test datasets from becoming 'latest'

        Args:
            df: Dataset DataFrame
            metadata: Dataset metadata with row counts

        Returns:
            True if dataset should update latest symlinks
        """
        # Gate 1: Test mode - never update latest
        if os.getenv("NO_LATEST_SYMLINK") == "1":
            LOGGER.info("NO_LATEST_SYMLINK=1 detected - skipping latest symlink update (test mode)")
            return False

        # Gate 2: Minimum row threshold
        rows = df.height
        min_codes = int(os.getenv("LATEST_MIN_CODES", "80"))
        floor = int(os.getenv("LATEST_MIN_ROWS_FLOOR", "100000"))

        # Calculate expected minimum rows (70% of days × min_codes)
        # For 1-year dataset: ~250 days × 80 codes × 0.7 = 14,000 rows
        days_output = metadata.get("n_days_output", 0)
        if days_output > 0:
            threshold = max(floor, int(0.7 * days_output * min_codes))
        else:
            threshold = floor

        if rows < threshold:
            LOGGER.warning(
                "Dataset too small for latest update: %d rows < %d threshold "
                "(days=%d, min_codes=%d, floor=%d). Skipping latest symlink.",
                rows,
                threshold,
                days_output,
                min_codes,
                floor,
            )
            return False

        return True

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
