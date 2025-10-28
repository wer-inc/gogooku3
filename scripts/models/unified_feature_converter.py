#!/usr/bin/env python3
"""
Unified Feature Converter for gogooku3 → ATFT-GAT-FAN
Converts ML dataset to ATFT-compatible format with proper time-series structure
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import polars as pl
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class UnifiedFeatureConverter:
    """統合特徴量変換システム：gogooku3 → ATFT-GAT-FAN"""

    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizons: List[int] = None,
        min_samples_per_code: int = 100,
    ):
        """
        Initialize converter

        Args:
            sequence_length: Length of time series sequences (default: 60)
            prediction_horizons: Prediction horizons in days (default: [1, 5, 10, 20])
            min_samples_per_code: Minimum samples required per stock code
        """
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons or [1, 5, 10, 20]
        self.min_samples_per_code = min_samples_per_code

        # Required columns
        self.required_columns = ["Code", "Date", "Close", "Volume"]

        # Feature columns to preserve (will be auto-detected)
        self.feature_columns = []

        # Target column mapping
        self.target_columns = {
            "feat_ret_1d": "target_1d",
            "returns_1d": "target_1d",
            "feat_ret_5d": "target_5d",
            "returns_5d": "target_5d",
            "feat_ret_10d": "target_10d",
            "returns_10d": "target_10d",
            "feat_ret_20d": "target_20d",
            "returns_20d": "target_20d",
        }

    def _validate_dataset(self, df: pl.DataFrame) -> None:
        """Validate input dataset"""
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure standardized target columns exist (creates target_{h}d if needed)
        df = self._ensure_target_columns(df)

        # Check again for standardized targets
        standardized = [f"target_{h}d" for h in self.prediction_horizons]
        present_std = [c for c in standardized if c in df.columns]
        if not present_std:
            # As a last resort, accept base returns columns if any
            base_any = any(col in df.columns for col in self.target_columns.keys())
            if not base_any:
                raise ValueError(
                    "No target columns found. Need at least one of: "
                    f"{list(self.target_columns.keys()) + standardized}"
                )

        logger.info(f"Dataset validated: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(
            "Target columns present: "
            + ", ".join([c for c in standardized if c in df.columns])
        )
        # Store back (caller may reuse)
        return None

    def _ensure_target_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create standardized target_{h}d columns.

        Priority per horizon h:
          1) target_{h}d already exists → keep
          2) returns_{h}d / feat_ret_{h}d → copy to target_{h}d
          3) returns_1d → forward rolling sum over next h days per Code
        """
        out = df
        # If we already have all targets, return as-is
        need = []
        for h in self.prediction_horizons:
            col = f"target_{h}d"
            if col not in out.columns:
                need.append(h)
        if not need:
            return out

        # Map from existing future-return columns first, then legacy returns
        for h in need[:]:
            for cand in (f"feat_ret_{h}d", f"returns_{h}d"):
                if cand in out.columns:
                    out = out.with_columns(pl.col(cand).alias(f"target_{h}d"))
                    need.remove(h)
                    break

        # If still missing, try to derive from returns_1d
        if need and "returns_1d" in out.columns:
            # Ensure sorted by Code, Date for forward rolling
            out = out.sort(["Code", "Date"])  # stable sort
            base = pl.col("returns_1d").cast(pl.Float64)
            for h in need[:]:
                # forward h-day sum (exclude today): shift(-1), then rolling_sum window=h over each Code
                try:
                    derived = (
                        base.shift(-1)
                        .over("Code")
                        .rolling_sum(window_size=h)
                        .alias(f"target_{h}d")
                    )
                    out = out.with_columns(derived)
                    need.remove(h)
                except Exception as e:
                    logger.warning(f"Failed to derive target_{h}d from returns_1d: {e}")

        if need:
            logger.warning(f"Could not create targets for horizons: {need}")
        return out

    def _detect_feature_columns(self, df: pl.DataFrame) -> List[str]:
        """Auto-detect feature columns (numeric columns excluding metadata)"""
        exclude_cols = {"Code", "Date", "code", "date", "index", "split_fold"}
        # Exclude obvious label/return columns to prevent leakage
        def _is_feature(c: str) -> bool:
            if c in exclude_cols:
                return False
            name = c.lower()
            # Remove targets/returns variants
            leakage_tokens = [
                "target_", "returns_", "log_returns_", "feat_ret_", "future_returns_",
                "label", "_y", "_target",
            ]
            if any(tok in name for tok in leakage_tokens):
                return False
            # Only numeric dtypes are eligible
            return df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]

        numeric_cols = [c for c in df.columns if _is_feature(c)]

        logger.info(f"Detected {len(numeric_cols)} feature columns")
        return numeric_cols

    def _apply_cross_sectional_zscores(
        self,
        frame: pl.DataFrame,
        feature_cols: Sequence[str],
    ) -> pl.DataFrame:
        """Append *_cs_z columns computed per date group and drop raw columns."""
        if not feature_cols:
            return frame

        exprs: list[pl.Expr] = []
        for col in feature_cols:
            mean_expr = pl.col(col).mean().over(self.date_column)
            std_expr = pl.col(col).std().over(self.date_column)
            z_expr = ((pl.col(col) - mean_expr) / (std_expr + 1e-8)).fill_nan(0.0)
            exprs.append(z_expr.alias(f"{col}_cs_z"))

        frame = frame.with_columns(exprs)
        # Ensure no residual nulls
        cs_cols = [f"{c}_cs_z" for c in feature_cols]
        frame = frame.with_columns(
            [pl.col(c).fill_null(0.0).alias(c) for c in cs_cols]
        )
        return frame

    def _create_sequences(
        self,
        df: pl.DataFrame,
        code: str
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Create time-series sequences for a single stock

        Returns:
            features: Array of shape (n_samples, sequence_length, n_features)
            targets: Array of shape (n_samples, n_horizons)
            dates: List of dates for each sequence
        """
        code_df = df.filter(pl.col("Code") == code).sort("Date")

        if len(code_df) < self.sequence_length + max(self.prediction_horizons):
            return np.array([]), np.array([]), []

        # Get feature values
        feature_values = code_df.select(self.all_feature_columns).to_numpy()

        # Prepare per-horizon target vectors (prefer standardized target_{h}d)
        target_series_by_h: Dict[int, np.ndarray] = {}
        for h in self.prediction_horizons:
            std_col = f"target_{h}d"
            if std_col in code_df.columns:
                target_series_by_h[h] = code_df[std_col].to_numpy()
                continue
            # fallback: returns_{h}d / feat_ret_{h}d
            for cand in (f"returns_{h}d", f"feat_ret_{h}d"):
                if cand in code_df.columns:
                    target_series_by_h[h] = code_df[cand].to_numpy()
                    break
            # last resort: use returns_1d forward rolling sum (should have been created above)
            if h not in target_series_by_h and "returns_1d" in code_df.columns:
                try:
                    # as polars expressions are not available here, approximate via numpy
                    r1 = code_df["returns_1d"].to_numpy().astype(float)
                    # forward rolling: sum r1[t+1 : t+h]
                    # compute cumulative sum and then diff with offset
                    csum = np.cumsum(np.concatenate([[0.0], r1]))
                    # Align to t: for index t, window sum = csum[t+1+h] - csum[t+1]
                    fwd = np.empty_like(r1)
                    fwd[:] = np.nan
                    for t in range(0, len(r1)):
                        t1 = t + 1
                        t2 = t + h
                        if t2 < len(csum):
                            fwd[t] = csum[t2] - csum[t1]
                    target_series_by_h[h] = fwd
                except Exception as e:
                    logger.warning(f"Failed to derive per-code forward returns for h={h}: {e}")

        dates = code_df["Date"].to_list()

        # Create sequences
        sequences = []
        targets = []
        seq_dates = []

        for i in range(len(feature_values) - self.sequence_length - max(self.prediction_horizons) + 1):
            # Feature sequence
            seq = feature_values[i:i + self.sequence_length]
            sequences.append(seq)

            # Multi-horizon targets
            horizon_targets = []
            for h in self.prediction_horizons:
                # Align standardized targets at the sequence end (t): use value at t
                series = target_series_by_h.get(h, None)
                if series is None:
                    horizon_targets.append(np.nan)
                    continue
                t_idx = i + self.sequence_length - 1
                val = series[t_idx] if t_idx < len(series) else np.nan
                horizon_targets.append(val)
            targets.append(horizon_targets)

            # Date of the last point in the sequence
            seq_dates.append(dates[i + self.sequence_length - 1])

        return np.array(sequences), np.array(targets), seq_dates

    def convert_to_atft_format(
        self,
        df: pl.DataFrame,
        output_dir: str = "output/atft_data",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        feature_columns: Optional[Sequence[str]] = None,
        static_columns: Optional[Sequence[str]] = None,
        regime_columns: Optional[Sequence[str]] = None,
        mask_columns: Optional[Sequence[str]] = None,
    ) -> Dict[str, Union[List[str], Dict]]:
        """
        Convert ML dataset to ATFT format and save as parquet files

        Args:
            df: Input DataFrame from ML dataset
            output_dir: Output directory for converted files
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            feature_columns: Optional explicit list of feature columns to use
            static_columns: Optional list of columns to expose as static features
            regime_columns: Optional list of columns for regime features (aggregated downstream)

        Returns:
            Dictionary with file paths and metadata
        """
        # Validate and normalize targets
        self._validate_dataset(df)
        df = self._ensure_target_columns(df)

        # Drop duplicated *_right columns if they sneaked through older builds
        dup_cols = [c for c in df.columns if c.endswith("_right")]
        if dup_cols:
            logger.info(f"Dropping duplicate columns: {dup_cols[:5]}{'...' if len(dup_cols) > 5 else ''}")
            df = df.drop(dup_cols)

        # Detect or override feature columns (used by downstream metadata)
        override_features: List[str] = []
        if feature_columns:
            unique_override = list(dict.fromkeys(feature_columns))
            override_features = [c for c in unique_override if c in df.columns]
            missing_override = [c for c in unique_override if c not in df.columns]
            if missing_override:
                logger.warning(
                    "Skipping %d requested feature columns not present in dataset (e.g. %s)",
                    len(missing_override),
                    ", ".join(missing_override[:5])
                    + ("..." if len(missing_override) > 5 else ""),
                )
            if override_features:
                logger.info(
                    "Using %d curated feature columns (override supplied)",
                    len(override_features),
                )
        if override_features:
            base_override = [
                col.removesuffix("_cs_z") if col.endswith("_cs_z") else col
                for col in override_features
            ]
            # Preserve order while dropping duplicates
            seen_base: dict[str, None] = {}
            base_ordered = [
                seen_base.setdefault(col, None) or col
                for col in base_override
                if col not in seen_base
            ]
            self.raw_feature_columns = base_ordered
            self.feature_columns = override_features
        else:
            detected = self._detect_feature_columns(df)
            self.raw_feature_columns = list(detected)
            self.feature_columns = detected
        logger.info(f"Using {len(self.feature_columns)} feature columns")

        mask_columns = [c for c in (mask_columns or []) if c in df.columns]
        static_columns = [c for c in (static_columns or []) if c in df.columns]
        missing_static = [c for c in (static_columns or []) if c not in df.columns]
        if missing_static:
            logger.warning(
                "Dropping %d static columns not present in dataset (e.g. %s)",
                len(missing_static),
                ", ".join(missing_static[:5])
                + ("..." if len(missing_static) > 5 else ""),
            )
        regime_columns = [c for c in (regime_columns or []) if c in df.columns]
        missing_regime = [c for c in (regime_columns or []) if c not in df.columns]
        if missing_regime:
            logger.warning(
                "Dropping %d regime columns not present in dataset (e.g. %s)",
                len(missing_regime),
                ", ".join(missing_regime[:5])
                + ("..." if len(missing_regime) > 5 else ""),
            )

        self.mask_columns = mask_columns
        self.raw_feature_columns = list(self.raw_feature_columns)
        self.dynamic_feature_columns = [
            f"{col}_cs_z" for col in self.raw_feature_columns
        ]
        combined_feature_columns = self.dynamic_feature_columns + self.mask_columns
        self.feature_columns = combined_feature_columns
        self.all_feature_columns = combined_feature_columns
        self.date_column = "Date" if "Date" in df.columns else "date"
        if self.date_column not in df.columns:
            raise ValueError(
                "Dataset must contain a Date/date column for cross-sectional normalization"
            )

        # Normalize ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if total_ratio <= 0:
            raise ValueError("Sum of train/val/test ratios must be > 0")
        train_ratio_norm = train_ratio / total_ratio
        val_ratio_norm = val_ratio / total_ratio
        test_ratio_norm = test_ratio / total_ratio

        # Sort for deterministic slicing
        df = df.sort(["Code", "Date"])

        # Prepare output directories
        output_path = Path(output_dir)
        train_dir = output_path / "train"
        val_dir = output_path / "val"
        test_dir = output_path / "test"
        for dir_path in (train_dir, val_dir, test_dir):
            dir_path.mkdir(parents=True, exist_ok=True)

        max_horizon = max(self.prediction_horizons)
        min_rows_required = max(self.sequence_length, self.min_samples_per_code)

        codes = df["Code"].unique().to_list()
        logger.info(f"Processing {len(codes)} unique stocks")

        train_files: List[str] = []
        val_files: List[str] = []
        test_files: List[str] = []

        diagnostics: List[Dict[str, Union[str, int, float, bool]]] = []

        stats = {
            "train_rows": 0,
            "val_rows": 0,
            "test_rows": 0,
            "skipped_codes": 0,
            "dropped_val_short": 0,
            "dropped_test_short": 0,
            "dropped_both_short": 0,
            "merged_val_into_test": 0,
            "merged_test_into_val": 0,
        }

        # Default to preserving codes by merging short splits, unless explicitly opted in.
        drop_short_splits = os.getenv("DROP_SHORT_SPLITS", "0").lower() in {"1", "true", "yes", "on"}

        # Minimum rows required for validation/test slices. Defaults to sequence_length.
        # Can be overridden for experimentation via env.
        try:
            min_val_test_rows = int(os.getenv("MIN_VAL_TEST_ROWS", str(self.sequence_length)))
        except ValueError:
            min_val_test_rows = self.sequence_length

        desc = "Writing ATFT splits"
        for code in tqdm(codes, desc=desc):
            code_df = df.filter(pl.col("Code") == code).sort("Date")
            n_rows = code_df.height

            if n_rows < min_rows_required:
                stats["skipped_codes"] += 1
                logger.warning(
                    "Skipping %s (rows=%d < min_required=%d)",
                    code,
                    n_rows,
                    min_rows_required,
                )
                continue

            diag_entry: Dict[str, Union[str, int, float, bool]] = {
                "code": code,
                "rows": n_rows,
                "train_rows": 0,
                "val_rows": 0,
                "test_rows": 0,
                "merged_val": False,
                "merged_test": False,
                "winsorized": False,
                "winsor_clip_min": float("nan"),
                "winsor_clip_max": float("nan"),
                "dropped_short_split": False,
                "drop_reason": "",
            }

            # Determine split boundaries
            train_end_target = int(round(n_rows * train_ratio_norm))
            val_end_target = int(round(n_rows * (train_ratio_norm + val_ratio_norm)))

            train_end = max(self.sequence_length + max_horizon - 1, train_end_target)
            train_end = min(train_end, n_rows)

            val_start = train_end
            if val_ratio_norm > 0:
                val_end = max(val_start + self.sequence_length, val_end_target)
            else:
                val_end = val_start
            val_end = min(val_end, n_rows)

            test_start = val_end
            test_end = n_rows

            winsor_applied = False
            winsor_clip_min: Optional[float] = None
            winsor_clip_max: Optional[float] = None

            code_df, winsor_meta = self._maybe_winsorize(code_df, train_end)
            if winsor_meta is not None:
                winsor_applied = True
                winsor_clip_min, winsor_clip_max = winsor_meta
                diag_entry["winsorized"] = True
                diag_entry["winsor_clip_min"] = winsor_clip_min
                diag_entry["winsor_clip_max"] = winsor_clip_max

            # Guard: handle short val/test slices with smarter merging before deciding to drop
            merged_val = False
            merged_test = False

            val_size = max(0, val_end - val_start)
            test_size = max(0, test_end - test_start)
            val_ok = (val_ratio_norm > 0) and (val_size >= min_val_test_rows)
            test_ok = (test_ratio_norm > 0) and (test_size >= min_val_test_rows)

            if not val_ok and test_ok and val_ratio_norm > 0:
                # Prefer preserving code: fold a tiny val into test
                stats["merged_val_into_test"] += 1
                logger.info(
                    "Val split for %s too short (%d rows < %d); merging into test.",
                    code,
                    val_size,
                    min_val_test_rows,
                )
                test_start = val_start
                val_start, val_end = val_start, val_start
                merged_val = True
                val_size = 0
                val_ok = False

            # Re-evaluate test after potential change in boundaries
            test_size = max(0, test_end - test_start)
            test_ok = (test_ratio_norm > 0) and (test_size >= min_val_test_rows)

            if not test_ok and (val_end > val_start):
                # Merge tiny test into existing validation if possible
                if (val_end > val_start) and (val_end - val_start) >= 0:
                    stats["merged_test_into_val"] += 1
                    logger.info(
                        "Test split for %s too short (%d rows < %d); merging into val.",
                        code,
                        test_size,
                        min_val_test_rows,
                    )
                    val_end = test_end
                    merged_test = True
                    test_start = test_end
                    test_size = 0
                    test_ok = False

            # If both val and test remain too short, decide whether to drop or keep all in train
            if (val_ratio_norm > 0 and val_end - val_start < min_val_test_rows) and (
                test_ratio_norm > 0 and test_end - test_start < min_val_test_rows
            ):
                if drop_short_splits:
                    stats["dropped_both_short"] += 1
                    diag_entry["dropped_short_split"] = True
                    diag_entry["drop_reason"] = "both_short"
                    logger.warning(
                        "Both val/test for %s too short (val=%d, test=%d; min=%d); dropping due to DROP_SHORT_SPLITS",
                        code,
                        max(0, val_end - val_start),
                        max(0, test_end - test_start),
                        min_val_test_rows,
                    )
                    diagnostics.append(diag_entry)
                    continue
                else:
                    logger.info(
                        "Both val/test for %s too short (val=%d, test=%d; min=%d); keeping all in train.",
                        code,
                        max(0, val_end - val_start),
                        max(0, test_end - test_start),
                        min_val_test_rows,
                    )
                    train_end = n_rows
                    val_start, val_end = train_end, train_end
                    test_start, test_end = train_end, train_end

            # Slice per split and persist
            train_slice: Optional[pl.DataFrame] = None
            val_slice: Optional[pl.DataFrame] = None
            test_slice: Optional[pl.DataFrame] = None

            train_slice = code_df.slice(0, train_end)
            train_slice = self._apply_cross_sectional_zscores(
                train_slice, self.raw_feature_columns
            )
            if train_slice.height >= self.sequence_length:
                train_path = train_dir / f"{code}.parquet"
                train_slice.write_parquet(train_path)
                train_files.append(str(train_path))
                stats["train_rows"] += train_slice.height
                diag_entry["train_rows"] = train_slice.height
            else:
                logger.warning("Train split for %s too short after adjustments; skipping", code)

            if val_end > val_start:
                val_slice = code_df.slice(val_start, val_end - val_start)
                val_slice = self._apply_cross_sectional_zscores(
                    val_slice, self.raw_feature_columns
                )
                if val_slice.height >= self.sequence_length:
                    val_path = val_dir / f"{code}.parquet"
                    val_slice.write_parquet(val_path)
                    val_files.append(str(val_path))
                    stats["val_rows"] += val_slice.height
                    diag_entry["val_rows"] = val_slice.height
                else:
                    logger.warning("Validation split for %s still too short; skipped", code)

            if test_end > test_start:
                test_slice = code_df.slice(test_start, test_end - test_start)
                test_slice = self._apply_cross_sectional_zscores(
                    test_slice, self.raw_feature_columns
                )
                if test_slice.height >= self.sequence_length:
                    test_path = test_dir / f"{code}.parquet"
                    test_slice.write_parquet(test_path)
                    test_files.append(str(test_path))
                    stats["test_rows"] += test_slice.height
                    diag_entry["test_rows"] = test_slice.height
                else:
                    logger.warning("Test split for %s still too short; skipped", code)

            diag_entry["merged_val"] = merged_val
            diag_entry["merged_test"] = merged_test
            diagnostics.append(diag_entry)

        combined_feature_columns = self.feature_columns
        self.all_feature_columns = combined_feature_columns

        metadata = {
            "sequence_length": self.sequence_length,
            "prediction_horizons": self.prediction_horizons,
            "n_features": len(combined_feature_columns),
            "feature_columns": combined_feature_columns,
            "raw_feature_columns": self.raw_feature_columns,
            "static_columns": static_columns,
            "regime_columns": regime_columns,
            "mask_columns": self.mask_columns,
            "n_codes_processed": len(codes),
            "train_ratio": train_ratio_norm,
            "val_ratio": val_ratio_norm,
            "test_ratio": test_ratio_norm,
            "row_stats": stats,
        }

        logger.info("✅ Conversion complete")
        logger.info(
            "Saved %d train, %d val, %d test files (skipped %d codes)",
            len(train_files), len(val_files), len(test_files), stats["skipped_codes"],
        )

        self._write_quality_report(diagnostics, output_path)

        return {
            "train_files": train_files,
            "val_files": val_files,
            "test_files": test_files,
            "metadata": metadata,
            "output_dir": str(output_path),
        }

    def _maybe_winsorize(
        self,
        code_df: pl.DataFrame,
        train_end: int,
    ) -> Tuple[pl.DataFrame, Optional[Tuple[float, float]]]:
        """Apply per-code winsorization using statistics from the training slice."""

        enable = os.getenv("ENABLE_WINSORIZE", "1").lower() in {"1", "true", "yes", "on"}
        if not enable or not self.feature_columns:
            return code_df, None

        try:
            lower_pct = float(os.getenv("FEATURE_WINSOR_LOWER_PCT", "0.01"))
            upper_pct = float(os.getenv("FEATURE_WINSOR_UPPER_PCT", "0.99"))
        except ValueError:
            logger.warning("Invalid FEATURE_WINSOR_* percentile; skipping winsorization")
            return code_df, None

        if not (0.0 <= lower_pct < upper_pct <= 1.0):
            logger.warning("FEATURE_WINSOR percentiles out of range; skipping winsorization")
            return code_df, None

        train_slice = code_df.slice(0, train_end)
        if train_slice.height <= 0:
            return code_df, None

        feature_names = self.raw_feature_columns
        if not feature_names:
            feature_names = [
                col.removesuffix("_cs_z") if col.endswith("_cs_z") else col
                for col in self.feature_columns
            ]
        feature_names = [col for col in feature_names if col in train_slice.columns]
        if not feature_names:
            return code_df, None

        feature_matrix = (
            train_slice.select(feature_names)
            .to_numpy()
            .astype(np.float64, copy=False)
        )

        if feature_matrix.size == 0:
            return code_df, None

        with np.errstate(invalid="ignore"):
            lower_bounds = np.nanquantile(feature_matrix, lower_pct, axis=0)
            upper_bounds = np.nanquantile(feature_matrix, upper_pct, axis=0)

        if np.isnan(lower_bounds).all() or np.isnan(upper_bounds).all():
            logger.warning("Winsor quantiles produced all-NaN bounds; skipping winsorization")
            return code_df, None

        clip_exprs: List[pl.Expr] = []
        min_clip = float("inf")
        max_clip = float("-inf")
        for idx, column in enumerate(feature_names):
            lo = lower_bounds[idx]
            hi = upper_bounds[idx]
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            if hi < lo:
                lo, hi = hi, lo
            clip_exprs.append(pl.col(column).clip(lo, hi))
            min_clip = min(min_clip, float(lo))
            max_clip = max(max_clip, float(hi))

        if not clip_exprs:
            return code_df, None

        clipped = code_df.with_columns(clip_exprs)
        return clipped, (min_clip, max_clip)

    def _write_quality_report(
        self,
        diagnostics: List[Dict[str, Union[str, int, float, bool]]],
        output_path: Path,
    ) -> None:
        """Persist per-code diagnostics (parquet + summary json) under _logs."""

        if not diagnostics:
            return

        report_dir = Path("_logs") / "data_quality"
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"atft_quality_{timestamp}.parquet"
        summary_path = report_dir / f"atft_quality_summary_{timestamp}.json"

        diag_df = pl.DataFrame(diagnostics)
        try:
            diag_df.write_parquet(report_path)
        except Exception as exc:
            logger.error("Failed to write quality report parquet: %s", exc)

        summary = {
            "total_codes": int(diag_df.height),
            "winsorized_codes": int(diag_df.filter(pl.col("winsorized") == True).height),
            "merged_val_codes": int(diag_df.filter(pl.col("merged_val") == True).height),
            "merged_test_codes": int(diag_df.filter(pl.col("merged_test") == True).height),
            "dropped_short_codes": int(diag_df.filter(pl.col("dropped_short_split") == True).height),
            "total_rows": int(diag_df.select(pl.col("rows").sum()).item()),
            "train_rows": int(diag_df.select(pl.col("train_rows").sum()).item()),
            "val_rows": int(diag_df.select(pl.col("val_rows").sum()).item()),
            "test_rows": int(diag_df.select(pl.col("test_rows").sum()).item()),
            "output_dir": str(output_path),
        }

        try:
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        except Exception as exc:
            logger.error("Failed to write quality report summary: %s", exc)

        logger.info("Quality report saved to %s (summary %s)", report_path, summary_path)

    def _save_chunked_parquet(
        self,
        df: pl.DataFrame,
        output_dir: Path,
        prefix: str,
        chunk_size: int = 100000
    ) -> List[str]:
        """Save DataFrame as chunked parquet files"""
        files = []
        n_chunks = (len(df) + chunk_size - 1) // chunk_size

        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(df))
            chunk = df[start:end]

            file_path = output_dir / f"{prefix}_chunk_{i:04d}.parquet"
            chunk.write_parquet(file_path)
            files.append(str(file_path))

        return files

    def convert_from_file(
        self,
        input_path: str,
        output_dir: str = "output/atft_data",
        **kwargs
    ) -> Dict[str, Union[List[str], Dict]]:
        """
        Convenience method to convert from a parquet file

        Args:
            input_path: Path to input ML dataset parquet
            output_dir: Output directory for converted files
            **kwargs: Additional arguments for convert_to_atft_format

        Returns:
            Dictionary with file paths and metadata
        """
        logger.info(f"Loading dataset from {input_path}")
        df = pl.read_parquet(input_path)

        return self.convert_to_atft_format(df, output_dir, **kwargs)


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Convert ML dataset to ATFT format")
    parser.add_argument(
        "--input",
        type=str,
        default="output/ml_dataset_latest_full.parquet",
        help="Input ML dataset parquet file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/atft_data",
        help="Output directory for ATFT format files",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Sequence length for time series",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for testing (optional)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load dataset
    df = pl.read_parquet(args.input)

    # Sample if requested
    if args.sample_size:
        df = df.sample(args.sample_size)
        logger.info(f"Sampled {args.sample_size} rows for testing")

    # Convert
    converter = UnifiedFeatureConverter(sequence_length=args.sequence_length)
    result = converter.convert_to_atft_format(df, args.output)

    logger.info("Conversion completed successfully")
    logger.info(f"Output directory: {result['output_dir']}")
    logger.info(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    main()
