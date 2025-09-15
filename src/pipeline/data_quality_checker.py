from __future__ import annotations

"""Pipeline data quality validation utilities (Polars-based)."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import polars as pl


@dataclass
class QualityConfig:
    outlier_threshold: float = 3.0
    missing_threshold: float = 0.1
    save_report: bool = True


class DataQualityChecker:
    """Run lightweight data quality checks on the final dataset."""

    def __init__(self, cfg: Optional[QualityConfig] = None) -> None:
        self.cfg = cfg or QualityConfig()

    def validate_dataset(self, df: pl.DataFrame) -> Dict[str, Any]:
        num_cols = [c for c, dt in zip(df.columns, df.dtypes) if pl.datatypes.is_numeric(dt)]
        date_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Date or dt == pl.Datetime]

        results: Dict[str, Any] = {}

        # Missingness
        total = df.height * len(df.columns)
        miss_by_col = {c: int(df[c].null_count()) for c in df.columns}
        miss_total = int(sum(miss_by_col.values()))
        missing_ratio = (miss_total / total) if total else 0.0
        results["missing_ratio"] = missing_ratio
        results["missing_by_col"] = miss_by_col

        # Outliers via z-score cutoff (per column)
        outlier_stats: Dict[str, float] = {}
        for c in num_cols:
            try:
                s = df[c].cast(pl.Float64)
                mean = float(s.mean() or 0.0)
                std = float(s.std() or 0.0)
                if std == 0.0:
                    outlier_stats[c] = 0.0
                    continue
                z = (s - mean) / std
                frac = float((z.abs() > self.cfg.outlier_threshold).sum() / df.height)
                outlier_stats[c] = frac
            except Exception:
                outlier_stats[c] = 0.0
        results["outlier_fraction_by_col"] = outlier_stats

        # Temporal consistency (coarse):
        temporal: Dict[str, Any] = {}
        if date_cols:
            tcol = date_cols[0]
            try:
                s = df[tcol].sort().drop_nulls()
                gaps = 0
                if s.len() > 1:
                    # treat as daily frequency; count gaps where delta_days > 1
                    deltas = (s.cast(pl.Date).diff().dt.total_days()).fill_null(0)
                    gaps = int((deltas > 1).sum())
                temporal["date_col"] = tcol
                temporal["gap_days_over_1"] = gaps
                temporal["min_date"] = str(s.min()) if s.len() else None
                temporal["max_date"] = str(s.max()) if s.len() else None
            except Exception:
                temporal["date_col"] = tcol
                temporal["gap_days_over_1"] = None
        results["temporal_consistency"] = temporal

        # Feature distribution summary for numeric columns
        dist: Dict[str, Dict[str, float]] = {}
        for c in num_cols:
            try:
                s = df[c].cast(pl.Float64)
                dist[c] = {
                    "mean": float(s.mean() or 0.0),
                    "std": float(s.std() or 0.0),
                    "p01": float(s.quantile(0.01, "nearest") or 0.0),
                    "p50": float(s.quantile(0.50, "nearest") or 0.0),
                    "p99": float(s.quantile(0.99, "nearest") or 0.0),
                }
            except Exception:
                continue
        results["feature_distribution"] = dist

        # Health flags
        results["health_flags"] = {
            "missing_exceeds_threshold": bool(missing_ratio > self.cfg.missing_threshold),
        }

        return results

    def generate_quality_report(self, results: Dict[str, Any]) -> str:
        lines = []
        lines.append("# Data Quality Report")
        lines.append("")
        lines.append(f"Missing ratio: {results.get('missing_ratio', 0.0):.4f}")
        flags = results.get("health_flags", {})
        if flags.get("missing_exceeds_threshold"):
            lines.append("- ⚠️ Missing ratio exceeds threshold")
        else:
            lines.append("- ✅ Missing ratio within threshold")

        temporal = results.get("temporal_consistency", {})
        if temporal:
            lines.append("")
            lines.append("Temporal Consistency:")
            lines.append(f"- Date col: {temporal.get('date_col')}")
            lines.append(f"- Min date: {temporal.get('min_date')}")
            lines.append(f"- Max date: {temporal.get('max_date')}")
            lines.append(f"- Gap days (>1): {temporal.get('gap_days_over_1')}")

        lines.append("")
        lines.append("Numeric Feature Distribution (mean/std/p01/p50/p99):")
        for c, st in list(results.get("feature_distribution", {}).items())[:25]:
            lines.append(
                f"- {c}: mean={st['mean']:.4f}, std={st['std']:.4f}, p01={st['p01']:.4f}, p50={st['p50']:.4f}, p99={st['p99']:.4f}"
            )
        return "\n".join(lines)

