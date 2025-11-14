"""Dataset quality checks shared by CLI and builders."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import polars as pl


def _ensure_datetime(value):
    if isinstance(value, date):
        return value
    if hasattr(value, "date"):
        return value.date()
    raise TypeError(f"Unsupported date type: {type(value)!r}")


def parse_asof_specs(specs: Sequence[str]) -> List[Tuple[str, str]]:
    """Convert CLI-style 'column<=reference' specs into tuples."""
    pairs: List[Tuple[str, str]] = []
    for spec in specs:
        item = spec.strip()
        if not item:
            continue
        if "<=" not in item:
            raise ValueError(f"Invalid as-of spec '{spec}'. Expected format 'col<=reference_col'.")
        left, right = item.split("<=", 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            raise ValueError(f"Invalid as-of spec '{spec}'.")
        pairs.append((left, right))
    return pairs


@dataclass
class CheckResult:
    status: str
    details: Dict[str, object] = field(default_factory=dict)
    message: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"status": self.status}
        if self.message:
            payload["message"] = self.message
        if self.details:
            payload["details"] = self.details
        return payload


def _check_primary_key(
    lf: pl.LazyFrame, date_col: str, code_col: str, sample_rows: int
) -> CheckResult:
    dup = (
        lf.group_by([date_col, code_col])
        .agg(pl.len().alias("row_count"))
        .filter(pl.col("row_count") > 1)
    )
    dup_df = dup.collect()
    if dup_df.is_empty():
        return CheckResult(status="ok")

    extra_rows = int((dup_df["row_count"] - 1).sum())
    sample = dup_df.head(sample_rows).select([date_col, code_col, "row_count"]).to_dicts()
    return CheckResult(
        status="error",
        message=f"{extra_rows} duplicated rows detected for ({date_col}, {code_col})",
        details={"sample_duplicates": sample, "duplicate_groups": dup_df.height, "extra_rows": extra_rows},
    )


def _check_future_dates(
    lf: pl.LazyFrame,
    date_col: str,
    code_col: str,
    allow_future_days: int,
    sample_rows: int,
) -> CheckResult:
    cutoff = date.today() + timedelta(days=allow_future_days)
    future = (
        lf.filter(pl.col(date_col) > pl.lit(cutoff))
        .select([date_col, code_col])
    )
    future_rows = future.limit(sample_rows).collect()
    total_future = future.select(pl.len()).collect().item()
    if total_future == 0:
        return CheckResult(status="ok")
    return CheckResult(
        status="error",
        message=f"{total_future} rows have {date_col} beyond {cutoff.isoformat()}",
        details={"sample_rows": future_rows.to_dicts()},
    )


def _check_target_nulls(lf: pl.LazyFrame, targets: Sequence[str]) -> CheckResult:
    if not targets:
        return CheckResult(status="ok")
    available = set(lf.columns)
    missing = [col for col in targets if col not in available]
    existing = [col for col in targets if col in available]
    if not existing:
        return CheckResult(
            status="warning",
            message=f"Target columns not found: {', '.join(missing)}",
        )

    exprs = [pl.col(target).is_null().sum().alias(target) for target in existing]
    counts = lf.select(exprs).collect().row(0)
    nulls = {target: int(counts[i]) for i, target in enumerate(existing)}
    offenders = {k: v for k, v in nulls.items() if v > 0}
    status = "ok" if not offenders else "error"
    message = None if status == "ok" else f"Targets contain nulls: {offenders}"
    details: Dict[str, object] = nulls
    if missing:
        status = "warning" if status == "ok" else status
        message = (message or "Targets checked").strip()
        details = {**details, "missing_targets": missing}
        if status == "warning" and not offenders:
            message = f"Targets missing: {missing}"
    return CheckResult(status=status, message=message, details=details)


def _check_asof_pairs(
    lf: pl.LazyFrame,
    pairs: Sequence[Tuple[str, str]],
    sample_rows: int,
) -> CheckResult:
    if not pairs:
        return CheckResult(status="ok")

    available = set(lf.columns)
    violations: Dict[str, Dict[str, object]] = {}
    missing_pairs: Dict[str, List[str]] = {}
    for left, right in pairs:
        if left not in available or right not in available:
            missing_pairs[f"{left}<={right}"] = [col for col in (left, right) if col not in available]
            continue
        cond = (
            pl.col(left).is_not_null()
            & pl.col(right).is_not_null()
            & (pl.col(left) > pl.col(right))
        )
        bad = (
            lf.filter(cond)
            .select([left, right])
        )
        total = bad.select(pl.len()).collect().item()
        if total > 0:
            violations[f"{left}<={right}"] = {
                "rows": total,
                "sample": bad.limit(sample_rows).collect().to_dicts(),
            }
    if not violations:
        if missing_pairs:
            return CheckResult(
                status="warning",
                message="Skipped as-of checks (columns missing)",
                details=missing_pairs,
            )
        return CheckResult(status="ok")
    return CheckResult(
        status="error",
        message="As-of ordering violations detected",
        details=violations,
    )


def run_quality_checks(
    dataset_path: Path | str,
    *,
    date_col: str = "date",
    code_col: str = "code",
    targets: Optional[Sequence[str]] = None,
    asof_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    allow_future_days: int = 0,
    sample_rows: int = 5,
) -> Tuple[Dict[str, Dict[str, object]], List[str], List[str]]:
    """Run all dataset quality checks and return summary + errors + warnings."""

    lf = pl.scan_parquet(str(dataset_path))

    results: Dict[str, CheckResult] = {}
    errors: List[str] = []
    warnings: List[str] = []

    pk_res = _check_primary_key(lf, date_col, code_col, sample_rows)
    results["primary_key"] = pk_res
    if pk_res.status == "error":
        errors.append(pk_res.message or "primary key violation")
    elif pk_res.status == "warning":
        warnings.append(pk_res.message or "primary key warning")

    fut_res = _check_future_dates(lf, date_col, code_col, allow_future_days, sample_rows)
    results["future_dates"] = fut_res
    if fut_res.status == "error":
        errors.append(fut_res.message or "future date violation")
    elif fut_res.status == "warning":
        warnings.append(fut_res.message or "future date warning")

    target_res = _check_target_nulls(lf, targets or [])
    results["targets"] = target_res
    if target_res.status == "error":
        errors.append(target_res.message or "target nulls detected")
    elif target_res.status == "warning":
        warnings.append(target_res.message or "target warning")

    asof_res = _check_asof_pairs(lf, asof_pairs or [], sample_rows)
    results["asof"] = asof_res
    if asof_res.status == "error":
        errors.append(asof_res.message or "as-of ordering violation")
    elif asof_res.status == "warning":
        warnings.append(asof_res.message or "as-of warning")

    summary = {name: result.to_dict() for name, result in results.items()}
    return summary, errors, warnings
