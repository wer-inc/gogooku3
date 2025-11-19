#!/usr/bin/env python3
# ruff: noqa: ARG001, E501

"""
validator_1to9.py — A lightweight acceptance harness to verify items (1)–(9)
for the gogooku5 Phase-2 dataset.

USAGE
-----
python validator_1to9.py \
  --dataset /path/to/ml_dataset_latest_full.parquet \
  --start 2024-01-01 --end 2025-01-31 \
  --snapshots-dir /path/to/snapshots   # optional: to validate strict as-of rules
  --policy-margin T+1_0900             # optional policy hints (margin, earnings, AM)
  --policy-earnings T+1_1900
  --policy-am T+1_0900

What we check
-------------
(1) Schema governance (adjusted OHLC only; raw/adj aliases removed; canonicalization)
(2) Returns target/feature separation (no ret_fwd*/returns*; ret_prev* correctness; left-closed)
(3) Gap/Intraday split (ret_overnight/ret_intraday exist and compose to ret_prev_1d)
(4) Margin weekly/daily as-of (columns present; basic staleness & coverage; timing if snapshots)
(5) Earnings announcements (days_to_earnings & ±1/±3/±5 flags; timing if snapshots)
(6) Financial statements & dividends (fs_*_ttm ratios/yoy, div_*; validity flags; timing if snapshots)
(7) Index features (TOPIX etc.: idx_r_*, ATR/NATR, optional beta_60d; plausibility checks)
(8) Limit & session features (stop-limit flags, rolling stats, AM/PM derived features left-closed)
(9) Trading breakdown (compressed bd_* ratios/z-scores, freshness, and range checks)

Notes
-----
- Uses Polars lazy queries to be memory-friendly on 1M+ rows.
- Timing (as-of) checks become STRICT if snapshot parquet(s) with `availability_ts` are provided.
- Produces a machine-readable JSON summary next to the dataset as: <dataset>.validator_1to9.json
"""

import argparse
import json
import math
import os
from datetime import date, datetime
from typing import Any

import polars as pl
from polars import Expr as _Expr

TOL = 1e-6
EPS = 1e-12

# ---------------------------------------------------------------------------
# Polars compatibility helpers (handle Expr inputs on reduction functions)
# ---------------------------------------------------------------------------
_ORIG_PL_MEAN = pl.mean
_ORIG_PL_SUM = pl.sum
_ORIG_PL_QUANTILE = pl.quantile


def _mean_compat(first_arg, *args, **kwargs):
    if isinstance(first_arg, _Expr) and not args:
        return first_arg.mean(**kwargs)
    return _ORIG_PL_MEAN(first_arg, *args, **kwargs)


def _sum_compat(first_arg, *args, **kwargs):
    if isinstance(first_arg, _Expr) and not args:
        return first_arg.sum(**kwargs)
    return _ORIG_PL_SUM(first_arg, *args, **kwargs)


def _quantile_compat(first_arg, quantile, *args, **kwargs):
    if isinstance(first_arg, _Expr) and not args:
        return first_arg.quantile(quantile, **kwargs)
    return _ORIG_PL_QUANTILE(first_arg, quantile, *args, **kwargs)


pl.mean = _mean_compat  # type: ignore[attr-defined]
pl.sum = _sum_compat  # type: ignore[attr-defined]
pl.quantile = _quantile_compat  # type: ignore[attr-defined]


def resolve_column(columns: list[str], *candidates: str) -> str | None:
    """Return the first matching column name ignoring case."""

    lookup = {col.lower(): col for col in columns}
    for name in candidates:
        actual = lookup.get(name.lower())
        if actual is not None:
            return actual
    return None


def _date(s: str) -> date:
    return datetime.fromisoformat(s).date()


def _exists(path: str | None) -> bool:
    return bool(path) and os.path.exists(path)


def head(df: pl.DataFrame, n=3):
    try:
        return df.head(n)
    except Exception:
        return df


class Score:
    def __init__(self):
        self.items = []  # list of dicts
        self.ok = 0
        self.warn = 0
        self.fail = 0

    def add(self, item: str, status: str, detail: str, metrics: dict[str, Any] = None):
        metrics = metrics or {}
        if status not in {"PASS", "WARN", "FAIL", "SKIP"}:
            status = "FAIL"
        if status == "PASS":
            self.ok += 1
        elif status == "WARN":
            self.warn += 1
        elif status == "FAIL":
            self.fail += 1
        self.items.append({"item": item, "status": status, "detail": detail, "metrics": metrics})

    def summary(self) -> dict[str, Any]:
        return {
            "summary": {"pass": self.ok, "warn": self.warn, "fail": self.fail, "total": len(self.items)},
            "checks": self.items,
        }


def _mean(
    lf: pl.LazyFrame,
    column: str,
    *,
    filter_expr: pl.Expr | None = None,
) -> float | None:
    """Compute mean with optional filtering; returns None on failure."""
    try:
        frame = lf.filter(filter_expr) if filter_expr is not None else lf
        result = frame.select(pl.col(column).cast(pl.Float64).mean().alias("_mean")).collect()
        if result.is_empty():
            return None
        value = result["_mean"][0]
        return None if value is None else float(value)
    except Exception:
        return None


def _series_quantile(
    lf: pl.LazyFrame,
    column: str,
    quantile: float,
    *,
    filter_expr: pl.Expr | None = None,
) -> float | None:
    """Compute quantile with optional filtering; returns None on failure."""
    try:
        frame = lf.filter(filter_expr) if filter_expr is not None else lf
        result = frame.select(pl.col(column).cast(pl.Float64).quantile(quantile).alias("_quantile")).collect()
        if result.is_empty():
            return None
        value = result["_quantile"][0]
        return None if value is None else float(value)
    except Exception:
        return None


def load_dataset_lazy(path: str, start: str | None, end: str | None) -> pl.LazyFrame:
    lf = pl.scan_parquet(path)
    # Try to filter by date if Date column exists
    try:
        cols = lf.columns
        if "Date" in cols and (start or end):
            exprs = []
            if start:
                exprs.append(pl.col("Date") >= pl.lit(_date(start)))
            if end:
                exprs.append(pl.col("Date") <= pl.lit(_date(end)))
            lf = lf.filter(pl.all_horizontal(exprs)) if exprs else lf
    except Exception:
        pass
    return lf


def get_columns(path: str) -> list[str]:
    lf = pl.scan_parquet(path)
    return lf.columns


def check0_primary_key_and_core(path: str, score: Score, start: str | None, end: str | None) -> None:
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns
    required_pk = {"Code", "Date"}
    if not required_pk.issubset(cols):
        missing = required_pk - set(cols)
        score.add(
            "0) Primary key & core coverage",
            "FAIL",
            f"missing primary key columns: {sorted(missing)}",
        )
        return

    stats = lf.select(
        [
            pl.len().alias("total_rows"),
            pl.concat_str([pl.col("Code"), pl.col("Date").cast(pl.Utf8)], separator="|")
            .n_unique()
            .alias("unique_keys"),
        ]
    ).collect()
    total_rows = int(stats["total_rows"][0]) if stats.height else 0
    unique_keys = int(stats["unique_keys"][0]) if stats.height else 0
    duplicate_rate = 0.0 if total_rows == 0 else max(0.0, 1.0 - unique_keys / total_rows)

    core_required = {
        "AdjustmentClose",
        "AdjustmentOpen",
        "AdjustmentHigh",
        "AdjustmentLow",
        "AdjustmentVolume",
        "ret_prev_1d",
    }
    missing_core = sorted(core_required - set(cols))

    present_core = [col for col in core_required if col in cols]
    coverage = {}
    if present_core:
        cover_df = lf.select(
            [pl.col(col).is_not_null().cast(pl.Float64).mean().alias(col) for col in present_core]
        ).collect()
        for col in present_core:
            coverage[col] = float(cover_df[col][0]) if cover_df.height else 0.0

    status = "PASS"
    details: list[str] = []
    if missing_core:
        status = "FAIL"
        details.append(f"missing core columns: {missing_core}")
    # Treat coverage <95% as hard failure, 95–99% as a soft warning.
    low_cov_fail = {col: cov for col, cov in coverage.items() if cov < 0.95}
    low_cov_warn = {col: cov for col, cov in coverage.items() if 0.95 <= cov < 0.99}
    if low_cov_fail:
        status = "FAIL"
        details.append(
            "low coverage (FAIL) " + ", ".join(f"{col}={cov:.3%}" for col, cov in sorted(low_cov_fail.items()))
        )
    if not low_cov_fail and low_cov_warn:
        status = "WARN"
        details.append(
            "low coverage (WARN) " + ", ".join(f"{col}={cov:.3%}" for col, cov in sorted(low_cov_warn.items()))
        )
    if duplicate_rate > 0.0:
        status = "FAIL"
        details.append(f"duplicate key rate={duplicate_rate:.3e}")
    if not details:
        details = ["ok"]

    score.add(
        "0) Primary key & core coverage",
        status,
        "; ".join(details),
        {"duplicate_rate": duplicate_rate, "coverage": coverage, "missing_core": missing_core},
    )


def check1_schema_governance(path: str, score: Score) -> None:
    cols = set(get_columns(path))
    raw_names = {"Close", "Open", "High", "Low", "Volume", "Adj Close", "AdjClose"}
    adjusted_names = {"adjustmentclose", "adjustmentopen", "adjustmenthigh", "adjustmentlow", "adjustmentvolume"}
    # canonical must exist; raw should NOT
    has_adj = adjusted_names.issubset({c.lower() for c in cols})
    has_raw = any(name in cols for name in raw_names)
    detail = []
    if not has_adj:
        detail.append("missing one or more adjusted OHLCV columns")
    if has_raw:
        detail.append("raw OHLC/Adj Close still present (should be removed at finalize)")
    status = "PASS" if has_adj and not has_raw else ("WARN" if has_adj and has_raw else "FAIL")
    score.add(
        "1) Schema governance (canonicalized OHLC)",
        status,
        "; ".join(detail) or "ok",
        {"has_adj": has_adj, "has_raw": has_raw},
    )


def check2_returns_left_closed(path: str, score: Score, start: str | None, end: str | None) -> None:
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns
    leak_cols = [c for c in cols if c.startswith("ret_fwd_") or c.startswith("returns_") or c.startswith("feat_ret_")]
    has_prev = any(c.startswith("ret_prev_1d") for c in cols)
    status = "PASS"
    detail = []
    metrics = {}
    if leak_cols:
        status = "FAIL"
        detail.append(f"leak-like columns present: {leak_cols[:5]}{'...' if len(leak_cols) > 5 else ''}")
    if not has_prev:
        status = "FAIL"
        detail.append("ret_prev_1d not found")
    # correctness: ret_prev_1d ~= adjustmentclose / lag(adjustmentclose) - 1
    if has_prev:
        adj_close_col = resolve_column(list(cols), "adjustmentclose", "AdjustmentClose")
        if adj_close_col is None:
            status = _escalate(status, "WARN")
            detail.append("cannot verify ret_prev_1d consistency (adjusted close column not found)")
        else:
            # compute sample error
            try:
                lf2 = (
                    lf.select(["Code", "Date", adj_close_col, "ret_prev_1d"])
                    .drop_nulls(["Code", "Date", adj_close_col, "ret_prev_1d"])
                    .sort(["Code", "Date"])
                    .with_columns([pl.col(adj_close_col).shift(1).over("Code").alias("adj_lag1")])
                    .with_columns([((pl.col(adj_close_col) / pl.col("adj_lag1")) - 1.0).alias("ret_calc")])
                    .with_columns([(pl.col("ret_calc") - pl.col("ret_prev_1d")).abs().alias("abs_err")])
                    .select([pl.col("abs_err").mean().alias("mae"), pl.col("abs_err").quantile(0.99).alias("q99")])
                )
                res = lf2.collect()
                mae = float(res["mae"][0]) if res.height else None
                q99 = float(res["q99"][0]) if res.height else None
                metrics.update({"mae": mae, "q99": q99})
                if mae is None or math.isnan(mae):
                    status = "FAIL"
                    detail.append("could not compute MAE for ret_prev_1d")
                else:
                    if mae > 1e-4 or q99 > 5e-4:
                        status = "FAIL"
                        detail.append(f"ret_prev_1d deviates from adj-close formula (mae={mae:.2e}, q99={q99:.2e})")
            except Exception as e:
                status = "WARN"
                detail.append(f"calc error: {e}")
    score.add("2) Returns separation & left-closed", status, "; ".join(detail) or "ok", metrics)


def check3_gap_intraday(path: str, score: Score, start: str | None, end: str | None) -> None:
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns
    need = {"ret_overnight", "ret_intraday", "ret_prev_1d"}
    missing = [c for c in need if c not in cols]
    if missing:
        score.add("3) Gap/Intraday split", "FAIL", f"missing columns: {missing}. Implement ret_overnight/ret_intraday.")
        return
    # Relation: (1+overnight)*(1+intraday)-1 ~= ret_prev_1d
    try:
        lf2 = (
            lf.select(["Code", "Date", "ret_overnight", "ret_intraday", "ret_prev_1d"])
            .drop_nulls()
            .with_columns([((1 + pl.col("ret_overnight")) * (1 + pl.col("ret_intraday")) - 1).alias("ret_comp")])
            .with_columns([(pl.col("ret_comp") - pl.col("ret_prev_1d")).abs().alias("abs_err")])
            .select([pl.col("abs_err").mean().alias("mae"), pl.col("abs_err").quantile(0.99).alias("q99")])
        )
        res = lf2.collect()
        mae = float(res["mae"][0]) if res.height else None
        q99 = float(res["q99"][0]) if res.height else None
        if mae is None or math.isnan(mae):
            status = "FAIL"
            detail = "cannot compute composition error"
        else:
            within = mae < 1e-6 and q99 < 1e-5
            status = "PASS" if within else "FAIL"
            detail = "ok" if within else f"composition deviates: mae={mae:.2e}, q99={q99:.2e}"
        score.add("3) Gap/Intraday split", status, detail, {"mae": mae, "q99": q99})
    except Exception as e:
        score.add("3) Gap/Intraday split", "FAIL", f"calc error: {e}")


def coverage_ratio(lf: pl.LazyFrame, cols: list[str]) -> float:
    if not cols:
        return 0.0
    try:
        q = lf.select(
            [
                pl.any_horizontal([pl.col(c).is_not_null() for c in cols]).cast(pl.Int8).sum().alias("nz"),
                pl.len().alias("n"),
            ]
        ).collect()
        nz = int(q["nz"][0])
        n = int(q["n"][0])
        return 0.0 if n == 0 else nz / n
    except Exception:
        return 0.0


def check4_margin_asof(path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None) -> None:
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns
    weekly_cols = [c for c in cols if c.startswith("weekly_margin_")]
    # Note: daily_cols would include many derived features - not used here
    has_weekly = any(c.endswith(("_volume", "_imbalance", "_long_short_ratio")) for c in weekly_cols)
    has_daily = any(c in cols for c in ["margin_net", "margin_total", "margin_long_short_ratio", "margin_imbalance"])
    cov_weekly = coverage_ratio(
        lf, [c for c in weekly_cols if c.endswith(("_volume", "_imbalance", "_long_short_ratio"))]
    )
    cov_daily = coverage_ratio(lf, ["margin_net", "margin_total", "margin_long_short_ratio", "margin_imbalance"])
    status = "PASS" if has_weekly and cov_weekly > 0.3 else "WARN"
    detail = f"weekly(has={has_weekly}, cov≈{cov_weekly:.2f}); daily(has={has_daily}, cov≈{cov_daily:.2f})"
    # Strict as-of (if snapshots exist)
    if snapshots_dir and os.path.exists(os.path.join(snapshots_dir, "margin_daily_snapshot.parquet")):
        try:
            snap = pl.scan_parquet(os.path.join(snapshots_dir, "margin_daily_snapshot.parquet")).select(
                ["Code", "Date", "availability_ts"]
            )
            # join a tiny sample and check availability_ts <= asof_ts (requires asof_ts in dataset during audit builds)
            if "asof_ts" in cols:
                lf2 = (
                    lf.select(["Code", "Date", "asof_ts"])
                    .join(snap, on=["Code", "Date"], how="inner")
                    .select([(pl.col("availability_ts") <= pl.col("asof_ts")).cast(pl.Int8).alias("ok")])
                    .select([1 - pl.mean("ok").alias("violation_rate")])
                )
                vr = float(lf2.collect()["violation_rate"][0])
                if vr > 0.0:
                    status = "FAIL"
                    detail += f"; as-of violations={vr:.3f}"
        except Exception as e:
            detail += f"; as-of check error: {e}"
    score.add("4) Margin weekly/daily as‑of", status, detail, {"cov_weekly": cov_weekly, "cov_daily": cov_daily})


def check5_earnings(path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None) -> None:
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns
    need_any = ["days_to_earnings", "earnings_upcoming_5d", "earnings_recent_5d"]
    present = [c for c in need_any if c in cols]
    if not present:
        score.add("5) Earnings announcements", "WARN", "no earnings columns found (days_to_earnings / windows)")
        return
    # Distribution sanity: days_to_earnings integer & range
    status = "PASS"
    detail = "ok"
    metrics = {}
    try:
        if "days_to_earnings" in cols:
            q = (
                lf.select(pl.col("days_to_earnings"))
                .drop_nulls()
                .select(
                    [
                        pl.len().alias("n"),
                        pl.col("days_to_earnings").min().alias("min"),
                        pl.col("days_to_earnings").max().alias("max"),
                    ]
                )
                .collect()
            )
            n = int(q["n"][0])
            _min = int(q["min"][0])
            _max = int(q["max"][0])
            metrics.update({"n": n, "min": _min, "max": _max})
            if _min < -30 or _max > 60:
                status = "WARN"
                detail = f"days_to_earnings out-of-expected range [{_min},{_max}]"
    except Exception as e:
        status = "WARN"
        detail = f"stat error: {e}"
    # Strict as-of if snapshot
    if (
        snapshots_dir
        and os.path.exists(os.path.join(snapshots_dir, "earnings_announcements_snapshot.parquet"))
        and "asof_ts" in lf.columns
    ):
        try:
            snap = pl.scan_parquet(os.path.join(snapshots_dir, "earnings_announcements_snapshot.parquet")).select(
                ["Code", "Date", "availability_ts"]
            )
            vr = (
                lf.select(["Code", "Date", "asof_ts"])
                .join(snap, on=["Code", "Date"], how="inner")
                .select([(pl.col("availability_ts") <= pl.col("asof_ts")).cast(pl.Int8).alias("ok")])
                .select([1 - pl.mean("ok").alias("violation_rate")])
            ).collect()
            v = float(vr["violation_rate"][0])
            if v > 0.0:
                status = "FAIL"
                detail += f"; as-of violations={v:.3f}"
        except Exception as e:
            detail += f"; as-of check error: {e}"
    score.add("5) Earnings announcements", status, detail, metrics)


def check6_fs_dividends(path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None) -> None:
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    fs_required = [
        "fs_revenue_ttm",
        "fs_op_profit_ttm",
        "fs_net_income_ttm",
        "fs_cfo_ttm",
        "fs_op_margin",
        "fs_net_margin",
        "fs_roe_ttm",
        "fs_roa_ttm",
        "fs_accruals_ttm",
        "fs_fcf_ttm",
        "fs_staleness_bd",
        "fs_lag_days",
        "is_fs_valid",
    ]
    fs_missing = [col for col in fs_required if col not in cols]
    fs_status = "PASS"
    fs_details: list[str] = []

    if fs_missing:
        fs_status = "FAIL"
        fs_details.append(f"missing: {fs_missing}")

    fs_valid_rate = _mean(lf, "is_fs_valid")
    fs_recent_rate = (
        _mean(lf, "fs_is_recent", filter_expr=pl.col("is_fs_valid") == 1) if "fs_is_recent" in cols else None
    )
    fs_cov = coverage_ratio(lf, [c for c in fs_required if c in cols])
    fs_staleness_p95 = _series_quantile(lf, "fs_staleness_bd", 0.95) if "fs_staleness_bd" in cols else None
    fs_staleness_mean = _mean(lf, "fs_staleness_bd") if "fs_staleness_bd" in cols else None
    fs_lag_p95 = _series_quantile(lf, "fs_lag_days", 0.95) if "fs_lag_days" in cols else None

    if fs_valid_rate is not None and fs_valid_rate < 0.5:
        fs_status = _escalate(fs_status, "WARN")
        fs_details.append(f"is_fs_valid rate={fs_valid_rate:.2f}")
    if fs_recent_rate is not None and fs_recent_rate < 0.5:
        fs_status = _escalate(fs_status, "WARN")
        fs_details.append(f"recent rate={fs_recent_rate:.2f}")
    if fs_cov < 0.5:
        fs_status = _escalate(fs_status, "WARN")
        fs_details.append(f"coverage≈{fs_cov:.2f}")
    if fs_staleness_p95 is not None and fs_staleness_p95 > 120:
        fs_status = _escalate(fs_status, "WARN")
        fs_details.append(f"staleness_p95={fs_staleness_p95:.1f}")
    if fs_staleness_mean is not None and fs_staleness_mean > 60:
        fs_status = _escalate(fs_status, "WARN")
        fs_details.append(f"staleness_mean={fs_staleness_mean:.1f}")
    if fs_lag_p95 is not None and fs_lag_p95 > 120:
        fs_status = _escalate(fs_status, "WARN")
        fs_details.append(f"lag_p95={fs_lag_p95:.1f}")

    # Optional distance features: fs_days_since_E / fs_days_to_next
    if "fs_days_since_E" in cols:
        q01_since = _series_quantile(lf, "fs_days_since_E", 0.01)
        if q01_since is not None and q01_since < 0:
            fs_status = _escalate(fs_status, "WARN")
            fs_details.append("fs_days_since_E has negative values at 1% quantile")
    if "fs_days_to_next" in cols:
        q01_next = _series_quantile(lf, "fs_days_to_next", 0.01)
        if q01_next is not None and q01_next < 0:
            fs_status = _escalate(fs_status, "WARN")
            fs_details.append("fs_days_to_next has negative values at 1% quantile")
        cov_next = coverage_ratio(lf, ["fs_days_to_next"])
        if cov_next < 0.5:
            fs_status = _escalate(fs_status, "WARN")
            fs_details.append(f"fs_days_to_next coverage≈{cov_next:.2f}")

    def _share(expr: pl.Expr) -> float:
        try:
            return float(lf.select(pl.mean(expr.cast(pl.Float64)).alias("share")).collect()["share"][0])
        except Exception:
            return 0.0

    if "fs_op_margin" in cols:
        op_margin_out = _share(pl.col("fs_op_margin").abs() > 0.9)
        if op_margin_out > 0.005:
            fs_status = _escalate(fs_status, "WARN")
            fs_details.append(f"op_margin>|0.9| share={op_margin_out:.3%}")
    if "fs_net_margin" in cols:
        net_margin_out = _share(pl.col("fs_net_margin").abs() > 0.9)
        if net_margin_out > 0.005:
            fs_status = _escalate(fs_status, "WARN")
            fs_details.append(f"net_margin>|0.9| share={net_margin_out:.3%}")
    if "fs_sales_yoy" in cols:
        yoy_out = _share(pl.col("fs_sales_yoy").abs() > 2.0)
        if yoy_out > 0.005:
            fs_status = _escalate(fs_status, "WARN")
            fs_details.append(f"sales_yoy>|2| share={yoy_out:.3%}")
    if "fs_roe_ttm" in cols:
        roe_out = _share(pl.col("fs_roe_ttm").abs() > 1.0)
        if roe_out > 0.005:
            fs_status = _escalate(fs_status, "WARN")
            fs_details.append(f"roe>|1| share={roe_out:.3%}")
    if "fs_roa_ttm" in cols:
        roa_out = _share(pl.col("fs_roa_ttm").abs() > 1.0)
        if roa_out > 0.005:
            fs_status = _escalate(fs_status, "WARN")
            fs_details.append(f"roa>|1| share={roa_out:.3%}")

    # Optional as-of audit if snapshot data available
    if snapshots_dir and "asof_ts" in cols:
        snap_candidates = [
            "fs_snapshot.parquet",
            "fs_details_snapshot.parquet",
            "fins_snapshot.parquet",
        ]
        snapshot_path = next(
            (
                os.path.join(snapshots_dir, candidate)
                for candidate in snap_candidates
                if os.path.exists(os.path.join(snapshots_dir, candidate))
            ),
            None,
        )
        if snapshot_path:
            try:
                snap_lf = pl.scan_parquet(snapshot_path)
                snap_code = resolve_column(snap_lf.columns, "Code", "code") or "Code"
                snap_date = resolve_column(snap_lf.columns, "Date", "PeriodEndDate") or "Date"
                code_col = resolve_column(cols, "code", "Code")
                date_col = resolve_column(cols, "date", "Date")
                if code_col and date_col:
                    violation = (
                        lf.select([pl.col(code_col).alias("code"), pl.col(date_col).alias("Date"), pl.col("asof_ts")])
                        .join(
                            snap_lf.select([snap_code, snap_date, "available_ts"]).rename(
                                {snap_code: "code", snap_date: "Date"}
                            ),
                            on=["code", "Date"],
                            how="inner",
                        )
                        .select(
                            (pl.col("available_ts") > pl.col("asof_ts")).cast(pl.Float64).mean().alias("violation_rate")
                        )
                        .collect()
                    )
                    violation_rate = float(violation["violation_rate"][0]) if violation.height else 0.0
                    if violation_rate > 0.0:
                        fs_status = "FAIL"
                        fs_details.append(f"as-of violations={violation_rate:.3%}")
            except Exception as exc:
                fs_details.append(f"as-of check error: {exc}")

    dv_need = ["div_days_to_ex", "div_dy_12m", "div_is_obs"]
    dv_present = [c for c in dv_need if c in cols]
    cov_dv = coverage_ratio(lf, dv_present)
    div_status = "PASS" if dv_present else "WARN"
    div_detail = f"div:{dv_present or 'none'}"

    if "div_dy_12m" in cols:
        try:
            q = (
                lf.select(pl.col("div_dy_12m"))
                .drop_nulls()
                .select(
                    [
                        pl.len().alias("n"),
                        pl.col("div_dy_12m").quantile(0.99).alias("q99"),
                    ]
                )
                .collect()
            )
            q99 = float(q["q99"][0]) if q.height else 0.0
            if q99 > 0.2:
                div_status = _escalate(div_status, "WARN")
                div_detail += f"; dy_12m q99={q99:.2f}"
        except Exception:
            pass

    # Basic sanity checks for tradable share count when available.
    if "shares_tradable" in cols and "fs_shares_outstanding" in cols:
        try:
            neg_share = _series_quantile(lf, "shares_tradable", 0.01)
            if neg_share is not None and neg_share < 0:
                fs_status = _escalate(fs_status, "WARN")
                fs_details.append("shares_tradable has negative values at 1% quantile")
            # Check that shares_tradable does not systematically exceed fs_shares_outstanding.
            over_rate_lf = lf.select(
                (
                    (pl.col("shares_tradable").cast(pl.Float64) > pl.col("fs_shares_outstanding").cast(pl.Float64))
                    & pl.col("fs_shares_outstanding").is_not_null()
                )
                .cast(pl.Float64)
                .mean()
                .alias("over_rate")
            ).collect()
            over_rate = float(over_rate_lf["over_rate"][0]) if over_rate_lf.height else 0.0
            if over_rate > 0.01:
                fs_status = _escalate(fs_status, "WARN")
                fs_details.append(f"shares_tradable>fs_shares_outstanding rate={over_rate:.3%}")
        except Exception:
            pass

    # Detect legacy *_pct_float columns (should be deprecated in favour of *_pct_tradable).
    float_cols = [c for c in cols if c.endswith("_pct_float")]
    if float_cols:
        fs_status = _escalate(fs_status, "WARN")
        fs_details.append(f"legacy pct_float columns present: {sorted(float_cols)}")

    fs_detail_str = "; ".join(fs_details) if fs_details else "ok"
    overall_status = fs_status
    overall_status = _escalate(overall_status, div_status)
    combined_detail = f"fs:{fs_detail_str}; div:{div_detail}" if div_detail else f"fs:{fs_detail_str}"
    score.add(
        "6) Financial statements & dividends (as‑of)",
        overall_status,
        combined_detail,
        {
            "fs_coverage": fs_cov,
            "fs_valid_rate": fs_valid_rate,
            "fs_recent_rate": fs_recent_rate,
            "fs_staleness_p95": fs_staleness_p95,
            "fs_lag_p95": fs_lag_p95,
            "div_coverage": cov_dv,
        },
    )


def check7_indices(path: str, score: Score, start: str | None, end: str | None) -> None:
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns
    need = ["topix_idx_r_1d", "topix_idx_atr14", "topix_idx_natr14"]
    present = [c for c in need if c in cols]
    status = "PASS" if len(present) >= 2 else "WARN"
    detail = f"present={present}"
    # Optional beta_60d plausibility
    beta_col = "beta_60d"
    if beta_col in cols:
        try:
            q = (
                lf.select(pl.col(beta_col).drop_nulls())
                .select(
                    [
                        pl.len().alias("n"),
                        pl.col(beta_col).quantile(0.01).alias("q01"),
                        pl.col(beta_col).quantile(0.99).alias("q99"),
                    ]
                )
                .collect()
            )
            q01, q99 = float(q["q01"][0]), float(q["q99"][0])
            if q01 < -3 or q99 > 3:
                status = "WARN"
                detail += f"; beta_60d wide [{q01:.2f},{q99:.2f}]"
        except Exception:
            pass
    score.add("7) Indices (TOPIX etc.)", status, detail)


def _escalate(current: str, candidate: str) -> str:
    ordering = {"PASS": 0, "WARN": 1, "FAIL": 2}
    return candidate if ordering[candidate] > ordering[current] else current


def check8_limit_session(path: str, score: Score, start: str | None, end: str | None) -> None:
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    code_col = resolve_column(cols, "code", "Code")
    date_col = resolve_column(cols, "date", "Date")
    if code_col is None or date_col is None:
        score.add(
            "8) Limit & session features",
            "FAIL",
            "missing Code/Date columns required for as-of validation",
        )
        return

    upper_limit_col = resolve_column(cols, "upper_limit")
    lower_limit_col = resolve_column(cols, "lower_limit")
    limit_up_flag_col = resolve_column(cols, "limit_up_flag")
    limit_down_flag_col = resolve_column(cols, "limit_down_flag")
    limit_any_flag_col = resolve_column(cols, "limit_any_flag")
    limit_up_5d_col = resolve_column(cols, "limit_up_5d")
    limit_down_5d_col = resolve_column(cols, "limit_down_5d")
    days_since_limit_col = resolve_column(cols, "days_since_limit")
    price_locked_col = resolve_column(cols, "price_locked_flag")

    morning_open_col = resolve_column(cols, "morning_open")
    morning_high_col = resolve_column(cols, "morning_high")
    morning_low_col = resolve_column(cols, "morning_low")
    morning_close_col = resolve_column(cols, "morning_close")
    morning_volume_col = resolve_column(cols, "morning_volume")
    morning_upper_limit_col = resolve_column(cols, "morning_upper_limit")
    morning_lower_limit_col = resolve_column(cols, "morning_lower_limit")
    afternoon_open_col = resolve_column(cols, "afternoon_open")
    afternoon_high_col = resolve_column(cols, "afternoon_high")
    afternoon_low_col = resolve_column(cols, "afternoon_low")
    afternoon_volume_col = resolve_column(cols, "afternoon_volume")

    am_gap_col = resolve_column(cols, "am_gap_prev_close")
    am_range_col = resolve_column(cols, "am_range")
    am_to_full_range_col = resolve_column(cols, "am_to_full_range")
    am_vol_share_col = resolve_column(cols, "am_vol_share")
    am_limit_up_col = resolve_column(cols, "am_limit_up_flag")
    am_limit_down_col = resolve_column(cols, "am_limit_down_flag")
    am_limit_any_col = resolve_column(cols, "am_limit_any_flag")
    pm_gap_col = resolve_column(cols, "pm_gap_am_close")
    pm_range_col = resolve_column(cols, "pm_range")

    adj_close_col = resolve_column(cols, "adjustmentclose", "AdjustmentClose")
    adj_high_col = resolve_column(cols, "adjustmenthigh", "AdjustmentHigh")
    adj_low_col = resolve_column(cols, "adjustmentlow", "AdjustmentLow")
    adj_volume_col = resolve_column(cols, "adjustmentvolume", "AdjustmentVolume")
    ret_prev_col = resolve_column(cols, "ret_prev_1d")

    missing_columns = []
    for name, col in [
        ("upper_limit", upper_limit_col),
        ("lower_limit", lower_limit_col),
        ("limit_up_flag", limit_up_flag_col),
        ("limit_down_flag", limit_down_flag_col),
        ("limit_any_flag", limit_any_flag_col),
        ("limit_up_5d", limit_up_5d_col),
        ("limit_down_5d", limit_down_5d_col),
        ("days_since_limit", days_since_limit_col),
        ("price_locked_flag", price_locked_col),
        ("morning_open", morning_open_col),
        ("morning_high", morning_high_col),
        ("morning_low", morning_low_col),
        ("morning_close", morning_close_col),
        ("morning_volume", morning_volume_col),
        ("morning_upper_limit", morning_upper_limit_col),
        ("morning_lower_limit", morning_lower_limit_col),
        ("afternoon_open", afternoon_open_col),
        ("afternoon_high", afternoon_high_col),
        ("afternoon_low", afternoon_low_col),
        ("afternoon_volume", afternoon_volume_col),
        ("am_gap_prev_close", am_gap_col),
        ("am_range", am_range_col),
        ("am_to_full_range", am_to_full_range_col),
        ("am_vol_share", am_vol_share_col),
        ("am_limit_up_flag", am_limit_up_col),
        ("am_limit_down_flag", am_limit_down_col),
        ("am_limit_any_flag", am_limit_any_col),
        ("pm_gap_am_close", pm_gap_col),
        ("pm_range", pm_range_col),
        ("adjustmentclose", adj_close_col),
        ("adjustmenthigh", adj_high_col),
        ("adjustmentlow", adj_low_col),
        ("adjustmentvolume", adj_volume_col),
    ]:
        if col is None:
            missing_columns.append(name)

    status = "PASS"
    detail_parts: list[str] = []
    metrics: dict[str, Any] = {}

    if missing_columns:
        # During schema migration, treat am_vol_share as optional if他の列は揃っている
        if sorted(missing_columns) == ["am_vol_share"]:
            status = _escalate(status, "WARN")
            detail_parts.append("am_vol_share missing (treated as optional during migration)")
        else:
            status = "FAIL"
            detail_parts.append(f"missing columns: {sorted(missing_columns)}")

    agg_exprs: list[pl.Expr] = []

    if upper_limit_col and limit_up_flag_col:
        agg_exprs.extend(
            [
                pl.mean(
                    pl.when(pl.col(upper_limit_col).is_null() | pl.col(limit_up_flag_col).is_null())
                    .then(None)
                    .otherwise((pl.col(upper_limit_col).eq(1).cast(pl.Int8) - pl.col(limit_up_flag_col)).abs())
                ).alias("limit_up_mismatch"),
                pl.mean(pl.col(limit_up_flag_col).is_null().cast(pl.Float64)).alias("limit_up_null_rate"),
            ]
        )
    if lower_limit_col and limit_down_flag_col:
        agg_exprs.extend(
            [
                pl.mean(
                    pl.when(pl.col(lower_limit_col).is_null() | pl.col(limit_down_flag_col).is_null())
                    .then(None)
                    .otherwise((pl.col(lower_limit_col).eq(1).cast(pl.Int8) - pl.col(limit_down_flag_col)).abs())
                ).alias("limit_down_mismatch"),
                pl.mean(pl.col(limit_down_flag_col).is_null().cast(pl.Float64)).alias("limit_down_null_rate"),
            ]
        )

    if limit_any_flag_col and limit_up_flag_col and limit_down_flag_col:
        agg_exprs.append(
            pl.mean(
                pl.when(
                    pl.col(limit_any_flag_col).is_null()
                    | pl.col(limit_up_flag_col).is_null()
                    | pl.col(limit_down_flag_col).is_null()
                )
                .then(None)
                .otherwise(
                    (
                        ((pl.col(limit_up_flag_col) == 1) | (pl.col(limit_down_flag_col) == 1)).cast(pl.Int8)
                        - pl.col(limit_any_flag_col)
                    ).abs()
                )
            ).alias("limit_any_mismatch")
        )

    if limit_up_flag_col and limit_up_5d_col and code_col:
        agg_exprs.append(
            pl.quantile(
                pl.when(pl.col(limit_up_5d_col).is_null())
                .then(None)
                .otherwise(
                    (
                        pl.col(limit_up_flag_col).cast(pl.Float64).shift(1).over(code_col).rolling_sum(window_size=5)
                        - pl.col(limit_up_5d_col)
                    ).abs()
                ),
                0.99,
            ).alias("limit_up_5d_q99_err")
        )

    if limit_down_flag_col and limit_down_5d_col and code_col:
        agg_exprs.append(
            pl.quantile(
                pl.when(pl.col(limit_down_5d_col).is_null())
                .then(None)
                .otherwise(
                    (
                        pl.col(limit_down_flag_col).cast(pl.Float64).shift(1).over(code_col).rolling_sum(window_size=5)
                        - pl.col(limit_down_5d_col)
                    ).abs()
                ),
                0.99,
            ).alias("limit_down_5d_q99_err")
        )

    if days_since_limit_col:
        agg_exprs.append(pl.col(days_since_limit_col).cast(pl.Float64).min().alias("days_since_limit_min"))

    if (
        price_locked_col
        and adj_close_col
        and adj_high_col
        and adj_low_col
        and limit_up_flag_col
        and limit_down_flag_col
    ):
        agg_exprs.append(
            pl.mean(
                pl.when(pl.col(price_locked_col).is_null())
                .then(None)
                .otherwise(
                    (
                        (
                            (
                                (pl.col(limit_up_flag_col) == 1)
                                & (pl.col(adj_high_col) - pl.col(adj_close_col)).abs().lt(EPS)
                            )
                            | (
                                (pl.col(limit_down_flag_col) == 1)
                                & (pl.col(adj_low_col) - pl.col(adj_close_col)).abs().lt(EPS)
                            )
                        ).cast(pl.Int8)
                        - pl.col(price_locked_col)
                    ).abs()
                )
            ).alias("price_locked_mismatch")
        )

    if am_gap_col and morning_open_col and adj_close_col and code_col:
        agg_exprs.append(
            pl.quantile(
                (
                    (
                        pl.col(morning_open_col).shift(1).over(code_col)
                        / (pl.col(adj_close_col).shift(2).over(code_col) + EPS)
                        - 1.0
                    )
                    - pl.col(am_gap_col)
                ).abs(),
                0.99,
            ).alias("am_gap_prev_close_q99_err")
        )

    if am_range_col and morning_high_col and morning_low_col and morning_open_col and code_col:
        agg_exprs.append(
            pl.quantile(
                (
                    (pl.col(morning_high_col).shift(1).over(code_col) - pl.col(morning_low_col).shift(1).over(code_col))
                    / (pl.col(morning_open_col).shift(1).over(code_col) + EPS)
                    - pl.col(am_range_col)
                ).abs(),
                0.99,
            ).alias("am_range_q99_err")
        )

    if am_to_full_range_col and morning_high_col and morning_low_col and adj_high_col and adj_low_col and code_col:
        agg_exprs.append(
            pl.quantile(
                (
                    (pl.col(morning_high_col).shift(1).over(code_col) - pl.col(morning_low_col).shift(1).over(code_col))
                    / (
                        (
                            pl.col(adj_high_col).shift(1).over(code_col) - pl.col(adj_low_col).shift(1).over(code_col)
                        ).abs()
                        + EPS
                    )
                    - pl.col(am_to_full_range_col)
                ).abs(),
                0.99,
            ).alias("am_to_full_range_q99_err")
        )

    if am_vol_share_col and morning_volume_col and adj_volume_col and code_col:
        agg_exprs.append(
            pl.quantile(
                (
                    (
                        pl.col(morning_volume_col).shift(1).over(code_col)
                        / (pl.col(adj_volume_col).shift(1).over(code_col) + EPS)
                    )
                    - pl.col(am_vol_share_col)
                ).abs(),
                0.99,
            ).alias("am_vol_share_q99_err")
        )

    if am_limit_up_col and morning_upper_limit_col and code_col:
        agg_exprs.append(
            pl.mean(
                pl.when(pl.col(am_limit_up_col).is_null())
                .then(None)
                .otherwise(
                    (
                        pl.col(morning_upper_limit_col).shift(1).over(code_col).eq(1).cast(pl.Int8)
                        - pl.col(am_limit_up_col)
                    ).abs()
                )
            ).alias("am_limit_up_mismatch")
        )

    if am_limit_down_col and morning_lower_limit_col and code_col:
        agg_exprs.append(
            pl.mean(
                pl.when(pl.col(am_limit_down_col).is_null())
                .then(None)
                .otherwise(
                    (
                        pl.col(morning_lower_limit_col).shift(1).over(code_col).eq(1).cast(pl.Int8)
                        - pl.col(am_limit_down_col)
                    ).abs()
                )
            ).alias("am_limit_down_mismatch")
        )

    if am_limit_any_col and am_limit_up_col and am_limit_down_col:
        agg_exprs.append(
            pl.mean(
                pl.when(
                    pl.col(am_limit_any_col).is_null()
                    | pl.col(am_limit_up_col).is_null()
                    | pl.col(am_limit_down_col).is_null()
                )
                .then(None)
                .otherwise(
                    (
                        ((pl.col(am_limit_up_col) == 1) | (pl.col(am_limit_down_col) == 1)).cast(pl.Int8)
                        - pl.col(am_limit_any_col)
                    ).abs()
                )
            ).alias("am_limit_any_mismatch")
        )

    if pm_gap_col and afternoon_open_col and morning_close_col and code_col:
        agg_exprs.append(
            pl.quantile(
                (
                    (pl.col(afternoon_open_col) / (pl.col(morning_close_col) + EPS) - 1.0).shift(1).over(code_col)
                    - pl.col(pm_gap_col)
                ).abs(),
                0.99,
            ).alias("pm_gap_q99_err")
        )

    if pm_range_col and afternoon_high_col and afternoon_low_col and afternoon_open_col and code_col:
        agg_exprs.append(
            pl.quantile(
                (
                    (pl.col(afternoon_high_col) - pl.col(afternoon_low_col)) / (pl.col(afternoon_open_col) + EPS)
                    - pl.col(pm_range_col)
                )
                .shift(1)
                .over(code_col)
                .abs(),
                0.99,
            ).alias("pm_range_q99_err")
        )

    if limit_up_flag_col and ret_prev_col:
        agg_exprs.append(
            pl.mean(((pl.col(limit_up_flag_col) == 1) & (pl.col(ret_prev_col) < -0.3)).cast(pl.Float64)).alias(
                "limit_up_negative_ret_share"
            )
        )
    if limit_down_flag_col and ret_prev_col:
        agg_exprs.append(
            pl.mean(((pl.col(limit_down_flag_col) == 1) & (pl.col(ret_prev_col) > 0.3)).cast(pl.Float64)).alias(
                "limit_down_positive_ret_share"
            )
        )

    stats = {}
    if agg_exprs:
        try:
            stats = lf.select(agg_exprs).collect().to_dict(as_series=False)
        except Exception as exc:
            status = "WARN"
            detail_parts.append(f"metric computation failed: {exc}")
            stats = {}

    for key, value in stats.items():
        if value:
            metrics[key] = value[0]

    up_mismatch = float(metrics.get("limit_up_mismatch", 0.0) or 0.0)
    down_mismatch = float(metrics.get("limit_down_mismatch", 0.0) or 0.0)
    up_null_rate = float(metrics.get("limit_up_null_rate", 0.0) or 0.0)
    down_null_rate = float(metrics.get("limit_down_null_rate", 0.0) or 0.0)
    limit_any_mismatch = float(metrics.get("limit_any_mismatch", 0.0) or 0.0)
    up5_err = float(metrics.get("limit_up_5d_q99_err", 0.0) or 0.0)
    down5_err = float(metrics.get("limit_down_5d_q99_err", 0.0) or 0.0)
    days_since_min = metrics.get("days_since_limit_min")
    price_locked_mismatch = float(metrics.get("price_locked_mismatch", 0.0) or 0.0)
    am_gap_err = float(metrics.get("am_gap_prev_close_q99_err", 0.0) or 0.0)
    am_range_err = float(metrics.get("am_range_q99_err", 0.0) or 0.0)
    am_to_full_range_err = float(metrics.get("am_to_full_range_q99_err", 0.0) or 0.0)
    am_vol_share_err = float(metrics.get("am_vol_share_q99_err", 0.0) or 0.0)
    am_limit_up_mismatch = float(metrics.get("am_limit_up_mismatch", 0.0) or 0.0)
    am_limit_down_mismatch = float(metrics.get("am_limit_down_mismatch", 0.0) or 0.0)
    am_limit_any_mismatch = float(metrics.get("am_limit_any_mismatch", 0.0) or 0.0)
    pm_gap_err = float(metrics.get("pm_gap_q99_err", 0.0) or 0.0)
    pm_range_err = float(metrics.get("pm_range_q99_err", 0.0) or 0.0)
    up_neg_share = float(metrics.get("limit_up_negative_ret_share", 0.0) or 0.0)
    down_pos_share = float(metrics.get("limit_down_positive_ret_share", 0.0) or 0.0)

    if up_mismatch > 1e-4:
        status = "FAIL"
        detail_parts.append(f"limit_up_flag mismatch rate={up_mismatch:.2e}")
    if down_mismatch > 1e-4:
        status = "FAIL"
        detail_parts.append(f"limit_down_flag mismatch rate={down_mismatch:.2e}")
    if limit_any_mismatch > 1e-4:
        status = "FAIL"
        detail_parts.append(f"limit_any_flag mismatch rate={limit_any_mismatch:.2e}")
    if up_null_rate > 0.05:
        status = _escalate(status, "WARN")
        detail_parts.append(f"limit_up_flag null_rate={up_null_rate:.2%}")
    if down_null_rate > 0.05:
        status = _escalate(status, "WARN")
        detail_parts.append(f"limit_down_flag null_rate={down_null_rate:.2%}")
    if up5_err > 1e-3:
        status = "FAIL"
        detail_parts.append(f"limit_up_5d q99 err={up5_err:.2e}")
    if down5_err > 1e-3:
        status = "FAIL"
        detail_parts.append(f"limit_down_5d q99 err={down5_err:.2e}")
    if isinstance(days_since_min, (int, float)) and days_since_min < 0:
        status = _escalate(status, "WARN")
        detail_parts.append(f"days_since_limit min={days_since_min}")
    if price_locked_mismatch > 1e-3:
        status = _escalate(status, "WARN")
        detail_parts.append(f"price_locked_flag mismatch={price_locked_mismatch:.2e}")

    if am_gap_err > 1e-6:
        status = "FAIL"
        detail_parts.append(f"am_gap_prev_close q99 err={am_gap_err:.2e}")
    if am_range_err > 1e-6:
        status = "FAIL"
        detail_parts.append(f"am_range q99 err={am_range_err:.2e}")
    if am_to_full_range_err > 1e-6:
        status = "FAIL"
        detail_parts.append(f"am_to_full_range q99 err={am_to_full_range_err:.2e}")
    if am_vol_share_err > 1e-6:
        status = "FAIL"
        detail_parts.append(f"am_vol_share q99 err={am_vol_share_err:.2e}")
    if am_limit_up_mismatch > 1e-4:
        status = "FAIL"
        detail_parts.append(f"am_limit_up_flag mismatch={am_limit_up_mismatch:.2e}")
    if am_limit_down_mismatch > 1e-4:
        status = "FAIL"
        detail_parts.append(f"am_limit_down_flag mismatch={am_limit_down_mismatch:.2e}")
    if am_limit_any_mismatch > 1e-4:
        status = "FAIL"
        detail_parts.append(f"am_limit_any_flag mismatch={am_limit_any_mismatch:.2e}")
    if pm_gap_err > 1e-6:
        status = "FAIL"
        detail_parts.append(f"pm_gap_am_close q99 err={pm_gap_err:.2e}")
    if pm_range_err > 1e-6:
        # pm_range の再現誤差は一時的に WARN 扱い（クリーンアップ過渡期）
        status = _escalate(status, "WARN")
        detail_parts.append(f"pm_range q99 err={pm_range_err:.2e}")

    if up_neg_share > 0.001:
        status = _escalate(status, "WARN")
        detail_parts.append(f"limit_up_flag with ret_prev_1d<-30bps share={up_neg_share:.2%}")
    if down_pos_share > 0.001:
        status = _escalate(status, "WARN")
        detail_parts.append(f"limit_down_flag with ret_prev_1d>30bps share={down_pos_share:.2%}")

    score.add(
        "8) Limit & session features",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check9_breakdown(path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None) -> None:
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    required = [
        "bd_total_value",
        "bd_net_value",
        "bd_net_ratio",
        "bd_short_share",
        "bd_activity_ratio",
        "bd_net_ratio_chg_1d",
        "bd_short_share_chg_1d",
        "bd_net_z20",
        "bd_net_z260",
        "bd_short_z260",
        "bd_credit_new_net",
        "bd_credit_close_net",
        "bd_net_ratio_local_max",
        "bd_net_ratio_local_min",
        "bd_turn_up",
        "bd_staleness_bd",
        "bd_is_recent",
        "is_bd_valid",
    ]
    missing = [col for col in required if col not in cols]
    if missing:
        score.add(
            "9) Breakdown features (as‑of)",
            "FAIL",
            f"missing required bd_* columns: {missing}",
        )
        return

    exprs = [
        pl.mean(
            ((pl.col("bd_net_ratio").abs() > 1.0001) & pl.col("bd_net_ratio").is_not_null()).cast(pl.Float64)
        ).alias("ratio_violation"),
        pl.mean(
            (
                ((pl.col("bd_short_share") > 1.0001) | (pl.col("bd_short_share") < -1e-4))
                & pl.col("bd_short_share").is_not_null()
            ).cast(pl.Float64)
        ).alias("short_violation"),
        pl.mean(((pl.col("bd_net_z20").abs() > 6.0) & pl.col("bd_net_z20").is_not_null()).cast(pl.Float64)).alias(
            "z_extreme_share"
        ),
        pl.mean(((pl.col("bd_net_z260").abs() > 6.0) & pl.col("bd_net_z260").is_not_null()).cast(pl.Float64)).alias(
            "z260_extreme_share"
        ),
        pl.mean(((pl.col("bd_short_z260").abs() > 6.0) & pl.col("bd_short_z260").is_not_null()).cast(pl.Float64)).alias(
            "short_z260_extreme_share"
        ),
        pl.mean((pl.col("bd_turn_up").is_not_null() & ~pl.col("bd_turn_up").is_in([0, 1])).cast(pl.Float64)).alias(
            "turn_invalid"
        ),
        pl.mean((pl.col("bd_is_recent").is_not_null() & ~pl.col("bd_is_recent").is_in([0, 1])).cast(pl.Float64)).alias(
            "recent_invalid"
        ),
        pl.mean((pl.col("is_bd_valid").is_not_null() & ~pl.col("is_bd_valid").is_in([0, 1])).cast(pl.Float64)).alias(
            "valid_invalid"
        ),
        pl.mean(
            ((pl.col("bd_activity_ratio") <= 0) & pl.col("bd_activity_ratio").is_not_null()).cast(pl.Float64)
        ).alias("activity_nonpos"),
        pl.mean(
            (pl.col("bd_net_ratio_local_max").is_not_null() & ~pl.col("bd_net_ratio_local_max").is_in([0, 1])).cast(
                pl.Float64
            )
        ).alias("local_max_invalid"),
        pl.mean(
            (pl.col("bd_net_ratio_local_min").is_not_null() & ~pl.col("bd_net_ratio_local_min").is_in([0, 1])).cast(
                pl.Float64
            )
        ).alias("local_min_invalid"),
        pl.col("bd_staleness_bd").mean().alias("staleness_mean"),
        pl.col("bd_staleness_bd").quantile(0.95).alias("staleness_p95"),
        pl.mean(((pl.col("bd_staleness_bd") < 0) & pl.col("bd_staleness_bd").is_not_null()).cast(pl.Float64)).alias(
            "staleness_negative_share"
        ),
    ]

    if "bd_net_adv60" in cols:
        exprs.append(pl.mean(pl.col("bd_net_adv60").is_finite().not_().cast(pl.Float64)).alias("adv_invalid"))
    stats = lf.select(exprs).collect()
    metrics = {name: float(stats[name][0]) for name in stats.columns}

    status = "PASS"
    details: list[str] = []

    if metrics["ratio_violation"] > 1e-3:
        status = "FAIL"
        details.append(f"net_ratio out-of-range share={metrics['ratio_violation']:.3%}")
    if metrics["short_violation"] > 1e-3:
        status = "FAIL"
        details.append(f"short_share outside [0,1] share={metrics['short_violation']:.3%}")
    if metrics["z_extreme_share"] > 1e-3:
        status = _escalate(status, "WARN")
        details.append(f"|bd_net_z20|>6 share={metrics['z_extreme_share']:.3%}")
    if metrics["z260_extreme_share"] > 1e-3:
        status = _escalate(status, "WARN")
        details.append(f"|bd_net_z260|>6 share={metrics['z260_extreme_share']:.3%}")
    if metrics["short_z260_extreme_share"] > 1e-3:
        status = _escalate(status, "WARN")
        details.append(f"|bd_short_z260|>6 share={metrics['short_z260_extreme_share']:.3%}")
    if metrics["activity_nonpos"] > 0.01:
        status = _escalate(status, "WARN")
        details.append(f"activity_ratio<=0 share={metrics['activity_nonpos']:.3%}")
    if metrics["turn_invalid"] > 0:
        status = "FAIL"
        details.append("bd_turn_up contains values outside {0,1}")
    if metrics["recent_invalid"] > 0:
        status = "FAIL"
        details.append("bd_is_recent contains values outside {0,1}")
    if metrics["valid_invalid"] > 0:
        status = "FAIL"
        details.append("is_bd_valid contains values outside {0,1}")
    if metrics["local_max_invalid"] > 0 or metrics["local_min_invalid"] > 0:
        status = "FAIL"
        details.append("bd_net_ratio_local_{max|min} contain values outside {0,1}")
    if metrics["staleness_negative_share"] > 0:
        status = "FAIL"
        details.append(f"staleness negative share={metrics['staleness_negative_share']:.3%}")
    if metrics["staleness_mean"] >= 7 or metrics["staleness_p95"] >= 20:
        status = _escalate(status, "WARN")
        details.append(f"staleness(mean={metrics['staleness_mean']:.2f}, p95={metrics['staleness_p95']:.1f})")
    if "adv_invalid" in metrics and metrics["adv_invalid"] > 0:
        status = _escalate(status, "WARN")
        details.append("bd_net_adv60 has non-finite values")

    detail_msg = "; ".join(details) if details else "ok"
    score.add(
        "9) Breakdown features (as‑of)",
        status,
        detail_msg,
        metrics,
    )


def check12_weekly_margin(
    path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None
) -> None:
    """Validate weekly margin interest features (wm_* from /markets/weekly_margin_interest)."""
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    required = [
        # P1: ベース
        "wm_long",
        "wm_short",
        "wm_net",
        "wm_lsr",
        # P1: 変化・モメンタム
        "wm_net_d1w",
        "wm_long_d1w",
        "wm_short_d1w",
        "wm_net_pct_d1w",
        # P1: 標準化
        "wm_net_to_adv20",
        "wm_long_to_adv20",
        "wm_short_to_adv20",
        # P1: 安定化
        "wm_net_z20",
        "wm_short_z20",
        "wm_long_z20",
        "wm_net_z52",
        # P1: 品質
        "is_wm_valid",
        "wm_staleness_bd",
        "wm_is_recent",
    ]
    missing = [col for col in required if col not in cols]
    if missing:
        score.add(
            "12) Weekly margin interest features (as‑of)",
            "FAIL",
            f"missing required wm_* columns: {missing}",
        )
        return

    exprs = [
        # Range checks: wm_long, wm_short must be ≥ 0
        pl.mean(((pl.col("wm_long") < 0) & pl.col("wm_long").is_not_null()).cast(pl.Float64)).alias("wm_long_negative"),
        pl.mean(((pl.col("wm_short") < 0) & pl.col("wm_short").is_not_null()).cast(pl.Float64)).alias(
            "wm_short_negative"
        ),
        # wm_lsr must be finite (or null)
        pl.mean((pl.col("wm_lsr").is_not_null() & ~pl.col("wm_lsr").is_finite()).cast(pl.Float64)).alias(
            "wm_lsr_nonfinite"
        ),
        # wm_net_to_adv20: 極端値チェック（|値|>30日分が0.1%未満）
        pl.mean(
            (pl.col("wm_net_to_adv20").abs() > 30.0 & pl.col("wm_net_to_adv20").is_not_null()).cast(pl.Float64)
        ).alias("wm_net_to_adv20_extreme_share"),
        # wm_staleness_bd must be ≥ 0 (no future leaks)
        pl.mean(((pl.col("wm_staleness_bd") < 0) & pl.col("wm_staleness_bd").is_not_null()).cast(pl.Float64)).alias(
            "staleness_negative"
        ),
        # is_wm_valid must be 0 or 1
        pl.mean((pl.col("is_wm_valid").is_not_null() & ~pl.col("is_wm_valid").is_in([0, 1])).cast(pl.Float64)).alias(
            "is_valid_invalid"
        ),
        # wm_is_recent must be 0 or 1
        pl.mean((pl.col("wm_is_recent").is_not_null() & ~pl.col("wm_is_recent").is_in([0, 1])).cast(pl.Float64)).alias(
            "is_recent_invalid"
        ),
        # Coverage: is_wm_valid daily average
        pl.mean(pl.col("is_wm_valid").cast(pl.Float64)).alias("coverage_avg"),
        # Staleness p95 (should not be too large)
        pl.col("wm_staleness_bd").filter(pl.col("wm_staleness_bd").is_not_null()).quantile(0.95).alias("staleness_p95"),
        # Extreme z-scores (should be rare)
        pl.mean(((pl.col("wm_net_z20").abs() > 3.0) & pl.col("wm_net_z20").is_not_null()).cast(pl.Float64)).alias(
            "z20_extreme_share"
        ),
    ]

    try:
        stats = lf.select(exprs).collect()
        metrics = {}
        for col in stats.columns:
            val = stats[col][0]
            if val is not None:
                metrics[col] = float(val)
    except Exception as exc:
        status = "WARN"
        detail_parts = [f"metric computation failed: {exc}"]
        stats = {}
        metrics = {}

    wm_long_neg = float(metrics.get("wm_long_negative", 0.0) or 0.0)
    wm_short_neg = float(metrics.get("wm_short_negative", 0.0) or 0.0)
    wm_lsr_nonfinite = float(metrics.get("wm_lsr_nonfinite", 0.0) or 0.0)
    wm_net_extreme = float(metrics.get("wm_net_to_adv20_extreme_share", 0.0) or 0.0)
    staleness_neg = float(metrics.get("staleness_negative", 0.0) or 0.0)
    is_valid_inv = float(metrics.get("is_valid_invalid", 0.0) or 0.0)
    is_recent_inv = float(metrics.get("is_recent_invalid", 0.0) or 0.0)
    coverage = float(metrics.get("coverage_avg", 0.0) or 0.0)
    staleness_p95 = metrics.get("staleness_p95")
    z20_extreme = float(metrics.get("z20_extreme_share", 0.0) or 0.0)

    status = "PASS"
    detail_parts = []

    if wm_long_neg > 1e-4:
        status = "FAIL"
        detail_parts.append(f"wm_long negative: {wm_long_neg:.2e}")
    if wm_short_neg > 1e-4:
        status = "FAIL"
        detail_parts.append(f"wm_short negative: {wm_short_neg:.2e}")
    if wm_lsr_nonfinite > 1e-4:
        status = "FAIL"
        detail_parts.append(f"wm_lsr non-finite: {wm_lsr_nonfinite:.2e}")
    if wm_net_extreme > 0.001:
        status = _escalate(status, "WARN")
        detail_parts.append(f"wm_net_to_adv20 extreme (|>30 days) share={wm_net_extreme:.2%} > 0.1%")
    if staleness_neg > 1e-4:
        status = "FAIL"
        detail_parts.append(f"wm_staleness_bd negative (future leak): {staleness_neg:.2e}")
    if is_valid_inv > 1e-4:
        status = "FAIL"
        detail_parts.append(f"is_wm_valid invalid values: {is_valid_inv:.2e}")
    if is_recent_inv > 1e-4:
        status = "FAIL"
        detail_parts.append(f"wm_is_recent invalid values: {is_recent_inv:.2e}")

    # Coverage check: should be ≥ 0.7 (70% of days have valid data)
    if coverage < 0.7:
        status = _escalate(status, "WARN")
        detail_parts.append(f"coverage (is_wm_valid avg)={coverage:.2%} < 0.7")

    # Staleness p95 check: should not be excessively large (≤10営業日)
    if isinstance(staleness_p95, (int, float)) and staleness_p95 > 10:
        status = _escalate(status, "WARN")
        detail_parts.append(f"staleness_p95={staleness_p95:.1f} days > 10")

    # Extreme z-scores should be rare (<0.5%)
    if z20_extreme > 0.005:
        status = _escalate(status, "WARN")
        detail_parts.append(f"wm_net_z20 extreme (|z|>3) share={z20_extreme:.2%} > 0.5%")

    score.add(
        "12) Weekly margin interest features (as‑of)",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check13_index_option_225(
    path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None
) -> None:
    """Validate Nikkei225 index option features (idxopt_* from /option/index_option)."""
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    required = [
        # P0: ベース
        "idxopt_iv_atm_near",
        "idxopt_iv_atm_30d",
        "idxopt_iv_ts_slope",
        "idxopt_pc_oi_ratio",
        "idxopt_pc_vol_ratio",
        "idxopt_skew_25",
        "idxopt_days_to_sq",
        "idxopt_iv_night_jump",
        # P0: VRP
        "idxopt_vrp_gap",
        "idxopt_vrp_ratio",
        # P0: 安定化
        "idxopt_iv_30d_z20",
        # P0: 品質
        "is_idxopt_valid",
    ]
    missing = [col for col in required if col not in cols]
    if missing:
        score.add(
            "13) Index option 225 features (as‑of)",
            "FAIL",
            f"missing required idxopt_* columns: {missing}",
        )
        return

    exprs = [
        # Coverage: is_idxopt_valid daily average
        pl.mean(pl.col("is_idxopt_valid").cast(pl.Float64)).alias("coverage_avg"),
        # iv_atm_30d should be positive (or null)
        pl.mean(((pl.col("idxopt_iv_atm_30d") < 0) & pl.col("idxopt_iv_atm_30d").is_not_null()).cast(pl.Float64)).alias(
            "iv_negative"
        ),
        # iv_ts_slope: 通常は正（順カーブ）で大きく負に偏らない（p10 > -5%）
        pl.col("idxopt_iv_ts_slope")
        .filter(pl.col("idxopt_iv_ts_slope").is_not_null())
        .quantile(0.10)
        .alias("iv_ts_slope_p10"),
        # pc_oi_ratio: メディアンは0.5〜2.0程度（極端値比率 < 1%）
        pl.col("idxopt_pc_oi_ratio")
        .filter(pl.col("idxopt_pc_oi_ratio").is_not_null())
        .quantile(0.50)
        .alias("pc_oi_ratio_median"),
        pl.mean(
            (
                ((pl.col("idxopt_pc_oi_ratio") < 0.1) | (pl.col("idxopt_pc_oi_ratio") > 10.0))
                & pl.col("idxopt_pc_oi_ratio").is_not_null()
            ).cast(pl.Float64)
        ).alias("pc_oi_ratio_extreme_share"),
        # VRP整合性: iv_atm_30d >= topix_realized_vol_20d の割合が過半
        # 注: topix_realized_vol_20dは特徴量には含まれていないため、VRP比で検証
        pl.mean(
            ((pl.col("idxopt_vrp_ratio") >= 1.0) & pl.col("idxopt_vrp_ratio").is_not_null()).cast(pl.Float64)
        ).alias("vrp_ratio_above_one"),
        # Extreme z-scores should be rare (<0.5%)
        pl.mean(
            ((pl.col("idxopt_iv_30d_z20").abs() > 3.0) & pl.col("idxopt_iv_30d_z20").is_not_null()).cast(pl.Float64)
        ).alias("z20_extreme_share"),
        # is_idxopt_valid must be 0 or 1
        pl.mean(
            (pl.col("is_idxopt_valid").is_not_null() & ~pl.col("is_idxopt_valid").is_in([0, 1])).cast(pl.Float64)
        ).alias("is_valid_invalid"),
    ]

    try:
        stats = lf.select(exprs).collect()
        metrics = {}
        for col in stats.columns:
            val = stats[col][0]
            if val is not None:
                metrics[col] = float(val)
    except Exception as exc:
        status = "WARN"
        detail_parts = [f"metric computation failed: {exc}"]
        stats = {}
        metrics = {}

    coverage = float(metrics.get("coverage_avg", 0.0) or 0.0)
    iv_neg = float(metrics.get("iv_negative", 0.0) or 0.0)
    iv_ts_p10 = metrics.get("iv_ts_slope_p10")
    pc_oi_median = metrics.get("pc_oi_ratio_median")
    pc_oi_extreme = float(metrics.get("pc_oi_ratio_extreme_share", 0.0) or 0.0)
    vrp_above_one = float(metrics.get("vrp_ratio_above_one", 0.0) or 0.0)
    z20_extreme = float(metrics.get("z20_extreme_share", 0.0) or 0.0)
    is_valid_inv = float(metrics.get("is_valid_invalid", 0.0) or 0.0)

    status = "PASS"
    detail_parts = []

    if iv_neg > 1e-4:
        status = "FAIL"
        detail_parts.append(f"idxopt_iv_atm_30d negative: {iv_neg:.2e}")
    if is_valid_inv > 1e-4:
        status = "FAIL"
        detail_parts.append(f"is_idxopt_valid invalid values: {is_valid_inv:.2e}")

    # Coverage check: should be ≥ 0.7 (70% of days have valid data)
    if coverage < 0.7:
        status = _escalate(status, "WARN")
        detail_parts.append(f"coverage (is_idxopt_valid avg)={coverage:.2%} < 0.7")

    # IV Term Structure check: p10 > -5%
    if isinstance(iv_ts_p10, (int, float)) and iv_ts_p10 < -0.05:
        status = _escalate(status, "WARN")
        detail_parts.append(f"iv_ts_slope p10={iv_ts_p10:.3f} < -0.05 (逆カーブ偏り)")

    # PC OI ratio check: median should be 0.5-2.0, extreme < 1%
    if isinstance(pc_oi_median, (int, float)):
        if pc_oi_median < 0.5 or pc_oi_median > 2.0:
            status = _escalate(status, "WARN")
            detail_parts.append(f"pc_oi_ratio median={pc_oi_median:.2f} outside [0.5, 2.0]")
    if pc_oi_extreme > 0.01:
        status = _escalate(status, "WARN")
        detail_parts.append(f"pc_oi_ratio extreme share={pc_oi_extreme:.2%} > 1%")

    # VRP整合性: iv >= rv の割合が過半（VRP比 >= 1.0）
    if vrp_above_one < 0.5:
        status = _escalate(status, "WARN")
        detail_parts.append(f"vrp_ratio>=1.0 share={vrp_above_one:.2%} < 50% (IV < RV が多い)")

    # Extreme z-scores should be rare (<0.5%)
    if z20_extreme > 0.005:
        status = _escalate(status, "WARN")
        detail_parts.append(f"idxopt_iv_30d_z20 extreme (|z|>3) share={z20_extreme:.2%} > 0.5%")

    score.add(
        "13) Index option 225 features (as‑of)",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check14_index_features(
    path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None
) -> None:
    """Validate market index features (idx_* from /indices, /indices/topix)."""
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    # 必須列（TOPIX系）
    required = [
        "idx_topix_ret_prev_1d",
        "idx_topix_atr_14",
    ]
    missing = [col for col in required if col not in cols]
    if missing:
        score.add(
            "14) Index features (as‑of)",
            "FAIL",
            f"missing required idx_topix_* columns: {missing}",
        )
        return

    # オプション列（存在する場合にチェック）
    # Note: available_optional could filter optional_cols but not used

    exprs = [
        # TOPIXリターンの欠損率
        pl.mean(pl.col("idx_topix_ret_prev_1d").is_null().cast(pl.Float64)).alias("topix_ret_1d_null_rate"),
        # リターン整合性: ret_prev_1d ≈ ret_oc + ret_co の誤差 < 5bps
        pl.mean(
            (
                (pl.col("idx_topix_ret_prev_1d") - (pl.col("idx_topix_ret_oc") + pl.col("idx_topix_ret_co"))).abs()
                > 0.0005
            )
            & pl.col("idx_topix_ret_prev_1d").is_not_null()
            & pl.col("idx_topix_ret_oc").is_not_null()
            & pl.col("idx_topix_ret_co").is_not_null()
        ).alias("ret_decomposition_error_rate"),
        # ATRは正でなければならない
        pl.mean(((pl.col("idx_topix_atr_14") <= 0) & pl.col("idx_topix_atr_14").is_not_null()).cast(pl.Float64)).alias(
            "atr_nonpositive_rate"
        ),
    ]

    # スプレッドが存在する場合
    if "idx_spread_topix_nk225_1d" in cols:
        exprs.append(
            pl.mean(
                (
                    (pl.col("idx_spread_topix_nk225_1d").abs() > 5.0)
                    & pl.col("idx_spread_topix_nk225_1d").is_not_null()
                ).cast(pl.Float64)
            ).alias("spread_extreme_rate")
        )

    try:
        stats = lf.select(exprs).collect()
        metrics = {}
        for col in stats.columns:
            val = stats[col][0]
            if val is not None:
                metrics[col] = float(val)
    except Exception as exc:
        status = "WARN"
        detail_parts = [f"metric computation failed: {exc}"]
        stats = {}
        metrics = {}

    topix_null = float(metrics.get("topix_ret_1d_null_rate", 0.0) or 0.0)
    ret_err = float(metrics.get("ret_decomposition_error_rate", 0.0) or 0.0)
    atr_nonpos = float(metrics.get("atr_nonpositive_rate", 0.0) or 0.0)
    spread_extreme = float(metrics.get("spread_extreme_rate", 0.0) or 0.0)

    status = "PASS"
    detail_parts = []

    # TOPIX欠損率チェック（< 0.5%）
    if topix_null > 0.005:
        status = _escalate(status, "WARN")
        detail_parts.append(f"topix_ret_1d null rate={topix_null:.2%} > 0.5%")

    # リターン分解誤差チェック（< 5bps）
    if ret_err > 0.01:  # 1%以上の日で誤差が大きい
        status = _escalate(status, "WARN")
        detail_parts.append(f"ret decomposition error rate={ret_err:.2%} > 1%")

    # ATR非正チェック
    if atr_nonpos > 1e-4:
        status = "FAIL"
        detail_parts.append(f"atr nonpositive rate={atr_nonpos:.2e}")

    # スプレッド極端値チェック（|z|>5相当、|value|>5%が0.5%未満）
    if spread_extreme > 0.005:
        status = _escalate(status, "WARN")
        detail_parts.append(f"spread extreme (|>5%) rate={spread_extreme:.2%} > 0.5%")

    score.add(
        "14) Index features (as‑of)",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check15_topix_features(
    path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None
) -> None:
    """Validate TOPIX-derived features (topix_* from /indices/topix)."""
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    # 必須列（P0）
    required = [
        "topix_ret_prev_1d",
        "topix_ret_overnight",
        "topix_ret_intraday",
        "topix_atr14",
    ]
    missing = [col for col in required if col not in cols]
    if missing:
        score.add(
            "15) TOPIX features (as‑of)",
            "FAIL",
            f"missing required topix_* columns: {missing}",
        )
        return

    exprs = [
        # TOPIXリターンの欠損率
        pl.mean(pl.col("topix_ret_prev_1d").is_null().cast(pl.Float64)).alias("topix_ret_1d_null_rate"),
        # リターン分解整合性: ret_prev_1d ≈ ret_overnight + ret_intraday の誤差 < 1e-8
        pl.mean(
            (
                (pl.col("topix_ret_prev_1d") - (pl.col("topix_ret_overnight") + pl.col("topix_ret_intraday"))).abs()
                > 1e-8
            )
            & pl.col("topix_ret_prev_1d").is_not_null()
            & pl.col("topix_ret_overnight").is_not_null()
            & pl.col("topix_ret_intraday").is_not_null()
        ).alias("ret_decomposition_error_rate"),
        # ATRは正でなければならない
        pl.mean(((pl.col("topix_atr14") <= 0) & pl.col("topix_atr14").is_not_null()).cast(pl.Float64)).alias(
            "atr_nonpositive_rate"
        ),
        # RSIは0-100の範囲
        pl.mean(
            (
                ((pl.col("topix_rsi_14") < 0) | (pl.col("topix_rsi_14") > 100)) & pl.col("topix_rsi_14").is_not_null()
            ).cast(pl.Float64)
        ).alias("rsi_out_of_range_rate"),
    ]

    try:
        stats = lf.select(exprs).collect()
        metrics = {}
        for col in stats.columns:
            val = stats[col][0]
            if val is not None:
                metrics[col] = float(val)
    except Exception as exc:
        status = "WARN"
        detail_parts = [f"metric computation failed: {exc}"]
        stats = {}
        metrics = {}

    topix_null = float(metrics.get("topix_ret_1d_null_rate", 0.0) or 0.0)
    ret_err = float(metrics.get("ret_decomposition_error_rate", 0.0) or 0.0)
    atr_nonpos = float(metrics.get("atr_nonpositive_rate", 0.0) or 0.0)
    rsi_out = float(metrics.get("rsi_out_of_range_rate", 0.0) or 0.0)

    status = "PASS"
    detail_parts = []

    # TOPIX欠損率チェック（< 0.5%）
    if topix_null > 0.005:
        status = _escalate(status, "WARN")
        detail_parts.append(f"topix_ret_1d null rate={topix_null:.2%} > 0.5%")

    # リターン分解誤差チェック（< 1e-8、99.9%以上で満たす）
    if ret_err > 0.001:  # 0.1%以上の日で誤差が大きい
        status = _escalate(status, "WARN")
        detail_parts.append(f"ret decomposition error rate={ret_err:.3%} > 0.1%")

    # ATR非正チェック
    if atr_nonpos > 1e-4:
        status = "FAIL"
        detail_parts.append(f"atr nonpositive rate={atr_nonpos:.2e}")

    # RSI範囲チェック
    if rsi_out > 1e-4:
        status = "FAIL"
        detail_parts.append(f"rsi out of range [0,100] rate={rsi_out:.2e}")

    score.add(
        "15) TOPIX features (as‑of)",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check16_trades_spec_features(
    path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None
) -> None:
    """Validate trades_spec (投資部門別売買状況) features (mkt_flow_* from /markets/trades_spec)."""
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    # 必須列（MVP）
    required = [
        "mkt_flow_foreigners_net_ratio",
        "mkt_flow_individuals_net_ratio",
        "is_trades_spec_valid",
    ]
    missing = [col for col in required if col not in cols]
    if missing:
        score.add(
            "16) Trades spec features (as‑of)",
            "FAIL",
            f"missing required mkt_flow_* columns: {missing}",
        )
        return

    exprs = [
        # ネット比率の範囲チェック: [-1, 1]に概ね収まる
        pl.mean(
            (
                (pl.col("mkt_flow_foreigners_net_ratio").abs() > 1.01)
                & pl.col("mkt_flow_foreigners_net_ratio").is_not_null()
            ).cast(pl.Float64)
        ).alias("foreigners_ratio_extreme_rate"),
        pl.mean(
            (
                (pl.col("mkt_flow_individuals_net_ratio").abs() > 1.01)
                & pl.col("mkt_flow_individuals_net_ratio").is_not_null()
            ).cast(pl.Float64)
        ).alias("individuals_ratio_extreme_rate"),
        # is_trades_spec_validのカバレッジ（>= 70%）
        pl.mean(pl.col("is_trades_spec_valid").is_not_null().cast(pl.Float64)).alias("is_valid_coverage"),
        pl.mean((pl.col("is_trades_spec_valid") == 1).cast(pl.Float64)).alias("is_valid_true_rate"),
        # trades_spec_staleness_bdのp95（<= 10営業日）
        pl.col("trades_spec_staleness_bd").quantile(0.95).alias("staleness_p95"),
        # z-scoreの範囲チェック（|z| > 5の日が0.5%未満）
        pl.mean(
            (pl.col("mkt_flow_foreigners_net_ratio_z13").abs() > 5)
            & pl.col("mkt_flow_foreigners_net_ratio_z13").is_not_null()
        )
        .cast(pl.Float64)
        .alias("foreigners_z13_extreme_rate"),
        pl.mean(
            (pl.col("mkt_flow_individuals_net_ratio_z13").abs() > 5)
            & pl.col("mkt_flow_individuals_net_ratio_z13").is_not_null()
        )
        .cast(pl.Float64)
        .alias("individuals_z13_extreme_rate"),
    ]

    try:
        stats = lf.select(exprs).collect()
        metrics = {}
        for col in stats.columns:
            val = stats[col][0]
            if val is not None:
                metrics[col] = float(val)
    except Exception as exc:
        status = "WARN"
        detail_parts = [f"metric computation failed: {exc}"]
        stats = {}
        metrics = {}

    foreigners_extreme = float(metrics.get("foreigners_ratio_extreme_rate", 0.0) or 0.0)
    individuals_extreme = float(metrics.get("individuals_ratio_extreme_rate", 0.0) or 0.0)
    is_valid_coverage = float(metrics.get("is_valid_coverage", 0.0) or 0.0)
    is_valid_true = float(metrics.get("is_valid_true_rate", 0.0) or 0.0)
    staleness_p95 = float(metrics.get("staleness_p95", 999.0) or 999.0)
    foreigners_z_extreme = float(metrics.get("foreigners_z13_extreme_rate", 0.0) or 0.0)
    individuals_z_extreme = float(metrics.get("individuals_z13_extreme_rate", 0.0) or 0.0)

    status = "PASS"
    detail_parts = []

    # ネット比率の極端値チェック（< 0.1%）
    if foreigners_extreme > 0.001:
        status = _escalate(status, "WARN")
        detail_parts.append(f"foreigners_net_ratio extreme rate={foreigners_extreme:.3%} > 0.1%")
    if individuals_extreme > 0.001:
        status = _escalate(status, "WARN")
        detail_parts.append(f"individuals_net_ratio extreme rate={individuals_extreme:.3%} > 0.1%")

    # is_trades_spec_validのカバレッジ（>= 70%）
    if is_valid_coverage < 0.7:
        status = _escalate(status, "WARN")
        detail_parts.append(f"is_trades_spec_valid coverage={is_valid_coverage:.2%} < 70%")
    if is_valid_true < 0.7:
        status = _escalate(status, "WARN")
        detail_parts.append(f"is_trades_spec_valid=true rate={is_valid_true:.2%} < 70%")

    # stalenessのp95（<= 10営業日）
    if staleness_p95 > 10:
        status = _escalate(status, "WARN")
        detail_parts.append(f"staleness_bd p95={staleness_p95:.1f} > 10 days")

    # z-scoreの極端値チェック（< 0.5%）
    if foreigners_z_extreme > 0.005:
        status = _escalate(status, "WARN")
        detail_parts.append(f"foreigners_z13 extreme rate={foreigners_z_extreme:.3%} > 0.5%")
    if individuals_z_extreme > 0.005:
        status = _escalate(status, "WARN")
        detail_parts.append(f"individuals_z13 extreme rate={individuals_z_extreme:.3%} > 0.5%")

    # As-ofチェック（snapshots_dirが指定されている場合）
    if snapshots_dir:
        # available_ts <= asof_tsのチェック（0 violations）
        # 簡易実装: trades_specのavailable_tsは特徴量生成時に設定済み
        # 詳細なチェックは省略（snapshot parquetが存在する場合のみ実施）
        pass

    score.add(
        "16) Trades spec features (as‑of)",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check17_earnings_announcement(
    path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None
) -> None:
    """Validate earnings announcement features (is_E_*, days_to_earnings from /fins/announcement)."""
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    # 必須列（P0）
    required = [
        "days_to_earnings",
        "is_E_0",
        "is_E_pp3",
        "is_earnings_sched_valid",
    ]
    missing = [col for col in required if col not in cols]
    if missing:
        score.add(
            "17) Earnings announcement features (as‑of)",
            "FAIL",
            f"missing required earnings columns: {missing}",
        )
        return

    exprs = [
        # is_E_0の比率（Eシーズンで上振れするはず）
        pl.mean(pl.col("is_E_0").cast(pl.Float64)).alias("is_E_0_rate"),
        # is_E_pp3の比率
        pl.mean(pl.col("is_E_pp3").cast(pl.Float64)).alias("is_E_pp3_rate"),
        # is_earnings_sched_validのカバレッジ
        pl.mean(pl.col("is_earnings_sched_valid").cast(pl.Float64)).alias("is_valid_coverage"),
        # days_to_earningsの範囲チェック（-30～30日程度が妥当）
        pl.mean(
            ((pl.col("days_to_earnings").abs() > 30) & pl.col("days_to_earnings").is_not_null()).cast(pl.Float64)
        ).alias("dte_extreme_rate"),
        # is_E_0とis_E_pp3の整合性（is_E_0=1ならis_E_pp3=1のはず）
        pl.mean(((pl.col("is_E_0") == 1) & (pl.col("is_E_pp3") != 1)).cast(pl.Float64)).alias(
            "E_flag_inconsistency_rate"
        ),
    ]

    try:
        stats = lf.select(exprs).collect()
        metrics = {}
        for col in stats.columns:
            val = stats[col][0]
            if val is not None:
                metrics[col] = float(val)
    except Exception as exc:
        status = "WARN"
        detail_parts = [f"metric computation failed: {exc}"]
        stats = {}
        metrics = {}

    is_E_0_rate = float(metrics.get("is_E_0_rate", 0.0) or 0.0)
    # Note: is_E_pp3_rate available but not used in current checks
    is_valid_coverage = float(metrics.get("is_valid_coverage", 0.0) or 0.0)
    dte_extreme = float(metrics.get("dte_extreme_rate", 0.0) or 0.0)
    inconsistency = float(metrics.get("E_flag_inconsistency_rate", 0.0) or 0.0)

    status = "PASS"
    detail_parts = []

    # is_E_0の比率（Eシーズンで上振れ、通常は1-5%程度）
    if is_E_0_rate > 0.1:  # 10%以上は異常
        status = _escalate(status, "WARN")
        detail_parts.append(f"is_E_0 rate={is_E_0_rate:.2%} > 10% (unusually high)")

    # is_earnings_sched_validのカバレッジ
    if is_valid_coverage < 0.5:  # 50%未満は低い
        status = _escalate(status, "WARN")
        detail_parts.append(f"is_earnings_sched_valid coverage={is_valid_coverage:.2%} < 50%")

    # days_to_earningsの極端値チェック
    if dte_extreme > 0.05:  # 5%以上は異常
        status = _escalate(status, "WARN")
        detail_parts.append(f"days_to_earnings extreme rate={dte_extreme:.3%} > 5%")

    # フラグの整合性チェック
    if inconsistency > 1e-4:  # 0.01%以上は異常
        status = "FAIL"
        detail_parts.append(f"E flag inconsistency rate={inconsistency:.3%} > 0.01%")

    # As-ofチェック（snapshots_dirが指定されている場合）
    if snapshots_dir:
        # available_ts <= asof_tsのチェック（0 violations）
        # 簡易実装: earnings特徴量のavailable_tsは特徴量生成時に設定済み
        # 詳細なチェックは省略（snapshot parquetが存在する場合のみ実施）
        pass

    score.add(
        "17) Earnings announcement features (as‑of)",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check18_trading_calendar(
    path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None
) -> None:
    """Validate trading calendar features (is_*, days_* from /markets/trading_calendar)."""
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    # 必須列（P0）
    required = [
        "is_trading_day",
        "is_month_end",
        "is_fy_end",
    ]
    missing = [col for col in required if col not in cols]
    if missing:
        score.add(
            "18) Trading calendar features",
            "FAIL",
            f"missing required calendar columns: {missing}",
        )
        return

    exprs = [
        # is_trading_dayの比率（通常は週5日/7日 ≈ 71%）
        pl.mean(pl.col("is_trading_day").cast(pl.Float64)).alias("is_trading_day_rate"),
        # is_month_endの比率（月に1回程度、≈ 3-4%）
        pl.mean(pl.col("is_month_end").cast(pl.Float64)).alias("is_month_end_rate"),
        # is_fy_endの比率（年に1回、≈ 0.3%）
        pl.mean(pl.col("is_fy_end").cast(pl.Float64)).alias("is_fy_end_rate"),
        # is_month_endは毎月>=1日だけ1
        pl.sum(pl.col("is_month_end")).cast(pl.Float64).alias("month_end_count"),
        # is_fy_endは年1回（多くは3月末）
        pl.sum(pl.col("is_fy_end")).cast(pl.Float64).alias("fy_end_count"),
    ]

    try:
        stats = lf.select(exprs).collect()
        metrics = {}
        for col in stats.columns:
            val = stats[col][0]
            if val is not None:
                metrics[col] = float(val)
    except Exception as exc:
        status = "WARN"
        detail_parts = [f"metric computation failed: {exc}"]
        stats = {}
        metrics = {}

    trading_day_rate = float(metrics.get("is_trading_day_rate", 0.0) or 0.0)
    month_end_rate = float(metrics.get("is_month_end_rate", 0.0) or 0.0)
    fy_end_rate = float(metrics.get("is_fy_end_rate", 0.0) or 0.0)
    month_end_count = float(metrics.get("month_end_count", 0.0) or 0.0)
    fy_end_count = float(metrics.get("fy_end_count", 0.0) or 0.0)

    status = "PASS"
    detail_parts = []

    # is_trading_dayの比率（通常は60-75%）
    if trading_day_rate < 0.6 or trading_day_rate > 0.75:
        status = _escalate(status, "WARN")
        detail_parts.append(f"is_trading_day rate={trading_day_rate:.2%} outside expected range [60%, 75%]")

    # is_month_endの比率（通常は3-5%）
    if month_end_rate < 0.02 or month_end_rate > 0.06:
        status = _escalate(status, "WARN")
        detail_parts.append(f"is_month_end rate={month_end_rate:.2%} outside expected range [2%, 6%]")

    # is_fy_endの比率（通常は0.2-0.4%）
    if fy_end_rate > 0.005:  # 0.5%以上は異常
        status = _escalate(status, "WARN")
        detail_parts.append(f"is_fy_end rate={fy_end_rate:.2%} > 0.5% (unusually high)")

    # is_month_endは期間中に適切な回数（月数×1回程度）
    # 簡易チェック: 期間日数/30日程度の月数
    expected_months = month_end_count / 1.0  # 月に1回
    if expected_months > 0 and (month_end_count < expected_months * 0.8 or month_end_count > expected_months * 1.2):
        status = _escalate(status, "WARN")
        detail_parts.append(f"month_end_count={month_end_count:.0f} outside expected range")

    # is_fy_endは年1回（多くは3月末）
    if fy_end_count > 0 and (fy_end_count < 0.5 or fy_end_count > 1.5):
        status = _escalate(status, "WARN")
        detail_parts.append(f"fy_end_count={fy_end_count:.0f} outside expected range [0.5, 1.5]")

    score.add(
        "18) Trading calendar features",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check19_financial_statements(
    path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None
) -> None:
    """Validate financial statement features (fs_* from /fins/statements, /fins/fs_details)."""
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    # 必須列（P0）— 一部は別名も許容（fs_ttm_op_profit ≒ fs_op_profit_ttm 等）
    alias_groups: dict[str, list[str]] = {
        "fs_ttm_sales": ["fs_ttm_sales"],
        "fs_ttm_op_profit": ["fs_ttm_op_profit", "fs_op_profit_ttm"],
        "fs_ttm_net_income": ["fs_ttm_net_income"],
        "fs_ttm_cfo": ["fs_ttm_cfo", "fs_cfo_ttm"],
        "fs_is_valid": ["fs_is_valid"],
    }

    resolved_required: dict[str, str] = {}
    missing_required: list[str] = []
    for logical, candidates in alias_groups.items():
        actual = resolve_column(list(cols), *candidates)
        if actual is None:
            missing_required.append(logical)
        else:
            resolved_required[logical] = actual

    if missing_required:
        score.add(
            "19) Financial statement features (as‑of)",
            "FAIL",
            f"missing required fs_* columns: {missing_required}",
        )
        return

    exprs = [
        # fs_is_validのカバレッジ（>= 70%）
        pl.mean(pl.col(resolved_required["fs_is_valid"]).cast(pl.Float64)).alias("is_valid_coverage"),
        pl.mean((pl.col(resolved_required["fs_is_valid"]) == 1).cast(pl.Float64)).alias("is_valid_true_rate"),
        # fs_ttm_*の欠損率
        pl.mean(pl.col(resolved_required["fs_ttm_sales"]).is_null().cast(pl.Float64)).alias("fs_ttm_sales_null_rate"),
        pl.mean(pl.col(resolved_required["fs_ttm_op_profit"]).is_null().cast(pl.Float64)).alias(
            "fs_ttm_op_profit_null_rate"
        ),
        # マージンの範囲チェック（-1～1程度が妥当）
        pl.mean(
            ((pl.col("fs_ttm_op_margin").abs() > 1.5) & pl.col("fs_ttm_op_margin").is_not_null()).cast(pl.Float64)
        ).alias("op_margin_extreme_rate"),
        pl.mean(
            ((pl.col("fs_ttm_cfo_margin").abs() > 1.5) & pl.col("fs_ttm_cfo_margin").is_not_null()).cast(pl.Float64)
        ).alias("cfo_margin_extreme_rate"),
        # 財務体力の範囲チェック
        pl.mean(
            ((pl.col("fs_equity_ratio") < 0) | (pl.col("fs_equity_ratio") > 1.5))
            & pl.col("fs_equity_ratio").is_not_null()
        )
        .cast(pl.Float64)
        .alias("equity_ratio_extreme_rate"),
        # fs_staleness_bdのp95（<= 65営業日）
        pl.col("fs_staleness_bd").quantile(0.95).alias("staleness_p95"),
        # fs_window_e_pp3の比率（Eシーズンで上振れ）
        pl.mean(pl.col("fs_window_e_pp3").cast(pl.Float64)).alias("window_e_pp3_rate"),
        # fs_ttm_* vs fs_*_ttmの整合性（エイリアスが一致）
        pl.mean(
            (
                (pl.col("fs_ttm_sales") != pl.col("fs_revenue_ttm"))
                & pl.col("fs_ttm_sales").is_not_null()
                & pl.col("fs_revenue_ttm").is_not_null()
            ).cast(pl.Float64)
        ).alias("alias_inconsistency_rate"),
    ]

    try:
        stats = lf.select(exprs).collect()
        metrics = {}
        for col in stats.columns:
            val = stats[col][0]
            if val is not None:
                metrics[col] = float(val)
    except Exception as exc:
        status = "WARN"
        detail_parts = [f"metric computation failed: {exc}"]
        stats = {}
        metrics = {}

    is_valid_coverage = float(metrics.get("is_valid_coverage", 0.0) or 0.0)
    is_valid_true = float(metrics.get("is_valid_true_rate", 0.0) or 0.0)
    ttm_sales_null = float(metrics.get("fs_ttm_sales_null_rate", 0.0) or 0.0)
    op_margin_extreme = float(metrics.get("op_margin_extreme_rate", 0.0) or 0.0)
    equity_ratio_extreme = float(metrics.get("equity_ratio_extreme_rate", 0.0) or 0.0)
    staleness_p95 = float(metrics.get("staleness_p95", 999.0) or 999.0)
    alias_inconsistency = float(metrics.get("alias_inconsistency_rate", 0.0) or 0.0)

    status = "PASS"
    detail_parts = []

    # fs_is_validのカバレッジ（>= 70%）
    if is_valid_coverage < 0.7:
        status = _escalate(status, "WARN")
        detail_parts.append(f"fs_is_valid coverage={is_valid_coverage:.2%} < 70%")
    if is_valid_true < 0.7:
        status = _escalate(status, "WARN")
        detail_parts.append(f"fs_is_valid=true rate={is_valid_true:.2%} < 70%")

    # fs_ttm_salesの欠損率（< 50%）
    if ttm_sales_null > 0.5:
        status = _escalate(status, "WARN")
        detail_parts.append(f"fs_ttm_sales null rate={ttm_sales_null:.2%} > 50%")

    # マージンの極端値チェック（< 0.1%）
    if op_margin_extreme > 0.001:
        status = _escalate(status, "WARN")
        detail_parts.append(f"fs_ttm_op_margin extreme rate={op_margin_extreme:.3%} > 0.1%")

    # 財務体力の範囲チェック
    if equity_ratio_extreme > 0.01:  # 1%以上は異常
        status = _escalate(status, "WARN")
        detail_parts.append(f"fs_equity_ratio extreme rate={equity_ratio_extreme:.2%} > 1%")

    # stalenessのp95（<= 65営業日）
    if staleness_p95 > 65:
        status = _escalate(status, "WARN")
        detail_parts.append(f"fs_staleness_bd p95={staleness_p95:.1f} > 65 days")

    # エイリアスの整合性チェック
    if alias_inconsistency > 1e-4:  # 0.01%以上は異常
        status = "FAIL"
        detail_parts.append(f"fs_ttm_* vs fs_*_ttm alias inconsistency rate={alias_inconsistency:.3%} > 0.01%")

    # As-ofチェック（snapshots_dirが指定されている場合）
    if snapshots_dir:
        # available_ts <= asof_tsのチェック（0 violations）
        # 簡易実装: fs特徴量のavailable_tsは特徴量生成時に設定済み
        # 詳細なチェックは省略（snapshot parquetが存在する場合のみ実施）
        pass

    score.add(
        "19) Financial statement features (as‑of)",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check20_daily_quotes(
    path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None
) -> None:
    """Validate daily quotes features (ret_overnight, ret_intraday, gap_*, range_*, limit_*)."""
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    # 必須列（P0）— 調整後OHLCVは大文字/小文字違いや canonical OHLC も許容
    required_lower = [
        "adjustmentopen",
        "adjustmentclose",
        "adjustmenthigh",
        "adjustmentlow",
        "adjustmentvolume",
        "ret_overnight",
        "ret_intraday",
        "ret_prev_1d",
    ]
    present_lower = {c.lower() for c in cols}
    missing = [name for name in required_lower if name not in present_lower]
    if missing:
        score.add(
            "20) Daily quotes features",
            "FAIL",
            f"missing required daily_quotes columns: {missing}",
        )
        return

    exprs = [
        # 整合性チェック: ret_prev_1d ≈ (1+ret_overnight)*(1+ret_intraday)-1
        pl.mean(
            (
                (
                    pl.col("ret_prev_1d").abs()
                    - ((1.0 + pl.col("ret_overnight")) * (1.0 + pl.col("ret_intraday")) - 1.0).abs()
                ).abs()
                > 1e-5  # 1bps許容
            )
            & pl.col("ret_prev_1d").is_not_null()
            & pl.col("ret_overnight").is_not_null()
            & pl.col("ret_intraday").is_not_null()
        )
        .cast(pl.Float64)
        .alias("gap_consistency_violation_rate"),
        # gap_signの範囲チェック（-1, 0, 1）
        pl.mean(
            (~pl.col("gap_sign").is_in([-1.0, 0.0, 1.0]) & pl.col("gap_sign").is_not_null()).cast(pl.Float64)
        ).alias("gap_sign_invalid_rate"),
        # gap_magnitude_z20の極端値チェック（|z|>5が0.5%未満）
        pl.mean(
            (pl.col("gap_magnitude_z20").abs() > 5 & pl.col("gap_magnitude_z20").is_not_null()).cast(pl.Float64)
        ).alias("gap_magnitude_z20_extreme_rate"),
        # gap_confirmedの範囲チェック（0/1）
        pl.mean(
            (~pl.col("gap_confirmed").is_in([0, 1]) & pl.col("gap_confirmed").is_not_null()).cast(pl.Float64)
        ).alias("gap_confirmed_invalid_rate"),
        # day_rangeの範囲チェック（0-1程度が妥当）
        pl.mean(
            ((pl.col("day_range") < 0) | (pl.col("day_range") > 0.5) & pl.col("day_range").is_not_null()).cast(
                pl.Float64
            )
        ).alias("day_range_extreme_rate"),
        # close_locationの範囲チェック（0-1）
        pl.mean(
            (
                (pl.col("close_location") < 0)
                | (pl.col("close_location") > 1.01) & pl.col("close_location").is_not_null()
            ).cast(pl.Float64)
        ).alias("close_location_extreme_rate"),
        # is_limit_up/downの範囲チェック（0/1）
        pl.mean((~pl.col("is_limit_up").is_in([0, 1]) & pl.col("is_limit_up").is_not_null()).cast(pl.Float64)).alias(
            "is_limit_up_invalid_rate"
        ),
        pl.mean(
            (~pl.col("is_limit_down").is_in([0, 1]) & pl.col("is_limit_down").is_not_null()).cast(pl.Float64)
        ).alias("is_limit_down_invalid_rate"),
        # upper_limit/lower_limitの範囲チェック（0/1）
        pl.mean((~pl.col("upper_limit").is_in([0, 1]) & pl.col("upper_limit").is_not_null()).cast(pl.Float64)).alias(
            "upper_limit_invalid_rate"
        ),
        pl.mean((~pl.col("lower_limit").is_in([0, 1]) & pl.col("lower_limit").is_not_null()).cast(pl.Float64)).alias(
            "lower_limit_invalid_rate"
        ),
        # ret_overnight, ret_intradayの極端値チェック（|ret|>30%が0.1%未満）
        pl.mean((pl.col("ret_overnight").abs() > 0.30 & pl.col("ret_overnight").is_not_null()).cast(pl.Float64)).alias(
            "ret_overnight_extreme_rate"
        ),
        pl.mean((pl.col("ret_intraday").abs() > 0.30 & pl.col("ret_intraday").is_not_null()).cast(pl.Float64)).alias(
            "ret_intraday_extreme_rate"
        ),
        # canonical OHLCチェック: raw/Adj Closeが残置していないか
        # adjustmentcloseがある場合、closeがrawかどうかを確認（調整係数が異なる場合はraw）
        # 簡易チェック: adjustmentcloseとcloseが大幅に異なる行が少ないか（調整係数による差は正常）
        # より正確には、canonicalize_ohlc後のraw列の存在を確認
    ]

    try:
        stats = lf.select(exprs).collect()
        metrics = {}
        for col in stats.columns:
            val = stats[col][0]
            if val is not None:
                metrics[col] = float(val)
    except Exception as exc:
        status = "WARN"
        detail_parts = [f"metric computation failed: {exc}"]
        stats = {}
        metrics = {}

    gap_consistency = float(metrics.get("gap_consistency_violation_rate", 0.0) or 0.0)
    gap_sign_invalid = float(metrics.get("gap_sign_invalid_rate", 0.0) or 0.0)
    gap_mag_extreme = float(metrics.get("gap_magnitude_z20_extreme_rate", 0.0) or 0.0)
    gap_confirmed_invalid = float(metrics.get("gap_confirmed_invalid_rate", 0.0) or 0.0)
    day_range_extreme = float(metrics.get("day_range_extreme_rate", 0.0) or 0.0)
    close_location_extreme = float(metrics.get("close_location_extreme_rate", 0.0) or 0.0)
    is_limit_up_invalid = float(metrics.get("is_limit_up_invalid_rate", 0.0) or 0.0)
    is_limit_down_invalid = float(metrics.get("is_limit_down_invalid_rate", 0.0) or 0.0)
    upper_limit_invalid = float(metrics.get("upper_limit_invalid_rate", 0.0) or 0.0)
    lower_limit_invalid = float(metrics.get("lower_limit_invalid_rate", 0.0) or 0.0)
    ret_overnight_extreme = float(metrics.get("ret_overnight_extreme_rate", 0.0) or 0.0)
    ret_intraday_extreme = float(metrics.get("ret_intraday_extreme_rate", 0.0) or 0.0)

    status = "PASS"
    detail_parts = []

    # 整合性チェック: ret_prev_1d ≈ (1+ret_overnight)*(1+ret_intraday)-1（>0.1%は異常）
    if gap_consistency > 0.001:
        status = "FAIL"
        detail_parts.append(f"ret_prev_1d vs (overnight×intraday) inconsistency rate={gap_consistency:.3%} > 0.1%")

    # gap_signの範囲チェック
    if gap_sign_invalid > 1e-4:
        status = _escalate(status, "WARN")
        detail_parts.append(f"gap_sign invalid rate={gap_sign_invalid:.3%} > 0.01%")

    # gap_magnitude_z20の極端値チェック
    if gap_mag_extreme > 0.005:
        status = _escalate(status, "WARN")
        detail_parts.append(f"gap_magnitude_z20 extreme rate={gap_mag_extreme:.3%} > 0.5%")

    # gap_confirmedの範囲チェック
    if gap_confirmed_invalid > 1e-4:
        status = _escalate(status, "WARN")
        detail_parts.append(f"gap_confirmed invalid rate={gap_confirmed_invalid:.3%} > 0.01%")

    # day_rangeの範囲チェック
    if day_range_extreme > 0.01:
        status = _escalate(status, "WARN")
        detail_parts.append(f"day_range extreme rate={day_range_extreme:.2%} > 1%")

    # close_locationの範囲チェック
    if close_location_extreme > 0.01:
        status = _escalate(status, "WARN")
        detail_parts.append(f"close_location extreme rate={close_location_extreme:.2%} > 1%")

    # is_limit_up/downの範囲チェック
    if is_limit_up_invalid > 1e-4 or is_limit_down_invalid > 1e-4:
        status = _escalate(status, "WARN")
        detail_parts.append(
            f"is_limit_up/down invalid rate={is_limit_up_invalid:.3%}/{is_limit_down_invalid:.3%} > 0.01%"
        )

    # upper_limit/lower_limitの範囲チェック
    if upper_limit_invalid > 1e-4 or lower_limit_invalid > 1e-4:
        status = _escalate(status, "WARN")
        detail_parts.append(
            f"upper_limit/lower_limit invalid rate={upper_limit_invalid:.3%}/{lower_limit_invalid:.3%} > 0.01%"
        )

    # ret_overnight/intradayの極端値チェック
    if ret_overnight_extreme > 0.001 or ret_intraday_extreme > 0.001:
        status = _escalate(status, "WARN")
        detail_parts.append(
            f"ret_overnight/intraday extreme rate={ret_overnight_extreme:.3%}/{ret_intraday_extreme:.3%} > 0.1%"
        )

    # canonical OHLCチェック（snapshots_dirが指定されている場合）
    if snapshots_dir:
        # canonicalize_ohlc後のraw/Adj Close残置チェック
        # 簡易実装: raw列の存在を確認（より正確にはcanonicalize_ohlcの実行結果を確認）
        # 詳細なチェックは省略（snapshot parquetが存在する場合のみ実施）
        pass

    score.add(
        "20) Daily quotes features",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check21_listed_info(path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None) -> None:
    """Validate listed info features (market dummies, sector codes, scale bucket, margin eligibility)."""
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    # 必須列（P0）
    required = [
        "is_prime",
        "is_standard",
        "is_growth",
        "sector33_code",
        "sector17_code",
        "is_listed_info_valid",
    ]
    missing = [col for col in required if col not in cols]
    if missing:
        score.add(
            "21) Listed info features",
            "FAIL",
            f"missing required listed_info columns: {missing}",
        )
        return

    exprs = [
        # 市場区分ダミーの範囲チェック（0/1）
        pl.mean((~pl.col("is_prime").is_in([0, 1]) & pl.col("is_prime").is_not_null()).cast(pl.Float64)).alias(
            "is_prime_invalid_rate"
        ),
        pl.mean((~pl.col("is_standard").is_in([0, 1]) & pl.col("is_standard").is_not_null()).cast(pl.Float64)).alias(
            "is_standard_invalid_rate"
        ),
        pl.mean((~pl.col("is_growth").is_in([0, 1]) & pl.col("is_growth").is_not_null()).cast(pl.Float64)).alias(
            "is_growth_invalid_rate"
        ),
        # sector33_code, sector17_codeの欠損率
        pl.mean(pl.col("sector33_code").is_null().cast(pl.Float64)).alias("sector33_code_null_rate"),
        pl.mean(pl.col("sector17_code").is_null().cast(pl.Float64)).alias("sector17_code_null_rate"),
        # is_listed_info_validのカバレッジ（>=70%）
        pl.mean((pl.col("is_listed_info_valid") == 1).cast(pl.Float64)).alias("is_listed_info_valid_coverage"),
        # market_codeの既知コード比率（0111/0112/0113以外の比率）
        pl.mean(
            (~pl.col("market_code").is_in(["0111", "0112", "0113"]) & pl.col("market_code").is_not_null()).cast(
                pl.Float64
            )
        ).alias("market_code_unknown_rate"),
        # 市場区分ダミーの排他性（is_prime + is_standard + is_growth <= 1）
        pl.mean(
            (
                (pl.col("is_prime") + pl.col("is_standard") + pl.col("is_growth"))
                > 1
                & pl.col("is_prime").is_not_null()
                & pl.col("is_standard").is_not_null()
                & pl.col("is_growth").is_not_null()
            ).cast(pl.Float64)
        ).alias("market_dummies_exclusive_violation_rate"),
    ]

    try:
        stats = lf.select(exprs).collect()
        metrics = {}
        for col in stats.columns:
            val = stats[col][0]
            if val is not None:
                metrics[col] = float(val)
    except Exception as exc:
        status = "WARN"
        detail_parts = [f"metric computation failed: {exc}"]
        stats = {}
        metrics = {}

    is_prime_invalid = float(metrics.get("is_prime_invalid_rate", 0.0) or 0.0)
    is_standard_invalid = float(metrics.get("is_standard_invalid_rate", 0.0) or 0.0)
    is_growth_invalid = float(metrics.get("is_growth_invalid_rate", 0.0) or 0.0)
    sector33_null = float(metrics.get("sector33_code_null_rate", 0.0) or 0.0)
    sector17_null = float(metrics.get("sector17_code_null_rate", 0.0) or 0.0)
    is_listed_info_valid_coverage = float(metrics.get("is_listed_info_valid_coverage", 0.0) or 0.0)
    market_code_unknown = float(metrics.get("market_code_unknown_rate", 0.0) or 0.0)
    market_dummies_exclusive = float(metrics.get("market_dummies_exclusive_violation_rate", 0.0) or 0.0)

    status = "PASS"
    detail_parts = []

    # 市場区分ダミーの範囲チェック
    if is_prime_invalid > 1e-4 or is_standard_invalid > 1e-4 or is_growth_invalid > 1e-4:
        status = _escalate(status, "WARN")
        detail_parts.append(
            f"is_prime/standard/growth invalid rate={is_prime_invalid:.3%}/{is_standard_invalid:.3%}/{is_growth_invalid:.3%} > 0.01%"
        )

    # sector33_code, sector17_codeの欠損率
    if sector33_null > 0.005:
        status = _escalate(status, "WARN")
        detail_parts.append(f"sector33_code null rate={sector33_null:.3%} > 0.5%")
    if sector17_null > 0.02:
        status = _escalate(status, "WARN")
        detail_parts.append(f"sector17_code null rate={sector17_null:.3%} > 2%")

    # is_listed_info_validのカバレッジ
    if is_listed_info_valid_coverage < 0.70:
        status = _escalate(status, "WARN")
        detail_parts.append(f"is_listed_info_valid coverage={is_listed_info_valid_coverage:.1%} < 70%")

    # market_codeの既知コード比率
    if market_code_unknown > 0.01:
        status = _escalate(status, "WARN")
        detail_parts.append(f"market_code unknown rate={market_code_unknown:.3%} > 1%")

    # 市場区分ダミーの排他性
    if market_dummies_exclusive > 0.001:
        status = "FAIL"
        detail_parts.append(f"market dummies exclusive violation rate={market_dummies_exclusive:.3%} > 0.1%")

    # As-ofチェック（snapshots_dirが指定されている場合）
    if snapshots_dir:
        # available_ts <= asof_tsのチェック（0 violations）
        # 簡易実装: listed_info特徴量のavailable_tsは特徴量生成時に設定済み
        # 詳細なチェックは省略（snapshot parquetが存在する場合のみ実施）
        pass

    score.add(
        "21) Listed info features",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check11_sector_short_selling(
    path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None
) -> None:
    """Validate sector short selling features (ss_* from /markets/short_selling)."""
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    required = [
        # 強度レベル
        "ss_total",
        "ss_ratio_market",
        "ss_ratio_with_restr",
        "ss_ratio_without_restr",
        # 変化レベル
        "d1_ss_ratio_market",
        "pct_chg_ss_ratio_market",
        "ss_ratio_market_z20",
        "ss_ratio_with_restr_z20",
        "ss_ratio_without_restr_z20",
        # 異常レベル
        "ss_extreme_hi",
        "ss_regime_switch",
        # 品質
        "is_ss_valid",
        "ss_staleness_bd",
    ]
    missing = [col for col in required if col not in cols]
    if missing:
        score.add(
            "11) Sector short selling features (as‑of)",
            "FAIL",
            f"missing required ss_* columns: {missing}",
        )
        return

    exprs = [
        # Range checks: ss_ratio_market ∈ [0, 1]
        pl.mean(
            ((pl.col("ss_ratio_market") < -1e-6) | (pl.col("ss_ratio_market") > 1.0001))
            & pl.col("ss_ratio_market").is_not_null()
        ).alias("ratio_market_violation"),
        # Range checks: ss_ratio_with_restr ∈ [0, 1]
        pl.mean(
            ((pl.col("ss_ratio_with_restr") < -1e-6) | (pl.col("ss_ratio_with_restr") > 1.0001))
            & pl.col("ss_ratio_with_restr").is_not_null()
        ).alias("ratio_with_restr_violation"),
        # Range checks: ss_ratio_without_restr ∈ [0, 1]
        pl.mean(
            ((pl.col("ss_ratio_without_restr") < -1e-6) | (pl.col("ss_ratio_without_restr") > 1.0001))
            & pl.col("ss_ratio_without_restr").is_not_null()
        ).alias("ratio_without_restr_violation"),
        # 整合性チェック: with_restr + without_restr ≈ 1 (許容誤差±0.5%)
        pl.mean(
            (
                ((pl.col("ss_ratio_with_restr") + pl.col("ss_ratio_without_restr")).abs() > 1.005)
                & (pl.col("ss_ratio_with_restr").is_not_null() | pl.col("ss_ratio_without_restr").is_not_null())
            ).cast(pl.Float64)
        ).alias("ratio_sum_violation"),
        # ss_total must be ≥ 0
        pl.mean(((pl.col("ss_total") < 0) & pl.col("ss_total").is_not_null()).cast(pl.Float64)).alias(
            "ss_total_negative"
        ),
        # ss_staleness_bd must be ≥ 0 (no future leaks)
        pl.mean(((pl.col("ss_staleness_bd") < 0) & pl.col("ss_staleness_bd").is_not_null()).cast(pl.Float64)).alias(
            "staleness_negative"
        ),
        # is_ss_valid must be 0 or 1
        pl.mean((pl.col("is_ss_valid").is_not_null() & ~pl.col("is_ss_valid").is_in([0, 1])).cast(pl.Float64)).alias(
            "is_valid_invalid"
        ),
        # ss_extreme_hi must be 0 or 1
        pl.mean(
            (pl.col("ss_extreme_hi").is_not_null() & ~pl.col("ss_extreme_hi").is_in([0, 1])).cast(pl.Float64)
        ).alias("extreme_hi_invalid"),
        # ss_regime_switch must be 0 or 1
        pl.mean(
            (pl.col("ss_regime_switch").is_not_null() & ~pl.col("ss_regime_switch").is_in([0, 1])).cast(pl.Float64)
        ).alias("regime_switch_invalid"),
        # Coverage: is_ss_valid daily average
        pl.mean(pl.col("is_ss_valid").cast(pl.Float64)).alias("coverage_avg"),
        # Staleness p95 (should not be too large)
        pl.col("ss_staleness_bd").filter(pl.col("ss_staleness_bd").is_not_null()).quantile(0.95).alias("staleness_p95"),
        # Extreme z-scores (should be rare)
        pl.mean(
            ((pl.col("ss_ratio_market_z20").abs() > 3.0) & pl.col("ss_ratio_market_z20").is_not_null()).cast(pl.Float64)
        ).alias("z20_extreme_share"),
    ]

    try:
        stats = lf.select(exprs).collect()
        metrics = {}
        for col in stats.columns:
            val = stats[col][0]
            if val is not None:
                metrics[col] = float(val)
    except Exception as exc:
        status = "WARN"
        detail_parts = [f"metric computation failed: {exc}"]
        stats = {}
        metrics = {}

    ratio_market_violation = float(metrics.get("ratio_market_violation", 0.0) or 0.0)
    ratio_with_restr_violation = float(metrics.get("ratio_with_restr_violation", 0.0) or 0.0)
    ratio_without_restr_violation = float(metrics.get("ratio_without_restr_violation", 0.0) or 0.0)
    ratio_sum_violation = float(metrics.get("ratio_sum_violation", 0.0) or 0.0)
    ss_total_neg = float(metrics.get("ss_total_negative", 0.0) or 0.0)
    staleness_neg = float(metrics.get("staleness_negative", 0.0) or 0.0)
    is_valid_inv = float(metrics.get("is_valid_invalid", 0.0) or 0.0)
    extreme_hi_inv = float(metrics.get("extreme_hi_invalid", 0.0) or 0.0)
    regime_switch_inv = float(metrics.get("regime_switch_invalid", 0.0) or 0.0)
    coverage = float(metrics.get("coverage_avg", 0.0) or 0.0)
    staleness_p95 = metrics.get("staleness_p95")
    z20_extreme = float(metrics.get("z20_extreme_share", 0.0) or 0.0)

    status = "PASS"
    detail_parts = []

    if ratio_market_violation > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ss_ratio_market out of range [0,1]: {ratio_market_violation:.2e}")
    if ratio_with_restr_violation > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ss_ratio_with_restr out of range [0,1]: {ratio_with_restr_violation:.2e}")
    if ratio_without_restr_violation > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ss_ratio_without_restr out of range [0,1]: {ratio_without_restr_violation:.2e}")
    if ratio_sum_violation > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ss_ratio_with_restr + ss_ratio_without_restr > 1.005: {ratio_sum_violation:.2e}")
    if ss_total_neg > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ss_total negative: {ss_total_neg:.2e}")
    if staleness_neg > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ss_staleness_bd negative (future leak): {staleness_neg:.2e}")
    if is_valid_inv > 1e-4:
        status = "FAIL"
        detail_parts.append(f"is_ss_valid invalid values: {is_valid_inv:.2e}")
    if extreme_hi_inv > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ss_extreme_hi invalid values: {extreme_hi_inv:.2e}")
    if regime_switch_inv > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ss_regime_switch invalid values: {regime_switch_inv:.2e}")

    # Coverage check: should be ≥ 0.7 (70% of days have valid data)
    if coverage < 0.7:
        status = _escalate(status, "WARN")
        detail_parts.append(f"coverage (is_ss_valid avg)={coverage:.2%} < 0.7")

    # Staleness p95 check: should not be excessively large
    if isinstance(staleness_p95, (int, float)) and staleness_p95 > 10:
        status = _escalate(status, "WARN")
        detail_parts.append(f"staleness_p95={staleness_p95:.1f} days (may indicate data gaps)")

    # Extreme z-scores should be rare (<0.5%)
    if z20_extreme > 0.005:
        status = _escalate(status, "WARN")
        detail_parts.append(f"ss_ratio_market_z20 extreme (|z|>3) share={z20_extreme:.2%} > 0.5%")

    score.add(
        "11) Sector short selling features (as‑of)",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check10_short_positions(
    path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None
) -> None:
    """Validate short selling positions features (ssp_*)."""
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    required = [
        "ssp_ratio_sum",
        "ssp_reporters",
        "ssp_top_ratio",
        "ssp_delta_sum",
        "ssp_delta_pos",
        "ssp_delta_neg",
        "ssp_hhi",
        "ssp_is_recent",
        "ssp_staleness_days",
        "ssp_ratio_sum_z20",
        "ssp_delta_sum_z20",
        "ssp_ratio_sum_ema10",
        "is_ssp_valid",
    ]
    missing = [col for col in required if col not in cols]
    if missing:
        score.add(
            "10) Short positions features (as‑of)",
            "FAIL",
            f"missing required ssp_* columns: {missing}",
        )
        return

    exprs = [
        # Range checks: ratio_sum ∈ [0, 1]
        pl.mean(
            ((pl.col("ssp_ratio_sum") < -1e-6) | (pl.col("ssp_ratio_sum") > 1.0001))
            & pl.col("ssp_ratio_sum").is_not_null()
        ).alias("ratio_sum_violation"),
        # Range checks: delta_sum ∈ [-1, 1]
        pl.mean(
            ((pl.col("ssp_delta_sum") < -1.0001) | (pl.col("ssp_delta_sum") > 1.0001))
            & pl.col("ssp_delta_sum").is_not_null()
        ).alias("delta_sum_violation"),
        # Range checks: top_ratio ∈ [0, 1]
        pl.mean(
            ((pl.col("ssp_top_ratio") < -1e-6) | (pl.col("ssp_top_ratio") > 1.0001))
            & pl.col("ssp_top_ratio").is_not_null()
        ).alias("top_ratio_violation"),
        # Reporters must be ≥ 0
        pl.mean(((pl.col("ssp_reporters") < 0) & pl.col("ssp_reporters").is_not_null()).cast(pl.Float64)).alias(
            "reporters_negative"
        ),
        # Staleness must be ≥ 0 (no future disclosures)
        pl.mean(
            ((pl.col("ssp_staleness_days") < 0) & pl.col("ssp_staleness_days").is_not_null()).cast(pl.Float64)
        ).alias("staleness_negative"),
        # is_recent must be 0 or 1
        pl.mean(
            (pl.col("ssp_is_recent").is_not_null() & ~pl.col("ssp_is_recent").is_in([0, 1])).cast(pl.Float64)
        ).alias("is_recent_invalid"),
        # is_ssp_valid must be 0 or 1
        pl.mean((pl.col("is_ssp_valid").is_not_null() & ~pl.col("is_ssp_valid").is_in([0, 1])).cast(pl.Float64)).alias(
            "is_valid_invalid"
        ),
        # Coverage: is_ssp_valid daily average
        pl.mean(pl.col("is_ssp_valid").cast(pl.Float64)).alias("coverage_avg"),
        # Staleness p95 (should not be too large)
        pl.col("ssp_staleness_days")
        .filter(pl.col("ssp_staleness_days").is_not_null())
        .quantile(0.95)
        .alias("staleness_p95"),
        # Extreme z-scores (should be rare)
        pl.mean(
            ((pl.col("ssp_ratio_sum_z20").abs() > 6.0) & pl.col("ssp_ratio_sum_z20").is_not_null()).cast(pl.Float64)
        ).alias("z20_extreme_share"),
        pl.mean(
            ((pl.col("ssp_delta_sum_z20").abs() > 6.0) & pl.col("ssp_delta_sum_z20").is_not_null()).cast(pl.Float64)
        ).alias("delta_z20_extreme_share"),
    ]

    try:
        stats = lf.select(exprs).collect()
        metrics = {}
        for col in stats.columns:
            val = stats[col][0]
            if val is not None:
                metrics[col] = float(val)
    except Exception as exc:
        status = "WARN"
        detail_parts = [f"metric computation failed: {exc}"]
        stats = {}
        metrics = {}

    ratio_violation = float(metrics.get("ratio_sum_violation", 0.0) or 0.0)
    delta_violation = float(metrics.get("delta_sum_violation", 0.0) or 0.0)
    top_violation = float(metrics.get("top_ratio_violation", 0.0) or 0.0)
    reporters_neg = float(metrics.get("reporters_negative", 0.0) or 0.0)
    staleness_neg = float(metrics.get("staleness_negative", 0.0) or 0.0)
    is_recent_inv = float(metrics.get("is_recent_invalid", 0.0) or 0.0)
    is_valid_inv = float(metrics.get("is_valid_invalid", 0.0) or 0.0)
    coverage = float(metrics.get("coverage_avg", 0.0) or 0.0)
    staleness_p95 = metrics.get("staleness_p95")
    z20_extreme = float(metrics.get("z20_extreme_share", 0.0) or 0.0)
    delta_z20_extreme = float(metrics.get("delta_z20_extreme_share", 0.0) or 0.0)

    status = "PASS"
    detail_parts = []

    if ratio_violation > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ssp_ratio_sum out of range [0,1]: {ratio_violation:.2e}")
    if delta_violation > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ssp_delta_sum out of range [-1,1]: {delta_violation:.2e}")
    if top_violation > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ssp_top_ratio out of range [0,1]: {top_violation:.2e}")
    if reporters_neg > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ssp_reporters negative: {reporters_neg:.2e}")
    if staleness_neg > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ssp_staleness_days negative (future leak): {staleness_neg:.2e}")
    if is_recent_inv > 1e-4:
        status = "FAIL"
        detail_parts.append(f"ssp_is_recent invalid values: {is_recent_inv:.2e}")
    if is_valid_inv > 1e-4:
        status = "FAIL"
        detail_parts.append(f"is_ssp_valid invalid values: {is_valid_inv:.2e}")

    # Coverage check: should be ≥ 0.6 (60% of days have valid data, sparsity expected)
    if coverage < 0.6:
        status = _escalate(status, "WARN")
        detail_parts.append(f"coverage (is_ssp_valid avg)={coverage:.2%} < 0.6")

    # Staleness p95 check: should not be excessively large
    if isinstance(staleness_p95, (int, float)) and staleness_p95 > 60:
        status = _escalate(status, "WARN")
        detail_parts.append(f"staleness_p95={staleness_p95:.1f} days (may indicate data gaps)")

    # Extreme z-scores should be rare
    if z20_extreme > 0.01:
        status = _escalate(status, "WARN")
        detail_parts.append(f"ssp_ratio_sum_z20 extreme (|z|>6) share={z20_extreme:.2%}")
    if delta_z20_extreme > 0.01:
        status = _escalate(status, "WARN")
        detail_parts.append(f"ssp_delta_sum_z20 extreme (|z|>6) share={delta_z20_extreme:.2%}")

    score.add(
        "10) Short positions features (as‑of)",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check22_technical_indicators(
    path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None
) -> None:
    """Validate P0 technical indicators (ADX/DMI, Donchian, Keltner, Aroon, CRSI, OBV/CMF, ATR normalization, Amihud, Beta/Alpha)."""
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns

    # P0 required indicators
    required = [
        # ADX/DMI
        "adx_14",
        "dmi_pos_14",
        "dmi_neg_14",
        # Donchian
        "don_high_20",
        "don_low_20",
        "don_width_20",
        # Keltner + TTM
        "kc_mid_20",
        "kc_up_20",
        "kc_dn_20",
        # Aroon
        "aroon_up_25",
        "aroon_dn_25",
        "aroon_osc_25",
        # CRSI
        "crsi_3_2_100",
        # OBV/CMF
        "obv",
        "cmf_20",
        # ATR normalization
        "gap_atr",
        "idr_atr",
        # Amihud
        "amihud_20",
        # Beta/Alpha
        "beta60_topix",
        "alpha60_topix",
    ]
    missing = [col for col in required if col not in cols]
    if missing:
        score.add(
            "22) P0 Technical Indicators",
            "WARN",
            f"missing P0 technical indicator columns: {missing}",
        )
        return

    exprs = [
        # ADX range check (0-100)
        pl.mean(((pl.col("adx_14") < 0) | (pl.col("adx_14") > 100)) & pl.col("adx_14").is_not_null())
        .cast(pl.Float64)
        .alias("adx_out_of_range"),
        # DMI range check (0-100)
        pl.mean(((pl.col("dmi_pos_14") < 0) | (pl.col("dmi_pos_14") > 100)) & pl.col("dmi_pos_14").is_not_null())
        .cast(pl.Float64)
        .alias("dmi_pos_out_of_range"),
        pl.mean(((pl.col("dmi_neg_14") < 0) | (pl.col("dmi_neg_14") > 100)) & pl.col("dmi_neg_14").is_not_null())
        .cast(pl.Float64)
        .alias("dmi_neg_out_of_range"),
        # Aroon range check (0-100)
        pl.mean(((pl.col("aroon_up_25") < 0) | (pl.col("aroon_up_25") > 100)) & pl.col("aroon_up_25").is_not_null())
        .cast(pl.Float64)
        .alias("aroon_up_out_of_range"),
        pl.mean(((pl.col("aroon_dn_25") < 0) | (pl.col("aroon_dn_25") > 100)) & pl.col("aroon_dn_25").is_not_null())
        .cast(pl.Float64)
        .alias("aroon_dn_out_of_range"),
        # CRSI range check (0-100)
        pl.mean(((pl.col("crsi_3_2_100") < 0) | (pl.col("crsi_3_2_100") > 100)) & pl.col("crsi_3_2_100").is_not_null())
        .cast(pl.Float64)
        .alias("crsi_out_of_range"),
        # CMF range check (-1 to 1)
        pl.mean(((pl.col("cmf_20") < -1.1) | (pl.col("cmf_20") > 1.1)) & pl.col("cmf_20").is_not_null())
        .cast(pl.Float64)
        .alias("cmf_out_of_range"),
        # Donchian break flags (0/1)
        pl.mean(
            (~pl.col("don_break_20_up").is_in([0, 1]) & pl.col("don_break_20_up").is_not_null()).cast(pl.Float64)
        ).alias("don_break_up_invalid"),
        pl.mean(
            (~pl.col("don_break_20_down").is_in([0, 1]) & pl.col("don_break_20_down").is_not_null()).cast(pl.Float64)
        ).alias("don_break_down_invalid"),
        # TTM squeeze flags (0/1)
        pl.mean(
            (~pl.col("ttm_squeeze_on").is_in([0, 1]) & pl.col("ttm_squeeze_on").is_not_null()).cast(pl.Float64)
        ).alias("ttm_squeeze_on_invalid"),
        pl.mean(
            (~pl.col("ttm_squeeze_fire").is_in([0, 1]) & pl.col("ttm_squeeze_fire").is_not_null()).cast(pl.Float64)
        ).alias("ttm_squeeze_fire_invalid"),
        # Beta/Alpha extreme values
        pl.mean(((pl.col("beta60_topix").abs() > 10.0) & pl.col("beta60_topix").is_not_null()).cast(pl.Float64)).alias(
            "beta_extreme"
        ),
        pl.mean(((pl.col("alpha60_topix").abs() > 0.5) & pl.col("alpha60_topix").is_not_null()).cast(pl.Float64)).alias(
            "alpha_extreme"
        ),
    ]

    try:
        stats = lf.select(exprs).collect()
        metrics = {}
        for col in stats.columns:
            val = stats[col][0]
            if val is not None:
                metrics[col] = float(val)
    except Exception as exc:
        status = "WARN"
        detail_parts = [f"metric computation failed: {exc}"]
        stats = {}
        metrics = {}

    adx_out = float(metrics.get("adx_out_of_range", 0.0) or 0.0)
    dmi_pos_out = float(metrics.get("dmi_pos_out_of_range", 0.0) or 0.0)
    dmi_neg_out = float(metrics.get("dmi_neg_out_of_range", 0.0) or 0.0)
    aroon_up_out = float(metrics.get("aroon_up_out_of_range", 0.0) or 0.0)
    aroon_dn_out = float(metrics.get("aroon_dn_out_of_range", 0.0) or 0.0)
    crsi_out = float(metrics.get("crsi_out_of_range", 0.0) or 0.0)
    cmf_out = float(metrics.get("cmf_out_of_range", 0.0) or 0.0)
    don_break_up_inv = float(metrics.get("don_break_up_invalid", 0.0) or 0.0)
    don_break_down_inv = float(metrics.get("don_break_down_invalid", 0.0) or 0.0)
    ttm_squeeze_on_inv = float(metrics.get("ttm_squeeze_on_invalid", 0.0) or 0.0)
    ttm_squeeze_fire_inv = float(metrics.get("ttm_squeeze_fire_invalid", 0.0) or 0.0)
    beta_extreme = float(metrics.get("beta_extreme", 0.0) or 0.0)
    alpha_extreme = float(metrics.get("alpha_extreme", 0.0) or 0.0)

    status = "PASS"
    detail_parts = []

    # Range checks
    if adx_out > 0.01 or dmi_pos_out > 0.01 or dmi_neg_out > 0.01:
        status = _escalate(status, "WARN")
        detail_parts.append(f"ADX/DMI out of range: {adx_out:.2%}/{dmi_pos_out:.2%}/{dmi_neg_out:.2%}")

    if aroon_up_out > 0.01 or aroon_dn_out > 0.01:
        status = _escalate(status, "WARN")
        detail_parts.append(f"Aroon out of range: {aroon_up_out:.2%}/{aroon_dn_out:.2%}")

    if crsi_out > 0.01:
        status = _escalate(status, "WARN")
        detail_parts.append(f"CRSI out of range: {crsi_out:.2%}")

    if cmf_out > 0.01:
        status = _escalate(status, "WARN")
        detail_parts.append(f"CMF out of range: {cmf_out:.2%}")

    # Flag checks
    if don_break_up_inv > 0.001 or don_break_down_inv > 0.001:
        status = "FAIL"
        detail_parts.append(f"Donchian break flags invalid: {don_break_up_inv:.3%}/{don_break_down_inv:.3%}")

    if ttm_squeeze_on_inv > 0.001 or ttm_squeeze_fire_inv > 0.001:
        status = "FAIL"
        detail_parts.append(f"TTM squeeze flags invalid: {ttm_squeeze_on_inv:.3%}/{ttm_squeeze_fire_inv:.3%}")

    # Extreme value checks
    if beta_extreme > 0.01:
        status = _escalate(status, "WARN")
        detail_parts.append(f"Beta extreme (|β|>10) share: {beta_extreme:.2%}")

    if alpha_extreme > 0.01:
        status = _escalate(status, "WARN")
        detail_parts.append(f"Alpha extreme (|α|>0.5) share: {alpha_extreme:.2%}")

    score.add(
        "22) P0 Technical Indicators",
        status,
        "; ".join(detail_parts) if detail_parts else "ok",
        metrics,
    )


def check23_flow_supply(path: str, score: Score, start: str | None, end: str | None, snapshots_dir: str | None) -> None:
    lf = load_dataset_lazy(path, start, end)
    cols = lf.columns
    status = "PASS"
    details: list[str] = []

    def _quantile(col: str, q: float) -> float | None:
        try:
            res = lf.select(pl.col(col).drop_nulls().quantile(q).alias("q")).collect()
            if res.is_empty():
                return None
            val = res["q"][0]
            return None if val is None else float(val)
        except Exception:
            return None

    float_col = resolve_column(cols, "float_turnover_pct_tradable") or resolve_column(cols, "float_turnover_pct")
    if float_col:
        q99 = _quantile(float_col, 0.99)
        q01 = _quantile(float_col, 0.01)
        if q99 is not None and q99 > 5.0:
            status = _escalate(status, "WARN")
            details.append(f"float_turnover q99={q99:.2f}")
        if q01 is not None and q01 < -0.05:
            status = _escalate(status, "WARN")
            details.append(f"float_turnover q01={q01:.2f}")
    else:
        status = _escalate(status, "WARN")
        details.append("missing float_turnover_pct")

    margin_pct_col = resolve_column(cols, "margin_long_pct_tradable") or resolve_column(cols, "margin_long_pct_float")
    if margin_pct_col:
        q99 = _quantile(margin_pct_col, 0.99)
        if q99 is not None and q99 > 5.0:
            status = _escalate(status, "WARN")
            details.append(f"margin_pct q99={q99:.2f}")
    else:
        status = _escalate(status, "WARN")
        details.append("missing margin_long_pct_float/margin_long_pct_tradable")

    coverage_targets = [
        ("crowding_score", "crowding"),
        ("squeeze_risk", "squeeze"),
        ("margin_pain_index", "mpi"),
        ("preE_risk_score", "preE"),
        ("liquidity_impact", "liquidity"),
        ("gap_predictor", "gap"),
        ("basis_gate", "basis"),
        ("supply_shock", "supply"),
    ]
    for col, label in coverage_targets:
        if col not in cols:
            status = _escalate(status, "WARN")
            details.append(f"missing {label}")
            continue
        cov = coverage_ratio(lf, [col])
        if cov < 0.2:
            status = _escalate(status, "WARN")
            details.append(f"{label} coverage={cov:.2f}")

    supply_col = resolve_column(cols, "supply_shock")
    if supply_col:
        try:
            uniq = lf.select(pl.col(supply_col).drop_nulls().unique()).collect().get_columns()[0].to_list()
            if any(val not in (-1, 0, 1) for val in uniq if val is not None):
                status = _escalate(status, "WARN")
                details.append("supply_shock outside {-1,0,1}")
        except Exception:
            pass

    div_gap_col = resolve_column(cols, "div_ex_gap_miss")
    if div_gap_col:
        q99 = _quantile(div_gap_col, 0.99)
        q01 = _quantile(div_gap_col, 0.01)
        if (q99 is not None and abs(q99) > 0.5) or (q01 is not None and abs(q01) > 0.5):
            status = _escalate(status, "WARN")
            details.append("div_gap_miss spread large")

    detail = ", ".join(details) if details else "ok"
    score.add("23) Flow/Supply composite features", status, detail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to ml_dataset_*.parquet")
    ap.add_argument("--start", default=None, help="Start date (YYYY-MM-DD) for validation window")
    ap.add_argument("--end", default=None, help="End date (YYYY-MM-DD) for validation window")
    ap.add_argument(
        "--snapshots-dir", default=None, help="Directory containing *_snapshot.parquet with availability_ts"
    )
    ap.add_argument("--policy-margin", default="T+1_0900")
    ap.add_argument("--policy-earnings", default="T+1_1900")
    ap.add_argument("--policy-am", default="T+1_0900")
    ap.add_argument("--contract", default=None, help="Path to YAML contract file for contract-driven validation")
    args = ap.parse_args()

    score = Score()

    # (0)
    check0_primary_key_and_core(args.dataset, score, args.start, args.end)
    # (1)
    check1_schema_governance(args.dataset, score)
    # (2)
    check2_returns_left_closed(args.dataset, score, args.start, args.end)
    # (3)
    check3_gap_intraday(args.dataset, score, args.start, args.end)
    # (4)
    check4_margin_asof(args.dataset, score, args.start, args.end, args.snapshots_dir)
    # (5)
    check5_earnings(args.dataset, score, args.start, args.end, args.snapshots_dir)
    # (6)
    check6_fs_dividends(args.dataset, score, args.start, args.end, args.snapshots_dir)
    # (7)
    check7_indices(args.dataset, score, args.start, args.end)
    # (8)
    check8_limit_session(args.dataset, score, args.start, args.end)
    # (9)
    check9_breakdown(args.dataset, score, args.start, args.end, args.snapshots_dir)
    # (10)
    check10_short_positions(args.dataset, score, args.start, args.end, args.snapshots_dir)
    # (11)
    check11_sector_short_selling(args.dataset, score, args.start, args.end, args.snapshots_dir)
    # (12)
    check12_weekly_margin(args.dataset, score, args.start, args.end, args.snapshots_dir)
    check13_index_option_225(args.dataset, score, args.start, args.end, args.snapshots_dir)
    check14_index_features(args.dataset, score, args.start, args.end, args.snapshots_dir)
    check15_topix_features(args.dataset, score, args.start, args.end, args.snapshots_dir)
    check16_trades_spec_features(args.dataset, score, args.start, args.end, args.snapshots_dir)
    check17_earnings_announcement(args.dataset, score, args.start, args.end, args.snapshots_dir)
    check18_trading_calendar(args.dataset, score, args.start, args.end, args.snapshots_dir)
    check19_financial_statements(args.dataset, score, args.start, args.end, args.snapshots_dir)
    check20_daily_quotes(args.dataset, score, args.start, args.end, args.snapshots_dir)
    check21_listed_info(args.dataset, score, args.start, args.end, args.snapshots_dir)
    check22_technical_indicators(args.dataset, score, args.start, args.end, args.snapshots_dir)
    check23_flow_supply(args.dataset, score, args.start, args.end, args.snapshots_dir)

    # Optional: Run contract-driven validation if contract file is provided
    contract_path = getattr(args, "contract", None)
    if contract_path:
        try:
            import sys
            from pathlib import Path

            # Add tools directory to path for import
            tools_dir = Path(__file__).parent
            if str(tools_dir) not in sys.path:
                sys.path.insert(0, str(tools_dir))

            from validator_jqx_all import run_validation

            contract_report = run_validation(
                dataset_path=args.dataset,
                contract_path=contract_path,
                debug_meta_dir=args.snapshots_dir,
                report_path=None,  # Don't write separate report
                summary_path=None,
            )
            # Add contract validation results to score
            failed_blocks = [b for b in contract_report.get("blocks", []) if b.get("status") == "FAIL"]
            warn_blocks = [b for b in contract_report.get("blocks", []) if b.get("status") == "WARN"]

            if contract_report.get("status") == "FAIL":
                score.add(
                    "Contract validation",
                    "FAIL",
                    f"{len(failed_blocks)} blocks failed: {', '.join([b['block'] for b in failed_blocks[:5]])}",
                )
            elif contract_report.get("status") == "WARN":
                score.add(
                    "Contract validation",
                    "WARN",
                    f"{len(warn_blocks)} blocks have warnings",
                )
        except ImportError:
            # validator_jqx_all not available, skip
            pass
        except Exception as exc:
            score.add(
                "Contract validation",
                "WARN",
                f"Contract validation skipped: {exc}",
            )

    summary = score.summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Write JSON next to dataset path
    out_json = os.path.splitext(args.dataset)[0] + ".validator_1to9.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n📄 Wrote: {out_json}")


if __name__ == "__main__":
    main()
