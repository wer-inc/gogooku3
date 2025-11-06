#!/usr/bin/env python3
"""Phase 2 dataset acceptance gate with fundamentals & breakdown checks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

DATASET_PATH = Path("gogooku5/data/output/ml_dataset_latest_full.parquet")
REPORT_PATH = Path("/tmp/phase2_dataset_accept_report.json")


def _load_dataset(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pl.read_parquet(str(path))


def _series_quantile(df: pl.DataFrame, column: str, q: float, *, absolute: bool = False) -> float | None:
    if column not in df.columns:
        return None
    series = df.select(pl.col(column)).to_series().drop_nulls()
    if series.is_empty():
        return None
    if absolute:
        series = series.abs()
    try:
        return float(series.quantile(q, interpolation="nearest"))
    except Exception:
        return None


def _mean(df: pl.DataFrame, column: str, *, filter_expr: pl.Expr | None = None) -> float | None:
    if column not in df.columns:
        return None
    expr = pl.col(column).cast(pl.Float64)
    if filter_expr is not None:
        result = df.lazy().filter(filter_expr).select(expr.mean()).collect()
    else:
        result = df.select(expr.mean())
    value = result.item() if result.height else None
    if value is None:
        return None
    return float(value)


@dataclass
class CheckResult:
    status: str
    details: dict[str, Any]


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def run_checks(df: pl.DataFrame) -> dict[str, CheckResult]:
    cols = df.columns
    lower_cols = {col.lower(): col for col in cols}

    results: dict[str, CheckResult] = {}

    # 1. Canonical OHLC enforcement
    banned = {"close", "open", "high", "low", "volume", "adj close", "adjclose"}
    banned_present = [col for col in cols if col.lower() in banned]
    canonical_map = {
        "adjustmentclose": {"adjustmentclose", "AdjustmentClose"},
        "adjustmentopen": {"adjustmentopen", "AdjustmentOpen"},
        "adjustmenthigh": {"adjustmenthigh", "AdjustmentHigh"},
        "adjustmentlow": {"adjustmentlow", "AdjustmentLow"},
        "adjustmentvolume": {"adjustmentvolume", "AdjustmentVolume"},
    }
    canonical_missing = [
        base
        for base, aliases in canonical_map.items()
        if not any(alias in lower_cols or alias in cols for alias in aliases)
    ]
    ohlc_pass = not banned_present and not canonical_missing
    results["canonical_ohlc"] = CheckResult(
        status=_status(ohlc_pass),
        details={
            "banned_columns": banned_present,
            "canonical_missing": canonical_missing,
        },
    )

    # 2. Gap decomposition sanity
    gap_cols = {
        "gap_ov_prev1",
        "gap_id_prev1",
        "gap_amplify_ratio_prev1",
        "gap_sign_concord_prev1",
        "gap_fill_ratio_prev1",
        "gap_filled_prev1_flag",
    }
    gap_missing = [col for col in gap_cols if col not in cols]
    gap_today = [col for col in ("gap_ov_today", "gap_id_today") if col in cols]
    gap_identity_alert = None
    gap_pass = not gap_missing and not gap_today
    if gap_pass and {"ret_prev_1d"}.issubset(cols):
        mask = (
            pl.col("gap_ov_prev1").is_not_null()
            & pl.col("gap_id_prev1").is_not_null()
            & pl.col("ret_prev_1d").is_not_null()
        )
        residuals = (
            df.lazy()
            .filter(mask)
            .with_columns(
                ((1 + pl.col("gap_ov_prev1")) * (1 + pl.col("gap_id_prev1")) - (1 + pl.col("ret_prev_1d")))
                .abs()
                .alias("_gap_residual")
            )
            .select(pl.col("_gap_residual").gt(1e-3).mean().alias("violation_rate"))
            .collect()
        )
        violation_rate = float(residuals.item()) if residuals.height else 0.0
        gap_identity_alert = violation_rate
        gap_pass = gap_pass and violation_rate <= 0.01
    results["gap_decomposition"] = CheckResult(
        status=_status(gap_pass),
        details={
            "missing": gap_missing,
            "forbidden_today_cols": gap_today,
            "residual_gt_1e-3_rate": gap_identity_alert,
        },
    )

    # 3. Financial statements coverage
    fs_required = [
        "fs_revenue_ttm",
        "fs_op_profit_ttm",
        "fs_net_income_ttm",
        "fs_cfo_ttm",
        "fs_capex_ttm",
        "fs_fcf_ttm",
        "fs_sales_yoy",
        "fs_op_margin",
        "fs_net_margin",
        "fs_roe_ttm",
        "fs_roa_ttm",
        "fs_accruals_ttm",
        "fs_cfo_to_ni",
        "fs_observation_count",
        "fs_lag_days",
        "fs_is_recent",
        "fs_staleness_bd",
        "is_fs_valid",
    ]
    fs_missing = [col for col in fs_required if col not in cols]
    fs_valid_rate = _mean(df, "is_fs_valid")
    fs_recent_rate = _mean(df, "fs_is_recent", filter_expr=pl.col("is_fs_valid") == 1)
    fs_staleness_p95 = _series_quantile(df, "fs_staleness_bd", 0.95)
    fs_lag_p95 = _series_quantile(df, "fs_lag_days", 0.95)
    fs_pass = (
        not fs_missing
        and (fs_valid_rate is not None and fs_valid_rate >= 0.5)
        and (fs_recent_rate is not None and fs_recent_rate >= 0.5)
        and (fs_staleness_p95 is not None and fs_staleness_p95 <= 120)
        and (fs_lag_p95 is None or fs_lag_p95 <= 120)
    )
    results["financial_statements"] = CheckResult(
        status=_status(fs_pass),
        details={
            "missing": fs_missing,
            "valid_rate": fs_valid_rate,
            "recent_rate": fs_recent_rate,
            "staleness_p95": fs_staleness_p95,
            "lag_p95": fs_lag_p95,
        },
    )

    # 4. Dividend governance
    div_required = [
        "div_days_to_ex",
        "div_pre1",
        "div_pre3",
        "div_pre5",
        "div_post1",
        "div_post3",
        "div_post5",
        "div_is_ex0",
        "div_dy_12m",
        "div_is_obs",
        "div_is_special",
        "div_staleness_bd",
    ]
    div_missing = [col for col in div_required if col not in cols]
    div_obs_rate = _mean(df, "div_is_obs")
    div_abs_max = _series_quantile(df, "div_days_to_ex", 1.0, absolute=True)
    div_staleness_p95 = _series_quantile(df, "div_staleness_bd", 0.95)
    div_pass = (
        not div_missing
        and (div_abs_max is None or div_abs_max <= 260)
        and (div_staleness_p95 is None or div_staleness_p95 <= 60)
    )
    results["dividend_features"] = CheckResult(
        status=_status(div_pass),
        details={
            "missing": div_missing,
            "obs_rate": div_obs_rate,
            "abs_days_to_ex_max": div_abs_max,
            "staleness_p95": div_staleness_p95,
        },
    )

    # 5. Investor breakdown block
    bd_required = [
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
    bd_missing = [col for col in bd_required if col not in cols]
    bd_valid_rate = _mean(df, "is_bd_valid")
    bd_is_recent_rate = _mean(df, "bd_is_recent")
    bd_staleness_p95 = _series_quantile(df, "bd_staleness_bd", 0.95)
    bd_net_ratio_p99 = _series_quantile(df, "bd_net_ratio", 0.99, absolute=True)
    bd_short_share_p99 = _series_quantile(df, "bd_short_share", 0.99, absolute=True)
    bd_net_z260_p99 = _series_quantile(df, "bd_net_z260", 0.99, absolute=True)
    bd_short_z260_p99 = _series_quantile(df, "bd_short_z260", 0.99, absolute=True)
    bd_pass = (
        not bd_missing
        and (bd_valid_rate is not None and bd_valid_rate >= 0.6)
        and (bd_staleness_p95 is not None and bd_staleness_p95 <= 5)
        and (bd_net_ratio_p99 is not None and bd_net_ratio_p99 <= 1.05)
        and (bd_short_share_p99 is not None and bd_short_share_p99 <= 1.05)
        and (bd_net_z260_p99 is not None and bd_net_z260_p99 <= 6.0)
        and (bd_short_z260_p99 is not None and bd_short_z260_p99 <= 6.0)
    )
    results["investor_breakdown"] = CheckResult(
        status=_status(bd_pass),
        details={
            "missing": bd_missing,
            "valid_rate": bd_valid_rate,
            "recent_rate": bd_is_recent_rate,
            "staleness_p95": bd_staleness_p95,
            "net_ratio_p99": bd_net_ratio_p99,
            "short_share_p99": bd_short_share_p99,
            "net_z260_p99": bd_net_z260_p99,
            "short_z260_p99": bd_short_z260_p99,
        },
    )

    # 6. Margin blocks
    margin_daily_required = [
        "dmi_net_adv60",
        "dmi_delta_net_adv60",
        "dmi_delta_net_adv60_z20",
        "dmi_long_short_ratio",
        "is_margin_daily_valid",
    ]
    md_missing = [col for col in margin_daily_required if col not in cols]
    md_valid_rate = _mean(df, "is_margin_daily_valid")
    md_net_adv_p99 = _series_quantile(df, "dmi_net_adv60", 0.99, absolute=True)
    md_pass = (
        not md_missing
        and (md_valid_rate is not None and md_valid_rate >= 0.85)
        and (md_net_adv_p99 is not None and md_net_adv_p99 <= 5.0)
    )
    results["margin_daily"] = CheckResult(
        status=_status(md_pass),
        details={
            "missing": md_missing,
            "valid_rate": md_valid_rate,
            "net_adv60_abs_p99": md_net_adv_p99,
        },
    )

    margin_weekly_required = [
        "wmi_net_adv5d",
        "wmi_delta_net_adv5d",
        "wmi_delta_net_adv5d_z52",
        "wmi_imbalance",
        "wmi_long_short_ratio",
        "is_margin_weekly_valid",
    ]
    mw_missing = [col for col in margin_weekly_required if col not in cols]
    mw_valid_rate = _mean(df, "is_margin_weekly_valid")
    mw_net_adv_p99 = _series_quantile(df, "wmi_net_adv5d", 0.99, absolute=True)
    mw_pass = (
        not mw_missing
        and (mw_valid_rate is not None and mw_valid_rate >= 0.5)
        and (mw_net_adv_p99 is not None and mw_net_adv_p99 <= 5.0)
    )
    results["margin_weekly"] = CheckResult(
        status=_status(mw_pass),
        details={
            "missing": mw_missing,
            "valid_rate": mw_valid_rate,
            "net_adv5d_abs_p99": mw_net_adv_p99,
        },
    )

    return results


def main() -> int:
    try:
        df = _load_dataset(DATASET_PATH)
    except FileNotFoundError as exc:
        print(f"❌ FAIL: {exc}")
        return 1

    print(f"📊 Validating dataset: {DATASET_PATH.name} ({df.height:,} rows, {len(df.columns)} columns)")
    print("=" * 80)

    checks = run_checks(df)

    statuses = []
    for name, result in checks.items():
        statuses.append(result.status)
        if result.status == "PASS":
            print(f"✅ PASS: {name}")
        else:
            print(f"❌ FAIL: {name}")
            for key, value in result.details.items():
                if value:
                    print(f"   - {key}: {value}")

    overall = "PASS" if all(status == "PASS" for status in statuses) else "FAIL"
    print("=" * 80)
    if overall == "PASS":
        print("🎉 ACCEPTANCE PASSED - Dataset ready for downstream tasks")
    else:
        print("❌ ACCEPTANCE FAILED - Resolve the issues above before proceeding")

    report = {
        "status": overall,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(DATASET_PATH),
            "rows": df.height,
            "cols": len(df.columns),
        },
        "checks": {
            name: {
                "status": result.status,
                "details": result.details,
            }
            for name, result in checks.items()
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"📄 Report saved: {REPORT_PATH}")

    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
