#!/usr/bin/env python3
"""
Build a lightweight ML dataset Parquet from data_v2.ml_dataset view.

This is designed to be:
- Fast to compute (narrow feature set, simple SQL)
- Reusable by multiple models (APEX-Ranker, ATFT-GAT-FAN, etc.)

Default preset: `fast_v1` for 20d-ahead ranking (target_20d).

Example:

PYTHONPATH=. python data_v2/scripts/build_ml_dataset_fast.py \\
  --db data_v2/output/jquants.duckdb \\
  --out data/output/datasets/apex_fast_20d.parquet \\
  --targets 20 \\
  --feature-preset fast_v1 \\
  --min-date 2019-01-04 \\
  --max-date 2025-11-20 \\
  --require-has-price-core \\
  --require-has-target \\
  --require-has-sessions \\
  --require-has-breakdown \\
  --require-has-long-window-60
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb

TARGET_COLUMN_MAP: dict[int, str] = {
    1: "fwd_ret_1d",
    5: "fwd_ret_5d",
    10: "fwd_ret_10d",
    20: "fwd_ret_20d",
}

TARGET_SETS: dict[str, list[int]] = {
    "ret_20": [20],
    "ret_multi": [1, 5, 10, 20],
    "ret_short": [1, 5],
}


FEATURE_PRESETS: dict[str, list[str]] = {
    # v1 fast preset for 20d-ahead ranking.
    # 価格・ボラ・出来高・需給・セッションをバランス良く 30本程度に絞ったセット。
    "fast_v1": [
        # 価格・モメンタム
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "logret_1d",
        "price_pos_20",
        # ボラ
        "rv_5",
        "rv_20",
        "rv_ratio",
        "range_z_20",
        # 出来高・流動性
        "vol_ratio_5",
        "vol_ratio_20",
        "val_ratio_5",
        "val_ratio_20",
        "log_amihud_20",
        # テクニカル
        "rsi_14",
        "macd_hist",
        "macd_hist_diff_1",
        "adx_14",
        # 需給（Breakdown + 信用）
        "breakdown_buy_ratio",
        "sell_short_ratio",
        "inst_short_ratio_value",
        "retail_lev_ratio_value",
        "flow_imbalance",
        "margin_flow_intensity",
        "short_overhang_days",
        "short_squeeze_risk",
        # マージンキャピチュレーション
        "margin_capitulation_ratio",
        "margin_capitulation_spike",
        # セッション（日中マイクロストラクチャ）
        "ret_morning",
        "ret_afternoon",
        "session_corr_20",
        "morning_trend_eff",
        "session_volume_ratio",
    ],
    # v2 preset: fast_v1 + listed_meta 基本セット（セクター・市場・規模・α）
    "fast_v2": [
        # fast_v1 ベース
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "logret_1d",
        "price_pos_20",
        "rv_5",
        "rv_20",
        "rv_ratio",
        "range_z_20",
        "vol_ratio_5",
        "vol_ratio_20",
        "val_ratio_5",
        "val_ratio_20",
        "log_amihud_20",
        "rsi_14",
        "macd_hist",
        "macd_hist_diff_1",
        "adx_14",
        "breakdown_buy_ratio",
        "sell_short_ratio",
        "inst_short_ratio_value",
        "retail_lev_ratio_value",
        "flow_imbalance",
        "margin_flow_intensity",
        "short_overhang_days",
        "short_squeeze_risk",
        "margin_capitulation_ratio",
        "margin_capitulation_spike",
        "ret_morning",
        "ret_afternoon",
        "session_corr_20",
        "morning_trend_eff",
        "session_volume_ratio",
        # ここから listed_meta 由来（セクター / 市場 / 規模 / α）
        # セクター α & 分位（5d/20d）
        "sector_alpha_ret_5d",
        "sector_alpha_ret_20d",
        "sector_pct_ret_5d",
        "sector_rank_ret_5d",
        # セクター平均のスケール感（任意だが入れておく）
        "sector_ret_5d",
        "sector_ret_20d",
        # 市場区分フラグ
        "is_prime",
        "is_standard",
        "is_growth",
        # 規模バケット
        "is_small_cap",
        "is_mid_cap",
        "is_large_cap",
        # 貸借可能フラグ
        "is_shortable",
    ],
}

# v3 preset: fast_v2 + K1-style fundamental factors (fund_* from financial_features).
FEATURE_PRESETS["fast_v3_fundamental"] = FEATURE_PRESETS["fast_v2"] + [
    # Core K1 net cash & valuation
    "fund_net_cash_k1",
    "fund_net_cash_ratio_k1",
    "fund_per_ttm",
    "fund_pbr",
    "fund_cash_neutral_per",
    # Growth & quality (FY YoY, ROE, margin)
    "fund_yoy_sales",
    "fund_yoy_profit",
    "fund_roe",
    "fund_op_margin",
    # K1-style flags
    "fund_is_cash_rich",
    # Extended growth/quality (optional but useful)
    "fund_yoy_cfo",
    "fund_accruals",
    "fund_yoy_net_cash",
    "fund_value_trap_flag",
]

# v4 preset: fast_v3_fundamental + Phase 3 advanced features (Risk/Alpha/Regime)
FEATURE_PRESETS["fast_v4"] = FEATURE_PRESETS["fast_v3_fundamental"] + [
    # Phase 3.1: Risk/Drawdown features (8)
    "drawdown_from_high_20",
    "drawdown_from_high_60",
    "max_drawdown_20",
    "max_drawdown_60",
    "risk_adj_mom_20",
    "risk_adj_mom_5",
    "underwater_days_20",
    "recovery_rate_20",
    # Phase 3.2: Market Alpha features (7)
    "market_alpha_1d",
    "market_alpha_5d",
    "market_alpha_20d",
    "rel_strength_5d",
    "rel_strength_20d",
    "cum_alpha_20d",
    "info_ratio_20d",
    # Phase 3.3: Volatility Regime features (5)
    "vol_expansion",
    "vol_trend_20",
    "vol_regime",
    "cs_vol_percentile",
    "vol_weighted_mom_20",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a lightweight ML dataset Parquet from data_v2.features_daily.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data_v2/output/jquants.duckdb"),
        help="Path to DuckDB file containing ml_dataset view (default: data_v2/output/jquants.duckdb)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output Parquet path (e.g. data/output/datasets/apex_fast_20d.parquet)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="20",
        help="Comma-separated horizons to include as targets (subset of 1,5,10,20). Default: 20",
    )
    parser.add_argument(
        "--target-set",
        type=str,
        default="custom",
        choices=["custom"] + sorted(TARGET_SETS.keys()),
        help="Named target set to use (overrides --targets unless 'custom').",
    )
    parser.add_argument(
        "--feature-preset",
        type=str,
        default="fast_v1",
        choices=sorted(FEATURE_PRESETS.keys()),
        help="Feature preset to use (default: fast_v1)",
    )
    parser.add_argument(
        "--min-date",
        type=str,
        default=None,
        help="Minimum Date (YYYY-MM-DD) to include (optional)",
    )
    parser.add_argument(
        "--max-date",
        type=str,
        default=None,
        help="Maximum Date (YYYY-MM-DD) to include (optional)",
    )
    # Row-level filters (all optional; can be combined)
    parser.add_argument(
        "--require-has-price-core",
        action="store_true",
        help="Require has_price_core = 1 in ml_dataset",
    )
    parser.add_argument(
        "--require-has-target",
        action="store_true",
        help="Require has_target_<h>d = 1 for all selected horizons",
    )
    parser.add_argument(
        "--require-has-sessions",
        action="store_true",
        help="Require has_sessions = 1 (ret_morning/ret_afternoon both available)",
    )
    parser.add_argument(
        "--require-has-breakdown",
        action="store_true",
        help="Require has_breakdown = 1 (breakdown-derived features available)",
    )
    parser.add_argument(
        "--require-has-long-window-60",
        action="store_true",
        help="Require has_long_window_60 = 1 (60d window features fully defined)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Do not write sidecar metadata JSON (APEX-Ranker v0 expects metadata by default).",
    )
    parser.add_argument(
        "--out-metadata",
        type=Path,
        default=None,
        help="Optional explicit metadata JSON path. Default: <out>_metadata.json",
    )
    return parser.parse_args()


def build_select_clause(horizons: list[int], feature_preset: str) -> str:
    """Build the SELECT clause for the dataset."""

    # Meta + targets
    select_parts: list[str] = [
        "date AS Date",
        "code AS Code",
    ]

    for h in horizons:
        src_col = TARGET_COLUMN_MAP[h]
        select_parts.append(f"{src_col} AS target_{h}d")

    # Feature columns from preset
    features = FEATURE_PRESETS[feature_preset]
    select_parts.extend(features)

    return ",\n  ".join(select_parts)


def build_where_clause(
    horizons: list[int],
    *,
    min_date: str | None,
    max_date: str | None,
    require_has_price_core: bool,
    require_has_target: bool,
    require_has_sessions: bool,
    require_has_breakdown: bool,
    require_has_long_window_60: bool,
) -> str:
    """Build WHERE clause with optional filters."""

    clauses: list[str] = []

    if min_date is not None:
        clauses.append(f"date >= DATE '{min_date}'")
    if max_date is not None:
        clauses.append(f"date <= DATE '{max_date}'")

    if require_has_price_core:
        clauses.append("has_price_core = 1")

    if require_has_target:
        for h in horizons:
            clauses.append(f"has_target_{h}d = 1")

    if require_has_sessions:
        clauses.append("has_sessions = 1")

    if require_has_breakdown:
        clauses.append("has_breakdown = 1")

    if require_has_long_window_60:
        clauses.append("has_long_window_60 = 1")

    if not clauses:
        return ""

    return "WHERE\n  " + "\n  AND ".join(clauses)


def resolve_horizons(args: argparse.Namespace) -> list[int]:
    """Resolve target horizons from CLI args."""

    if args.target_set != "custom":
        horizons = TARGET_SETS[args.target_set]
    else:
        try:
            horizons = sorted({int(h) for h in args.targets.split(",") if h.strip()})
        except ValueError as exc:  # pragma: no cover - defensive
            raise SystemExit(f"Invalid --targets value: {args.targets}") from exc

    unsupported = [h for h in horizons if h not in TARGET_COLUMN_MAP]
    if unsupported:
        raise SystemExit(f"Unsupported horizon(s): {unsupported} (allowed: 1,5,10,20)")

    return horizons


def main() -> None:
    args = parse_args()

    # Resolve horizons (target set or custom)
    horizons = resolve_horizons(args)

    # Build SQL
    select_clause = build_select_clause(horizons, args.feature_preset)
    target_columns = [f"target_{h}d" for h in horizons]
    feature_columns = FEATURE_PRESETS[args.feature_preset]
    where_clause = build_where_clause(
        horizons,
        min_date=args.min_date,
        max_date=args.max_date,
        require_has_price_core=args.require_has_price_core,
        require_has_target=args.require_has_target,
        require_has_sessions=args.require_has_sessions,
        require_has_breakdown=args.require_has_breakdown,
        require_has_long_window_60=args.require_has_long_window_60,
    )

    # Use logical ML view.
    sql_inside = f"SELECT\n  {select_clause}\nFROM ml_dataset\n{where_clause}"

    # Ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Run COPY in DuckDB for speed
    con = duckdb.connect(str(args.db))
    copy_sql = f"COPY ({sql_inside}) TO '{args.out}' (FORMAT PARQUET)"
    con.execute(copy_sql)

    # Quick summary
    rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{args.out}')").fetchone()[0]
    print(f"Wrote {rows:,} rows to {args.out}")

    # Optionally write minimal metadata JSON for APEX-Ranker and other consumers.
    if not args.no_metadata:
        try:
            import polars as pl

            # Compute simple schema hash (column name + dtype)
            df_head = pl.read_parquet(args.out, n_rows=0)
            schema_items = [f"{name}:{dtype}" for name, dtype in zip(df_head.columns, df_head.dtypes)]
            schema_blob = "\n".join(schema_items).encode("utf-8")
            schema_hash = hashlib.sha256(schema_blob).hexdigest()

            # Compute dataset hash (sha256 of Parquet file)
            h = hashlib.sha256()
            with args.out.open("rb") as fh:  # type: ignore[arg-type]
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            dataset_hash = h.hexdigest()

            metadata = {
                "dataset_hash": dataset_hash,
                "dataset_hash_algorithm": "sha256",
                "feature_schema_version": schema_hash,
                "feature_index": {
                    "schema_hash": schema_hash,
                },
                "row_count": rows,
                "path": str(args.out),
                "target_columns": target_columns,
                "feature_columns": feature_columns,
                "args": {
                    "targets": horizons,
                    "target_set": args.target_set,
                    "feature_preset": args.feature_preset,
                    "min_date": args.min_date,
                    "max_date": args.max_date,
                    "require_has_price_core": args.require_has_price_core,
                    "require_has_target": args.require_has_target,
                    "require_has_sessions": args.require_has_sessions,
                    "require_has_breakdown": args.require_has_breakdown,
                    "require_has_long_window_60": args.require_has_long_window_60,
                },
            }
            meta_path = args.out_metadata or args.out.with_name(args.out.stem + "_metadata.json")
            meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Wrote metadata JSON to {meta_path}")
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Failed to write metadata JSON: {exc}")


if __name__ == "__main__":
    main()
