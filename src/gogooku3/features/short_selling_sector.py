from __future__ import annotations

"""
業種別空売り比率特徴量 → 需給・圧力をセクター粒度で日次パネルに統合.

This module implements sector-wise short selling features with:
- T+1 effective date computation for next business day availability
- Sector-level short selling ratios and restriction ratios
- Market-wide aggregation and relative features
- Momentum, acceleration, and Z-score anomaly detection
- As-of backward join to prevent future data leakage
- Individual stock relative features vs sector and market

Key features generated:
- ss_sec33_short_share: 空売り比率（売りに占める空売り）
- ss_sec33_restrict_share: 価格規制下比率（パニック度合い）
- ss_sec33_short_share_d1: 空売り比率の日次変化
- ss_sec33_short_mom5: 5日間のモメンタム
- ss_sec33_short_share_z60: 60日間のZ-score異常度
- ss_mkt_short_share: 市場全体の空売り比率
- ss_mkt_restrict_share: 市場全体の価格規制比率
- ss_mkt_short_breadth_q80: 極値セクターの広がり
- ss_rel_short_share: セクター vs 市場の相対強さ
- is_ss_valid: データ有効性マスク

Public API:
- build_sector_short_features(ss_df, calendar_next_bday) -> sector aggregated features
- attach_sector_short_to_quotes(quotes, sector_feats, sector_map) -> individual stock features
- add_sector_short_selling_block(quotes, ss_df, listed_info_df, enable_z_scores=True)

References:
- 業種別空売り比率は1-5営業日で効きやすい需給・圧力指標
- T+1 rule: 公表時点はEOD後とみなし、翌営業日から使用
- セクター集計 → 個別銘柄as-of配布でリーク防止
"""

import datetime as _dt
from collections.abc import Callable
from typing import Optional

import polars as pl

EPS = 1e-12


def _next_business_day_jp(date_col: pl.Expr) -> pl.Expr:
    """日本の営業日ベースで翌営業日を計算 (for T+1 rule)."""
    wd = date_col.dt.weekday()
    return (
        pl.when(wd <= 3)  # Mon-Thu
        .then(date_col + pl.duration(days=1))
        .when(wd == 4)  # Fri
        .then(date_col + pl.duration(days=3))  # Next Mon
        .when(wd == 5)  # Sat
        .then(date_col + pl.duration(days=2))  # Next Mon
        .otherwise(date_col + pl.duration(days=1))  # Sun -> Mon
    )


def build_sector_short_features(
    ss_df: pl.DataFrame,
    calendar_next_bday: Optional[Callable[[pl.Expr], pl.Expr]] = None,
    enable_z_scores: bool = True
) -> pl.DataFrame:
    """
    Build sector-level short selling features (業種33 × 日付).

    Args:
        ss_df: Raw sector short selling data from J-Quants API
        calendar_next_bday: Function to compute next business day

    Returns:
        DataFrame with sector-level features and T+1 effective dates
    """
    if ss_df.is_empty():
        return ss_df

    calendar_fn = calendar_next_bday or _next_business_day_jp

    # 1) データ整形とセクター内集計（列名の方言に寛容）
    cols = set(ss_df.columns)
    # Build tolerant expressions for three key columns
    if "SellingExcludingShortSellingTurnoverValue" in cols:
        sell_ex_short_expr = pl.col("SellingExcludingShortSellingTurnoverValue").cast(pl.Float64)
    elif {"Selling", "ShortSelling"}.issubset(cols):
        sell_ex_short_expr = pl.col("Selling").cast(pl.Float64) - pl.col("ShortSelling").cast(pl.Float64)
    else:
        sell_ex_short_expr = pl.lit(None, dtype=pl.Float64)

    if "ShortSellingWithRestrictionsTurnoverValue" in cols:
        short_with_expr = pl.col("ShortSellingWithRestrictionsTurnoverValue").cast(pl.Float64)
    elif "ShortSellingWithPriceRestriction" in cols:
        short_with_expr = pl.col("ShortSellingWithPriceRestriction").cast(pl.Float64)
    else:
        short_with_expr = pl.lit(None, dtype=pl.Float64)

    if "ShortSellingWithoutRestrictionsTurnoverValue" in cols:
        short_without_expr = pl.col("ShortSellingWithoutRestrictionsTurnoverValue").cast(pl.Float64)
    elif {"ShortSelling", "ShortSellingWithPriceRestriction"}.issubset(cols):
        short_without_expr = (
            pl.col("ShortSelling").cast(pl.Float64) - pl.col("ShortSellingWithPriceRestriction").cast(pl.Float64)
        )
    else:
        short_without_expr = pl.lit(None, dtype=pl.Float64)

    s = (ss_df
         .with_columns([
            pl.col("Date").cast(pl.Date).alias("Date"),
            pl.col("Sector33Code").cast(pl.Utf8).alias("sec33"),
            sell_ex_short_expr.alias("sell_ex_short"),
            short_with_expr.alias("short_with"),
            short_without_expr.alias("short_without"),
         ])
         .group_by(["Date", "sec33"]).agg([
            pl.col("sell_ex_short").sum(),
            pl.col("short_with").sum(),
            pl.col("short_without").sum(),
         ])
    )

    # 2) セクター内派生特徴量
    s = s.with_columns([
        # 空売り総額と売り代金総額
        (pl.col("short_with") + pl.col("short_without")).alias("short_turnover"),
        (pl.col("sell_ex_short") + pl.col("short_with") + pl.col("short_without")).alias("total_selling"),
    ]).with_columns([
        # 基本比率
        (pl.col("short_turnover") / (pl.col("total_selling") + EPS)).alias("ss_sec33_short_share"),
        (pl.col("short_with") / (pl.col("short_turnover") + EPS)).alias("ss_sec33_restrict_share"),
        pl.col("short_turnover").alias("ss_sec33_short_turnover"),
    ]).sort(["sec33", "Date"])

    # 短小サンプルではウィンドウ/差分をスキップ（旧Polarsの制約回避）
    if s.height >= 2:
        s = s.with_columns([
            pl.col("ss_sec33_short_share").diff().over("sec33").alias("ss_sec33_short_share_d1"),
            pl.col("ss_sec33_restrict_share").diff().over("sec33").alias("ss_sec33_restrict_share_d1"),
            (pl.col("ss_sec33_short_share") - pl.col("ss_sec33_short_share").rolling_mean(5).over("sec33")).alias("ss_sec33_short_mom5"),
            (pl.col("ss_sec33_short_share").diff().over("sec33") - pl.col("ss_sec33_short_share").diff().over("sec33").shift(1).over("sec33")).alias("ss_sec33_short_accel"),
        ])
    else:
        s = s.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("ss_sec33_short_share_d1"),
            pl.lit(None, dtype=pl.Float64).alias("ss_sec33_restrict_share_d1"),
            pl.lit(None, dtype=pl.Float64).alias("ss_sec33_short_mom5"),
            pl.lit(None, dtype=pl.Float64).alias("ss_sec33_short_accel"),
        ])

    # Z-score features (optional, can be disabled for performance)
    if enable_z_scores:
        s = s.with_columns([
            # Z-score異常度 (イベント耐性)
            ((pl.col("ss_sec33_short_share") - pl.col("ss_sec33_short_share").rolling_mean(60).over("sec33")) /
             (pl.col("ss_sec33_short_share").rolling_std(60).over("sec33") + EPS)).alias("ss_sec33_short_share_z60"),
            ((pl.col("short_turnover") - pl.col("short_turnover").rolling_mean(252).over("sec33")) /
             (pl.col("short_turnover").rolling_std(252).over("sec33") + EPS)).alias("ss_sec33_short_turnover_z252"),
        ])

    # 3) 市場集計特徴量（全銘柄に同じ値を配布）
    mkt = (s.group_by("Date").agg([
            # マクロの空売り圧（地合い）
            (pl.col("short_turnover").sum() / (pl.col("total_selling").sum() + EPS)).alias("ss_mkt_short_share"),
            (pl.col("short_with").sum() / (pl.col("short_turnover").sum() + EPS)).alias("ss_mkt_restrict_share"),
        ]))

    # 当日分位点80%のbreadth (極値セクターの広がり)
    q80 = (s.group_by("Date")
             .agg(pl.col("ss_sec33_short_share").quantile(0.8).alias("q80")))

    breadth = (s.join(q80, on="Date", how="left")
                .with_columns([
                    (pl.col("ss_sec33_short_share") > pl.col("q80")).cast(pl.Int8).alias("is_extreme")
                ])
                .group_by("Date")
                .agg((pl.col("is_extreme").mean()).alias("ss_mkt_short_breadth_q80")))

    # 市場集計を結合
    s = s.join(mkt, on="Date", how="left").join(breadth, on="Date", how="left")

    # 4) T+1有効化 (公表時点はEOD後とみなし、翌営業日から使用)
    s = s.with_columns([
        calendar_fn(pl.col("Date")).alias("effective_date"),
        pl.lit(1).cast(pl.Int8).alias("is_ss_valid")
    ])

    # Select columns based on whether Z-scores are enabled
    base_columns = [
        "Date", "effective_date", "sec33",
        "ss_sec33_short_share", "ss_sec33_restrict_share", "ss_sec33_short_turnover",
        "ss_sec33_short_share_d1", "ss_sec33_restrict_share_d1", "ss_sec33_short_mom5", "ss_sec33_short_accel",
        "ss_mkt_short_share", "ss_mkt_restrict_share", "ss_mkt_short_breadth_q80",
        "is_ss_valid"
    ]

    if enable_z_scores:
        base_columns.extend(["ss_sec33_short_share_z60", "ss_sec33_short_turnover_z252"])

    return s.select(base_columns)


def attach_sector_short_to_quotes(
    quotes: pl.DataFrame,
    sector_feats: pl.DataFrame,
    sector_map: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    """
    Attach sector short selling features to individual quotes via as-of join.

    Args:
        quotes: Daily quotes (Code, Date, ...)
        sector_feats: Sector features with effective_date
        sector_map: Sector mapping (Code, valid_from, valid_to, sector33_code)

    Returns:
        Daily quotes with sector short selling features attached
    """
    if sector_feats.is_empty():
        return quotes

    # 1) セクターマッピングの適用
    if sector_map is not None:
        # listed_info からのas-ofマッピング
        q = (quotes.join(sector_map, on="Code", how="left")
                  .filter((pl.col("Date") >= pl.col("valid_from")) &
                          (pl.col("Date") <= pl.col("valid_to")))
                  .with_columns([pl.col("sector33_code").cast(pl.Utf8).alias("sec33")])
                  .drop(["valid_from", "valid_to", "sector33_code"]))
    else:
        # セクター情報が既にある場合
        if "sec33" not in quotes.columns and "sector33_code" in quotes.columns:
            q = quotes.with_columns([pl.col("sector33_code").cast(pl.Utf8).alias("sec33")])
        else:
            q = quotes

    if "sec33" not in q.columns:
        # セクター情報がない場合、nullで埋める
        null_features = [c for c in sector_feats.columns if c.startswith("ss_")]
        null_exprs = [pl.lit(None).alias(c) for c in null_features]
        null_exprs.append(pl.lit(0).cast(pl.Int8).alias("is_ss_valid"))
        return quotes.with_columns(null_exprs)

    # 2) セクター × effective_date での as-of 結合 (T+1 leak-safe)
    q = (q.sort(["sec33", "Date"])
          .join_asof(
             sector_feats.sort(["sec33", "effective_date"]),
             left_on="Date",
             right_on="effective_date",
             by="sec33",
             strategy="backward"
          ))

    # 3) 相対化特徴量の追加 (効きどころ)
    q = q.with_columns([
        # セクター vs 市場の相対強さ
        (pl.col("ss_sec33_short_share") - pl.col("ss_mkt_short_share")).alias("ss_rel_short_share"),
        (pl.col("ss_sec33_restrict_share") - pl.col("ss_mkt_restrict_share")).alias("ss_rel_restrict_share"),
    ])

    # 4) 条件付きシグナル特徴量 (個別×セクターの相対化)
    has_z = "ss_sec33_short_share_z60" in q.columns
    if "returns_1d" in q.columns and has_z:
        q = q.with_columns([
            # 下落×空売り過熱の継続シグナル
            ((pl.col("returns_1d") < 0) & (pl.col("ss_sec33_short_share_z60") > 1.0)).cast(pl.Int8).alias("ss_cond_pressure"),
        ])
    elif "returns_1d" in q.columns and not has_z:
        q = q.with_columns([pl.lit(0).cast(pl.Int8).alias("ss_cond_pressure")])

    if "returns_5d" in q.columns and has_z:
        q = q.with_columns([
            ((pl.col("returns_5d") > 0) & (pl.col("ss_sec33_short_share_z60") > 1.0)).cast(pl.Int8).alias("ss_squeeze_setup"),
        ])
    elif "returns_5d" in q.columns and not has_z:
        q = q.with_columns([pl.lit(0).cast(pl.Int8).alias("ss_squeeze_setup")])

    # 5) 有効性フラグの調整
    q = q.with_columns([
        pl.col("is_ss_valid").fill_null(0).cast(pl.Int8)
    ])

    return q


def add_sector_short_selling_block(
    quotes: pl.DataFrame,
    ss_df: Optional[pl.DataFrame],
    listed_info_df: Optional[pl.DataFrame] = None,
    *,
    enable_z_scores: bool = True,
    enable_relative_features: bool = True,
    calendar_next_bday: Optional[Callable[[pl.Expr], pl.Expr]] = None,
) -> pl.DataFrame:
    """
    Complete sector short selling features integration pipeline.

    Args:
        quotes: Base daily quotes DataFrame
        ss_df: Sector short selling data from J-Quants API
        listed_info_df: Listed info data for sector mapping
        enable_z_scores: Whether to compute Z-score features
        enable_relative_features: Whether to compute relative features

    Returns:
        DataFrame with sector short selling features attached
    """
    if ss_df is None or ss_df.is_empty():
        # Add null features for consistency
        null_cols = [
            "ss_sec33_short_share", "ss_sec33_restrict_share", "ss_sec33_short_turnover",
            "ss_sec33_short_share_d1", "ss_sec33_restrict_share_d1", "ss_sec33_short_mom5", "ss_sec33_short_accel",
            "ss_mkt_short_share", "ss_mkt_restrict_share", "ss_mkt_short_breadth_q80",
            "ss_rel_short_share", "ss_rel_restrict_share"
        ]

        if enable_z_scores:
            null_cols.extend(["ss_sec33_short_share_z60", "ss_sec33_short_turnover_z252"])

        null_exprs = [pl.lit(None).alias(c) for c in null_cols]
        null_exprs.extend([
            pl.lit(0).cast(pl.Int8).alias("ss_cond_pressure"),
            pl.lit(0).cast(pl.Int8).alias("ss_squeeze_setup"),
            pl.lit(0).cast(pl.Int8).alias("is_ss_valid")
        ])

        return quotes.with_columns(null_exprs)

    # Step 1: Build sector-level features (inject calendar-aware next BD when supplied)
    sector_feats = build_sector_short_features(
        ss_df,
        calendar_next_bday=calendar_next_bday,
        enable_z_scores=enable_z_scores,
    )

    # Step 2: Create sector mapping if available
    # Prefer existing sector assignment on quotes when present; otherwise fall back to listed_info
    sector_map = None
    if listed_info_df is not None and ("sec33" not in quotes.columns and "sector33_code" not in quotes.columns):
        sector_map = (
            listed_info_df
            .filter(pl.col("sector33_code").is_not_null())
            .select(["Code", "sector33_code", "Date"])  # Date is snapshot date
            .with_columns([
                pl.col("Date").alias("valid_from"),
                (pl.col("Date") + pl.duration(days=365)).alias("valid_to"),
            ])
            .unique(["Code"], keep="last")
        )

    # Step 3: Attach to quotes via as-of join
    result = attach_sector_short_to_quotes(quotes, sector_feats, sector_map)

    return result


def validate_sector_short_features(df: pl.DataFrame) -> dict:
    """
    Validate sector short selling features for quality and safety.

    Args:
        df: DataFrame with sector short selling features

    Returns:
        Dict with validation results and warnings
    """
    validation = {
        "warnings": [],
        "coverage": {},
        "extreme_values": {},
        "data_leaks": 0
    }

    if df.is_empty():
        validation["warnings"].append("Empty DataFrame")
        return validation

    # Coverage analysis
    if "is_ss_valid" in df.columns:
        coverage = df["is_ss_valid"].mean()
        validation["coverage"]["overall"] = coverage
        if coverage < 0.8:
            validation["warnings"].append(f"Low coverage: {coverage:.2%}")

    # Extreme value detection
    if "ss_sec33_restrict_share" in df.columns:
        restrict_p99 = df["ss_sec33_restrict_share"].quantile(0.999)
        validation["extreme_values"]["restrict_share_p99"] = restrict_p99
        if restrict_p99 > 0.9:
            validation["warnings"].append(f"Extreme restriction ratio: {restrict_p99:.3f}")

    # Data leak detection: flags rows where feature applied BEFORE effective date
    if all(c in df.columns for c in ["Date", "effective_date"]):
        leaks = df.filter(pl.col("Date") < pl.col("effective_date")).height
        validation["data_leaks"] = leaks
        if leaks > 0:
            validation["warnings"].append(f"Potential data leakage detected: {leaks} rows where Date < effective_date")

    return validation
