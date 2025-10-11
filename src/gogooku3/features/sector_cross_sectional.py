from __future__ import annotations

"""
Sector-relative cross-sectional features (Phase 2).

Computes per-Date × Sector statistics and attaches:
- Deviation from sector mean for returns (1d/5d)
- Percentile rank within sector for returns_1d
- Z-scores within sector for Volume and realized_vol_20 (if present)

Requirements: a sector column among ['sector33_code','sec33','Sector33Code','sector'].
No forward-fill; uses only same-day cross-sectional information.
"""

import polars as pl


def _find_sector_col(df: pl.DataFrame) -> str | None:
    candidates = ["sector33_code", "sec33", "Sector33Code", "sector", "Section"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_sector_cross_sectional_features(
    df: pl.DataFrame,
    *,
    sector_col: str | None = None,
    include_cols: list[str] | None = None,
) -> pl.DataFrame:
    if df.is_empty():
        return df
    sec = sector_col or _find_sector_col(df)
    if not sec or sec not in df.columns:
        # Sector info not present; no-op
        return df

    out = df
    keys = ["Date", sec]

    # Helper: within sector stats
    def _sector_mean(x: str, alias: str) -> pl.Expr:
        return pl.col(x).mean().over(keys).alias(alias)

    def _sector_std(x: str, alias: str) -> pl.Expr:
        return pl.col(x).std().over(keys).alias(alias)

    # Returns deviation, ranks, and z-scores within sector
    if "returns_1d" in out.columns:
        out = out.with_columns([
            _sector_mean("returns_1d", "_sec_mean_ret1d"),
            _sector_std("returns_1d", "_sec_std_ret1d"),
        ])
        out = out.with_columns([
            (pl.col("returns_1d") - pl.col("_sec_mean_ret1d")).alias("ret_1d_vs_sec"),
            ((pl.col("returns_1d") - pl.col("_sec_mean_ret1d")) / (pl.col("_sec_std_ret1d") + 1e-12)).alias("ret_1d_in_sec_z"),
        ])
        # Percentile rank within sector for returns_1d
        cnt = pl.count().over(keys)
        rk = pl.col("returns_1d").rank(method="average").over(keys)
        out = out.with_columns([
            pl.when(cnt > 1).then((rk - 1.0) / (cnt - 1.0)).otherwise(0.5).alias("ret_1d_rank_in_sec")
        ])
    if "returns_5d" in out.columns:
        out = out.with_columns([
            _sector_mean("returns_5d", "_sec_mean_ret5d"),
            _sector_std("returns_5d", "_sec_std_ret5d"),
        ])
        out = out.with_columns([
            (pl.col("returns_5d") - pl.col("_sec_mean_ret5d")).alias("ret_5d_vs_sec"),
            ((pl.col("returns_5d") - pl.col("_sec_mean_ret5d")) / (pl.col("_sec_std_ret5d") + 1e-12)).alias("ret_5d_in_sec_z"),
        ])
    if "returns_10d" in out.columns:
        out = out.with_columns([
            _sector_mean("returns_10d", "_sec_mean_ret10d"),
            _sector_std("returns_10d", "_sec_std_ret10d"),
        ])
        out = out.with_columns([
            (pl.col("returns_10d") - pl.col("_sec_mean_ret10d")).alias("ret_10d_vs_sec"),
            ((pl.col("returns_10d") - pl.col("_sec_mean_ret10d")) / (pl.col("_sec_std_ret10d") + 1e-12)).alias("ret_10d_in_sec_z"),
        ])

    # Z-scores within sector (Volume, realized_vol_20 if present)
    if "Volume" in out.columns:
        out = out.with_columns([
            _sector_mean("Volume", "_sec_mean_vol"),
            _sector_std("Volume", "_sec_std_vol"),
            ((pl.col("Volume") - pl.col("_sec_mean_vol")) / (pl.col("_sec_std_vol") + 1e-12)).alias("volume_in_sec_z"),
            # Sector percentile rank for volume
            pl.col("Volume").rank(method="average").over(keys).alias("_vol_rank"),
        ])
        cnt_v = pl.count().over(keys)
        rk_v = pl.col("_vol_rank")
        out = out.with_columns([
            pl.when(cnt_v > 1).then((rk_v - 1.0) / (cnt_v - 1.0)).otherwise(0.5).alias("volume_rank_in_sec")
        ])
    if "realized_vol_20" in out.columns:
        out = out.with_columns([
            _sector_mean("realized_vol_20", "_sec_mean_rv20"),
            _sector_std("realized_vol_20", "_sec_std_rv20"),
            ((pl.col("realized_vol_20") - pl.col("_sec_mean_rv20")) / (pl.col("_sec_std_rv20") + 1e-12)).alias("rv20_in_sec_z"),
        ])
    # RSI in sector z
    if "rsi_14" in out.columns:
        out = out.with_columns([
            _sector_mean("rsi_14", "_sec_mean_rsi14"),
            _sector_std("rsi_14", "_sec_std_rsi14"),
            ((pl.col("rsi_14") - pl.col("_sec_mean_rsi14")) / (pl.col("_sec_std_rsi14") + 1e-12)).alias("rsi14_in_sec_z"),
        ])
    # MACD slope in sector z（存在時）
    if "macd_hist_slope" in out.columns:
        out = out.with_columns([
            _sector_mean("macd_hist_slope", "_sec_mean_mhs"),
            _sector_std("macd_hist_slope", "_sec_std_mhs"),
            ((pl.col("macd_hist_slope") - pl.col("_sec_mean_mhs")) / (pl.col("_sec_std_mhs") + 1e-12)).alias("macd_slope_in_sec_z"),
        ])
    # Momentum×Volume, RSI×Vol interaction in sector z（存在時）
    if "vol_confirmed_mom" in out.columns:
        out = out.with_columns([
            _sector_mean("vol_confirmed_mom", "_sec_mean_vcm"),
            _sector_std("vol_confirmed_mom", "_sec_std_vcm"),
            ((pl.col("vol_confirmed_mom") - pl.col("_sec_mean_vcm")) / (pl.col("_sec_std_vcm") + 1e-12)).alias("vcm_in_sec_z"),
        ])
    if "rsi_vol_interact" in out.columns:
        out = out.with_columns([
            _sector_mean("rsi_vol_interact", "_sec_mean_rvint"),
            _sector_std("rsi_vol_interact", "_sec_std_rvint"),
            ((pl.col("rsi_vol_interact") - pl.col("_sec_mean_rvint")) / (pl.col("_sec_std_rvint") + 1e-12)).alias("rsi_vol_in_sec_z"),
        ])

    # Cleanup temp columns
    out = out.drop([c for c in out.columns if c.startswith("_sec_mean_") or c.startswith("_sec_std_") or c == "_vol_rank"]) 

    # Generic sector-relative for additional numeric columns (if present)
    # Compute diff and z for a shortlist of candidates not already handled
    default_candidates = [
        "rsi_14",
        "volume_accel",
        "returns_10d",
        "vol_confirmed_mom",
        "rsi_vol_interact",
        "realized_vol_20",
    ]
    cand = default_candidates
    if include_cols:
        # Allow external override/extension
        cand = list({*default_candidates, *include_cols})
    for col in cand:
        if col not in out.columns:
            continue
        diff_name = f"{col}_vs_sec"
        z_name = f"{col}_in_sec_z"
        # Skip if already computed above
        if diff_name in out.columns or z_name in out.columns:
            continue
        out = out.with_columns([
            _sector_mean(col, f"_sec_mean_{col}"),
            _sector_std(col, f"_sec_std_{col}"),
        ])
        out = out.with_columns([
            (pl.col(col) - pl.col(f"_sec_mean_{col}")).alias(diff_name),
            ((pl.col(col) - pl.col(f"_sec_mean_{col}")) / (pl.col(f"_sec_std_{col}") + 1e-12)).alias(z_name),
        ])

    # Validity flag
    valid_conds = []
    if "ret_1d_vs_sec" in out.columns:
        valid_conds.append(pl.col("ret_1d_vs_sec").is_not_null())
    if "ret_5d_vs_sec" in out.columns:
        valid_conds.append(pl.col("ret_5d_vs_sec").is_not_null())
    if valid_conds:
        cond = valid_conds[0]
        for c in valid_conds[1:]:
            cond = cond | c
        out = out.with_columns([cond.cast(pl.Int8).alias("is_sec_cs_valid")])
    else:
        out = out.with_columns([pl.lit(0).cast(pl.Int8).alias("is_sec_cs_valid")])

    return out
