from __future__ import annotations

"""
Weekly margin interest → daily panel integration (leak-safe, Polars).

This module implements the weekly-to-daily pipeline with:
 - effective_start computation (T+1 rule; optional conservative lag)
 - weekly-core features (stock, diffs, z-scores)
 - ADV20-based scaling
 - as-of backward join to attach to a daily (Code, Date) grid

Minimal public API:
 - build_weekly_effective(df_w, lag_bdays_weekly, next_business_day)
 - add_weekly_core_features(w)
 - attach_adv20_and_scale(w_feat, adv20)
 - asof_attach_to_daily(quotes, wk)
 - add_margin_weekly_block(quotes, weekly_df, adv20_df, *, lag_bdays_weekly=3)

Notes:
 - ADV20_shares must be derived from adjusted volume; if only Volume is
   available, a reasonable approximation can be used upstream.
 - No time leakage: values become valid starting from effective_start only.
"""

import datetime as _dt
from collections.abc import Callable

import polars as pl

EPS = 1e-12


def _next_business_day_jp(date_col: pl.Expr) -> pl.Expr:
    """Next business day for JP market (Mon-Fri); skips weekends.

    Weekday mapping: Monday=0 ... Sunday=6
    - Mon-Thu -> +1 day
    - Fri -> +3 days (next Monday)
    - Sat -> +2 days (next Monday)
    - Sun -> +1 day (next Monday)
    """
    wd = date_col.dt.weekday()
    return (
        pl.when(wd <= 3)
        .then(date_col + pl.duration(days=1))
        .when(wd == 4)
        .then(date_col + pl.duration(days=3))
        .when(wd == 5)
        .then(date_col + pl.duration(days=2))
        .otherwise(date_col + pl.duration(days=1))
    )


def _add_business_days_jp(d: _dt.date, n: int) -> _dt.date:
    """Add n Japanese business days to a date (Mon-Fri).

    Handles weekend rollovers; ignores holidays.
    """
    cur = d
    count = 0
    while count < n:
        cur = cur + _dt.timedelta(days=1)
        if cur.weekday() < 5:  # Mon-Fri
            count += 1
    return cur


def build_weekly_effective(
    df_w: pl.DataFrame,
    lag_bdays_weekly: int,
    next_business_day: Callable[[pl.Expr], pl.Expr] | None = None,
) -> pl.DataFrame:
    """
    Compute effective_start for weekly margin series.

    Rules:
      - If PublishedDate exists: effective_start = next_business_day(PublishedDate)
      - Else: next_business_day(Date) + (lag_bdays_weekly - 1) business days

    Returns a DataFrame sorted by (Code, effective_start).
    """
    # Use Python-level add_business_days to avoid version-specific Expr pitfalls
    def _eff_start(published, date):
        if published is not None:
            # PublishedDate present → T+1 business day
            return _add_business_days_jp(published, 1)
        # PublishedDate missing → next BD of Date plus (lag-1) BDs → total lag
        return _add_business_days_jp(date, max(lag_bdays_weekly, 1))

    w = df_w.with_columns(
        pl.struct(["PublishedDate", "Date"]).map_elements(
            lambda s: _eff_start(s["PublishedDate"], s["Date"]), return_dtype=pl.Date
        ).alias("effective_start")
    )

    w = w.sort(["Code", "effective_start"])
    return w


def add_weekly_core_features(w: pl.DataFrame) -> pl.DataFrame:
    """Derive weekly core features (stock, diffs, z-scores, composition)."""
    g = (
        w.with_columns(
            [
                pl.col("LongMarginTradeVolume").cast(pl.Float64).alias("margin_long_tot"),
                pl.col("ShortMarginTradeVolume").cast(pl.Float64).alias("margin_short_tot"),
            ]
        )
        .with_columns(
            [
                (pl.col("margin_long_tot") - pl.col("margin_short_tot")).alias(
                    "margin_net"
                ),
                (pl.col("margin_long_tot") + pl.col("margin_short_tot")).alias(
                    "margin_total_gross"
                ),
                (pl.col("margin_long_tot") / (pl.col("margin_short_tot") + EPS)).alias(
                    "margin_credit_ratio"
                ),
                (
                    (pl.col("margin_long_tot") - pl.col("margin_short_tot"))
                    / (pl.col("margin_long_tot") + pl.col("margin_short_tot") + EPS)
                ).alias("margin_imbalance"),
                (
                    pl.col("LongStandardizedMarginTradeVolume")
                    / (pl.col("margin_long_tot") + EPS)
                ).alias("margin_std_share_long"),
                (
                    pl.col("ShortStandardizedMarginTradeVolume")
                    / (pl.col("margin_short_tot") + EPS)
                ).alias("margin_std_share_short"),
                (
                    pl.col("LongNegotiableMarginTradeVolume")
                    / (pl.col("margin_long_tot") + EPS)
                ).alias("margin_neg_share_long"),
                (
                    pl.col("ShortNegotiableMarginTradeVolume")
                    / (pl.col("margin_short_tot") + EPS)
                ).alias("margin_neg_share_short"),
            ]
        )
        .with_columns(
            [
                pl.col("margin_long_tot").diff().over("Code").alias("margin_d_long_wow"),
                pl.col("margin_short_tot").diff().over("Code").alias("margin_d_short_wow"),
                pl.col("margin_net").diff().over("Code").alias("margin_d_net_wow"),
                pl.col("margin_credit_ratio").diff().over("Code").alias(
                    "margin_d_ratio_wow"
                ),
                (
                    pl.col("margin_total_gross")
                    - pl.col("margin_total_gross").rolling_mean(4).over("Code")
                ).alias("margin_gross_mom4"),
            ]
        )
    )

    for c, out in [
        ("margin_long_tot", "long_z52"),
        ("margin_short_tot", "short_z52"),
        ("margin_total_gross", "margin_gross_z52"),
        ("margin_credit_ratio", "ratio_z52"),
    ]:
        g = g.with_columns(
            [
                (
                    (pl.col(c) - pl.col(c).rolling_mean(52).over("Code"))
                    / (pl.col(c).rolling_std(52).over("Code") + EPS)
                ).alias(out)
            ]
        )
    return g


def attach_adv20_and_scale(w_feat: pl.DataFrame, adv20: pl.DataFrame) -> pl.DataFrame:
    """Join ADV20_shares (as-of at effective_start) and compute scaled features."""
    adv = adv20.select(["Code", "Date", "ADV20_shares"]).sort(["Code", "Date"])
    wk = w_feat.join_asof(
        adv, left_on="effective_start", right_on="Date", by="Code", strategy="backward"
    )
    # Drop right-side helper date to avoid downstream name collisions
    if "Date_right" in wk.columns:
        wk = wk.drop(["Date_right"])
    wk = wk.with_columns(
        [
            (pl.col("margin_long_tot") / (pl.col("ADV20_shares") + EPS)).alias(
                "margin_long_to_adv20"
            ),
            (pl.col("margin_short_tot") / (pl.col("ADV20_shares") + EPS)).alias(
                "margin_short_to_adv20"
            ),
            (
                pl.col("margin_d_long_wow")
                / (pl.col("ADV20_shares") * 5 + EPS)
            ).alias("margin_d_long_to_adv20"),
            (
                pl.col("margin_d_short_wow")
                / (pl.col("ADV20_shares") * 5 + EPS)
            ).alias("margin_d_short_to_adv20"),
        ]
    )
    return wk


def asof_attach_to_daily(quotes: pl.DataFrame, wk: pl.DataFrame) -> pl.DataFrame:
    """Attach weekly features to daily grid via backward as-of join."""
    q = quotes.sort(["Code", "Date"])
    w = wk.sort(["Code", "effective_start"])
    # Note: older Polars versions do not support `allow_exact_matches`.
    out = q.join_asof(
        w,
        left_on="Date",
        right_on="effective_start",
        by="Code",
        strategy="backward",
    ).with_columns(
        [
            (pl.col("Date") == pl.col("effective_start")).cast(pl.Int8).alias(
                "margin_impulse"
            ),
            (pl.col("Date") - pl.col("effective_start")).dt.days().alias(
                "margin_days_since"
            ),
            pl.when(pl.col("effective_start").is_null())
            .then(0)
            .otherwise(1)
            .alias("is_margin_valid"),
            pl.col("IssueType").cast(pl.Int8).alias("margin_issue_type"),
            (pl.col("IssueType").cast(pl.Int8) == 2).cast(pl.Int8).alias("is_borrowable"),
        ]
    )
    return out


def _compute_adv20_from_quotes(quotes: pl.DataFrame, adv_window_days: int = 20) -> pl.DataFrame:
    """Compute ADV20_shares from adjusted volume-like column.

    Prefers `AdjustmentVolume` if present; falls back to `Volume`.
    Returns a DataFrame with (Code, Date, ADV20_shares).
    """
    vol_col = "AdjustmentVolume" if "AdjustmentVolume" in quotes.columns else "Volume"
    if vol_col not in quotes.columns:
        # Create an empty ADV table to avoid crashes downstream
        return pl.DataFrame(schema={"Code": pl.Utf8, "Date": pl.Date, "ADV20_shares": pl.Float64})
    adv = (
        quotes.select(["Code", "Date", vol_col])
        .sort(["Code", "Date"])
        .with_columns(
            pl.col(vol_col)
            .rolling_mean(window_size=adv_window_days, min_periods=adv_window_days)
            .over("Code")
            .alias("ADV20_shares")
        )
        .select(["Code", "Date", "ADV20_shares"])
    )
    return adv


def add_margin_weekly_block(
    quotes: pl.DataFrame,
    weekly_df: pl.DataFrame,
    adv20_df: pl.DataFrame | None = None,
    *,
    lag_bdays_weekly: int = 3,
    next_business_day: Callable[[pl.Expr], pl.Expr] | None = None,
    adv_window_days: int = 20,
) -> pl.DataFrame:
    """High-level glue: weekly → features → scale → as-of attach → daily.

    - quotes: daily grid with (Code, Date, Volume/AdjustmentVolume)
    - weekly_df: normalized weekly margin table
    - adv20_df: optional precomputed ADV20_shares; if None it will be derived from quotes
    """
    if weekly_df.is_empty():
        return quotes

    w_eff = build_weekly_effective(
        weekly_df, lag_bdays_weekly=lag_bdays_weekly, next_business_day=next_business_day
    )
    w_feat = add_weekly_core_features(w_eff)
    adv = adv20_df if adv20_df is not None else _compute_adv20_from_quotes(quotes, adv_window_days)
    wk = attach_adv20_and_scale(w_feat, adv)
    out = asof_attach_to_daily(quotes, wk)
    return out
