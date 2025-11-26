"""Short selling positions-derived daily features."""

from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import polars as pl


def compute_short_positions_features(
    con: duckdb.DuckDBPyConnection,
    *,
    start: str,
    end: str,
    recent_window: int = 5,
) -> pl.DataFrame:
    """
    Compute event-driven short selling positions features:
    - ssp_disclosed_short_ratio: sum of ShortPositionsToSharesOutstandingRatio per code as of last disclosure
    - ssp_short_disclosure_flag_recent: 1 if disclosure in last `recent_window` trading days
    Effective date is the first trading day after disclosed_date to avoid leak.
    """
    # trading calendar for join_asof to find next trading day
    cal = con.execute(
        """
        SELECT date AS cal_date
        FROM trading_calendar
        WHERE holiday_division IN ('1','2')
        ORDER BY date
        """
    ).fetch_arrow_table()
    cal_df = pl.from_arrow(cal)
    if cal_df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "code": pl.Utf8})

    # raw positions within a superset window (start - recent_window to end)
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    window_start = (start_dt - timedelta(days=recent_window + 5)).strftime("%Y-%m-%d")
    pos_arrow = con.execute(
        """
        SELECT
            disclosed_date,
            calculated_date,
            code,
            short_positions_to_shares_outstanding_ratio
        FROM short_selling_positions
        WHERE disclosed_date BETWEEN ? AND ?
        """,
        [window_start, end],
    ).fetch_arrow_table()
    pos = pl.from_arrow(pos_arrow)
    if pos.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "code": pl.Utf8})

    pos = pos.with_columns(
        [
            pl.col("disclosed_date").cast(pl.Date, strict=False),
            pl.col("code").cast(pl.Utf8, strict=False),
        ]
    )

    # effective_date = next trading day after disclosed_date
    pos = pos.with_columns((pl.col("disclosed_date") + pl.duration(days=1)).alias("_disclosed_plus1"))
    pos = (
        pos.join_asof(cal_df, left_on="_disclosed_plus1", right_on="cal_date", strategy="forward")
        .rename({"cal_date": "effective_date"})
        .drop("_disclosed_plus1")
    )
    pos = pos.filter(pl.col("effective_date").is_not_null())

    # aggregate per code/effective_date: sum ratio
    agg = (
        pos.group_by(["code", "effective_date"], maintain_order=True)
        .agg(pl.col("short_positions_to_shares_outstanding_ratio").sum().alias("ssp_disclosed_short_ratio"))
        .sort(["code", "effective_date"])
    )

    # build disclosure date list per code for recent-flag calculation
    disc = pos.select(["code", "effective_date", "disclosed_date"]).unique().sort(["code", "effective_date"])
    codes_with_disc = disc.select(pl.col("code").unique()).to_series()

    # daily grid to join_asof
    pf_arrow = con.execute(
        """
        SELECT date, code
        FROM price_flow_features
        WHERE date BETWEEN ? AND ?
        ORDER BY code, date
        """,
        [start, end],
    ).fetch_arrow_table()
    pf = pl.from_arrow(pf_arrow)
    if pf.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "code": pl.Utf8})
    # If disclosures are sparse, limit to relevant codes to speed up join
    if codes_with_disc.len() > 0 and codes_with_disc.len() < pf["code"].n_unique():
        pf = pf.filter(pl.col("code").is_in(codes_with_disc))

    pf = pf.sort(["code", "date"])
    agg = agg.sort(["code", "effective_date"])

    # join_asof to forward-fill latest disclosed ratio
    with pl.Config(set_sorted=True):
        daily = pf.join_asof(
            agg,
            left_on="date",
            right_on="effective_date",
            by="code",
            strategy="backward",
        )

    # mark disclosure days, then rolling_sum over code to flag recent disclosures
    disc_flags = (
        disc.select(["code", "effective_date"])
        .rename({"effective_date": "date"})
        .with_columns(pl.lit(1).cast(pl.Int8).alias("_disc_flag"))
    )
    daily = daily.join(disc_flags, on=["code", "date"], how="left")
    daily = daily.with_columns(pl.col("_disc_flag").fill_null(0))
    daily = daily.with_columns(
        pl.col("_disc_flag")
        .rolling_sum(window_size=recent_window, min_periods=1)
        .over("code")
        .fill_null(0)
        .clip(0, 1)
        .cast(pl.Int8)
        .alias("ssp_short_disclosure_flag_recent")
    )
    daily = daily.drop("_disc_flag")

    # fill missing columns if any
    if "ssp_disclosed_short_ratio" not in daily.columns:
        daily = daily.with_columns(pl.lit(None).cast(pl.Float64).alias("ssp_disclosed_short_ratio"))
    if "ssp_short_disclosure_flag_recent" not in daily.columns:
        daily = daily.with_columns(pl.lit(0).cast(pl.Int8).alias("ssp_short_disclosure_flag_recent"))

    return daily.select(["date", "code", "ssp_disclosed_short_ratio", "ssp_short_disclosure_flag_recent"])
