"""Section-level flow features computed from markets/trades_spec."""

from __future__ import annotations

import duckdb
import polars as pl
from data_v2.etl.schemas import SECTION_FLOW_FEATURES_COLUMNS


def _compute_streaks(values: list[float]) -> tuple[list[int], list[int]]:
    """Return (buy_streak, sell_streak) where streaks reset on sign flip/zero."""
    buy: list[int] = []
    sell: list[int] = []
    b = s = 0
    for v in values:
        if v > 0:
            b += 1
        else:
            b = 0
        if v < 0:
            s += 1
        else:
            s = 0
        buy.append(b)
        sell.append(s)
    return buy, sell


def _add_group_streaks(df: pl.DataFrame) -> pl.DataFrame:
    """Compute streaks per section group (called via groupby.apply)."""
    buy_f, sell_f = _compute_streaks(df["foreigners_net"].to_list())
    buy_i, sell_i = _compute_streaks(df["individuals_net"].to_list())
    return df.with_columns(
        [
            pl.Series("foreigners_buy_streak", buy_f),
            pl.Series("foreigners_sell_streak", sell_f),
            pl.Series("individuals_buy_streak", buy_i),
            pl.Series("individuals_sell_streak", sell_i),
        ]
    )


def compute_trades_spec_features(
    con: duckdb.DuckDBPyConnection,
    *,
    start: str,
    end: str,
) -> pl.DataFrame:
    """
    Compute section-level flow features from trades_spec (weekly investor flows).

    - Uses PublishedDate as effective date (information availability).
    - Maps section flows to be broadcast later by section-to-market mapping.
    """
    arrow = con.execute(
        """
        SELECT *
        FROM trades_spec
        WHERE published_date BETWEEN ? AND ?
        """,
        [start, end],
    ).fetch_arrow_table()
    df = pl.from_arrow(arrow)
    if df.is_empty():
        return pl.DataFrame()

    df = df.sort(["section", "published_date"])
    EPS = 1e-9

    cal = con.execute(
        """
        SELECT date AS cal_date
        FROM trading_calendar
        WHERE holiday_division IN ('1', '2')
        ORDER BY date
        """
    ).fetch_arrow_table()
    cal_df = pl.from_arrow(cal)
    if cal_df.is_empty():
        raise RuntimeError("trading_calendar is empty; fetch calendar before computing trades_spec features")

    df = df.with_columns((pl.col("published_date") + pl.duration(days=1)).alias("_pub_plus1"))
    df = df.sort("_pub_plus1")
    df = (
        df.join_asof(
            cal_df,
            left_on="_pub_plus1",
            right_on="cal_date",
            strategy="forward",
        )
        .rename({"cal_date": "effective_date"})
        .drop("_pub_plus1")
    )

    df = df.with_columns(
        [
            (pl.col("foreigners_purchases") - pl.col("foreigners_sales")).alias("foreigners_net"),
            (pl.col("individuals_purchases") - pl.col("individuals_sales")).alias("individuals_net"),
            (pl.col("trust_banks_purchases") - pl.col("trust_banks_sales")).alias("trust_banks_net"),
            (pl.col("investment_trusts_purchases") - pl.col("investment_trusts_sales")).alias("investment_trusts_net"),
            pl.col("total_total").alias("total_turnover"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("foreigners_net") / (pl.col("total_turnover") + EPS)).alias("foreigners_net_ratio"),
            (pl.col("individuals_net") / (pl.col("total_turnover") + EPS)).alias("individuals_net_ratio"),
            (pl.col("trust_banks_net") / (pl.col("total_turnover") + EPS)).alias("trust_banks_net_ratio"),
        ]
    )
    df = df.with_columns(
        [
            (pl.col("foreigners_net_ratio") + pl.col("trust_banks_net_ratio")).alias("smart_net_ratio"),
            pl.col("individuals_net_ratio").alias("retail_net_ratio"),
        ]
    )
    df = df.with_columns((pl.col("smart_net_ratio") - pl.col("retail_net_ratio")).alias("smart_vs_retail"))

    df = df.with_columns(
        [
            pl.col("foreigners_net").rolling_sum(window_size=4).over("section").alias("foreigners_net_cum_4w"),
            pl.col("foreigners_net").rolling_sum(window_size=13).over("section").alias("foreigners_net_cum_13w"),
            pl.col("smart_vs_retail").rolling_sum(window_size=4).over("section").alias("smart_vs_retail_cum_4w"),
            pl.col("smart_vs_retail").rolling_sum(window_size=13).over("section").alias("smart_vs_retail_cum_13w"),
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col("foreigners_net_ratio").rolling_std(window_size=26).over("section") > 0)
            .then(
                (
                    pl.col("foreigners_net_ratio")
                    - pl.col("foreigners_net_ratio").rolling_mean(window_size=26).over("section")
                )
                / pl.col("foreigners_net_ratio").rolling_std(window_size=26).over("section")
            )
            .otherwise(None)
            .alias("foreigners_net_ratio_z_26w"),
            pl.when(pl.col("individuals_net_ratio").rolling_std(window_size=26).over("section") > 0)
            .then(
                (
                    pl.col("individuals_net_ratio")
                    - pl.col("individuals_net_ratio").rolling_mean(window_size=26).over("section")
                )
                / pl.col("individuals_net_ratio").rolling_std(window_size=26).over("section")
            )
            .otherwise(None)
            .alias("individuals_net_ratio_z_26w"),
            pl.when(pl.col("smart_vs_retail").rolling_std(window_size=26).over("section") > 0)
            .then(
                (pl.col("smart_vs_retail") - pl.col("smart_vs_retail").rolling_mean(window_size=26).over("section"))
                / pl.col("smart_vs_retail").rolling_std(window_size=26).over("section")
            )
            .otherwise(None)
            .alias("smart_vs_retail_z_26w"),
        ]
    )

    df = df.group_by("section", maintain_order=True).map_groups(_add_group_streaks).sort(["section", "published_date"])

    col_order = [name for name, _ in SECTION_FLOW_FEATURES_COLUMNS]  # type: ignore[name-defined]
    return df.select(col_order)
