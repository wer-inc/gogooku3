from __future__ import annotations

"""
Futures (index futures) feature engineering and safe attachment to equity panel.

Implements leak-safe ON (T+0) and EOD (T+1) features with optional basis/carry
when corresponding spot index series are present.

Notes:
- Designed for J-Quants /derivatives/futures daily dataset schema.
- Uses center contract only (CentralContractMonthFlag == "1").
- EOD features become effective on the next business day via effective_date.
"""


import polars as pl

EPS = 1e-12


def _num(col: str) -> pl.Expr:
    """Normalize numeric-like Utf8 columns: empty string -> NULL -> Float64."""
    return (
        pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).cast(pl.Float64)
    )


def prep_futures(df: pl.DataFrame, categories: list[str]) -> pl.DataFrame:
    """Prepare raw futures DataFrame for feature computation.

    - Filter to categories
    - Parse dates and normalize identifiers
    - Cast numeric fields
    - Resolve duplicates preferring Emergency=002 over 001
    - Restrict to center contract
    """
    if df.is_empty():
        return df

    f = (
        df.filter(pl.col("DerivativesProductCategory").is_in(categories))
          .with_columns(
            [
              pl.col("Date").str.strptime(pl.Date, strict=False).alias("Date"),
              pl.col("DerivativesProductCategory").alias("cat"),
              pl.col("CentralContractMonthFlag").cast(pl.Utf8).alias("ccf"),
              pl.col("EmergencyMarginTriggerDivision").cast(pl.Utf8).alias("emg"),
              pl.col("SpecialQuotationDay").str.strptime(pl.Date, strict=False).alias("SQ"),
              (pl.col("ContractMonth").cast(pl.Utf8) if "ContractMonth" in df.columns else pl.lit(None).cast(pl.Utf8)).alias("CM"),
            ]
        )
    )

    for c in [
        "NightSessionOpen",
        "NightSessionHigh",
        "NightSessionLow",
        "NightSessionClose",
        "DaySessionOpen",
        "DaySessionHigh",
        "DaySessionLow",
        "DaySessionClose",
        "WholeDayOpen",
        "WholeDayHigh",
        "WholeDayLow",
        "WholeDayClose",
        "SettlementPrice",
        "Volume",
        "OpenInterest",
    ]:
        if c in f.columns:
            f = f.with_columns([_num(c).alias(c)])

    # Resolve same-day duplicates by preferring emg==002 over 001
    # Sort so that 002 comes last and tail(1) keeps it
    f = (
        f.sort(["Date", "cat", "ccf", "emg"]).group_by(["Date", "cat", "ccf"]).tail(1)
    )
    # Keep center contract only
    f = f.filter(pl.col("ccf") == "1")
    return f.sort(["cat", "Date"])


def build_on_features(f: pl.DataFrame, on_z_window: int = 60) -> pl.DataFrame:
    """Compute ON features available same day (T+0)."""
    if f.is_empty():
        return f.select([])

    base = (
        f.with_columns(
            [
                (
                    (pl.col("NightSessionClose") - pl.col("DaySessionClose").shift(1).over("cat"))
                    / (pl.col("DaySessionClose").shift(1).over("cat") + EPS)
                ).alias("fut_on_ret"),
                (
                    pl.when(
                        pl.col("NightSessionHigh").is_not_null()
                        & pl.col("NightSessionLow").is_not_null()
                    )
                    .then((pl.col("NightSessionHigh") / pl.col("NightSessionLow")).log())
                    .otherwise(None)
                ).alias("fut_on_range"),
            ]
        )
        .with_columns(
            [
                (
                    (pl.col("fut_on_ret") - pl.col("fut_on_ret").rolling_mean(on_z_window).over("cat"))
                    / (pl.col("fut_on_ret").rolling_std(on_z_window).over("cat") + EPS)
                ).alias("fut_on_z"),
                (
                    pl.col("NightSessionClose").is_not_null()
                    & pl.col("DaySessionClose").shift(1).over("cat").is_not_null()
                )
                .cast(pl.Int8)
                .alias("is_fut_on_valid"),
                (pl.col("emg") == "001").cast(pl.Int8).alias("fut_emergency_flag"),
            ]
        )
        .select(
            [
                "Date",
                "cat",
                "fut_on_ret",
                "fut_on_range",
                "fut_on_z",
                "is_fut_on_valid",
                "fut_emergency_flag",
            ]
        )
    )
    return base


def build_eod_features(
    f: pl.DataFrame,
    *,
    spot_map: dict[str, pl.DataFrame],
    next_bday_expr: pl.Expr,
    z_window: int = 252,
    make_continuous_series: bool = False,
) -> pl.DataFrame:
    """Compute EOD features that become effective on next business day.

    spot_map: mapping from category to DataFrame with columns (Date, S)
    next_bday_expr: polars expression evaluating next business day for each row
    """
    if f.is_empty():
        return f.select([])

    g = f
    # Attach spot indices per category if provided
    for cat, s in spot_map.items():
        s_df = s
        if "Close" in s_df.columns and "S" not in s_df.columns:
            s_df = s_df.rename({"Close": "S"})
        if "Date" in s_df.columns and s_df["Date"].dtype == pl.Utf8:
            s_df = s_df.with_columns(pl.col("Date").str.strptime(pl.Date, strict=False))
        g = g.join(
            s_df.select(["Date", "S"]).rename({"S": f"S__{cat}"}),
            on="Date",
            how="left",
        )

    g = g.with_columns(
        [
            (
                (pl.col("DaySessionClose") - pl.col("NightSessionClose"))
                / (pl.col("NightSessionClose") + EPS)
            ).alias("fut_day_ret"),
            (
                (pl.col("WholeDayClose") - pl.col("WholeDayClose").shift(1).over("cat"))
                / (pl.col("WholeDayClose").shift(1).over("cat") + EPS)
            ).alias("fut_whole_ret"),
            (pl.col("SQ") - pl.col("Date")).dt.total_days().alias("ttm_days"),
        ]
    )

    # Compute basis/carry for categories with spot available
    cats: list[str] = g.select("cat").unique().to_series().to_list()  # type: ignore[assignment]
    parts: list[pl.DataFrame] = []
    for cat in cats:
        gi = g.filter(pl.col("cat") == cat)
        if f"S__{cat}" in gi.columns:
            gi = gi.with_columns(
                ((pl.col("DaySessionClose") - pl.col(f"S__{cat}")) / (pl.col(f"S__{cat}") + EPS)).alias(
                    "basis_close"
                )
            )
            gi = gi.with_columns(
                [
                    (
                        (pl.col("basis_close") - pl.col("basis_close").rolling_mean(z_window).over("cat"))
                        / (pl.col("basis_close").rolling_std(z_window).over("cat") + EPS)
                    ).alias("basis_close_z252"),
                    (
                        pl.col("basis_close")
                        / pl.max_horizontal(pl.col("ttm_days"), pl.lit(1))
                    ).alias("carry_per_day"),
                ]
            )
        else:
            gi = gi.with_columns(
                [
                    pl.lit(None).alias("basis_close"),
                    pl.lit(None).alias("basis_close_z252"),
                    pl.lit(None).alias("carry_per_day"),
                ]
            )
        # OI/Vol and z-scores
        gi = gi.with_columns(
            [
                pl.col("OpenInterest").alias("fut_oi"),
                (pl.col("OpenInterest") - pl.col("OpenInterest").shift(1).over("cat")).alias(
                    "fut_oi_delta"
                ),
                pl.col("Volume").alias("fut_vol"),
                (
                    (pl.col("OpenInterest") - pl.col("OpenInterest").rolling_mean(z_window).over("cat"))
                    / (pl.col("OpenInterest").rolling_std(z_window).over("cat") + EPS)
                ).alias("fut_oi_z252"),
                (
                    (pl.col("Volume") - pl.col("Volume").rolling_mean(z_window).over("cat"))
                    / (pl.col("Volume").rolling_std(z_window).over("cat") + EPS)
                ).alias("fut_vol_z252"),
            ]
        )
        # Price x OI quadrants
        gi = gi.with_columns(
            [
                ((pl.col("fut_day_ret") > 0) & (pl.col("fut_oi_delta") > 0))
                .cast(pl.Int8)
                .alias("price_up_oi_up"),
                ((pl.col("fut_day_ret") > 0) & (pl.col("fut_oi_delta") < 0))
                .cast(pl.Int8)
                .alias("price_up_oi_dn"),
                ((pl.col("fut_day_ret") < 0) & (pl.col("fut_oi_delta") > 0))
                .cast(pl.Int8)
                .alias("price_dn_oi_up"),
                ((pl.col("fut_day_ret") < 0) & (pl.col("fut_oi_delta") < 0))
                .cast(pl.Int8)
                .alias("price_dn_oi_dn"),
            ]
        )
        parts.append(gi)

    eod = pl.concat(parts).sort(["cat", "Date"]) if parts else g.head(0)
    # Optional ratio-linked continuous series
    if make_continuous_series and "CM" in g.columns and "WholeDayClose" in g.columns:
        # Compute link factors within each category
        def _cont_link(df_cat: pl.DataFrame) -> pl.DataFrame:
            dfc = df_cat.with_columns([
                (pl.col("CM") != pl.col("CM").shift(1)).fill_null(False).alias("is_roll"),
                pl.col("WholeDayClose").shift(1).alias("wdc_prev"),
            ])
            dfc = dfc.with_columns([
                pl.when(pl.col("is_roll")).then((pl.col("wdc_prev") / (pl.col("WholeDayClose") + EPS)).fill_null(1.0)).otherwise(1.0).alias("k")
            ])
            dfc = dfc.with_columns([pl.col("k").cumprod().alias("link_factor")])
            dfc = dfc.with_columns([
                (pl.col("WholeDayClose") * pl.col("link_factor")).alias("WholeDayClose_cont"),
            ])
            dfc = dfc.with_columns([
                ((pl.col("WholeDayClose_cont") - pl.col("WholeDayClose_cont").shift(1)) / (pl.col("WholeDayClose_cont").shift(1) + EPS)).alias("fut_whole_ret_cont")
            ])
            return dfc.select(["Date", "cat", "fut_whole_ret_cont"])  # avoid redundant joins

        cont_parts = []
        for cat in g.select("cat").unique().to_series().to_list():
            cont_parts.append(_cont_link(g.filter(pl.col("cat") == cat)))
        cont_df = pl.concat(cont_parts) if cont_parts else g.head(0)
        eod = eod.join(cont_df, on=["Date", "cat"], how="left")
    # Effective date = next business day
    eod = eod.with_columns([next_bday_expr.alias("effective_date")])
    return eod.select(
        [
            "effective_date",
            "Date",
            "cat",
            "fut_day_ret",
            "fut_whole_ret",
            "fut_whole_ret_cont",
            "basis_close",
            "basis_close_z252",
            "carry_per_day",
            "fut_oi",
            "fut_oi_delta",
            "fut_vol",
            "fut_oi_z252",
            "fut_vol_z252",
            "ttm_days",
            "price_up_oi_up",
            "price_up_oi_dn",
            "price_dn_oi_up",
            "price_dn_oi_dn",
        ]
    )


def attach_to_equity_panel(
    quotes: pl.DataFrame,
    on_df: pl.DataFrame,
    eod_df: pl.DataFrame,
    cats: list[str],
) -> pl.DataFrame:
    """Attach ON (Date join, T+0) and EOD (effective_date join, T+1) features.

    Joins are across Date only (global market-level features applied to all stocks).
    """
    q = quotes
    # ON features (T+0)
    for cat in cats:
        oni = (
            on_df.filter(pl.col("cat") == cat)
            .rename(
                {
                    c: f"{c}_{cat.lower()}"
                    for c in [
                        "fut_on_ret",
                        "fut_on_range",
                        "fut_on_z",
                        "is_fut_on_valid",
                        "fut_emergency_flag",
                    ]
                }
            )
            .drop("cat")
        )
        q = q.join(oni, on="Date", how="left")

    # EOD features (T+1) by shifting Date to effective_date pre-join
    for cat in cats:
        ei = (
            eod_df.filter(pl.col("cat") == cat)
            .rename(
                {
                    c: f"{c}_{cat.lower()}"
                    for c in [
                        "fut_day_ret",
                        "fut_whole_ret",
                        "fut_whole_ret_cont",
                        "basis_close",
                        "basis_close_z252",
                        "carry_per_day",
                        "fut_oi",
                        "fut_oi_delta",
                        "fut_vol",
                        "fut_oi_z252",
                        "fut_vol_z252",
                        "ttm_days",
                        "price_up_oi_up",
                        "price_up_oi_dn",
                        "price_dn_oi_up",
                        "price_dn_oi_dn",
                    ]
                }
            )
            .with_columns([pl.col("effective_date").alias("Date")])
            .drop(["cat"])
        )
        q = q.join(ei, on="Date", how="left")
        # Validity flag for EOD block
        q = q.with_columns(
            [
                pl.when(pl.col(f"basis_close_{cat.lower()}").is_null())
                .then(0)
                .otherwise(1)
                .cast(pl.Int8)
                .alias(f"is_fut_eod_valid_{cat.lower()}"),
            ]
        )
    return q


def build_next_bday_expr_from_quotes(quotes: pl.DataFrame) -> pl.Expr:
    """Build a next business day expression using available equity Dates as calendar.

    Treats all unique equity dates as business days and maps each to the next one.
    """
    dates = quotes.select("Date").unique().sort("Date")["Date"].to_list()
    # Map each date to next, last date maps to +1 day naive
    next_map = {}
    for i in range(len(dates) - 1):
        next_map[dates[i]] = dates[i + 1]
    if dates:
        last = dates[-1]
        # For polars Date, adding 1 day via duration
        # map_dict default handles others; here we set explicit default below
    return pl.col("Date").map_dict(next_map, default=pl.col("Date") + pl.duration(days=1))
