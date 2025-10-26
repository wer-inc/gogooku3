from __future__ import annotations

"""
Nikkei225 index option feature engineering (flat, one row per Code x Date).

Design focuses on:
- De-duplication preference (EmergencyMarginTriggerDivision 002 as EOD)
- Representative price selection with price_source encoding
- Time-to-maturity, (log) moneyness, flags
- IV, d1/d2, Greeks (delta/gamma/vega) and related signals
- Smile/term: intra-expiry percent ranks and z-scores; simple CMAT IV (30/60d)
- Flow (Volume/OpenInterest) and session (night/day) features
- Calendar flags and quality flags
"""

import math
from datetime import date as _date

import polars as pl

EPS = 1e-12


def _num(col: str) -> pl.Expr:
    """Normalize numeric-like Utf8 columns: empty string -> NULL -> Float64."""
    return pl.when(pl.col(col).cast(pl.Utf8) == "").then(None).otherwise(pl.col(col)).cast(pl.Float64)


def _safe_log_ratio(num: pl.Expr, den: pl.Expr) -> pl.Expr:
    return pl.when((num.is_not_null()) & (den.is_not_null()) & (den.abs() > 0)).then(
        (num / (den + EPS)).log()
    ).otherwise(None)


def _norm_cdf_expr(x: pl.Expr) -> pl.Expr:
    # N(x) via erf
    return x.map_elements(lambda v: 0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) if v is not None else None, return_dtype=pl.Float64)


def _norm_pdf_expr(x: pl.Expr) -> pl.Expr:
    inv_sqrt_2pi = 1.0 / math.sqrt(2.0 * math.pi)
    return x.map_elements(lambda v: inv_sqrt_2pi * math.exp(-0.5 * v * v) if v is not None else None, return_dtype=pl.Float64)


def build_index_option_features(opt_df: pl.DataFrame) -> pl.DataFrame:
    """Build flat per-contract daily features for Nikkei225 index options.

    Input columns expected (subset is OK; missing ones are filled with NULLs):
        Date, Code, ContractMonth, StrikePrice, PutCallDivision, EmergencyMarginTriggerDivision,
        WholeDay*, NightSession*, DaySession*, Volume, OpenInterest, TurnoverValue,
        SettlementPrice, TheoreticalPrice, BaseVolatility, ImpliedVolatility,
        UnderlyingPrice, InterestRate, LastTradingDay, SpecialQuotationDay,
        Volume(OnlyAuction)
    """
    if opt_df.is_empty():
        return opt_df

    df = opt_df
    # Normalize dates, strings, and numeric columns
    cols = df.columns

    # Rename auction column if present
    if "Volume(OnlyAuction)" in cols and "VolumeOnlyAuction" not in cols:
        df = df.rename({"Volume(OnlyAuction)": "VolumeOnlyAuction"})

    def _dtcol(name: str) -> pl.Expr:
        if name not in df.columns:
            return pl.lit(None, dtype=pl.Date).alias(name)
        # Simply cast to date, letting Polars handle string parsing
        return pl.col(name).cast(pl.Date, strict=False).alias(name)

    df = df.with_columns(
        [
            _dtcol("Date"),
            _dtcol("LastTradingDay"),
            _dtcol("SpecialQuotationDay"),
            pl.col("Code").cast(pl.Utf8) if "Code" in cols else pl.lit(None, dtype=pl.Utf8).alias("Code"),
            (pl.col("ContractMonth").cast(pl.Utf8) if "ContractMonth" in cols else pl.lit(None, dtype=pl.Utf8)).alias("ContractMonth"),
            (pl.col("EmergencyMarginTriggerDivision").cast(pl.Utf8) if "EmergencyMarginTriggerDivision" in cols else pl.lit(None, dtype=pl.Utf8)).alias("EmergencyMarginTriggerDivision"),
            (pl.col("PutCallDivision").cast(pl.Utf8) if "PutCallDivision" in cols else pl.lit(None, dtype=pl.Utf8)).alias("PutCallDivision"),
        ]
    )

    for c in [
        "WholeDayOpen",
        "WholeDayHigh",
        "WholeDayLow",
        "WholeDayClose",
        "NightSessionOpen",
        "NightSessionHigh",
        "NightSessionLow",
        "NightSessionClose",
        "DaySessionOpen",
        "DaySessionHigh",
        "DaySessionLow",
        "DaySessionClose",
        "Volume",
        "OpenInterest",
        "TurnoverValue",
        "SettlementPrice",
        "TheoreticalPrice",
        "BaseVolatility",
        "ImpliedVolatility",
        "UnderlyingPrice",
        "InterestRate",
        "StrikePrice",
        "VolumeOnlyAuction",
    ]:
        if c in df.columns:
            df = df.with_columns(_num(c).alias(c))
        else:
            # ensure presence with NULLs to simplify expressions
            df = df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

    # Resolve duplicates per (Date, Code) preferring EOD (002) over 001
    df = df.sort(["Date", "Code", "EmergencyMarginTriggerDivision"]).group_by(["Date", "Code"]).tail(1)

    # Representative price and price_source
    df = df.with_columns(
        [
            pl.when(pl.col("SettlementPrice") > 0)
            .then(pl.col("SettlementPrice"))
            .when(pl.col("WholeDayClose") > 0)
            .then(pl.col("WholeDayClose"))
            .otherwise(pl.col("TheoreticalPrice")).alias("price"),
            pl.when(pl.col("SettlementPrice") > 0)
            .then(pl.lit(1))
            .when(pl.col("WholeDayClose") > 0)
            .then(pl.lit(2))
            .otherwise(pl.lit(3))
            .cast(pl.Int8)
            .alias("price_source"),
            (pl.col("EmergencyMarginTriggerDivision") == "002").cast(pl.Int8).alias("is_eod"),
        ]
    )

    # Time-to-maturity and moneyness
    df = df.with_columns(
        [
            (pl.col("LastTradingDay") - pl.col("Date")).dt.total_days().clip(lower_bound=0).alias("tau_d"),
            ((pl.col("SpecialQuotationDay") - pl.col("Date")).dt.total_days()).alias("days_to_sq"),
        ]
    )
    df = df.with_columns([(pl.col("tau_d") / 365.0).alias("tau_y")])
    df = df.with_columns(
        [
            (pl.col("UnderlyingPrice") / (pl.col("StrikePrice") + EPS)).alias("moneyness"),
            _safe_log_ratio(pl.col("UnderlyingPrice"), pl.col("StrikePrice")).alias("log_moneyness"),
            (pl.col("PutCallDivision") == "1").cast(pl.Int8).alias("put_flag"),
        ]
    ).with_columns([
        # Now log_moneyness exists, so we can reference it
        (pl.col("log_moneyness").abs() < 0.01).cast(pl.Int8).alias("atm_flag"),
    ])

    # Returns (by Code)
    df = df.sort(["Code", "Date"]).with_columns(
        [
            _safe_log_ratio(pl.col("price"), pl.col("price").shift(1).over("Code")).alias("ret_1d"),
            _safe_log_ratio(pl.col("UnderlyingPrice"), pl.col("UnderlyingPrice").shift(1).over("Code")).alias("sx_ret_1d"),
        ]
    )
    df = df.with_columns(
        [
            _safe_log_ratio(pl.col("price"), pl.col("price").shift(5).over("Code")).alias("ret_5d"),
            # Realized vol (20d) for underlying, annualized approx
            (pl.col("sx_ret_1d").rolling_std(20).over("Code") * math.sqrt(252.0)).alias("sx_vol_20d"),
        ]
    )

    # IV and Greeks (q=0 approximation)
    df = df.with_columns(
        [
            (pl.col("ImpliedVolatility") / 100.0).alias("iv"),
            (pl.col("BaseVolatility") / 100.0).alias("base_vol"),
        ]
    ).with_columns([(pl.col("iv") - pl.col("base_vol")).alias("iv_minus_basevol")])

    sqrt_tau = pl.col("tau_y").map_elements(lambda t: math.sqrt(t) if t and t > 0 else None, return_dtype=pl.Float64)
    denom = (pl.col("iv") * (sqrt_tau + 0.0))
    d1 = (
        (pl.col("log_moneyness") + (pl.col("InterestRate") / 100.0 + 0.5 * pl.col("iv") * pl.col("iv")) * pl.col("tau_y"))
        / (denom + EPS)
    )
    d2 = d1 - pl.col("iv") * (sqrt_tau + 0.0)
    df = df.with_columns([d1.alias("d1"), d2.alias("d2")])
    Nd1 = _norm_cdf_expr(pl.col("d1"))
    Nd2 = _norm_cdf_expr(pl.col("d2"))
    pdf_d1 = _norm_pdf_expr(pl.col("d1"))
    df = df.with_columns(
        [
            # Delta (call: N(d1); put: N(d1)-1)
            pl.when(pl.col("put_flag") == 1).then(Nd1 - 1.0).otherwise(Nd1).alias("delta"),
            # Gamma and Vega
            (pdf_d1 / (pl.col("UnderlyingPrice") * pl.col("iv") * (sqrt_tau + 0.0) + EPS)).alias("gamma"),
            (pl.col("UnderlyingPrice") * pdf_d1 * (sqrt_tau + 0.0)).alias("vega"),
            (Nd2).alias("itm_prob_call"),
        ]
    )
    df = df.with_columns(
        [
            # ITM probability unified (call vs put)
            pl.when(pl.col("put_flag") == 1).then(_norm_cdf_expr(-pl.col("d2"))).otherwise(pl.col("itm_prob_call")).alias("itm_prob"),
            # Sigma-units moneyness (approx ~ d2 shift)
            (pl.col("log_moneyness") / (pl.col("iv") * (sqrt_tau + 0.0) + EPS)).alias("z_mny"),
            # Normalized greeks
            (pl.col("gamma") * pl.col("UnderlyingPrice")).alias("norm_gamma"),
            (pl.col("vega") / 100.0).alias("norm_vega"),
        ]
    )

    # Flow / microstructure
    df = df.with_columns(
        [
            pl.col("Volume").alias("vol"),
            pl.col("OpenInterest").alias("oi"),
            pl.col("TurnoverValue").alias("turnover"),
            (pl.col("OpenInterest") - pl.col("OpenInterest").shift(1).over("Code")).alias("oi_chg_1d"),
            (pl.col("Volume") / (pl.col("OpenInterest") + EPS)).alias("oi_turnover"),
            (pl.col("Volume").rolling_mean(5).over("Code")).alias("vol_ema_5"),
            (pl.col("Volume") * pl.col("price") * 1000.0).alias("dollar_vol"),
            (pl.col("VolumeOnlyAuction") / (pl.col("Volume") + EPS)).alias("auction_vol_ratio"),
            ((pl.col("TheoreticalPrice") - pl.col("price")) / (pl.col("price") + EPS)).alias("theo_gap"),
            ((pl.col("Volume") <= 0) | (pl.col("TurnoverValue") <= 0)).cast(pl.Int8).alias("illiquid_flag"),
        ]
    )

    # Session features and ranges
    df = df.with_columns(
        [
            _safe_log_ratio(pl.col("NightSessionClose"), pl.col("NightSessionOpen")).alias("overnight_ret"),
            _safe_log_ratio(pl.col("DaySessionClose"), pl.col("DaySessionOpen")).alias("intraday_ret"),
            ((pl.col("DaySessionOpen") - pl.col("NightSessionClose")) / (pl.col("NightSessionClose") + EPS)).alias("gap_ratio"),
            ((pl.col("WholeDayHigh") - pl.col("WholeDayLow")) / (pl.col("price") + EPS)).alias("wd_range"),
            ((pl.col("DaySessionHigh") - pl.col("DaySessionLow")) / (pl.col("price") + EPS)).alias("day_range"),
            ((pl.col("NightSessionHigh") - pl.col("NightSessionLow")) / (pl.col("price") + EPS)).alias("night_range"),
            (pl.col("WholeDayClose") - pl.col("price")).alias("wd_close_diff"),
        ]
    )

    # Calendar/time features
    df = df.with_columns(
        [
            pl.col("Date").dt.weekday().alias("dow"),
            pl.col("Date").dt.day().alias("dom"),
        ]
    )
    df = df.with_columns((((pl.col("dom") - 1) // 7) + 1).alias("wom"))
    df = df.with_columns(
        [
            (pl.col("LastTradingDay") - pl.col("Date")).dt.total_days().alias("days_to_last_trading_day"),
            (pl.col("LastTradingDay") == pl.col("Date")).cast(pl.Int8).alias("is_expiry_day"),
            ((pl.col("LastTradingDay") - pl.col("Date")).dt.total_days() <= 5).cast(pl.Int8).alias("is_expiry_week"),
            (pl.col("EmergencyMarginTriggerDivision") == "001").cast(pl.Int8).alias("is_emergency_margin"),
            (pl.col("Date") >= pl.lit(_date(2011, 2, 14))).cast(pl.Int8).alias("post_2011_session_flag"),
            (pl.col("Date") >= pl.lit(_date(2016, 7, 19))).cast(pl.Int8).alias("data_after_2016_07_19_flag"),
        ]
    )

    # Cross-sectional stats by (Date, ContractMonth)
    grp = ["Date", "ContractMonth"]
    cnt = pl.count().over(grp)
    # Percent ranks
    def _pct_rank(x: pl.Expr, name: str) -> pl.Expr:
        rk = x.rank("average").over(grp)
        return (pl.when(cnt > 1).then((rk - 1.0) / (cnt - 1.0)).otherwise(0.5)).alias(name)

    df = df.with_columns(
        [
            _pct_rank(pl.col("iv"), "iv_pct_rank_by_expiry"),
            _pct_rank(pl.col("delta"), "delta_pct_by_expiry"),
            _pct_rank(pl.col("oi"), "oi_pct_by_expiry"),
            _pct_rank(pl.col("price"), "price_pct_by_expiry"),
            _pct_rank(pl.col("theo_gap"), "theo_gap_pct_by_expiry"),
            ((pl.col("iv") - pl.col("iv").mean().over(grp)) / (pl.col("iv").std().over(grp) + EPS)).alias("iv_z_by_expiry"),
        ]
    )

    # Simple CMAT IV: per Date, use expiry medians (iv, tau_y) and interpolate to 30/60 days
    base = (
        df.group_by(["Date", "ContractMonth"]).agg([
            pl.col("tau_y").median().alias("tau_y"),
            pl.col("iv").median().alias("iv_med"),
        ])
        .sort(["Date", "tau_y"])
    )
    # For each Date, get nearest below/above for 30/60 days
    t30 = 30.0 / 365.0
    t60 = 60.0 / 365.0

    def _interp(df_day: pl.DataFrame, t: float) -> float | None:
        if df_day.is_empty():
            return None
        v = df_day.filter(pl.col("tau_y") == t)
        if v.height > 0:
            return float(v["iv_med"][0])
        below = df_day.filter(pl.col("tau_y") <= t).tail(1)
        above = df_day.filter(pl.col("tau_y") >= t).head(1)
        if below.is_empty() and above.is_empty():
            return None
        if below.is_empty():
            return float(above["iv_med"][0])
        if above.is_empty():
            return float(below["iv_med"][0])
        t0, v0 = float(below["tau_y"][0]), float(below["iv_med"][0])
        t1, v1 = float(above["tau_y"][0]), float(above["iv_med"][0])
        if abs(t1 - t0) < 1e-9:
            return v0
        w = (t - t0) / (t1 - t0)
        return v0 * (1 - w) + v1 * w

    # Apply interpolation per Date
    parts = []
    for d in base.select("Date").unique().to_series().to_list():
        di = base.filter(pl.col("Date") == d)
        cm30 = _interp(di, t30)
        cm60 = _interp(di, t60)
        parts.append(pl.DataFrame({"Date": [d], "iv_cmat_30d": [cm30], "iv_cmat_60d": [cm60]}))
    cmat = pl.concat(parts) if parts else base.select(["Date"]).with_columns([
        pl.lit(None).alias("iv_cmat_30d"), pl.lit(None).alias("iv_cmat_60d")
    ])
    df = df.join(cmat, on="Date", how="left").with_columns(
        [(pl.col("iv_cmat_60d") - pl.col("iv_cmat_30d")).alias("term_slope_30_60")]
    )

    # Final column selection (concise but representative subset)
    keep = [
        "Date", "Code", "ContractMonth", "StrikePrice", "PutCallDivision", "EmergencyMarginTriggerDivision",
        "is_eod", "tau_d", "tau_y", "days_to_sq", "moneyness", "log_moneyness", "atm_flag", "put_flag",
        "price", "price_source", "ret_1d", "ret_5d", "sx_ret_1d", "sx_vol_20d",
        "iv", "base_vol", "iv_minus_basevol", "d1", "d2", "delta", "gamma", "vega", "norm_gamma", "norm_vega", "itm_prob", "z_mny",
        "iv_pct_rank_by_expiry", "iv_z_by_expiry", "iv_cmat_30d", "iv_cmat_60d", "term_slope_30_60",
        "vol", "oi", "turnover", "oi_chg_1d", "oi_turnover", "vol_ema_5", "dollar_vol", "auction_vol_ratio",
        "theo_gap", "theo_gap_pct_by_expiry",
        "wd_range", "day_range", "night_range", "overnight_ret", "intraday_ret", "gap_ratio",
        "dow", "dom", "wom", "days_to_last_trading_day", "is_expiry_week", "is_expiry_day",
        "post_2011_session_flag", "is_emergency_margin", "illiquid_flag", "data_after_2016_07_19_flag",
    ]
    # Only keep those that exist in df
    keep = [c for c in keep if c in df.columns]
    return df.select(keep)


def build_option_market_aggregates(opt_feats: pl.DataFrame, next_bday_expr: pl.Expr | None = None) -> pl.DataFrame:
    """Build daily market-level aggregates from per-contract option features.

    Aggregation is across all contracts for a given Date (option market snapshot),
    then shifted to `effective_date` if `next_bday_expr` is provided.
    """
    if opt_feats.is_empty():
        return opt_feats.select([])

    s = opt_feats
    # Prefer EOD snapshot if available
    if "is_eod" in s.columns:
        s = s.filter(pl.col("is_eod") == 1)

    if "tau_d" in s.columns:
        s = s.with_columns(
            pl.col("tau_d")
            .fill_null(pl.col("tau_d").max().over("Date"))
            .rank("dense")
            .over("Date")
            .alias("tau_rank")
        )
    else:
        s = s.with_columns(pl.lit(1).alias("tau_rank"))

    # Aggregations (robust via median)
    agg = (
        s.group_by("Date")
        .agg(
            [
                pl.col("iv_cmat_30d").median().alias("opt_iv_cmat_30d"),
                pl.col("iv_cmat_60d").median().alias("opt_iv_cmat_60d"),
                pl.col("term_slope_30_60").median().alias("opt_term_slope_30_60"),
                pl.col("iv").filter(pl.col("atm_flag") == 1).median().alias("opt_iv_atm_median"),
                pl.col("oi").sum().alias("opt_oi_sum"),
                pl.col("vol").sum().alias("opt_vol_sum"),
                pl.col("dollar_vol").sum().alias("opt_dollar_vol_sum"),
                pl.col("iv")
                .filter((pl.col("tau_rank") == 1) & (pl.col("atm_flag") == 1))
                .median()
                .alias("opt_iv_atm_near"),
                pl.col("iv")
                .filter((pl.col("tau_rank") == 2) & (pl.col("atm_flag") == 1))
                .median()
                .alias("opt_iv_atm_next"),
                (
                    pl.col("iv")
                    .filter(
                        (pl.col("tau_rank") == 1)
                        & (pl.col("put_flag") == 1)
                        & (pl.col("moneyness") >= 0.93)
                        & (pl.col("moneyness") <= 0.97)
                    )
                    .median()
                    - pl.col("iv")
                    .filter(
                        (pl.col("tau_rank") == 1)
                        & (pl.col("put_flag") == 0)
                        & (pl.col("moneyness") >= 1.03)
                        & (pl.col("moneyness") <= 1.07)
                    )
                    .median()
                ).alias("opt_skew_5pct"),
                pl.col("iv")
                .filter(
                    (pl.col("tau_rank") == 1)
                    & ((pl.col("moneyness") - 1.0).abs() <= 0.1)
                )
                .std()
                .alias("opt_smile_width"),
                (
                    pl.col("oi")
                    .filter(
                        (pl.col("tau_rank") == 1)
                        & (pl.col("put_flag") == 1)
                        & ((pl.col("moneyness") - 1.0).abs() <= 0.1)
                    )
                    .sum()
                    / (
                        pl.col("oi")
                        .filter(
                            (pl.col("tau_rank") == 1)
                            & (pl.col("put_flag") == 0)
                            & ((pl.col("moneyness") - 1.0).abs() <= 0.1)
                        )
                        .sum()
                        + EPS
                    )
                ).alias("opt_oi_put_call_ratio"),
            ]
        )
        .sort("Date")
    )
    if next_bday_expr is not None:
        agg = agg.with_columns([next_bday_expr.alias("effective_date")])
        agg = agg.select(["effective_date", "Date", *[c for c in agg.columns if c not in ("Date", "effective_date")]])
    agg = agg.with_columns(
        [
            (pl.col("opt_iv_atm_next") - pl.col("opt_iv_atm_near")).alias("opt_ts_slope"),
            (
                (pl.col("opt_iv_atm_near")
                 - pl.col("opt_iv_atm_near").rolling_mean(window_size=20, min_periods=5))
                / (
                    pl.col("opt_iv_atm_near").rolling_std(window_size=20, min_periods=5)
                    + EPS
                )
            ).alias("opt_iv_shock"),
        ]
    )
    return agg


def attach_option_market_to_equity(quotes: pl.DataFrame, mkt: pl.DataFrame) -> pl.DataFrame:
    """Attach market-level option aggregates to equity panel as T+1 if effective_date provided.

    If `mkt` contains `effective_date`, it will be used for join; otherwise, Date join (T+0).
    """
    if mkt.is_empty():
        return quotes
    has_eff = "effective_date" in mkt.columns
    df = quotes
    if has_eff:
        join_df = mkt.rename({"effective_date": "Date"}).drop([c for c in ("Date",) if c in mkt.columns])
    else:
        join_df = mkt
    out = df.join(join_df, on="Date", how="left")
    if has_eff:
        # Validity flag for the EOD block
        out = out.with_columns([
            pl.when(pl.col("opt_iv_cmat_30d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("is_opt_mkt_valid")
        ])
    return out
