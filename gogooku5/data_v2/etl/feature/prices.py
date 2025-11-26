"""Feature computation from daily_quotes and breakdown."""

from __future__ import annotations

import math
from datetime import date, timedelta

import duckdb
import polars as pl

from ..schemas import FEATURES_DAILY_COLUMNS
from .daily_margin import compute_daily_margin_features
from .short_positions import compute_short_positions_features


def compute_price_features(
    con: duckdb.DuckDBPyConnection,
    *,
    start: str,
    end: str,
    warmup_days: int = 260,
) -> pl.DataFrame:
    """
    Compute daily price-based features from `daily_quotes` and `breakdown`.

    This includes:
    - Per-name time-series features (returns, volatility, technicals, TA)
    - Breakdown-derived demand/flow features (value/volume ratios, overhang, margin flow)
    - Cross-sectional ranks / percentiles / Z-scores per date
    - Session (前場/後場) microstructure features

    All features for a given (date, code) are materialised into a single
    DataFrame aligned to FEATURES_DAILY_COLUMNS; listed_info-derived
    sector/meta features are added in a separate step.
    """
    EPS = 1e-9
    start_dt = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)

    # Guard: ensure requested range exists in daily_quotes and adjust warmup if data is shorter.
    min_q_date = con.execute("SELECT min(date) FROM daily_quotes").fetchone()[0]
    if min_q_date is None:
        raise ValueError("daily_quotes is empty; fetch quotes before computing features.")
    if start_dt < min_q_date:
        raise ValueError(f"start={start} is earlier than earliest daily_quotes={min_q_date}")

    warmup_start_dt = start_dt - timedelta(days=warmup_days) if warmup_days > 0 else start_dt
    effective_start_dt = warmup_start_dt if warmup_start_dt >= min_q_date else min_q_date
    warmup_start = effective_start_dt.isoformat()
    if effective_start_dt > warmup_start_dt:
        print(
            f"[warn] warmup truncated: requested start-warmup={warmup_start_dt} "
            f"but daily_quotes starts at {min_q_date}; using warmup_start={warmup_start}"
        )

    # Load quotes from DuckDB
    # Fetch quotes with projection and filter applied in DuckDB, then process lazily.
    quotes_df = pl.read_database(
        """
        SELECT
            date,
            code,
            adjustment_close AS close,
            adjustment_volume AS volume,
            turnover_value AS turnover,
            adjustment_open AS open,
            adjustment_high AS high,
            adjustment_low AS low,
            open   AS raw_open,
            high   AS raw_high,
            low    AS raw_low,
            close  AS raw_close,
            upper_limit,
            lower_limit,
            morning_open,
            morning_high,
            morning_low,
            morning_close,
            morning_volume,
            afternoon_open,
            afternoon_high,
            afternoon_low,
            afternoon_close,
            afternoon_volume,
            adjustment_factor
        FROM daily_quotes
        WHERE date BETWEEN ? AND ?
        ORDER BY code, date
        """,
        connection=con,
        execute_options={"parameters": [warmup_start, end]},
    )
    if quotes_df.is_empty():
        return pl.DataFrame()

    q = (
        quotes_df.lazy()
        .sort(["code", "date"])
        .with_columns(
            [
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
                pl.col("turnover").cast(pl.Float64),
            ]
        )
    )

    # Basic returns and momentum
    q = q.with_columns(
        [
            (pl.col("close") / pl.col("close").shift(1).over("code") - 1.0).alias("ret_1d"),
            (pl.col("close") / pl.col("close").shift(5).over("code")).log().alias("ret_5d"),
            (pl.col("close") / pl.col("close").shift(20).over("code")).log().alias("ret_20d"),
            (pl.col("close") / pl.col("close").shift(60).over("code")).log().alias("ret_60d"),
            (pl.col("close") / pl.col("open")).log().alias("ret_intraday"),
            (pl.col("open") / pl.col("close").shift(1).over("code")).log().alias("ret_overnight"),
            (pl.col("close") / pl.col("close").shift(5).over("code") - 1.0).alias("mom_5"),
            (pl.col("close") / pl.col("close").shift(10).over("code") - 1.0).alias("mom_10"),
            (pl.col("close") / pl.col("close").shift(20).over("code") - 1.0).alias("mom_20"),
            (pl.col("close") / pl.col("close").shift(-1).over("code")).log().alias("fwd_ret_1d"),
            (pl.col("close") / pl.col("close").shift(-5).over("code")).log().alias("fwd_ret_5d"),
            (pl.col("close") / pl.col("close").shift(-10).over("code")).log().alias("fwd_ret_10d"),
            (pl.col("close") / pl.col("close").shift(-20).over("code")).log().alias("fwd_ret_20d"),
            (pl.col("close") / pl.col("close").shift(1).over("code")).log().alias("logret_1d"),
        ]
    )
    # simple momentum acceleration: day-to-day change in mom_5
    q = q.with_columns((pl.col("mom_5") - pl.col("mom_5").shift(1).over("code")).alias("mom_accel"))

    # Realized volatilities and ratios
    q = q.with_columns(
        [
            pl.col("ret_1d").rolling_std(window_size=5, min_periods=3).over("code").alias("rv_5"),
            pl.col("ret_1d").rolling_std(window_size=20, min_periods=10).over("code").alias("rv_20"),
        ]
    )
    q = q.with_columns(
        [
            pl.when(pl.col("rv_20").is_null() | (pl.col("rv_20") == 0))
            .then(None)
            .otherwise(pl.col("rv_5") / pl.col("rv_20"))
            .alias("rv_ratio"),
            (
                (pl.col("rv_5") - pl.col("rv_5").rolling_mean(window_size=20, min_periods=5).over("code"))
                / (pl.col("rv_5").rolling_std(window_size=20, min_periods=5).over("code") + EPS)
            ).alias("rv_z_20"),
        ]
    )

    # Range, moving averages and volume/value ratios
    q = q.with_columns(
        [
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range"),
            pl.col("volume").rolling_mean(window_size=5, min_periods=3).over("code").alias("mean_vol_5"),
            pl.col("volume").rolling_mean(window_size=20, min_periods=5).over("code").alias("mean_vol_20"),
            pl.col("volume").rolling_mean(window_size=20, min_periods=5).over("code").alias("adv_vol_20"),
            pl.col("turnover").rolling_mean(window_size=5, min_periods=3).over("code").alias("mean_val_5"),
            pl.col("turnover").rolling_mean(window_size=20, min_periods=5).over("code").alias("mean_val_20"),
            pl.col("close").rolling_mean(window_size=5, min_periods=3).over("code").alias("ma_5"),
            pl.col("close").rolling_mean(window_size=20, min_periods=5).over("code").alias("ma_20"),
            pl.col("close").rolling_mean(window_size=60, min_periods=10).over("code").alias("ma_60"),
            pl.col("close").rolling_min(window_size=20, min_periods=5).over("code").alias("_min_20"),
            pl.col("close").rolling_max(window_size=20, min_periods=5).over("code").alias("_max_20"),
            pl.col("close").rolling_min(window_size=252, min_periods=20).over("code").alias("_low_252"),
            pl.col("close").rolling_max(window_size=252, min_periods=20).over("code").alias("_high_252"),
        ]
    )
    q = q.with_columns(
        [
            ((pl.col("close") - pl.col("_min_20")) / (pl.col("_max_20") - pl.col("_min_20") + EPS)).alias(
                "price_pos_20"
            ),
            ((pl.col("close") - pl.col("_high_252")) / (pl.col("_high_252") + EPS)).alias("dist_from_high52"),
            ((pl.col("close") - pl.col("_low_252")) / (pl.col("_low_252") + EPS)).alias("dist_from_low52"),
        ]
    )

    q = q.with_columns(
        [
            (
                (
                    (
                        (pl.col("high") / pl.col("low"))
                        .log()
                        .pow(2)
                        .rolling_mean(window_size=20, min_periods=5)
                        .over("code")
                    )
                    / (4 * math.log(2))
                ).sqrt()
            ).alias("parkinson_20"),
            (
                (
                    (
                        0.5 * (pl.col("high") / pl.col("low")).log().pow(2)
                        - (2 * math.log(2) - 1.0) * (pl.col("close") / pl.col("open")).log().pow(2)
                    )
                    .rolling_mean(window_size=20, min_periods=5)
                    .over("code")
                ).sqrt()
            ).alias("gk_20"),
        ]
    )

    # EMA-based indicators (MACD: EMA version)
    alpha_ema_12 = 2.0 / (12.0 + 1.0)
    alpha_ema_26 = 2.0 / (26.0 + 1.0)
    alpha_macd_signal = 2.0 / (9.0 + 1.0)
    q = q.with_columns(
        [
            pl.col("close").ewm_mean(alpha=alpha_ema_12).over("code").alias("ema_12"),
            pl.col("close").ewm_mean(alpha=alpha_ema_26).over("code").alias("ema_26"),
        ]
    )
    q = q.with_columns((pl.col("ema_12") - pl.col("ema_26")).alias("macd"))
    q = q.with_columns(pl.col("macd").ewm_mean(alpha=alpha_macd_signal).over("code").alias("macd_signal"))
    q = q.with_columns((pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"))
    q = q.with_columns((pl.col("macd_hist") - pl.col("macd_hist").shift(1).over("code")).alias("macd_hist_diff_1"))

    # RSI14 (Wilder-style smoothing via EMA)
    q = q.with_columns((pl.col("close") - pl.col("close").shift(1).over("code")).alias("_delta"))
    q = q.with_columns(
        [
            pl.when(pl.col("_delta") > 0).then(pl.col("_delta")).otherwise(0.0).alias("_gain"),
            pl.when(pl.col("_delta") < 0).then(-pl.col("_delta")).otherwise(0.0).alias("_loss"),
        ]
    )
    alpha_rsi = 1.0 / 14.0
    q = q.with_columns(
        [
            pl.col("_gain").ewm_mean(alpha=alpha_rsi).over("code").alias("_avg_gain"),
            pl.col("_loss").ewm_mean(alpha=alpha_rsi).over("code").alias("_avg_loss"),
        ]
    )
    q = q.with_columns(
        pl.when(pl.col("_avg_loss") == 0)
        .then(None)
        .otherwise(100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / (pl.col("_avg_loss") + EPS)))
        .alias("rsi_14")
    )

    # Bollinger Bands 20, 2σ
    q = q.with_columns(pl.col("close").rolling_std(window_size=20, min_periods=5).over("code").alias("_std_20"))
    q = q.with_columns(
        [
            (pl.col("ma_20") + 2.0 * pl.col("_std_20")).alias("bb_upper_20_2"),
            pl.col("ma_20").alias("bb_mid_20_2"),
            (pl.col("ma_20") - 2.0 * pl.col("_std_20")).alias("bb_lower_20_2"),
        ]
    )
    q = q.with_columns(
        [
            ((pl.col("bb_upper_20_2") - pl.col("bb_lower_20_2")) / (pl.col("bb_mid_20_2") + EPS)).alias(
                "bb_width_20_2"
            ),
            (
                (pl.col("close") - pl.col("bb_lower_20_2")) / (pl.col("bb_upper_20_2") - pl.col("bb_lower_20_2") + EPS)
            ).alias("bb_percent_b_20_2"),
        ]
    )

    # Candlestick shape
    q = q.with_columns(
        [
            ((pl.col("close") - pl.col("open")).abs() / (pl.col("open") + EPS)).alias("cand_body"),
            (
                (pl.col("close") - pl.col("open")).abs()
                / (((pl.col("high") - pl.col("low")) / (pl.col("close") + EPS)) + EPS)
            ).alias("cand_body_ratio"),
            (
                (pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close")))
                / (pl.col("high") - pl.col("low") + EPS)
            ).alias("cand_upper_shadow"),
            (
                (pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low"))
                / (pl.col("high") - pl.col("low") + EPS)
            ).alias("cand_lower_shadow"),
            (pl.col("close") > pl.col("open")).cast(pl.Int8).alias("cand_is_bull"),
        ]
    )
    # Bull/bear run lengths and ratios
    q = q.with_columns(pl.arange(0, pl.len()).over("code").alias("_cand_idx"))
    q = q.with_columns(
        [
            pl.when(pl.col("cand_is_bull") == 0)
            .then(pl.col("_cand_idx"))
            .otherwise(None)
            .forward_fill()
            .over("code")
            .fill_null(-1)
            .alias("_last_bear_idx"),
            pl.when(pl.col("cand_is_bull") == 1)
            .then(pl.col("_cand_idx"))
            .otherwise(None)
            .forward_fill()
            .over("code")
            .fill_null(-1)
            .alias("_last_bull_idx"),
        ]
    )
    q = q.with_columns(
        [
            pl.when(pl.col("cand_is_bull") == 1)
            .then((pl.col("_cand_idx") - pl.col("_last_bear_idx")).cast(pl.Int32))
            .otherwise(pl.lit(0))
            .alias("bull_run_len"),
            pl.when(pl.col("cand_is_bull") == 0)
            .then((pl.col("_cand_idx") - pl.col("_last_bull_idx")).cast(pl.Int32))
            .otherwise(pl.lit(0))
            .alias("bear_run_len"),
            pl.col("cand_is_bull").rolling_mean(window_size=5, min_periods=1).over("code").alias("bull_ratio_5d"),
            pl.col("cand_is_bull").rolling_mean(window_size=20, min_periods=3).over("code").alias("bull_ratio_20d"),
        ]
    )

    q = q.with_columns(
        [
            (pl.col("volume") / pl.col("mean_vol_5")).alias("vol_ratio_5"),
            (pl.col("volume") / pl.col("mean_vol_20")).alias("vol_ratio_20"),
            (pl.col("mean_vol_5") / pl.col("mean_vol_20")).alias("vol_ratio_5_20"),
            (pl.col("turnover") / pl.col("mean_val_5")).alias("val_ratio_5"),
            (pl.col("turnover") / pl.col("mean_val_20")).alias("val_ratio_20"),
            ((pl.col("close") - pl.col("ma_5")) / pl.col("ma_5")).alias("price_dev_5"),
            ((pl.col("close") - pl.col("ma_20")) / pl.col("ma_20")).alias("price_dev_20"),
            (pl.col("close") * pl.col("volume")).alias("dollar_volume"),
            (pl.col("mom_5") - pl.col("mom_20")).alias("mom_chg_5_20"),
            ((pl.col("close") - pl.col("ma_60")) / pl.col("ma_60")).alias("dev_ma60"),
            (pl.col("turnover") / (pl.col("volume") + EPS)).alias("vwap"),
        ]
    )

    # Z-scores and vol_20d
    q = q.with_columns(
        [
            pl.col("volume").log().alias("_log_vol"),
            pl.col("turnover").log().alias("_log_val"),
            pl.col("close").forward_fill().over("code").alias("_close_ta_ff"),
            pl.col("high").forward_fill().over("code").alias("_high_ta_ff"),
            pl.col("low").forward_fill().over("code").alias("_low_ta_ff"),
        ]
    )
    q = q.with_columns(
        [
            (
                (pl.col("_log_vol") - pl.col("_log_vol").rolling_mean(window_size=20, min_periods=5).over("code"))
                / (pl.col("_log_vol").rolling_std(window_size=20, min_periods=5).over("code") + EPS)
            ).alias("vol_z_20"),
            (
                (pl.col("_log_val") - pl.col("_log_val").rolling_mean(window_size=20, min_periods=5).over("code"))
                / (pl.col("_log_val").rolling_std(window_size=20, min_periods=5).over("code") + EPS)
            ).alias("val_z_20"),
            (
                (pl.col("range") - pl.col("range").rolling_mean(window_size=20, min_periods=5).over("code"))
                / (pl.col("range").rolling_std(window_size=20, min_periods=5).over("code") + EPS)
            ).alias("range_z_20"),
            pl.col("ret_1d").rolling_std(window_size=20, min_periods=10).over("code").alias("vol_20d"),
        ]
    )

    # ATR14 and ADX14 (manual, Wilder-style smoothing via EMA)
    q = q.with_columns(
        [
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1).over("code")).abs(),
                (pl.col("low") - pl.col("close").shift(1).over("code")).abs(),
            ).alias("_true_range"),
            (pl.col("high") - pl.col("high").shift(1).over("code")).alias("_up_move"),
            (pl.col("low").shift(1).over("code") - pl.col("low")).alias("_down_move"),
        ]
    )
    q = q.with_columns(
        [
            pl.when((pl.col("_up_move") > pl.col("_down_move")) & (pl.col("_up_move") > 0))
            .then(pl.col("_up_move"))
            .otherwise(0.0)
            .alias("_plus_dm"),
            pl.when((pl.col("_down_move") > pl.col("_up_move")) & (pl.col("_down_move") > 0))
            .then(pl.col("_down_move"))
            .otherwise(0.0)
            .alias("_minus_dm"),
        ]
    )
    alpha_wilder = 1.0 / 14.0
    q = q.with_columns(
        [
            pl.col("_true_range").ewm_mean(alpha=alpha_wilder).over("code").alias("_atr14_smooth"),
            pl.col("_plus_dm").ewm_mean(alpha=alpha_wilder).over("code").alias("_plus_dm_smooth"),
            pl.col("_minus_dm").ewm_mean(alpha=alpha_wilder).over("code").alias("_minus_dm_smooth"),
        ]
    )
    q = q.with_columns(
        [
            (100.0 * pl.col("_plus_dm_smooth") / (pl.col("_atr14_smooth") + EPS)).alias("_plus_di"),
            (100.0 * pl.col("_minus_dm_smooth") / (pl.col("_atr14_smooth") + EPS)).alias("_minus_di"),
        ]
    )
    q = q.with_columns(
        (
            100.0 * (pl.col("_plus_di") - pl.col("_minus_di")).abs() / (pl.col("_plus_di") + pl.col("_minus_di") + EPS)
        ).alias("_dx")
    )
    q = q.with_columns(pl.col("_dx").ewm_mean(alpha=alpha_wilder).over("code").alias("adx_14"))
    q = q.with_columns((pl.col("adx_14") - pl.col("adx_14").shift(5).over("code")).alias("adx_slope_5"))
    # use smoothed ATR as atr_14
    q = q.with_columns(pl.col("_atr14_smooth").alias("atr_14"))

    # Amihud illiquidity measures
    amihud_scale = 1_000_000.0
    q = q.with_columns(
        [
            pl.when(pl.col("turnover") > 0)
            .then(pl.col("ret_1d").abs() / (pl.col("turnover") / amihud_scale))
            .otherwise(None)
            .alias("amihud_1d"),
        ]
    )
    q = q.with_columns(
        [
            pl.col("amihud_1d").rolling_mean(window_size=20, min_periods=5).over("code").alias("amihud_20"),
        ]
    )
    q = q.with_columns(
        [
            pl.when(pl.col("amihud_1d").is_null())
            .then(None)
            .otherwise((1.0 + pl.col("amihud_1d")).log())
            .alias("log_amihud_1d"),
            pl.when(pl.col("amihud_1d").is_null())
            .then(None)
            .otherwise((1.0 + pl.col("amihud_20")).log())
            .alias("log_amihud_20"),
        ]
    )

    # ========== Phase 3.1: Risk/Drawdown Features ==========
    # Max drawdown: (current - rolling_max) / rolling_max
    q = q.with_columns(
        [
            pl.col("close").rolling_max(window_size=20, min_periods=5).over("code").alias("_rolling_high_20"),
            pl.col("close").rolling_max(window_size=60, min_periods=10).over("code").alias("_rolling_high_60"),
        ]
    )
    q = q.with_columns(
        [
            ((pl.col("close") - pl.col("_rolling_high_20")) / (pl.col("_rolling_high_20") + EPS)).alias(
                "drawdown_from_high_20"
            ),
            ((pl.col("close") - pl.col("_rolling_high_60")) / (pl.col("_rolling_high_60") + EPS)).alias(
                "drawdown_from_high_60"
            ),
        ]
    )
    # Max drawdown over period (worst drawdown seen in window)
    q = q.with_columns(
        [
            pl.col("drawdown_from_high_20")
            .rolling_min(window_size=20, min_periods=5)
            .over("code")
            .alias("max_drawdown_20"),
            pl.col("drawdown_from_high_60")
            .rolling_min(window_size=60, min_periods=10)
            .over("code")
            .alias("max_drawdown_60"),
        ]
    )
    # Risk-adjusted momentum (Sharpe-like ratio: ret / vol)
    q = q.with_columns(
        [
            (pl.col("mom_20") / (pl.col("rv_20") + EPS)).alias("risk_adj_mom_20"),
            (pl.col("mom_5") / (pl.col("rv_5") + EPS)).alias("risk_adj_mom_5"),
        ]
    )
    # Underwater duration: days since last all-time high in 252-day window
    # _high_252 was already computed earlier
    q = q.with_columns(
        [
            (pl.col("close") >= pl.col("_rolling_high_20")).cast(pl.Int8).alias("_is_at_high_20"),
        ]
    )
    q = q.with_columns(pl.arange(0, pl.len()).over("code").alias("_uw_idx"))
    q = q.with_columns(
        pl.when(pl.col("_is_at_high_20") == 1)
        .then(pl.col("_uw_idx"))
        .otherwise(None)
        .forward_fill()
        .over("code")
        .fill_null(-1)
        .alias("_last_high_idx")
    )
    q = q.with_columns(
        pl.when(pl.col("_last_high_idx") < 0)
        .then(None)
        .otherwise((pl.col("_uw_idx") - pl.col("_last_high_idx")).cast(pl.Int32))
        .alias("underwater_days_20")
    )
    # Recovery rate: how much of the drawdown has been recovered
    q = q.with_columns(
        pl.when(pl.col("max_drawdown_20").abs() < EPS)
        .then(1.0)  # No drawdown = fully recovered
        .otherwise(1.0 - pl.col("drawdown_from_high_20").abs() / (pl.col("max_drawdown_20").abs() + EPS))
        .clip(0.0, 1.0)
        .alias("recovery_rate_20")
    )
    # Clean up temp columns (will be dropped later with other temps)

    # ========== Phase 3.2: Market Alpha Features ==========
    # Market-relative returns using cross-sectional mean as market proxy
    q = q.with_columns(
        [
            pl.col("ret_1d").mean().over("date").alias("_market_mean_ret_1d"),
            pl.col("ret_5d").mean().over("date").alias("_market_mean_ret_5d"),
            pl.col("ret_20d").mean().over("date").alias("_market_mean_ret_20d"),
        ]
    )
    q = q.with_columns(
        [
            (pl.col("ret_1d") - pl.col("_market_mean_ret_1d")).alias("market_alpha_1d"),
            (pl.col("ret_5d") - pl.col("_market_mean_ret_5d")).alias("market_alpha_5d"),
            (pl.col("ret_20d") - pl.col("_market_mean_ret_20d")).alias("market_alpha_20d"),
        ]
    )
    # Relative strength: stock momentum vs market momentum (ratio approach)
    q = q.with_columns(
        [
            pl.when(pl.col("_market_mean_ret_5d").abs() > EPS)
            .then(pl.col("ret_5d") / pl.col("_market_mean_ret_5d"))
            .otherwise(None)
            .alias("rel_strength_5d"),
            pl.when(pl.col("_market_mean_ret_20d").abs() > EPS)
            .then(pl.col("ret_20d") / pl.col("_market_mean_ret_20d"))
            .otherwise(None)
            .alias("rel_strength_20d"),
        ]
    )
    # Cumulative excess returns (alpha summed over rolling window)
    q = q.with_columns(
        pl.col("market_alpha_1d").rolling_sum(window_size=20, min_periods=5).over("code").alias("cum_alpha_20d")
    )
    # Information ratio-like metric: cumulative alpha / alpha volatility
    q = q.with_columns(
        [
            pl.col("market_alpha_1d").rolling_std(window_size=20, min_periods=5).over("code").alias("_alpha_std_20"),
        ]
    )
    q = q.with_columns((pl.col("cum_alpha_20d") / (pl.col("_alpha_std_20") + EPS)).alias("info_ratio_20d"))
    # Clean up Market Alpha temp columns (added to drop list below)

    # ========== Phase 3.3: Volatility Regime Features ==========
    # Volatility expansion flag: short-term vol > long-term vol
    q = q.with_columns(
        [
            (pl.col("rv_5") > pl.col("rv_20")).cast(pl.Int8).alias("vol_expansion"),
            # Volatility trend: change in rv_20 over past 20 days
            (pl.col("rv_20") - pl.col("rv_20").shift(20).over("code")).alias("vol_trend_20"),
        ]
    )
    # Volatility regime: compare current vol to historical distribution
    q = q.with_columns(
        [
            pl.col("rv_20")
            .rolling_quantile(quantile=0.8, window_size=252, min_periods=60)
            .over("code")
            .alias("_vol_80pct_252"),
            pl.col("rv_20")
            .rolling_quantile(quantile=0.2, window_size=252, min_periods=60)
            .over("code")
            .alias("_vol_20pct_252"),
        ]
    )
    q = q.with_columns(
        pl.when(pl.col("rv_20") > pl.col("_vol_80pct_252"))
        .then(1)  # High vol regime
        .when(pl.col("rv_20") < pl.col("_vol_20pct_252"))
        .then(-1)  # Low vol regime
        .otherwise(0)  # Normal regime
        .alias("vol_regime")
    )
    # Cross-sectional volatility percentile: how does this stock's vol compare to market today?
    q = q.with_columns(
        (pl.col("rv_20").rank(method="average").over("date") / pl.count().over("date")).alias("cs_vol_percentile")
    )
    # Volatility-adjusted return momentum: higher reward for gains in high vol environments
    q = q.with_columns((pl.col("ret_20d") * pl.col("rv_20")).alias("vol_weighted_mom_20"))
    # Clean up Volatility Regime temp columns (added to drop list below)

    # Session-based features if available
    schema_names = q.collect_schema().names()
    has_morning = "morning_open" in schema_names and "morning_close" in schema_names
    has_afternoon = "afternoon_open" in schema_names and "afternoon_close" in schema_names
    if has_morning:
        q = q.with_columns(
            [
                ((pl.col("morning_close") - pl.col("morning_open")) / pl.col("morning_open")).alias("ret_morning"),
                (pl.col("morning_volume") / pl.col("volume")).alias("morning_vol_share"),
                (pl.col("afternoon_volume") / pl.col("volume")).alias("afternoon_vol_share")
                if has_afternoon
                else pl.lit(None).alias("afternoon_vol_share"),
                (
                    (pl.col("morning_open") - pl.col("close").shift(1).over("code"))
                    / pl.col("close").shift(1).over("code")
                ).alias("overnight_gap"),
                (
                    (pl.col("morning_close") - pl.col("morning_open"))
                    / (pl.col("morning_high") - pl.col("morning_low") + EPS)
                ).alias("morning_trend_eff"),
            ]
        )
    if has_afternoon:
        q = q.with_columns(
            [
                ((pl.col("afternoon_close") - pl.col("afternoon_open")) / pl.col("afternoon_open")).alias(
                    "ret_afternoon"
                ),
                ((pl.col("afternoon_open") - pl.col("morning_close")) / pl.col("morning_close")).alias("lunch_gap")
                if has_morning
                else pl.lit(None).alias("lunch_gap"),
                (
                    (pl.col("afternoon_close") - pl.col("afternoon_open"))
                    / (pl.col("afternoon_high") - pl.col("afternoon_low") + EPS)
                ).alias("afternoon_trend_eff"),
            ]
        )
    if has_morning and has_afternoon:
        q = q.with_columns(
            [
                (pl.col("ret_afternoon") - pl.col("ret_morning")).alias("session_divergence"),
                ((pl.col("morning_high") - pl.col("morning_low")) / (pl.col("morning_close") + EPS)).alias(
                    "range_morning"
                ),
                ((pl.col("afternoon_high") - pl.col("afternoon_low")) / (pl.col("afternoon_close") + EPS)).alias(
                    "range_afternoon"
                ),
                (
                    ((pl.col("morning_high") - pl.col("morning_low")) / (pl.col("morning_close") + EPS))
                    / (((pl.col("afternoon_high") - pl.col("afternoon_low")) / (pl.col("afternoon_close") + EPS)) + EPS)
                ).alias("session_vol_ratio"),
            ]
        )
        # Session stats: win rates and 20-day correlation
        q = q.with_columns(
            [
                pl.col("ret_morning")
                .gt(0)
                .cast(pl.Float64)
                .rolling_mean(window_size=20, min_periods=5)
                .over("code")
                .alias("morning_win_rate_20"),
                pl.col("ret_afternoon")
                .gt(0)
                .cast(pl.Float64)
                .rolling_mean(window_size=20, min_periods=5)
                .over("code")
                .alias("afternoon_win_rate_20"),
            ]
        )
        q = q.with_columns(
            pl.when((pl.col("morning_volume") > 0) & (pl.col("afternoon_volume").is_not_null()))
            .then(pl.col("afternoon_volume") / pl.col("morning_volume"))
            .otherwise(None)
            .alias("session_volume_ratio"),
        )
    else:
        q = q.with_columns(
            [
                pl.lit(None).alias("ret_morning"),
                pl.lit(None).alias("ret_afternoon"),
                pl.lit(None).alias("lunch_gap"),
                pl.lit(None).alias("session_divergence"),
                pl.lit(None).alias("morning_vol_share"),
                pl.lit(None).alias("afternoon_vol_share"),
                pl.lit(None).alias("overnight_gap"),
                pl.lit(None).alias("morning_trend_eff"),
                pl.lit(None).alias("afternoon_trend_eff"),
                pl.lit(None).alias("range_morning"),
                pl.lit(None).alias("range_afternoon"),
                pl.lit(None).alias("session_vol_ratio"),
                pl.lit(None).alias("session_volume_ratio"),
                pl.lit(None).alias("morning_win_rate_20"),
                pl.lit(None).alias("afternoon_win_rate_20"),
                pl.lit(None).alias("session_corr_20"),
            ]
        )

    # Gap magnitude and direction from overnight_gap
    q = q.with_columns(
        [
            pl.when(pl.col("overnight_gap").is_null())
            .then(None)
            .otherwise(pl.col("overnight_gap").abs())
            .alias("gap_abs"),
            pl.when(pl.col("overnight_gap").is_null())
            .then(None)
            .when(pl.col("overnight_gap") > 0)
            .then(1.0)
            .when(pl.col("overnight_gap") < 0)
            .then(-1.0)
            .otherwise(0.0)
            .alias("gap_dir"),
        ]
    )

    # Limit flags and rolling counts
    q = q.with_columns(
        [
            (pl.col("upper_limit") == "1").cast(pl.Int8).alias("is_limit_up"),
            (pl.col("lower_limit") == "1").cast(pl.Int8).alias("is_limit_down"),
            (pl.col("adjustment_factor") != 1.0).cast(pl.Int8).alias("is_split"),
        ]
    )
    q = q.with_columns(
        [
            pl.col("is_limit_up").rolling_sum(window_size=20, min_periods=1).over("code").alias("limit_up_count_20"),
            pl.col("is_limit_down")
            .rolling_sum(window_size=20, min_periods=1)
            .over("code")
            .alias("limit_down_count_20"),
        ]
    )
    # days since last limit up/down event per code
    q = q.with_columns(pl.arange(0, pl.len()).over("code").alias("_limit_idx"))
    q = q.with_columns(
        [
            pl.when(pl.col("is_limit_up") == 1)
            .then(pl.col("_limit_idx"))
            .otherwise(None)
            .forward_fill()
            .over("code")
            .fill_null(-1)
            .alias("_last_limit_up_idx"),
            pl.when(pl.col("is_limit_down") == 1)
            .then(pl.col("_limit_idx"))
            .otherwise(None)
            .forward_fill()
            .over("code")
            .fill_null(-1)
            .alias("_last_limit_down_idx"),
        ]
    )
    q = q.with_columns(
        [
            pl.when(pl.col("_last_limit_up_idx") < 0)
            .then(None)
            .otherwise((pl.col("_limit_idx") - pl.col("_last_limit_up_idx")).cast(pl.Int32))
            .alias("days_since_limit_up"),
            pl.when(pl.col("_last_limit_down_idx") < 0)
            .then(None)
            .otherwise((pl.col("_limit_idx") - pl.col("_last_limit_down_idx")).cast(pl.Int32))
            .alias("days_since_limit_down"),
        ]
    )

    # Normalize NaN -> Null for downstream consistency (base features)
    nan_cols = [
        "macd",
        "macd_signal",
        "macd_hist",
        "adx_14",
        "rsi_14",
        "bb_mid_20_2",
        "bb_upper_20_2",
        "bb_lower_20_2",
        "bb_percent_b_20_2",
        "bb_width_20_2",
        "parkinson_20",
        "gk_20",
        "vol_20d",
        "rv_ratio",
    ]
    existing_nan = set(q.collect_schema().names())
    q = q.with_columns([pl.col(c).fill_nan(None) for c in nan_cols if c in existing_nan])

    # Drop temp columns
    q = q.drop(
        [
            "_log_vol",
            "_log_val",
            "_close_ta_ff",
            "_high_ta_ff",
            "_low_ta_ff",
            "_cand_idx",
            "_last_bear_idx",
            "_last_bull_idx",
            "_delta",
            "_gain",
            "_loss",
            "_avg_gain",
            "_avg_loss",
            "_std_20",
            "_up_move",
            "_down_move",
            "_plus_dm",
            "_minus_dm",
            "_atr14_smooth",
            "_plus_dm_smooth",
            "_minus_dm_smooth",
            "_plus_di",
            "_minus_di",
            "_dx",
            "mean_vol_5",
            "mean_vol_20",
            "mean_val_5",
            "mean_val_20",
            "ma_5",
            "ma_20",
            "ma_60",
            "_true_range",
            "_min_20",
            "_max_20",
            "_low_252",
            "_high_252",
            "_limit_idx",
            "_last_limit_up_idx",
            "_last_limit_down_idx",
            # Phase 3.1 Risk/Drawdown temp columns
            "_rolling_high_20",
            "_rolling_high_60",
            "_is_at_high_20",
            "_uw_idx",
            "_last_high_idx",
            # Phase 3.2 Market Alpha temp columns
            "_market_mean_ret_1d",
            "_market_mean_ret_5d",
            "_market_mean_ret_20d",
            "_alpha_std_20",
            # Phase 3.3 Volatility Regime temp columns
            "_vol_80pct_252",
            "_vol_20pct_252",
        ]
    )

    # Load breakdown for the same core range (no warmup needed)
    bd_arrow = con.execute(
        """
        SELECT date, code,
            long_sell_value, short_sell_without_margin_value,
            margin_sell_new_value, margin_sell_close_value,
            long_buy_value, margin_buy_new_value, margin_buy_close_value,
            long_sell_volume, short_sell_without_margin_volume,
            margin_sell_new_volume, margin_sell_close_volume,
            long_buy_volume, margin_buy_new_volume, margin_buy_close_volume
        FROM breakdown
        WHERE date BETWEEN ? AND ?
        """,
        [start, end],
    ).fetch_arrow_table()
    if bd_arrow.num_rows == 0:
        bd = pl.DataFrame(
            schema={
                "date": pl.Date,
                "code": pl.Utf8,
                "breakdown_buy_ratio": pl.Float64,
                "breakdown_sell_ratio": pl.Float64,
                "breakdown_volume_ratio": pl.Float64,
            }
        ).lazy()
    else:
        bd = pl.from_arrow(bd_arrow).lazy().sort(["code", "date"])
        value_cols = [
            "long_sell_value",
            "short_sell_without_margin_value",
            "margin_sell_new_value",
            "margin_sell_close_value",
            "long_buy_value",
            "margin_buy_new_value",
            "margin_buy_close_value",
        ]
        volume_cols = [
            "long_sell_volume",
            "short_sell_without_margin_volume",
            "margin_sell_new_volume",
            "margin_sell_close_volume",
            "long_buy_volume",
            "margin_buy_new_volume",
            "margin_buy_close_volume",
        ]
        # Cast to Float64 but keep NULLs (don't fill with 0.0 - preserves data quality)
        # NULL means "no breakdown data available", 0.0 means "actual zero value"
        bd = bd.with_columns([pl.col(c).cast(pl.Float64) for c in value_cols + volume_cols])

        # Add has_breakdown flag: True if any breakdown column has non-null data
        bd = bd.with_columns(
            pl.any_horizontal([pl.col(c).is_not_null() for c in value_cols + volume_cols]).alias("has_breakdown")
        )

        bd = bd.with_columns(
            [
                (
                    (pl.col("long_buy_value") + pl.col("margin_buy_new_value") + pl.col("margin_buy_close_value"))
                    / (
                        pl.col("long_buy_value")
                        + pl.col("margin_buy_new_value")
                        + pl.col("margin_buy_close_value")
                        + pl.col("long_sell_value")
                        + pl.col("short_sell_without_margin_value")
                        + pl.col("margin_sell_new_value")
                        + pl.col("margin_sell_close_value")
                        + EPS
                    )
                ).alias("breakdown_buy_ratio"),
                (
                    (pl.col("long_buy_volume") + pl.col("margin_buy_new_volume") + pl.col("margin_buy_close_volume"))
                    / (
                        pl.col("long_buy_volume")
                        + pl.col("margin_buy_new_volume")
                        + pl.col("margin_buy_close_volume")
                        + pl.col("long_sell_volume")
                        + pl.col("short_sell_without_margin_volume")
                        + pl.col("margin_sell_new_volume")
                        + pl.col("margin_sell_close_volume")
                        + EPS
                    )
                ).alias("breakdown_volume_ratio"),
            ]
        ).with_columns([(1.0 - pl.col("breakdown_buy_ratio")).alias("breakdown_sell_ratio")])

        # Base aggregates and simple ratios (no rolling)
        bd = bd.with_columns(
            [
                (
                    pl.col("long_sell_value")
                    + pl.col("short_sell_without_margin_value")
                    + pl.col("margin_sell_new_value")
                    + pl.col("margin_sell_close_value")
                ).alias("total_sell_value"),
                (pl.col("long_buy_value") + pl.col("margin_buy_new_value") + pl.col("margin_buy_close_value")).alias(
                    "total_buy_value"
                ),
                (
                    pl.col("long_sell_volume")
                    + pl.col("short_sell_without_margin_volume")
                    + pl.col("margin_sell_new_volume")
                    + pl.col("margin_sell_close_volume")
                ).alias("total_sell_volume"),
                (pl.col("long_buy_volume") + pl.col("margin_buy_new_volume") + pl.col("margin_buy_close_volume")).alias(
                    "total_buy_volume"
                ),
                (pl.col("short_sell_without_margin_value") + pl.col("margin_sell_new_value")).alias("short_flow_value"),
                (pl.col("short_sell_without_margin_volume") + pl.col("margin_sell_new_volume")).alias(
                    "short_flow_volume"
                ),
            ]
        )
        bd = bd.with_columns(
            [
                pl.when(pl.col("total_sell_value") > 0)
                .then(pl.col("short_sell_without_margin_value") / pl.col("total_sell_value"))
                .otherwise(None)
                .alias("inst_short_ratio_value"),
                pl.when(pl.col("total_sell_value") > 0)
                .then(pl.col("long_sell_value") / pl.col("total_sell_value"))
                .otherwise(None)
                .alias("sell_long_ratio"),
                pl.when(pl.col("total_sell_value") > 0)
                .then(pl.col("short_flow_value") / pl.col("total_sell_value"))
                .otherwise(None)
                .alias("sell_short_ratio"),
                pl.when(pl.col("total_sell_value") > 0)
                .then(pl.col("margin_sell_close_value") / pl.col("total_sell_value"))
                .otherwise(None)
                .alias("sell_margin_close_ratio"),
                pl.when((pl.col("long_buy_value") + pl.col("margin_buy_new_value")) > 0)
                .then(pl.col("margin_buy_new_value") / (pl.col("long_buy_value") + pl.col("margin_buy_new_value")))
                .otherwise(None)
                .alias("retail_lev_ratio_value"),
                pl.when(pl.col("total_buy_value") > 0)
                .then(pl.col("margin_buy_close_value") / pl.col("total_buy_value"))
                .otherwise(None)
                .alias("margin_capitulation_ratio"),
                pl.when(pl.col("total_buy_value") > 0)
                .then(pl.col("long_buy_value") / pl.col("total_buy_value"))
                .otherwise(None)
                .alias("buy_long_ratio"),
                pl.when(pl.col("total_buy_value") > 0)
                .then(pl.col("margin_buy_new_value") / pl.col("total_buy_value"))
                .otherwise(None)
                .alias("buy_margin_new_ratio"),
                pl.when(pl.col("total_buy_value") > 0)
                .then(pl.col("margin_buy_close_value") / pl.col("total_buy_value"))
                .otherwise(None)
                .alias("buy_margin_close_ratio"),
                pl.when(pl.col("total_buy_value") > 0)
                .then((pl.col("margin_buy_new_value") + pl.col("margin_buy_close_value")) / pl.col("total_buy_value"))
                .otherwise(None)
                .alias("credit_buy_share"),
                pl.when(pl.col("total_sell_value") > 0)
                .then(
                    (pl.col("margin_sell_new_value") + pl.col("margin_sell_close_value")) / pl.col("total_sell_value")
                )
                .otherwise(None)
                .alias("credit_sell_share"),
                pl.when((pl.col("total_buy_value") + pl.col("total_sell_value")) > 0)
                .then(
                    (
                        pl.col("margin_buy_new_value")
                        + pl.col("margin_buy_close_value")
                        + pl.col("margin_sell_new_value")
                        + pl.col("margin_sell_close_value")
                    )
                    / (pl.col("total_buy_value") + pl.col("total_sell_value"))
                )
                .otherwise(None)
                .alias("credit_turnover_share"),
                (pl.col("total_buy_value") - pl.col("total_sell_value")).alias("net_flow_value"),
                pl.when((pl.col("total_buy_value") + pl.col("total_sell_value")) > 0)
                .then(
                    (pl.col("total_buy_value") - pl.col("total_sell_value"))
                    / (pl.col("total_buy_value") + pl.col("total_sell_value"))
                )
                .otherwise(None)
                .alias("flow_imbalance"),
                (pl.col("long_buy_value") - pl.col("long_sell_value")).alias("net_long_value"),
                pl.when((pl.col("long_buy_value") + pl.col("long_sell_value")) > 0)
                .then(
                    (pl.col("long_buy_value") - pl.col("long_sell_value"))
                    / (pl.col("long_buy_value") + pl.col("long_sell_value"))
                )
                .otherwise(None)
                .alias("net_long_ratio"),
                (pl.col("margin_buy_new_value") - pl.col("margin_buy_close_value")).alias("net_margin_flow_value"),
                (pl.col("margin_buy_new_value") - pl.col("margin_sell_new_value")).alias("net_margin_new_value"),
                pl.when((pl.col("margin_buy_new_value") + pl.col("margin_sell_new_value")) > 0)
                .then(
                    (pl.col("margin_buy_new_value") - pl.col("margin_sell_new_value"))
                    / (pl.col("margin_buy_new_value") + pl.col("margin_sell_new_value"))
                )
                .otherwise(None)
                .alias("margin_new_sentiment"),
                pl.when(
                    (
                        pl.col("long_buy_value")
                        + pl.col("margin_buy_new_value")
                        + pl.col("long_sell_value")
                        + pl.col("short_flow_value")
                    )
                    > 0
                )
                .then(
                    (pl.col("long_buy_value") + pl.col("margin_buy_new_value"))
                    / (
                        pl.col("long_buy_value")
                        + pl.col("margin_buy_new_value")
                        + pl.col("long_sell_value")
                        + pl.col("short_flow_value")
                    )
                )
                .otherwise(None)
                .alias("bull_bear_ratio"),
                pl.when(pl.col("total_sell_volume") > 0)
                .then(pl.col("short_flow_volume") / pl.col("total_sell_volume"))
                .otherwise(None)
                .alias("sell_short_ratio_vol"),
                pl.when(pl.col("total_sell_volume") > 0)
                .then(pl.col("margin_sell_close_volume") / pl.col("total_sell_volume"))
                .otherwise(None)
                .alias("sell_margin_close_ratio_vol"),
                pl.when(pl.col("total_buy_volume") > 0)
                .then(pl.col("margin_buy_new_volume") / pl.col("total_buy_volume"))
                .otherwise(None)
                .alias("buy_margin_new_ratio_vol"),
                pl.when(pl.col("total_buy_volume") > 0)
                .then(pl.col("margin_buy_close_volume") / pl.col("total_buy_volume"))
                .otherwise(None)
                .alias("buy_margin_close_ratio_vol"),
            ]
        )
        # Time-series breakdown trends
        bd = bd.with_columns(
            [
                pl.col("sell_short_ratio")
                .rolling_mean(window_size=5, min_periods=3)
                .over("code")
                .alias("short_ratio_ma_5"),
                pl.col("sell_short_ratio")
                .rolling_mean(window_size=20, min_periods=5)
                .over("code")
                .alias("short_ratio_ma_20"),
                pl.col("buy_long_ratio")
                .rolling_mean(window_size=5, min_periods=3)
                .over("code")
                .alias("long_buy_ratio_ma_5"),
                pl.col("buy_long_ratio")
                .rolling_mean(window_size=20, min_periods=5)
                .over("code")
                .alias("long_buy_ratio_ma_20"),
                pl.col("margin_capitulation_ratio")
                .rolling_mean(window_size=5, min_periods=3)
                .over("code")
                .alias("margin_capitulation_ma5"),
            ]
        )
        bd = bd.with_columns(
            [
                (pl.col("sell_short_ratio") - pl.col("short_ratio_ma_5")).alias("short_ratio_dev_5"),
                (pl.col("sell_short_ratio") - pl.col("short_ratio_ma_20")).alias("short_ratio_dev_20"),
                (pl.col("buy_long_ratio") - pl.col("long_buy_ratio_ma_5")).alias("long_buy_ratio_dev_5"),
                (pl.col("buy_long_ratio") - pl.col("long_buy_ratio_ma_20")).alias("long_buy_ratio_dev_20"),
                (pl.col("sell_short_ratio") - pl.col("sell_short_ratio").shift(1).over("code")).alias(
                    "short_ratio_chg_1"
                ),
                (pl.col("sell_short_ratio") - pl.col("sell_short_ratio").shift(5).over("code")).alias(
                    "short_ratio_chg_5"
                ),
                (pl.col("buy_long_ratio") - pl.col("buy_long_ratio").shift(1).over("code")).alias(
                    "long_buy_ratio_chg_1"
                ),
                (pl.col("buy_long_ratio") - pl.col("buy_long_ratio").shift(5).over("code")).alias(
                    "long_buy_ratio_chg_5"
                ),
                (pl.col("margin_capitulation_ratio") - pl.col("margin_capitulation_ma5")).alias(
                    "margin_capitulation_spike"
                ),
            ]
        )
        bd = bd.with_columns(
            [
                (
                    (
                        pl.col("sell_short_ratio")
                        - pl.col("sell_short_ratio").rolling_mean(window_size=60, min_periods=10).over("code")
                    )
                    / (pl.col("sell_short_ratio").rolling_std(window_size=60, min_periods=10).over("code") + EPS)
                ).alias("short_ratio_z_60"),
                (
                    (
                        pl.col("buy_long_ratio")
                        - pl.col("buy_long_ratio").rolling_mean(window_size=60, min_periods=10).over("code")
                    )
                    / (pl.col("buy_long_ratio").rolling_std(window_size=60, min_periods=10).over("code") + EPS)
                ).alias("long_buy_ratio_z_60"),
                (
                    (
                        pl.col("flow_imbalance")
                        - pl.col("flow_imbalance").rolling_mean(window_size=60, min_periods=10).over("code")
                    )
                    / (pl.col("flow_imbalance").rolling_std(window_size=60, min_periods=10).over("code") + EPS)
                ).alias("flow_imbalance_z_60"),
            ]
        )
        # Margin overhang (volume-based)
        bd = bd.with_columns(
            [
                (pl.col("margin_buy_new_volume") - pl.col("margin_sell_close_volume")).alias("delta_margin_long_vol"),
                (pl.col("margin_sell_new_volume") - pl.col("margin_buy_close_volume")).alias("delta_margin_short_vol"),
            ]
        )
        bd = bd.with_columns(
            [
                pl.col("delta_margin_long_vol")
                .rolling_sum(window_size=60, min_periods=5)
                .over("code")
                .alias("cum_margin_long_60"),
                pl.col("delta_margin_short_vol")
                .rolling_sum(window_size=60, min_periods=5)
                .over("code")
                .alias("cum_margin_short_60"),
            ]
        )

    bd_cols = [
        "date",
        "code",
        "breakdown_buy_ratio",
        "breakdown_sell_ratio",
        "breakdown_volume_ratio",
        "inst_short_ratio_value",
        "sell_long_ratio",
        "sell_short_ratio",
        "sell_margin_close_ratio",
        "buy_long_ratio",
        "buy_margin_new_ratio",
        "buy_margin_close_ratio",
        "retail_lev_ratio_value",
        "margin_capitulation_ratio",
        "margin_capitulation_ma5",
        "margin_capitulation_spike",
        "credit_buy_share",
        "credit_sell_share",
        "credit_turnover_share",
        "net_flow_value",
        "flow_imbalance",
        "net_long_value",
        "net_long_ratio",
        "net_margin_flow_value",
        "net_margin_new_value",
        "margin_new_sentiment",
        "bull_bear_ratio",
        "short_flow_value",
        "sell_short_ratio_vol",
        "sell_margin_close_ratio_vol",
        "buy_margin_new_ratio_vol",
        "buy_margin_close_ratio_vol",
        "delta_margin_long_vol",
        "delta_margin_short_vol",
        "cum_margin_long_60",
        "cum_margin_short_60",
        "short_ratio_ma_5",
        "short_ratio_ma_20",
        "short_ratio_dev_5",
        "short_ratio_dev_20",
        "short_ratio_chg_1",
        "short_ratio_chg_5",
        "long_buy_ratio_ma_5",
        "long_buy_ratio_ma_20",
        "long_buy_ratio_dev_5",
        "long_buy_ratio_dev_20",
        "long_buy_ratio_chg_1",
        "long_buy_ratio_chg_5",
        "short_ratio_z_60",
        "long_buy_ratio_z_60",
        "flow_imbalance_z_60",
    ]

    out = q.join(bd.select(bd_cols), on=["date", "code"], how="left")
    out = out.with_columns(
        [
            pl.when(
                pl.col("breakdown_buy_ratio").is_null()
                & pl.col("breakdown_sell_ratio").is_null()
                & pl.col("breakdown_volume_ratio").is_null()
            )
            .then(0)
            .otherwise(1)
            .cast(pl.Int8)
            .alias("has_breakdown"),
            pl.when((pl.col("is_limit_up") == 1) | (pl.col("is_limit_down") == 1))
            .then(pl.col("volume") / (pl.col("adv_vol_20") + EPS))
            .otherwise(None)
            .alias("limit_volume_ratio"),
        ]
    )

    # Overhang and flow composites
    out = out.with_columns(
        [
            (pl.col("cum_margin_short_60") / (pl.col("adv_vol_20") + EPS)).alias("short_overhang_days"),
            (pl.col("cum_margin_long_60") / (pl.col("adv_vol_20") + EPS)).alias("long_overhang_days"),
        ]
    )
    out = out.with_columns(
        [
            (pl.col("short_overhang_days") * pl.when(pl.col("ret_5d") > 0).then(pl.col("ret_5d")).otherwise(0.0)).alias(
                "short_squeeze_risk"
            ),
            (pl.col("long_overhang_days") * pl.when(pl.col("ret_5d") < 0).then(-pl.col("ret_5d")).otherwise(0.0)).alias(
                "long_liquidation_risk"
            ),
            pl.when(pl.col("ret_1d") > 0)
            .then(1.0)
            .when(pl.col("ret_1d") < 0)
            .then(-1.0)
            .otherwise(0.0)
            .alias("_ret1d_sign"),
        ]
    )
    out = out.with_columns(
        [
            (pl.col("_ret1d_sign") * pl.col("net_flow_value")).alias("flow_price_alignment"),
            (pl.col("_ret1d_sign") * pl.col("short_flow_value")).alias("short_price_alignment"),
            (pl.col("net_flow_value") / (pl.col("dollar_volume") + EPS)).alias("flow_intensity"),
            (pl.col("short_flow_value") / (pl.col("dollar_volume") + EPS)).alias("short_intensity"),
            (pl.col("net_margin_flow_value") / (pl.col("dollar_volume") + EPS)).alias("margin_flow_intensity"),
        ]
    )
    out = out.with_columns(
        [
            (pl.col("flow_intensity") * pl.col("rv_5")).alias("flow_vol_combo"),
            (pl.col("flow_intensity") / (pl.col("rv_5") + EPS)).alias("flow_vol_ratio"),
        ]
    )

    # Sector-level short selling features (broadcast by sector33_code)
    # Attach sector33_code as-of metadata from listed_info for the join.
    sector_meta_arrow = con.execute(
        """
        SELECT date, code, sector33_code
        FROM listed_info
        WHERE date BETWEEN ? AND ?
        """,
        [start, end],
    ).fetch_arrow_table()
    sector_meta = pl.from_arrow(sector_meta_arrow)
    if not sector_meta.is_empty():
        out = out.join(sector_meta.lazy(), on=["date", "code"], how="left")

    ss_arrow = con.execute(
        """
        SELECT
            date,
            sector33_code,
            selling_excluding_short_selling_turnover_value,
            short_selling_with_restrictions_turnover_value,
            short_selling_without_restrictions_turnover_value
        FROM short_selling
        WHERE date BETWEEN ? AND ?
        """,
        [start, end],
    ).fetch_arrow_table()
    ss = pl.from_arrow(ss_arrow)
    if not ss.is_empty() and "sector33_code" in out.collect_schema().names():
        ss = ss.sort(["sector33_code", "date"])
        ss = ss.with_columns(
            [
                (
                    pl.col("short_selling_with_restrictions_turnover_value")
                    + pl.col("short_selling_without_restrictions_turnover_value")
                ).alias("_short_total"),
                (
                    pl.col("selling_excluding_short_selling_turnover_value")
                    + pl.col("short_selling_with_restrictions_turnover_value")
                    + pl.col("short_selling_without_restrictions_turnover_value")
                ).alias("_total_turnover"),
            ]
        )
        ss = ss.with_columns((pl.col("_short_total") / (pl.col("_total_turnover") + EPS)).alias("sector_short_ratio"))
        ss = ss.with_columns(
            (
                pl.col("sector_short_ratio")
                - pl.col("sector_short_ratio").rolling_mean(window_size=5, min_periods=1).over("sector33_code")
            ).alias("sector_short_trend")
        )
        ss = ss.select(["date", "sector33_code", "sector_short_ratio", "sector_short_trend"])
        out = out.join(ss.lazy(), on=["date", "sector33_code"], how="left")
    else:
        out = out.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("sector_short_ratio"),
                pl.lit(None).cast(pl.Float64).alias("sector_short_trend"),
            ]
        )

    # Daily margin interest-derived features (effective from next trading day)
    dmi = compute_daily_margin_features(con, start=start, end=end)
    if not dmi.is_empty():
        out = out.join(dmi.lazy(), on=["date", "code"], how="left")
    else:
        out = out.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("dmi_short_balance"),
                pl.lit(None).cast(pl.Float64).alias("dmi_long_balance"),
                pl.lit(None).cast(pl.Float64).alias("dmi_short_balance_listed_ratio"),
                pl.lit(None).cast(pl.Utf8).alias("dmi_regulation_code"),
                pl.lit(None).cast(pl.Utf8).alias("precaution_by_jsf"),
                pl.lit(None).cast(pl.Utf8).alias("restricted_by_jsf"),
            ]
        )

    out = out.with_columns(
        pl.when(pl.col("adv_vol_20").is_null() | (pl.col("adv_vol_20") <= 0))
        .then(None)
        .otherwise(pl.col("dmi_short_balance") / (pl.col("adv_vol_20") + EPS))
        .alias("dmi_short_days_to_cover_20")
    )
    out = out.with_columns(
        (
            (pl.col("dmi_short_balance") - pl.col("dmi_short_balance").shift(5).over("code"))
            / (pl.col("dmi_short_balance").shift(5).over("code").abs() + EPS)
        ).alias("_dmi_short_chg_5")
    )
    out = out.with_columns(
        [
            (
                (
                    pl.col("dmi_short_balance_listed_ratio")
                    - pl.col("dmi_short_balance_listed_ratio").rolling_mean(window_size=60, min_periods=10).over("code")
                )
                / (
                    pl.col("dmi_short_balance_listed_ratio").rolling_std(window_size=60, min_periods=10).over("code")
                    + EPS
                )
            ).alias("_dmi_ratio_z"),
            (
                (
                    pl.col("dmi_short_days_to_cover_20")
                    - pl.col("dmi_short_days_to_cover_20").rolling_mean(window_size=60, min_periods=10).over("code")
                )
                / (pl.col("dmi_short_days_to_cover_20").rolling_std(window_size=60, min_periods=10).over("code") + EPS)
            ).alias("_dmi_dtc_z"),
            (
                (
                    pl.col("_dmi_short_chg_5")
                    - pl.col("_dmi_short_chg_5").rolling_mean(window_size=60, min_periods=10).over("code")
                )
                / (pl.col("_dmi_short_chg_5").rolling_std(window_size=60, min_periods=10).over("code") + EPS)
            ).alias("_dmi_chg_z"),
        ]
    )
    dmi_components = ["_dmi_ratio_z", "_dmi_dtc_z", "_dmi_chg_z"]
    out = out.with_columns(
        pl.when(pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Float64) for c in dmi_components]) > 0)
        .then(
            pl.sum_horizontal([pl.col(c).fill_null(0.0) for c in dmi_components])
            / pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Float64) for c in dmi_components])
        )
        .otherwise(None)
        .clip(-5.0, 5.0)
        .alias("dmi_short_squeeze_score")
    )
    reg_codes = ["003", "004", "005", "006", "3", "4", "5", "6"]
    out = out.with_columns(
        [
            pl.when(pl.col("dmi_regulation_code").cast(pl.Utf8).is_in(reg_codes))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("dmi_has_margin_regulation"),
            (
                (pl.col("precaution_by_jsf").is_not_null() & (pl.col("precaution_by_jsf") != "0"))
                | (pl.col("restricted_by_jsf").is_not_null() & (pl.col("restricted_by_jsf") != "0"))
            )
            .cast(pl.Int8)
            .alias("dmi_is_jsf_flagged"),
        ]
    )

    # Availability flags for downstream filtering
    out = out.with_columns(
        [
            pl.when(pl.col("ret_1d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_price_core"),
            pl.when(pl.col("fwd_ret_1d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_target_1d"),
            pl.when(pl.col("fwd_ret_5d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_target_5d"),
            pl.when(pl.col("fwd_ret_10d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_target_10d"),
            pl.when(pl.col("fwd_ret_20d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_target_20d"),
            pl.when(pl.col("ret_morning").is_not_null() & pl.col("ret_afternoon").is_not_null())
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("has_sessions"),
            pl.when(pl.col("ret_60d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_long_window_60"),
        ]
    )

    # Cross-sectional percentiles (per date)
    # Minimum population threshold for reliable CS statistics
    MIN_CS_POPULATION = 500

    out = out.with_columns(
        [
            pl.count().over("date").alias("_n_date"),
            pl.col("ret_1d").rank(method="average").over("date").alias("_rank_ret_1d"),
            pl.col("ret_5d").rank(method="average").over("date").alias("_rank_ret_5d"),
            pl.col("dollar_volume").rank(method="average").over("date").alias("_rank_dollar_volume"),
            pl.col("sell_short_ratio").rank(method="average").over("date").alias("_rank_sell_short_ratio"),
            pl.col("flow_imbalance").rank(method="average").over("date").alias("_rank_flow_imbalance"),
            pl.col("credit_turnover_share").rank(method="average").over("date").alias("_rank_credit_turnover_share"),
            pl.col("short_overhang_days").rank(method="average").over("date").alias("_rank_short_overhang_days"),
        ]
    )

    # Add flag for sufficient CS population
    out = out.with_columns((pl.col("_n_date") >= MIN_CS_POPULATION).alias("_cs_population_ok"))

    # Helper function to apply CS population filter
    def cs_with_filter(expr: pl.Expr, alias: str) -> pl.Expr:
        """Apply CS feature calculation only when population >= MIN_CS_POPULATION."""
        return pl.when(pl.col("_cs_population_ok")).then(expr).otherwise(None).alias(alias)

    out = out.with_columns(
        [
            cs_with_filter(pl.col("_rank_ret_1d") / (pl.col("_n_date") - 1 + EPS), "cs_pct_ret_1d"),
            cs_with_filter(pl.col("_rank_dollar_volume") / (pl.col("_n_date") - 1 + EPS), "cs_pct_dollar_volume"),
            cs_with_filter(pl.col("_rank_sell_short_ratio") / (pl.col("_n_date") - 1 + EPS), "cs_pct_sell_short_ratio"),
            cs_with_filter(pl.col("_rank_flow_imbalance") / (pl.col("_n_date") - 1 + EPS), "cs_pct_flow_imbalance"),
            cs_with_filter(
                pl.col("_rank_credit_turnover_share") / (pl.col("_n_date") - 1 + EPS),
                "cs_pct_credit_turnover_share",
            ),
            cs_with_filter(
                pl.col("_rank_short_overhang_days") / (pl.col("_n_date") - 1 + EPS), "cs_pct_short_overhang_days"
            ),
            # rank versions (0-1) kept併記
            cs_with_filter(pl.col("_rank_ret_1d") / (pl.col("_n_date") - 1 + EPS), "cs_rank_ret_1d"),
            cs_with_filter(pl.col("_rank_ret_5d") / (pl.col("_n_date") - 1 + EPS), "cs_rank_ret_5d"),
            cs_with_filter(pl.col("_rank_sell_short_ratio") / (pl.col("_n_date") - 1 + EPS), "cs_rank_short_ratio"),
        ]
    )

    # Cross-sectional Z-scores per date (with population filter)
    out = out.with_columns(
        [
            cs_with_filter(
                (pl.col("ret_1d") - pl.col("ret_1d").mean().over("date")) / (pl.col("ret_1d").std().over("date") + EPS),
                "cs_z_ret_1d",
            ),
            cs_with_filter(
                (pl.col("vol_z_20") - pl.col("vol_z_20").mean().over("date"))
                / (pl.col("vol_z_20").std().over("date") + EPS),
                "cs_z_vol_z_20",
            ),
            cs_with_filter(
                (pl.col("sell_short_ratio") - pl.col("sell_short_ratio").mean().over("date"))
                / (pl.col("sell_short_ratio").std().over("date") + EPS),
                "cs_z_short_ratio",
            ),
            cs_with_filter(
                (pl.col("flow_imbalance") - pl.col("flow_imbalance").mean().over("date"))
                / (pl.col("flow_imbalance").std().over("date") + EPS),
                "cs_z_flow_imbalance",
            ),
        ]
    )

    # Flow run lengths (consecutive days of positive/negative flow_imbalance per code)
    out = out.with_columns(pl.arange(0, pl.len()).over("code").alias("_idx"))
    out = out.with_columns(
        [
            pl.when(pl.col("flow_imbalance") <= 0)
            .then(pl.col("_idx"))
            .otherwise(None)
            .forward_fill()
            .over("code")
            .fill_null(-1)
            .alias("_last_nonpos_idx"),
            pl.when(pl.col("flow_imbalance") >= 0)
            .then(pl.col("_idx"))
            .otherwise(None)
            .forward_fill()
            .over("code")
            .fill_null(-1)
            .alias("_last_nonneg_idx"),
        ]
    )
    out = out.with_columns(
        [
            pl.when(pl.col("flow_imbalance") > 0)
            .then((pl.col("_idx") - pl.col("_last_nonpos_idx")).cast(pl.Int32))
            .otherwise(pl.lit(0))
            .alias("pos_flow_run_len"),
            pl.when(pl.col("flow_imbalance") < 0)
            .then((pl.col("_idx") - pl.col("_last_nonneg_idx")).cast(pl.Int32))
            .otherwise(pl.lit(0))
            .alias("neg_flow_run_len"),
        ]
    )

    # Availability flags for downstream filtering
    out = out.with_columns(
        [
            pl.when(pl.col("ret_1d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_price_core"),
            pl.when(pl.col("ret_60d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_long_window_60"),
            pl.when(pl.col("fwd_ret_1d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_target_1d"),
            pl.when(pl.col("fwd_ret_5d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_target_5d"),
            pl.when(pl.col("fwd_ret_10d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_target_10d"),
            pl.when(pl.col("fwd_ret_20d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_target_20d"),
            pl.when(pl.col("ret_morning").is_not_null() & pl.col("ret_afternoon").is_not_null())
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("has_sessions"),
        ]
    )

    # Short selling positions-derived features (event-driven)
    ssp = compute_short_positions_features(con, start=start, end=end)
    if not ssp.is_empty():
        out = out.join(ssp.lazy(), on=["date", "code"], how="left")
    else:
        out = out.with_columns(
            [
                pl.lit(None).cast(pl.Float64).alias("ssp_disclosed_short_ratio"),
                pl.lit(0).cast(pl.Int8).alias("ssp_short_disclosure_flag_recent"),
            ]
        )

    # Align to FEATURES_DAILY_COLUMNS schema; any missing columns become null.
    schema_cols = [name for name, _ in FEATURES_DAILY_COLUMNS]
    existing_cols = set(out.collect_schema().names())
    out = out.select([pl.col(c) if c in existing_cols else pl.lit(None).alias(c) for c in schema_cols])
    out = out.filter(pl.col("date").is_between(start_dt, end_dt))
    # collect with streaming to reduce peak memory on large ranges; sort eagerly after
    df = out.collect(streaming=True).sort(["code", "date"])

    # Recompute session_corr_20 eagerly per code to ensure it is populated.
    # NOTE:
    # - Lazy-side window expressions for session_corr_20 were being pruned by
    #   the optimiser in some cases, resulting in an all-NULL column.
    # - To make behaviour robust, we recompute the 20-day rolling correlation
    #   here in eager mode after collect(), then re-align flags and schema.
    if "ret_morning" in df.columns and "ret_afternoon" in df.columns:
        from collections import deque

        def _add_session_corr(g: pl.DataFrame) -> pl.DataFrame:
            rm = g["ret_morning"].to_list()
            ra = g["ret_afternoon"].to_list()
            n = len(rm)
            if n == 0:
                return g.with_columns(pl.lit(None).alias("session_corr_20"))
            window: deque[tuple[float | None, float | None]] = deque()
            sx = sy = sx2 = sy2 = sxy = 0.0
            k = 0
            corr: list[float | None] = [None] * n
            eps = 1e-9
            for i in range(n):
                x = rm[i]
                y = ra[i]
                window.append((x, y))
                if x is not None and y is not None:
                    sx += x
                    sy += y
                    sx2 += x * x
                    sy2 += y * y
                    sxy += x * y
                    k += 1
                if len(window) > 20:
                    ox, oy = window.popleft()
                    if ox is not None and oy is not None:
                        sx -= ox
                        sy -= oy
                        sx2 -= ox * ox
                        sy2 -= oy * oy
                        sxy -= ox * oy
                        k -= 1
                if k >= 5:
                    mean_x = sx / k
                    mean_y = sy / k
                    mean_x2 = sx2 / k
                    mean_y2 = sy2 / k
                    mean_xy = sxy / k
                    var_x = max(mean_x2 - mean_x * mean_x, 0.0)
                    var_y = max(mean_y2 - mean_y * mean_y, 0.0)
                    denom = (var_x**0.5 * var_y**0.5) + eps
                    cov = mean_xy - mean_x * mean_y
                    corr[i] = cov / denom
            return g.with_columns(pl.Series("session_corr_20", corr))

        df = df.group_by("code", maintain_order=True).map_groups(_add_session_corr)

    # Recompute availability flags after the eager step to avoid loss through group operations.
    df = df.with_columns(
        [
            pl.when(pl.col("ret_1d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_price_core"),
            pl.when(pl.col("ret_60d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_long_window_60"),
            pl.when(pl.col("fwd_ret_1d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_target_1d"),
            pl.when(pl.col("fwd_ret_5d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_target_5d"),
            pl.when(pl.col("fwd_ret_10d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_target_10d"),
            pl.when(pl.col("fwd_ret_20d").is_null()).then(0).otherwise(1).cast(pl.Int8).alias("has_target_20d"),
            pl.when(pl.col("ret_morning").is_not_null() & pl.col("ret_afternoon").is_not_null())
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("has_sessions"),
        ]
    )

    # Final align to schema order
    schema_cols = [name for name, _ in FEATURES_DAILY_COLUMNS]
    df_cols = set(df.columns)
    df = df.select([pl.col(c) if c in df_cols else pl.lit(None).alias(c) for c in schema_cols])
    return df
