import polars as pl


def _alias_if_exists(df: pl.DataFrame, src: str, dst: str) -> pl.DataFrame:
    return df.with_columns(pl.col(src).alias(dst)) if src in df.columns and dst not in df.columns else df


def apply_spec_aliases(df: pl.DataFrame) -> pl.DataFrame:
    """仕様で絶対の列名を追加エイリアスとして付与（既存列は保持）。"""
    # returns / log_returns
    for k in [1, 5, 10, 20, 60, 120]:
        df = _alias_if_exists(df, f"px_returns_{k}d", f"ret_{k}d")
    for k in [1, 5, 10, 20]:
        df = _alias_if_exists(df, f"px_log_returns_{k}d", f"log_ret_{k}d")

    # volatility
    for k in [5, 10, 20, 60]:
        df = _alias_if_exists(df, f"px_volatility_{k}d", f"vol_{k}d")

    # realized vol (Parkinson 20d)
    df = _alias_if_exists(df, "px_park_vol_20d", "realized_vol_20")

    # price-ema dev
    for k in [5, 10, 20, 60, 200]:
        df = _alias_if_exists(df, f"px_price_ema{k}_dev", f"price_ema{k}_dev")

    # ema gaps and crosses
    df = _alias_if_exists(df, "px_ema_gap_5_20", "ma_gap_5_20")
    df = _alias_if_exists(df, "px_ema_gap_20_60", "ma_gap_20_60")
    df = _alias_if_exists(df, "px_ema_gap_60_200", "ma_gap_60_200")
    df = _alias_if_exists(df, "px_ema_cross_5_20", "ema_cross_5_20")

    # slope
    for k in [10, 20, 60]:
        df = _alias_if_exists(df, f"px_ema{k}_slope_3", f"ema{k}_slope_3")

    # positions and range
    df = _alias_if_exists(df, "px_high_low_ratio", "high_low_ratio")
    df = _alias_if_exists(df, "close_to_high", "close_to_high")
    df = _alias_if_exists(df, "close_to_low", "close_to_low")

    # volume
    df = _alias_if_exists(df, "px_volume_ratio_5", "vol_ratio_5d")
    df = _alias_if_exists(df, "px_volume_ratio_20", "vol_ratio_20d")
    df = _alias_if_exists(df, "px_volume_ma_5", "vol_ma_5")
    df = _alias_if_exists(df, "px_volume_ma_20", "vol_ma_20")
    df = _alias_if_exists(df, "px_turnover_rate", "turnover_rate")
    df = _alias_if_exists(df, "px_dollar_volume", "dollar_volume")

    # pandas-ta like
    df = _alias_if_exists(df, "px_rsi_2", "rsi_2")
    df = _alias_if_exists(df, "px_rsi_14", "rsi_14")
    df = _alias_if_exists(df, "px_rsi_delta", "rsi_delta")
    df = _alias_if_exists(df, "px_macd_signal", "macd_signal")
    df = _alias_if_exists(df, "px_macd_histogram", "macd_histogram")
    df = _alias_if_exists(df, "px_bb_position", "bb_pct_b")
    df = _alias_if_exists(df, "px_bb_width", "bb_bw")
    df = _alias_if_exists(df, "px_atr_14", "atr_14")
    df = _alias_if_exists(df, "px_adx", "adx_14")
    df = _alias_if_exists(df, "px_stoch_k", "stoch_k")

    # targets
    for h in [1, 5, 10, 20]:
        df = _alias_if_exists(df, f"y_{h}d", f"target_{h}d")
        df = _alias_if_exists(df, f"y_{h}d_bin", f"target_{h}d_binary")

    return df


