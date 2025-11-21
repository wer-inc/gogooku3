"""TOPIX派生特徴量生成モジュール（P0: 最小構成）"""

from __future__ import annotations

import logging
from typing import Optional

import polars as pl

from ..utils.asof_join import prepare_snapshot_pl
from ..utils.rolling import roll_mean_safe, roll_std_safe

LOGGER = logging.getLogger(__name__)


def build_topix_features(
    topix_df: pl.DataFrame,
    trading_calendar: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    TOPIX派生特徴量を生成（P0: 最小構成）。

    P0フィーチャー:
    - トレンド・リターン: ret_prev_{1d,5d,20d}, ret_overnight, ret_intraday
    - 移動平均: price_to_sma{20,60}, ma_gap_{5_20,20_60}
    - RSI: rsi_14
    - ボラ・レジーム: atr14, natr14, realized_vol_20, vol_z_252d
    - ドローダウン: drawdown_60d, time_since_peak_60d

    Args:
        topix_df: TOPIX OHLCデータ（Date, Open, High, Low, Close列が必要）
        trading_calendar: 営業日カレンダー（date列が必要、as-of計算用）

    Returns:
        特徴量DataFrame（date, available_ts, topix_* 列）
    """
    if topix_df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")})

    # 列名の正規化
    col_map = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    for old, new in col_map.items():
        if old in topix_df.columns and new not in topix_df.columns:
            topix_df = topix_df.rename({old: new})

    # 型の正規化
    if "date" in topix_df.columns:
        topix_df = topix_df.with_columns(pl.col("date").cast(pl.Date, strict=False))
    else:
        LOGGER.warning("No date column found in TOPIX data")
        return pl.DataFrame(schema={"date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")})

    for col in ["open", "high", "low", "close"]:
        if col in topix_df.columns:
            topix_df = topix_df.with_columns(pl.col(col).cast(pl.Float64, strict=False))
        else:
            LOGGER.warning(f"Required column '{col}' not found in TOPIX data")
            return pl.DataFrame(schema={"date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")})

    # 日付でソート
    topix_df = topix_df.sort("date")

    eps = 1e-9

    # 1. リターン特徴量（shift(1)でリーク防止）
    topix_df = topix_df.with_columns(
        [
            # ret_prev_{1d,5d,20d}
            ((pl.col("close") / (pl.col("close").shift(1) + eps)) - 1.0).alias("topix_ret_prev_1d"),
            ((pl.col("close") / (pl.col("close").shift(5) + eps)) - 1.0).alias("topix_ret_prev_5d"),
            ((pl.col("close") / (pl.col("close").shift(20) + eps)) - 1.0).alias("topix_ret_prev_20d"),
            # ret_overnight = open / close_prev - 1
            ((pl.col("open") / (pl.col("close").shift(1) + eps)) - 1.0).alias("topix_ret_overnight"),
            # ret_intraday = close / open - 1
            ((pl.col("close") / (pl.col("open") + eps)) - 1.0).alias("topix_ret_intraday"),
        ]
    )

    # 2. 移動平均特徴量
    topix_df = topix_df.with_columns(
        [
            roll_mean_safe(pl.col("close"), 5, min_periods=3, by=None).alias("_sma5"),
            roll_mean_safe(pl.col("close"), 20, min_periods=10, by=None).alias("_sma20"),
            roll_mean_safe(pl.col("close"), 60, min_periods=30, by=None).alias("_sma60"),
        ]
    )

    topix_df = topix_df.with_columns(
        [
            # price_to_sma{20,60}
            (pl.col("close") / (pl.col("_sma20") + eps) - 1.0).alias("topix_price_to_sma20"),
            (pl.col("close") / (pl.col("_sma60") + eps) - 1.0).alias("topix_price_to_sma60"),
            # ma_gap_{5_20,20_60}
            ((pl.col("_sma5") / (pl.col("_sma20") + eps)) - 1.0).alias("topix_ma_gap_5_20"),
            ((pl.col("_sma20") / (pl.col("_sma60") + eps)) - 1.0).alias("topix_ma_gap_20_60"),
        ]
    )

    # 3. RSI（14日）
    # RSI = 100 - 100 / (1 + RS), where RS = avg_gain / avg_loss
    # Wilder's smoothing method
    topix_df = topix_df.with_columns(
        [
            # 価格変化
            (pl.col("close") - pl.col("close").shift(1)).alias("_price_change"),
        ]
    )

    # 利得と損失
    topix_df = topix_df.with_columns(
        [
            pl.col("_price_change").clip(lower_bound=0.0).alias("_gain"),
            (-pl.col("_price_change")).clip(lower_bound=0.0).alias("_loss"),
        ]
    )

    # Wilder's smoothing (EMA with alpha = 1/period)
    # 簡易実装: 移動平均で近似
    topix_df = topix_df.with_columns(
        [
            roll_mean_safe(pl.col("_gain"), 14, min_periods=7, by=None).alias("_avg_gain"),
            roll_mean_safe(pl.col("_loss"), 14, min_periods=7, by=None).alias("_avg_loss"),
        ]
    )

    topix_df = topix_df.with_columns(
        [
            (
                pl.when(pl.col("_avg_loss").abs() > eps)
                .then(100.0 - 100.0 / (1.0 + pl.col("_avg_gain") / (pl.col("_avg_loss") + eps)))
                .otherwise(50.0)  # 損失がない場合はRSI=50
            ).alias("topix_rsi_14"),
        ]
    )

    # 4. ATR/NATR（14日）
    topix_df = topix_df.with_columns(
        [
            # True Range
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col("close").shift(1)).abs(),
                (pl.col("low") - pl.col("close").shift(1)).abs(),
            ).alias("_tr"),
        ]
    )

    topix_df = topix_df.with_columns(
        [
            roll_mean_safe(pl.col("_tr"), 14, min_periods=5, by=None).alias("topix_atr14"),
            roll_mean_safe(pl.col("_tr") / (pl.col("close") + eps), 14, min_periods=5, by=None).alias("topix_natr14"),
        ]
    )

    # 5. 実現ボラティリティ（20日）
    topix_df = topix_df.with_columns(
        [
            roll_std_safe(pl.col("topix_ret_prev_1d"), 20, min_periods=10, by=None)
            .fill_null(0.0)
            .mul(252.0**0.5)
            .alias("topix_realized_vol_20"),
        ]
    )

    # 6. ボラティリティZ-score（252日）
    topix_df = topix_df.with_columns(
        [
            roll_mean_safe(pl.col("topix_realized_vol_20"), 252, min_periods=126, by=None).alias("_vol_mean_252"),
            roll_std_safe(pl.col("topix_realized_vol_20"), 252, min_periods=126, by=None).alias("_vol_std_252"),
        ]
    )

    topix_df = topix_df.with_columns(
        [
            (
                pl.when(pl.col("_vol_std_252").abs() > eps)
                .then((pl.col("topix_realized_vol_20") - pl.col("_vol_mean_252")) / (pl.col("_vol_std_252") + eps))
                .otherwise(None)
            ).alias("topix_vol_z_252d"),
        ]
    )

    # 7. ドローダウン（60日）
    # 60日間の最高値を計算
    topix_df = topix_df.with_columns(
        [
            pl.col("close").rolling_max(window_size=60, min_periods=30).alias("_peak_60d"),
        ]
    )

    topix_df = topix_df.with_columns(
        [
            # drawdown = (close - peak) / peak
            ((pl.col("close") - pl.col("_peak_60d")) / (pl.col("_peak_60d") + eps)).alias("topix_drawdown_60d"),
        ]
    )

    # time_since_peak_60d: 最後にピークに達してからの営業日数
    # 簡易実装: ピーク日からの経過日数（calendar days）
    # 各日について、直近60日間の最高値が何日前かを計算
    topix_df = topix_df.with_columns(
        [
            # 各日のcloseが60日間の最高値と一致する日を特定
            (pl.col("close") == pl.col("_peak_60d")).cast(pl.Int8).alias("_is_peak"),
            # 日付を数値に変換（経過日数計算用）
            pl.col("date").cast(pl.Int64).alias("_date_int"),
        ]
    )

    # 最後のピーク日を後方fillで保持
    topix_df = topix_df.with_columns(
        [
            # ピーク日の日付を保持（非ピーク日はNone）
            pl.when(pl.col("_is_peak") == 1)
            .then(pl.col("_date_int"))
            .otherwise(None)
            .forward_fill()
            .alias("_last_peak_date"),
        ]
    )

    # 経過日数を計算
    topix_df = topix_df.with_columns(
        [
            (
                pl.when(pl.col("_last_peak_date").is_not_null())
                .then(pl.col("_date_int") - pl.col("_last_peak_date"))
                .otherwise(None)
            )
            .cast(pl.Int32)
            .alias("topix_time_since_peak_60d"),
        ]
    )

    # 一時列を削除
    cleanup_cols = [
        "_sma5",
        "_sma20",
        "_sma60",
        "_price_change",
        "_gain",
        "_loss",
        "_avg_gain",
        "_avg_loss",
        "_tr",
        "_vol_mean_252",
        "_vol_std_252",
        "_peak_60d",
        "_is_peak",
        "_date_int",
        "_last_peak_date",
    ]
    for col in cleanup_cols:
        if col in topix_df.columns:
            topix_df = topix_df.drop(col)

    # available_tsを設定（T+1 15:00 JST）
    # 引け後推論ならT+0でも可だが、安全のためにT+1を使用
    # prepare_snapshot_plは"date"列をフォールバックで認識するが、明示的に指定
    topix_df = prepare_snapshot_pl(
        topix_df,
        published_date_col="date",
        trading_calendar=trading_calendar,
        availability_hour=15,
        availability_minute=0,
    )

    # 選択列（date, available_ts, topix_*）
    feature_cols = [col for col in topix_df.columns if col.startswith("topix_") or col in ["date", "available_ts"]]
    result = topix_df.select(feature_cols)

    LOGGER.info(
        "Generated TOPIX features: %d rows, %d features",
        result.height,
        len(feature_cols) - 2,  # date, available_tsを除く
    )

    return result
