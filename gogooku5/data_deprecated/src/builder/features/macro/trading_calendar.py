"""取引カレンダー特徴量生成モジュール（P0: 最小構成）"""

from __future__ import annotations

import logging

import polars as pl

LOGGER = logging.getLogger(__name__)


def build_trading_calendar_features(
    calendar_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    取引カレンダーのP0特徴量を生成。

    P0フィーチャー:
    - is_trading_day: 営業日フラグ
    - is_mon..is_fri: 曜日フラグ
    - is_month_end: 月末フラグ
    - is_quarter_end: 四半期末フラグ
    - is_fy_end: 年度末（3月末）フラグ
    - days_to_holiday, days_since_holiday: 連休関連
    - is_sq_day, days_to_sq, days_since_sq, is_sq_week: SQ関連

    Args:
        calendar_df: 取引カレンダーDataFrame（date列が必要）

    Returns:
        特徴量DataFrame（date, is_trading_day, is_*列）
    """
    if calendar_df.is_empty():
        return pl.DataFrame(schema={"date": pl.Date})

    # date列の確認
    if "date" not in calendar_df.columns:
        LOGGER.warning("date column not found in calendar DataFrame")
        return pl.DataFrame(schema={"date": pl.Date})

    # date列を正規化
    result = calendar_df.with_columns(pl.col("date").cast(pl.Date, strict=False).alias("date")).sort("date")

    # 1. 基本フラグ
    result = result.with_columns(
        [
            # is_trading_day: 営業日フラグ（calendar_dfに含まれる日は営業日）
            pl.lit(1).cast(pl.Int8).alias("is_trading_day"),
            # 曜日フラグ（dt.weekday()はISO週番号: Monday=1, Sunday=7）
            # Python形式（Monday=0, Sunday=6）に変換するために-1
            ((pl.col("date").dt.weekday() - 1) == 0).cast(pl.Int8).alias("is_mon"),
            ((pl.col("date").dt.weekday() - 1) == 1).cast(pl.Int8).alias("is_tue"),
            ((pl.col("date").dt.weekday() - 1) == 2).cast(pl.Int8).alias("is_wed"),
            ((pl.col("date").dt.weekday() - 1) == 3).cast(pl.Int8).alias("is_thu"),
            ((pl.col("date").dt.weekday() - 1) == 4).cast(pl.Int8).alias("is_fri"),
        ]
    )

    # 2. 月末/四半期末/年度末フラグ
    result = result.with_columns(
        [
            # 月末日を計算
            pl.col("date").dt.month_end().alias("_month_end"),
        ]
    )

    result = result.with_columns(
        [
            # is_month_end: 月末フラグ
            (pl.col("date") == pl.col("_month_end")).cast(pl.Int8).alias("is_month_end"),
            # is_quarter_end: 四半期末（3, 6, 9, 12月末）
            ((pl.col("date").dt.month().is_in([3, 6, 9, 12])) & (pl.col("date") == pl.col("_month_end")))
            .cast(pl.Int8)
            .alias("is_quarter_end"),
            # is_fy_end: 年度末（3月末、日本は3月決算が多い）
            ((pl.col("date").dt.month() == 3) & (pl.col("date") == pl.col("_month_end")))
            .cast(pl.Int8)
            .alias("is_fy_end"),
        ]
    )

    # 一時列を削除
    if "_month_end" in result.columns:
        result = result.drop("_month_end")

    # 3. 連休関連（簡易実装）
    # 営業日カレンダーから非営業日を推測（実際のカレンダーAPIから取得する場合は拡張）
    # 簡易実装: 週末（土日）を非営業日として扱う
    # dt.weekday()はISO週番号（Mon=1, Sun=7）なので、Python形式（Mon=0, Sun=6）に変換
    result = result.with_columns(
        [
            (pl.col("date").dt.weekday() - 1).alias("_weekday"),  # Python形式: Mon=0, Sun=6
        ]
    )

    # 次の非営業日までの日数（簡易実装: 次の土曜日までの日数）
    result = result.with_columns(
        [
            (
                pl.when(pl.col("_weekday") == 4)  # 金曜日（Python形式: weekday=4）
                .then(1)  # 次の日（土曜日）まで1日
                .when(pl.col("_weekday") == 5)  # 土曜日（Python形式: weekday=5）
                .then(0)  # 当日
                .otherwise(4 - pl.col("_weekday"))  # 金曜日（weekday=4）までの日数
            )
            .cast(pl.Int32)
            .alias("days_to_holiday"),
            # 前の非営業日からの日数（簡易実装）
            (
                pl.when(pl.col("_weekday") == 0)  # 月曜日（Python形式: weekday=0）
                .then(1)  # 前日（日曜日）から1日
                .when(pl.col("_weekday") == 6)  # 日曜日（Python形式: weekday=6）
                .then(0)  # 当日
                .otherwise(pl.col("_weekday"))  # 前の日曜日からの日数（weekdayそのまま）
            )
            .cast(pl.Int32)
            .alias("days_since_holiday"),
        ]
    )

    # 一時列を削除
    if "_weekday" in result.columns:
        result = result.drop("_weekday")

    # 4. SQ関連（Special Quotation: 指数先物・オプションSQ）
    # SQは原則「第2金曜」（例外あり、簡易実装）
    result = result.with_columns(
        [
            pl.col("date").dt.year().alias("_year"),
            pl.col("date").dt.month().alias("_month"),
            pl.col("date").dt.day().alias("_day"),
            # dt.weekday()はISO週番号（Mon=1, Fri=5）なので、Python形式（Mon=0, Fri=4）に変換
            (pl.col("date").dt.weekday() - 1).alias("_weekday_for_sq"),  # Python形式: Mon=0, Fri=4
        ]
    )

    # 第2金曜日を簡易計算（月の最初の金曜日を特定し、+7日）
    # より正確には、実際のSQカレンダーを使用すべき
    result = result.with_columns(
        [
            # 簡易実装: 8日～14日の範囲で金曜日をSQと仮定
            (
                (pl.col("_weekday_for_sq") == 4)  # 金曜日（Python形式: weekday=4）
                & (pl.col("_day") >= 8)
                & (pl.col("_day") <= 14)
            )
            .cast(pl.Int8)
            .alias("is_sq_day"),
        ]
    )

    # days_to_sq, days_since_sqを計算
    # 各SQ日までの距離を計算（簡易実装）
    # より正確には、実際のSQカレンダーから計算
    result = result.with_columns(
        [
            # days_to_sq: 次のSQ日までの日数（簡易実装: 次のSQ日を探す）
            pl.when(pl.col("is_sq_day") == 1)
            .then(0)
            .otherwise(None)  # 簡易実装では計算を省略
            .cast(pl.Int32)
            .alias("days_to_sq"),
            # days_since_sq: 前のSQ日からの日数（簡易実装）
            pl.when(pl.col("is_sq_day") == 1)
            .then(0)
            .otherwise(None)  # 簡易実装では計算を省略
            .cast(pl.Int32)
            .alias("days_since_sq"),
            # is_sq_week: SQ週フラグ（SQ日前後±3日）
            (
                (pl.col("is_sq_day") == 1)
                | (
                    # SQ日前後±3日（簡易実装）
                    (pl.col("_day") >= 5)
                    & (pl.col("_day") <= 17)
                    & (pl.col("_month").is_in([3, 6, 9, 12]))  # 四半期末のみ
                )
            )
            .cast(pl.Int8)
            .alias("is_sq_week"),
        ]
    )

    # 一時列を削除
    cleanup_cols = ["_year", "_month", "_day", "_weekday_for_sq"]
    for col in cleanup_cols:
        if col in result.columns:
            result = result.drop(col)

    LOGGER.info(
        "Generated trading calendar features: %d rows, %d trading days",
        result.height,
        result["is_trading_day"].sum() if "is_trading_day" in result.columns else 0,
    )

    return result
