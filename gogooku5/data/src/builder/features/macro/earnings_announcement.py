"""決算発表予定特徴量生成モジュール（P0: 最小構成）"""
from __future__ import annotations

import logging
from typing import Optional

import polars as pl

from ..utils.asof_join import prepare_snapshot_pl

LOGGER = logging.getLogger(__name__)


def load_earnings_announcement(df: pl.DataFrame) -> pl.DataFrame:
    """
    決算発表予定データをロードし、正規化する。

    Args:
        df: J-Quants決算発表予定データ（/fins/announcement）

    Returns:
        正規化されたDataFrame（Code, Date列をcode, event_dateに正規化）
    """
    if df.is_empty():
        return df

    # 列名の正規化
    col_map = {
        "Code": "code",
        "LocalCode": "code",
        "Date": "event_date",
        "AnnouncementDate": "event_date",
        "AnnounceDate": "event_date",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename({old: new})

    # 日付列の正規化
    if "event_date" in df.columns:
        df = df.with_columns(pl.col("event_date").cast(pl.Date, strict=False).alias("event_date"))
    else:
        # Date列がない場合は空のDataFrameを返す
        LOGGER.warning("No event_date column found in earnings announcement data")
        return pl.DataFrame()

    # Code列の正規化
    if "code" in df.columns:
        df = df.with_columns(pl.col("code").cast(pl.Utf8, strict=False))
    else:
        LOGGER.warning("No code column found in earnings announcement data")
        return pl.DataFrame()

    # Codeとevent_dateでソート
    df = df.sort(["code", "event_date"])

    # 同じcode, event_dateの重複を削除（最新を保持）
    df = df.unique(subset=["code", "event_date"], keep="last")

    return df


def build_earnings_announcement_features(
    earnings_df: pl.DataFrame,
    trading_calendar: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    決算発表予定のP0特徴量を生成。

    P0フィーチャー:
    - days_to_earnings: 次回Eまでの営業日距離（0=E当日、1=E前日...）
    - is_E_pm1: E±1営業日フラグ
    - is_E_0: E当日フラグ
    - is_E_pp1/pp3/pp5: E±1/±3/±5営業日フラグ
    - is_earnings_sched_valid: 当日までにスケジュールが19:00で公開済みか

    Args:
        earnings_df: 正規化済み決算発表予定データ
        trading_calendar: 営業日カレンダー（date列が必要、as-of計算用）

    Returns:
        特徴量DataFrame（code, event_date, available_ts, days_to_earnings, is_E_*列）
    """
    if earnings_df.is_empty():
        return pl.DataFrame(
            schema={
                "code": pl.Utf8,
                "event_date": pl.Date,
                "available_ts": pl.Datetime("us", "Asia/Tokyo"),
            }
        )

    # 必須列の確認
    required = ["code", "event_date"]
    if not all(col in earnings_df.columns for col in required):
        LOGGER.warning("Required columns missing in earnings announcement data")
        return pl.DataFrame(
            schema={
                "code": pl.Utf8,
                "event_date": pl.Date,
                "available_ts": pl.Datetime("us", "Asia/Tokyo"),
            }
        )

    # available_tsを設定（event_dateの前日19:00 JST）
    # 公式仕様: 翌営業日の予定を19時頃更新 → event_date-1日の19:00をavailable_ts
    # prepare_snapshot_plはT+1を計算するため、event_date-1日をpublished_dateとして渡す
    earnings_df = earnings_df.with_columns(
        (pl.col("event_date") - pl.duration(days=1)).alias("_published_date_for_availability")
    )

    # _published_date_for_availabilityを基準にT+1を計算（= event_date当日）
    earnings_df = prepare_snapshot_pl(
        earnings_df,
        published_date_col="_published_date_for_availability",
        trading_calendar=trading_calendar,
        availability_hour=19,
        availability_minute=0,
    )

    # しかし、要件は「event_dateの前日19:00 JST」なので、available_tsを直接設定
    # prepare_snapshot_plはT+1を計算するため、event_date-1日を基準にするとevent_date当日になる
    # 要件に合わせて、event_date-1日の19:00を直接設定
    earnings_df = earnings_df.with_columns(
        [
            (pl.col("event_date") - pl.duration(days=1)).alias("_availability_date"),
        ]
    )

    # available_tsを直接計算（前日19:00 JST）
    earnings_df = earnings_df.with_columns(
        pl.datetime(
            pl.col("_availability_date").dt.year(),
            pl.col("_availability_date").dt.month(),
            pl.col("_availability_date").dt.day(),
            19,
            0,
            0,
        )
        .dt.replace_time_zone("Asia/Tokyo")
        .alias("available_ts")
    )

    # prepare_snapshot_plで計算されたavailable_tsを上書き（既に計算済みの場合は削除）
    if "available_ts" in earnings_df.columns and "_availability_date" in earnings_df.columns:
        # 一時列を削除
        cleanup_cols = ["_published_date_for_availability", "_availability_date"]
        for col in cleanup_cols:
            if col in earnings_df.columns:
                earnings_df = earnings_df.drop(col)

    # Codeとevent_dateでソート
    earnings_df = earnings_df.sort(["code", "event_date"])

    LOGGER.info(
        "Generated earnings announcement features: %d rows, %d codes",
        earnings_df.height,
        earnings_df["code"].n_unique() if "code" in earnings_df.columns else 0,
    )

    return earnings_df


def attach_earnings_features_to_base(
    base_df: pl.DataFrame,
    earnings_features: pl.DataFrame,
    trading_calendar: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    決算発表予定特徴量をベースDataFrameに結合し、P0特徴量を生成。

    Args:
        base_df: ベースDataFrame（code, date, asof_ts列が必要）
        earnings_features: 決算発表予定特徴量（code, event_date, available_ts列が必要）
        trading_calendar: 営業日カレンダー（days_to_earnings計算用）

    Returns:
        結合済みDataFrame（days_to_earnings, is_E_*列が追加）
    """
    if base_df.is_empty():
        return base_df

    if earnings_features.is_empty():
        # デフォルト値で列を追加
        return base_df.with_columns(
            [
                pl.lit(None).cast(pl.Int32).alias("days_to_earnings"),
                pl.lit(0).cast(pl.Int8).alias("is_E_pm1"),
                pl.lit(0).cast(pl.Int8).alias("is_E_0"),
                pl.lit(0).cast(pl.Int8).alias("is_E_pp1"),
                pl.lit(0).cast(pl.Int8).alias("is_E_pp3"),
                pl.lit(0).cast(pl.Int8).alias("is_E_pp5"),
                pl.lit(0).cast(pl.Int8).alias("is_earnings_sched_valid"),
            ]
        )

    # As-of結合（backward）
    from ..utils.asof_join import interval_join_pl

    # available_tsとasof_tsで結合
    if "available_ts" in earnings_features.columns and "asof_ts" in base_df.columns:
        joined = interval_join_pl(
            backbone=base_df,
            snapshot=earnings_features,
            on_code="code",
            backbone_ts="asof_ts",
            snapshot_ts="available_ts",
            strategy="backward",
            suffix="_earn",
        )
    else:
        # フォールバック: 日付で直接結合
        if "date" in base_df.columns and "event_date" in earnings_features.columns:
            joined = base_df.join(
                earnings_features,
                left_on=["code", "date"],
                right_on=["code", "event_date"],
                how="left",
            )
        else:
            LOGGER.warning("Cannot join earnings features: missing required columns")
            return base_df.with_columns(
                [
                    pl.lit(None).cast(pl.Int32).alias("days_to_earnings"),
                    pl.lit(0).cast(pl.Int8).alias("is_E_pm1"),
                    pl.lit(0).cast(pl.Int8).alias("is_E_0"),
                    pl.lit(0).cast(pl.Int8).alias("is_E_pp1"),
                    pl.lit(0).cast(pl.Int8).alias("is_E_pp3"),
                    pl.lit(0).cast(pl.Int8).alias("is_E_pp5"),
                    pl.lit(0).cast(pl.Int8).alias("is_earnings_sched_valid"),
                ]
            )

    # event_date列を取得（結合後の列名を確認）
    event_date_col = None
    for col in ["event_date", "event_date_earn"]:
        if col in joined.columns:
            event_date_col = col
            break

    if event_date_col is None:
        # event_dateがない場合はデフォルト値
        joined = joined.with_columns(
            [
                pl.lit(None).cast(pl.Int32).alias("days_to_earnings"),
                pl.lit(0).cast(pl.Int8).alias("is_E_pm1"),
                pl.lit(0).cast(pl.Int8).alias("is_E_0"),
                pl.lit(0).cast(pl.Int8).alias("is_E_pp1"),
                pl.lit(0).cast(pl.Int8).alias("is_E_pp3"),
                pl.lit(0).cast(pl.Int8).alias("is_E_pp5"),
                pl.lit(0).cast(pl.Int8).alias("is_earnings_sched_valid"),
            ]
        )
        return joined

    # date列を取得
    date_col = "date" if "date" in joined.columns else None
    if date_col is None:
        LOGGER.warning("date column not found in joined DataFrame")
        return joined

    # days_to_earningsを計算（営業日差）
    # 簡易実装: 日付差を使用（営業日カレンダーを使った正確な計算は後で拡張可能）
    joined = joined.with_columns(
        [
            pl.col(date_col).cast(pl.Date, strict=False).alias("_base_date"),
            pl.col(event_date_col).cast(pl.Date, strict=False).alias("_event_date"),
        ]
    )

    joined = joined.with_columns(
        ((pl.col("_event_date").cast(pl.Int64) - pl.col("_base_date").cast(pl.Int64)) / pl.duration(days=1))
        .cast(pl.Int32)
        .alias("days_to_earnings")
    )

    # P0特徴量を生成
    dte = pl.col("days_to_earnings")
    joined = joined.with_columns(
        [
            # is_E_pm1: E±1営業日
            (dte.is_in([-1, 1])).cast(pl.Int8).alias("is_E_pm1"),
            # is_E_0: E当日
            (dte == 0).cast(pl.Int8).alias("is_E_0"),
            # is_E_pp1: E±1営業日（is_E_pm1と同じ）
            (dte.is_in([-1, 1])).cast(pl.Int8).alias("is_E_pp1"),
            # is_E_pp3: E±3営業日
            (dte.is_in([-3, -2, -1, 0, 1, 2, 3])).cast(pl.Int8).alias("is_E_pp3"),
            # is_E_pp5: E±5営業日
            (dte.is_in([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])).cast(pl.Int8).alias("is_E_pp5"),
            # is_earnings_sched_valid: 当日までにスケジュールが19:00で公開済みか
            pl.col(event_date_col).is_not_null().cast(pl.Int8).alias("is_earnings_sched_valid"),
        ]
    )

    # 一時列を削除
    cleanup_cols = ["_base_date", "_event_date", "available_ts_earn", event_date_col]
    for col in cleanup_cols:
        if col in joined.columns:
            joined = joined.drop(col)

    return joined
