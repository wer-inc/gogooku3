"""TOPIX先物特徴量生成モジュール（P0: 最小構成）"""
from __future__ import annotations

import logging

import polars as pl

from ..utils.asof_join import prepare_snapshot_pl

LOGGER = logging.getLogger(__name__)


def load_futures(
    df: pl.DataFrame,
    category: str = "TOPIXF",
) -> pl.DataFrame:
    """
    TOPIX先物データをロードし、正規化する。

    Args:
        df: J-Quants先物データ（get_futures_dailyで取得）
        category: 商品区分（デフォルト: "TOPIXF"）

    Returns:
        正規化された先物DataFrame
    """
    if df.is_empty():
        LOGGER.warning("TOPIX futures loader received empty dataframe")
        return df

    # カテゴリでフィルタ（DerivativesProductCategory を最優先）
    category_columns = ("DerivativesProductCategory", "ProductCategory")
    for col in category_columns:
        if col in df.columns:
            df = df.filter(pl.col(col).cast(pl.Utf8, strict=False) == category)
            break

    # 日付列の正規化
    date_cols = ["Date", "date"]
    date_expr = None
    for col in date_cols:
        if col in df.columns:
            date_expr = pl.col(col).cast(pl.Date, strict=False)
            break

    if date_expr is None:
        LOGGER.warning("No date column found in futures data")
        return pl.DataFrame()

    df = df.with_columns(date_expr.alias("Date"))

    # 数値列の正規化（空文字→Null）
    numeric_cols = [
        "SettlementPrice",
        "WholeDayClose",
        "WholeDayOpen",
        "WholeDayHigh",
        "WholeDayLow",
        "NightSessionClose",
        "DaySessionOpen",
        "DaySessionClose",
        "OpenInterest",
        "Volume",
    ]

    for col in numeric_cols:
        if col in df.columns:
            cleaned = pl.col(col).cast(pl.Utf8, strict=False).str.strip_chars()
            df = df.with_columns(
                pl.when(cleaned == "")
                .then(pl.lit(None, dtype=pl.Float64))
                .otherwise(cleaned.cast(pl.Float64, strict=False))
                .alias(col)
            )

    # 文字列列の正規化
    string_cols = [
        "ContractMonth",
        "CentralContractMonthFlag",
        "LastTradingDay",
        "SpecialQuotationDay",
        "EmergencyMarginTriggerDivision",
    ]
    for col in string_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Utf8, strict=False))

    return df


def select_front_next(df: pl.DataFrame) -> pl.DataFrame:
    """
    同日・同カテゴリ内でfront/next限月を選定する。

    ルール:
    1. CentralContractMonthFlag=="1" を front
    2. 残りで OpenInterest 最大を next（同点は満期が遠い方）
    3. OpenInterest が空なら満期が近い順

    Args:
        df: 同日・同カテゴリの先物データ

    Returns:
        front/next を横持ちにしたDataFrame（date, front_settle, next_settle, ...）
    """
    if df.is_empty():
        return pl.DataFrame()

    # 日付を取得（1件目）
    date_val = df["Date"].head(1).item() if "Date" in df.columns else None
    if date_val is None:
        return pl.DataFrame()

    # Front限月の選定（CentralContractMonthFlag=="1"）
    front_candidates = df.filter(pl.col("CentralContractMonthFlag") == "1")
    if front_candidates.is_empty():
        # フォールバック: OpenInterest最大
        front_candidates = df.sort("OpenInterest", descending=True).head(1)
    if front_candidates.is_empty():
        # 最後のフォールバック: 満期が近い順
        front_candidates = df.sort("ContractMonth").head(1)

    if front_candidates.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "front_settle": pl.Float64, "next_settle": pl.Float64})

    front_row = front_candidates.head(1)
    front_settle = (
        front_row["SettlementPrice"].head(1).item()
        if "SettlementPrice" in front_row.columns
        else (front_row["WholeDayClose"].head(1).item() if "WholeDayClose" in front_row.columns else None)
    )
    front_oi = front_row["OpenInterest"].head(1).item() if "OpenInterest" in front_row.columns else None
    front_last_trading_day = (
        front_row["LastTradingDay"].head(1).item() if "LastTradingDay" in front_row.columns else None
    )
    front_contract_month = front_row["ContractMonth"].head(1).item() if "ContractMonth" in front_row.columns else None

    # Next限月の選定（front以外でOpenInterest最大）
    next_candidates = df.filter(
        (pl.col("ContractMonth") != front_contract_month) if front_contract_month else pl.lit(True)
    )

    if not next_candidates.is_empty():
        # OpenInterest最大を選ぶ
        next_candidates = next_candidates.sort("OpenInterest", descending=True)
        if not next_candidates.is_empty():
            next_row = next_candidates.head(1)
            next_settle = (
                next_row["SettlementPrice"].head(1).item()
                if "SettlementPrice" in next_row.columns
                else (next_row["WholeDayClose"].head(1).item() if "WholeDayClose" in next_row.columns else None)
            )
        else:
            next_settle = None
    else:
        next_settle = None

    # 結果を構築
    result = pl.DataFrame(
        {
            "date": [date_val],
            "front_settle": [front_settle],
            "next_settle": [next_settle],
            "front_oi": [front_oi],
            "front_last_trading_day": [front_last_trading_day],
        }
    ).with_columns(
        # FIX: Ensure date is pl.Date, not Object (prevents join type mismatch)
        pl.col("date").cast(pl.Date, strict=False)
    )

    # セッション価格も追加（front限月から）
    front_day_open_val = None
    front_day_close_val = None
    front_night_close_val = None

    if "DaySessionOpen" in front_row.columns:
        try:
            front_day_open_val = front_row["DaySessionOpen"].head(1).item()
        except Exception:
            pass
    if "DaySessionClose" in front_row.columns:
        try:
            front_day_close_val = front_row["DaySessionClose"].head(1).item()
        except Exception:
            pass
    if "NightSessionClose" in front_row.columns:
        try:
            front_night_close_val = front_row["NightSessionClose"].head(1).item()
        except Exception:
            pass

    result = result.with_columns(
        [
            pl.lit(front_day_open_val, dtype=pl.Float64).alias("front_day_open"),
            pl.lit(front_day_close_val, dtype=pl.Float64).alias("front_day_close"),
            pl.lit(front_night_close_val, dtype=pl.Float64).alias("front_night_close"),
        ]
    )

    return result


def build_futures_features(
    fut: pl.DataFrame,
    topix_df: pl.DataFrame | None = None,
    trading_calendar: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    TOPIX先物の特徴量を生成（P0: 最小構成）。

    Args:
        fut: 正規化済み先物データ
        topix_df: TOPIX現物データ（Date, Close列が必要）
        trading_calendar: 営業日カレンダー（date列が必要）

    Returns:
        特徴量DataFrame（date, available_ts, fut_topix_* 列）
    """
    if fut.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")})

    # Front/Next限月の選定（日次・カテゴリ別にグループ化）
    category_col = "ProductCategory" if "ProductCategory" in fut.columns else "DerivativesProductCategory"
    if category_col not in fut.columns:
        LOGGER.warning("No category column found in futures data")
        return pl.DataFrame(schema={"date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")})

    front_next_list = []
    for (date_val, cat_val), group in fut.group_by(["Date", category_col], maintain_order=True):
        fn_result = select_front_next(group)
        if not fn_result.is_empty():
            front_next_list.append(fn_result)

    if not front_next_list:
        LOGGER.warning("TOPIX futures features skipped: unable to determine front/next contracts")
        return pl.DataFrame(schema={"date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")})

    fn_df = pl.concat(front_next_list).sort("date")

    # 前日の価格を取得（shift(1)）
    fn_df = fn_df.with_columns(
        [
            pl.col("front_settle").shift(1).alias("prev_front_settle"),
            pl.col("front_night_close").shift(1).alias("prev_day_settle"),
        ]
    )

    # TOPIX現物との結合（ベーシス計算用）
    if topix_df is not None and not topix_df.is_empty():
        # Date列の名前を確認（Dateまたはdate）
        date_col = "Date" if "Date" in topix_df.columns else "date"
        close_col = "Close" if "Close" in topix_df.columns else "close"

        topix_clean = topix_df.select(
            [
                pl.col(date_col).cast(pl.Date, strict=False).alias("date"),
                pl.col(close_col).cast(pl.Float64, strict=False).alias("topix_close"),
            ]
        ).filter(pl.col("topix_close").is_not_null() & pl.col("date").is_not_null())

        fn_df = fn_df.join(topix_clean, on="date", how="left")
    else:
        fn_df = fn_df.with_columns(pl.lit(None).cast(pl.Float64).alias("topix_close"))

    # P0特徴量の生成
    fn_df = fn_df.with_columns(
        [
            # 市場レジーム（方向/変化）
            ((pl.col("front_settle") / pl.col("prev_front_settle")) - 1.0).alias("fut_topix_front_ret_1d"),
            # 日中/夜間の分解
            ((pl.col("front_day_close") / pl.col("front_day_open")) - 1.0).alias("fut_topix_intraday"),
            ((pl.col("front_night_close") / pl.col("prev_day_settle")) - 1.0).alias("fut_topix_overnight"),
            # ボラ指標（範囲率）
            ((pl.col("front_day_close") - pl.col("front_day_open")) / pl.col("front_settle")).alias(
                "fut_topix_range_pct"
            ),
            # 建玉の圧力
            (pl.col("front_oi") - pl.col("front_oi").shift(1)).alias("fut_topix_oi_chg_1d"),
            # 期近/期先の傾き
            ((pl.col("next_settle") / pl.col("front_settle")) - 1.0).alias("fut_topix_term_slope"),
            # 先物-現物の乖離（ベーシス）
            ((pl.col("front_settle") - pl.col("topix_close")) / pl.col("topix_close")).alias("fut_topix_basis"),
        ]
    )

    # ロール接近日数の計算
    if "front_last_trading_day" in fn_df.columns:
        # LastTradingDayを日付に変換
        fn_df = fn_df.with_columns(
            pl.col("front_last_trading_day")
            .cast(pl.Utf8, strict=False)
            .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            .alias("last_trading_day_date")
        )
        fn_df = fn_df.with_columns(
            (pl.col("last_trading_day_date") - pl.col("date")).dt.total_days().alias("fut_topix_days_to_roll")
        )
        # ロール窓フラグ
        fn_df = fn_df.with_columns(
            [
                (pl.col("fut_topix_days_to_roll") <= 3).alias("fut_topix_roll_soon_3"),
                (pl.col("fut_topix_days_to_roll") <= 5).alias("fut_topix_roll_soon_5"),
                (pl.col("fut_topix_days_to_roll") <= 10).alias("fut_topix_roll_soon_10"),
            ]
        )
    else:
        fn_df = fn_df.with_columns(
            [
                pl.lit(None).cast(pl.Int64).alias("fut_topix_days_to_roll"),
                pl.lit(False).alias("fut_topix_roll_soon_3"),
                pl.lit(False).alias("fut_topix_roll_soon_5"),
                pl.lit(False).alias("fut_topix_roll_soon_10"),
            ]
        )

    # キャリー近似（年率）- next_settleとfront_settleが両方ある場合のみ
    if "next_settle" in fn_df.columns and "front_settle" in fn_df.columns and "fut_topix_days_to_roll" in fn_df.columns:
        fn_df = fn_df.with_columns(
            (
                (pl.col("next_settle") / pl.col("front_settle")).log()
                / (pl.col("fut_topix_days_to_roll").abs() + 1e-9)
                * 365.0
            ).alias("fut_topix_carry_ann")
        )
    else:
        fn_df = fn_df.with_columns(pl.lit(None).cast(pl.Float64).alias("fut_topix_carry_ann"))

    # as-of スナップショット化（T+1 09:00 JST）
    # Date列をPublishedDateとして扱う
    fn_df = fn_df.with_columns(pl.col("date").alias("PublishedDate"))
    fn_df = prepare_snapshot_pl(
        fn_df,
        published_date_col="PublishedDate",
        trading_calendar=trading_calendar,
        availability_hour=9,
        availability_minute=0,
    )

    # 出力列を選択
    feature_cols = [
        "date",
        "available_ts",
        "fut_topix_front_ret_1d",
        "fut_topix_intraday",
        "fut_topix_overnight",
        "fut_topix_range_pct",
        "fut_topix_oi_chg_1d",
        "fut_topix_term_slope",
        "fut_topix_basis",
        "fut_topix_days_to_roll",
        "fut_topix_roll_soon_3",
        "fut_topix_roll_soon_5",
        "fut_topix_roll_soon_10",
        "fut_topix_carry_ann",
    ]

    # 存在する列のみ選択
    available_cols = [col for col in feature_cols if col in fn_df.columns]
    result = fn_df.select(available_cols)

    LOGGER.info(
        "Generated TOPIX futures features: %d rows, %d features",
        len(result),
        len(available_cols) - 2,  # date, available_tsを除く
    )

    return result
