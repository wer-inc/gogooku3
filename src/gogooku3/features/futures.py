"""
先物データから特徴量を生成するモジュール

先物データ（指数先物・ベーシス）をML用日次パネルに統合する機能:
- オーバーナイト（ON）シグナル: 夜間取引の変動を翌営業日に反映
- ベーシス特徴量: 先物価格 - 現物価格による裁定機会の測定
- 建玉・出来高特徴量: 市場の流動性とポジション状況
- ターム構造: 限月間の価格構造による市場センチメント測定
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import polars as pl

logger = logging.getLogger(__name__)


# スポット指数のマッピング
FUTURES_TO_SPOT_MAPPING = {
    "TOPIXF": "TOPIX",
    "TOPIXMF": "TOPIX",
    "NK225F": "NK225",
    "NK225MF": "NK225",
    "NK225MCF": "NK225",
    "JN400F": "JPX400",
    "REITF": "REIT",
    "MOTF": "MOTHERS",
}


def _next_business_day_jp(date: datetime) -> datetime:
    """日本の営業日ベースで翌営業日を計算"""
    if date.weekday() < 4:  # Mon-Thu
        return date + timedelta(days=1)
    elif date.weekday() == 4:  # Fri
        return date + timedelta(days=3)  # Next Mon
    elif date.weekday() == 5:  # Sat
        return date + timedelta(days=2)  # Next Mon
    else:  # Sun
        return date + timedelta(days=1)  # Next Mon


def build_central_contracts(futures_df: pl.DataFrame) -> pl.DataFrame:
    """
    中心限月のみを抽出・整理

    Args:
        futures_df: 正規化済み先物データ

    Returns:
        中心限月データ
    """
    if futures_df.is_empty():
        return futures_df

    # 中心限月のみをフィルタ
    central = futures_df.filter(
        pl.col("CentralContractMonthFlag") == "1"
    )

    if central.is_empty():
        logger.warning("No central contract month data found")
        return futures_df.head(0)  # Empty with schema

    # 日付順でソート
    central = central.sort(["ProductCategory", "Date"])

    logger.info(f"Extracted {len(central)} central contract records")
    return central


def add_overnight_signals(futures_df: pl.DataFrame) -> pl.DataFrame:
    """
    オーバーナイト（ON）シグナルを生成

    夜間セッションの価格変動を翌営業日の特徴量として追加
    """
    if futures_df.is_empty():
        return futures_df

    # 夜間セッション価格変動
    df_with_on = futures_df.with_columns([
        # 夜間リターン (T-1夜間Close → T日中Open)
        (
            pl.col("NightSessionClose") / pl.col("Close").shift(1).over("Code") - 1.0
        ).alias("futures_on_return"),

        # 夜間ボラティリティ
        (
            (pl.col("NightSessionHigh") - pl.col("NightSessionLow")) / pl.col("NightSessionOpen")
        ).alias("futures_on_volatility"),

        # 夜間出来高比率
        (
            pl.col("NightSessionVolume") / (pl.col("Volume") + pl.col("NightSessionVolume") + 1)
        ).alias("futures_on_volume_ratio"),

        # Gap (夜間終値 → 翌日始値)
        (
            pl.col("Open") / pl.col("NightSessionClose").shift(1).over("Code") - 1.0
        ).alias("futures_gap"),
    ])

    return df_with_on


def compute_basis_features(
    futures_df: pl.DataFrame,
    spot_indices_df: pl.DataFrame | None = None
) -> pl.DataFrame:
    """
    ベーシス特徴量を計算

    Args:
        futures_df: 先物データ
        spot_indices_df: スポット指数データ (Date, index_name, Close)

    Returns:
        ベーシス特徴量付きデータ
    """
    if futures_df.is_empty():
        return futures_df

    df = futures_df.copy()

    if spot_indices_df is not None and not spot_indices_df.is_empty():
        # スポット指数とのマージ
        for futures_category, spot_name in FUTURES_TO_SPOT_MAPPING.items():
            # 該当カテゴリの先物データ
            category_mask = pl.col("ProductCategory") == futures_category

            # スポット指数データ
            spot_data = spot_indices_df.filter(
                pl.col("index_name") == spot_name
            ).select(["Date", "Close"]).rename({"Close": f"{spot_name}_spot"})

            if not spot_data.is_empty():
                # Left joinでスポット価格を追加
                df = df.join(spot_data, on="Date", how="left")

                # ベーシス計算
                df = df.with_columns([
                    pl.when(category_mask)
                    .then(pl.col("Close") - pl.col(f"{spot_name}_spot"))
                    .otherwise(None)
                    .alias(f"futures_{futures_category.lower()}_basis"),

                    # ベーシス率
                    pl.when(category_mask & (pl.col(f"{spot_name}_spot") > 0))
                    .then((pl.col("Close") - pl.col(f"{spot_name}_spot")) / pl.col(f"{spot_name}_spot"))
                    .otherwise(None)
                    .alias(f"futures_{futures_category.lower()}_basis_ratio"),
                ])

                # 一時的なスポット価格列を削除
                df = df.drop(f"{spot_name}_spot")

    # 共通のベーシス近似特徴量（スポット指数がない場合のフォールバック）
    df = df.with_columns([
        # Forward premium approximation (簡易版)
        (pl.col("Close") / pl.col("Close").shift(1).over("Code") - 1.0).alias("futures_forward_premium"),

        # Convenience yield proxy (出来高とOpen Interestの関係)
        (
            pl.col("OpenInterest") / (pl.col("Volume") + 1)
        ).alias("futures_convenience_yield_proxy"),
    ])

    return df


def add_term_structure_features(futures_df: pl.DataFrame) -> pl.DataFrame:
    """
    ターム構造特徴量を追加

    限月間の価格構造から市場センチメントを測定
    """
    if futures_df.is_empty():
        return futures_df

    # 限月を数値化（YYYYMM形式を想定）
    df = futures_df.with_columns([
        pl.col("ContractMonth").str.slice(0, 6).cast(pl.Int32, strict=False).alias("contract_month_numeric")
    ])

    # 各商品カテゴリ・日付で限月構造を分析
    df = df.with_columns([
        # 当限月と次限月の価格比
        (
            pl.col("Close") / pl.col("Close").shift(-1).over(["ProductCategory", "Date"]) - 1.0
        ).alias("futures_calendar_spread"),

        # 建玉の限月分布
        (
            pl.col("OpenInterest") / pl.col("OpenInterest").sum().over(["ProductCategory", "Date"])
        ).alias("futures_oi_distribution"),

        # Volume weighted term structure slope
        (
            (pl.col("Close") - pl.col("Close").mean().over(["ProductCategory", "Date"])) *
            pl.col("Volume") / pl.col("Volume").sum().over(["ProductCategory", "Date"])
        ).alias("futures_term_slope_vw"),
    ])

    return df


def add_technical_features(futures_df: pl.DataFrame) -> pl.DataFrame:
    """
    先物固有のテクニカル特徴量を追加
    """
    if futures_df.is_empty():
        return futures_df

    # 基本的なテクニカル指標
    df = futures_df.with_columns([
        # Moving averages
        pl.col("Close").rolling_mean(5).over("Code").alias("futures_ma5"),
        pl.col("Close").rolling_mean(20).over("Code").alias("futures_ma20"),

        # Volatility (20-day rolling)
        pl.col("Close").pct_change().over("Code").rolling_std(20).over("Code").alias("futures_vol20"),

        # Volume indicators
        pl.col("Volume").rolling_mean(20).over("Code").alias("futures_volume_ma20"),
        (pl.col("Volume") / pl.col("Volume").rolling_mean(20).over("Code")).alias("futures_volume_ratio"),

        # Open Interest changes
        pl.col("OpenInterest").diff().over("Code").alias("futures_oi_change"),
        (pl.col("OpenInterest").diff().over("Code") / pl.col("OpenInterest")).alias("futures_oi_change_ratio"),

        # Price-Volume relationship
        (
            pl.col("Close").pct_change().over("Code") * pl.col("Volume")
        ).alias("futures_price_volume"),

        # Intraday range
        ((pl.col("High") - pl.col("Low")) / pl.col("Open")).alias("futures_intraday_range"),

        # Emergency margin stress indicator
        pl.col("emergency_margin_triggered").cast(pl.Int8),
    ])

    return df


def aggregate_by_category(futures_df: pl.DataFrame) -> pl.DataFrame:
    """
    商品カテゴリ別に日次集計特徴量を作成
    """
    if futures_df.is_empty():
        return pl.DataFrame()

    # カテゴリ別日次集計
    category_agg = (
        futures_df
        .group_by(["Date", "ProductCategory"])
        .agg([
            # Price features
            pl.col("Close").mean().alias("futures_close_mean"),
            pl.col("Close").std().alias("futures_close_std"),
            pl.col("futures_on_return").mean().alias("futures_on_return_mean"),
            pl.col("futures_on_volatility").mean().alias("futures_on_volatility_mean"),

            # Volume features
            pl.col("Volume").sum().alias("futures_total_volume"),
            pl.col("OpenInterest").sum().alias("futures_total_oi"),
            pl.col("TurnoverValue").sum().alias("futures_total_turnover"),

            # Technical features
            pl.col("futures_vol20").mean().alias("futures_avg_vol20"),
            pl.col("emergency_margin_triggered").max().alias("futures_emergency_margin_any"),

            # Market structure
            pl.col("futures_oi_distribution").std().alias("futures_oi_concentration"),
        ])
        .sort(["Date", "ProductCategory"])
    )

    return category_agg


def attach_futures_to_panel(
    quotes_panel: pl.DataFrame,
    futures_df: pl.DataFrame,
    category_agg_df: pl.DataFrame | None = None
) -> pl.DataFrame:
    """
    先物特徴量を株式日次パネルに統合

    Args:
        quotes_panel: 株式日次パネル (Code, Date, ...)
        futures_df: 処理済み先物データ
        category_agg_df: カテゴリ別集計データ（オプション）

    Returns:
        先物特徴量が追加された日次パネル
    """
    if futures_df.is_empty():
        logger.warning("Empty futures data, skipping futures features")
        return quotes_panel

    # TOPIX先物を代表として全銘柄に配布
    topix_futures = futures_df.filter(
        pl.col("ProductCategory").is_in(["TOPIXF", "TOPIXMF"])
    )

    if not topix_futures.is_empty():
        # 各日付でTOPIX先物の特徴量を作成
        daily_futures_features = (
            topix_futures
            .group_by("Date")
            .agg([
                # Overnight signals (T+0で利用可能)
                pl.col("futures_on_return").mean().alias("fut_on_return_topix"),
                pl.col("futures_on_volatility").mean().alias("fut_on_vol_topix"),
                pl.col("futures_gap").mean().alias("fut_gap_topix"),

                # Market structure (T+1で利用可能)
                pl.col("futures_forward_premium").mean().alias("fut_forward_premium_topix"),
                pl.col("futures_vol20").mean().alias("fut_vol20_topix"),
                pl.col("futures_volume_ratio").mean().alias("fut_vol_ratio_topix"),
                pl.col("futures_oi_change_ratio").mean().alias("fut_oi_change_topix"),

                # Emergency margin indicator
                pl.col("emergency_margin_triggered").max().alias("fut_emergency_margin"),
            ])
            .sort("Date")
        )

        # 株式パネルにjoin
        result = quotes_panel.join(
            daily_futures_features,
            on="Date",
            how="left"
        )

        logger.info(f"Attached TOPIX futures features to {len(result)} panel records")
    else:
        logger.warning("No TOPIX futures data found")
        result = quotes_panel

    # カテゴリ別集計データも追加（オプション）
    if category_agg_df is not None and not category_agg_df.is_empty():
        # 主要カテゴリを選択して追加
        for category in ["TOPIXF", "NK225F", "REITF"]:
            cat_data = category_agg_df.filter(
                pl.col("ProductCategory") == category
            ).select([
                "Date",
                pl.col("futures_close_mean").alias(f"fut_{category.lower()}_price"),
                pl.col("futures_on_return_mean").alias(f"fut_{category.lower()}_on_ret"),
                pl.col("futures_total_volume").alias(f"fut_{category.lower()}_volume"),
                pl.col("futures_emergency_margin_any").alias(f"fut_{category.lower()}_emergency"),
            ])

            if not cat_data.is_empty():
                result = result.join(cat_data, on="Date", how="left")

    return result


def process_futures_pipeline(
    futures_raw_df: pl.DataFrame,
    spot_indices_df: pl.DataFrame | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    先物データ処理の完全パイプライン

    Args:
        futures_raw_df: 生の先物データ
        spot_indices_df: スポット指数データ（オプション）

    Returns:
        Tuple[processed_futures, category_aggregates]
    """
    if futures_raw_df.is_empty():
        logger.warning("Empty futures raw data")
        return futures_raw_df, pl.DataFrame()

    logger.info(f"Processing {len(futures_raw_df)} raw futures records")

    # Step 1: 中心限月の抽出
    central_contracts = build_central_contracts(futures_raw_df)
    if central_contracts.is_empty():
        logger.error("No central contracts found")
        return futures_raw_df.head(0), pl.DataFrame()

    # Step 2: オーバーナイトシグナル
    with_on = add_overnight_signals(central_contracts)

    # Step 3: ベーシス特徴量
    with_basis = compute_basis_features(with_on, spot_indices_df)

    # Step 4: ターム構造
    with_term = add_term_structure_features(with_basis)

    # Step 5: テクニカル特徴量
    with_technical = add_technical_features(with_term)

    # Step 6: カテゴリ別集計
    category_agg = aggregate_by_category(with_technical)

    logger.info(f"Generated {len(with_technical)} processed futures records")
    logger.info(f"Generated {len(category_agg)} category aggregate records")

    return with_technical, category_agg
