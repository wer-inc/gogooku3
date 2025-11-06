"""Listed info (上場銘柄一覧) feature engineering."""
from __future__ import annotations

import polars as pl

# Note: prepare_snapshot_pl is not used here as we handle as-of directly


def build_listed_info_features(
    listed_df: pl.DataFrame,
    *,
    trading_calendar: pl.DataFrame | None = None,
    build_date: str | None = None,
) -> pl.DataFrame:
    """Build listed info features with as-of protection.

    Args:
        listed_df: Raw listed info DataFrame with columns:
            - Code, Date, MarketCode, Sector33Code, Sector17Code, ScaleCategory, MarginCode
        trading_calendar: Trading calendar for business day calculations
        build_date: Build date (YYYY-MM-DD) for determining "tomorrow"

    Returns:
        DataFrame with listed info features:
        - Market dummies: is_prime, is_standard, is_growth
        - Sector codes: sector33_code, sector17_code
        - Scale bucket: scale_bucket
        - Margin eligibility: is_margin_eligible
        - available_ts: As-of timestamp for leak prevention
    """
    if listed_df.is_empty():
        return pl.DataFrame(
            {
                "code": pl.Series([], dtype=pl.Utf8),
                "date": pl.Series([], dtype=pl.Date),
                "available_ts": pl.Series([], dtype=pl.Datetime("us", "Asia/Tokyo")),
            }
        )

    required = {"Code", "Date"}
    missing = [col for col in required if col not in listed_df.columns]
    if missing:
        raise ValueError(f"listed_info payload missing required columns: {missing}")

    # 正規化と列名の統一
    df = listed_df.with_columns(
        [
            pl.col("Code").cast(pl.Utf8, strict=False).alias("code"),
            pl.col("Date").cast(pl.Date, strict=False).alias("date"),
        ]
    )

    # MarketCode → market_code + ダミー
    if "MarketCode" in df.columns:
        df = df.with_columns(pl.col("MarketCode").cast(pl.Utf8, strict=False).alias("market_code"))
    elif "marketcode" in df.columns:
        df = df.with_columns(pl.col("marketcode").cast(pl.Utf8, strict=False).alias("market_code"))
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("market_code"))

    # 市場区分ダミー（MarketCode: 0111=プライム, 0112=スタンダード, 0113=グロース）
    df = df.with_columns(
        [
            (pl.col("market_code") == "0111").cast(pl.Int8).alias("is_prime"),
            (pl.col("market_code") == "0112").cast(pl.Int8).alias("is_standard"),
            (pl.col("market_code") == "0113").cast(pl.Int8).alias("is_growth"),
        ]
    )

    # Sector33Code → sector33_code
    if "Sector33Code" in df.columns:
        df = df.with_columns(
            pl.col("Sector33Code").cast(pl.Utf8, strict=False).fill_null("UNKNOWN").alias("sector33_code")
        )
    elif "sector33code" in df.columns:
        df = df.with_columns(
            pl.col("sector33code").cast(pl.Utf8, strict=False).fill_null("UNKNOWN").alias("sector33_code")
        )
    else:
        df = df.with_columns(pl.lit("UNKNOWN").cast(pl.Utf8).alias("sector33_code"))

    # Sector17Code → sector17_code
    if "Sector17Code" in df.columns:
        df = df.with_columns(
            pl.col("Sector17Code").cast(pl.Utf8, strict=False).fill_null("UNKNOWN").alias("sector17_code")
        )
    elif "sector17code" in df.columns:
        df = df.with_columns(
            pl.col("sector17code").cast(pl.Utf8, strict=False).fill_null("UNKNOWN").alias("sector17_code")
        )
    else:
        df = df.with_columns(pl.lit("UNKNOWN").cast(pl.Utf8).alias("sector17_code"))

    # ScaleCategory → scale_bucket
    if "ScaleCategory" in df.columns:
        df = df.with_columns(pl.col("ScaleCategory").cast(pl.Utf8, strict=False).alias("scale_bucket"))
    elif "scalecategory" in df.columns:
        df = df.with_columns(pl.col("scalecategory").cast(pl.Utf8, strict=False).alias("scale_bucket"))
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("scale_bucket"))

    # MarginCode → is_margin_eligible（取得可能なら）
    if "MarginCode" in df.columns:
        df = df.with_columns((pl.col("MarginCode") == "1").cast(pl.Int8).alias("is_margin_eligible"))
    elif "margincode" in df.columns:
        df = df.with_columns((pl.col("margincode") == "1").cast(pl.Int8).alias("is_margin_eligible"))
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Int8).alias("is_margin_eligible"))

    # 品質フラグ
    df = df.with_columns(
        (
            pl.col("sector33_code").is_not_null()
            & (pl.col("sector33_code") != "UNKNOWN")
            & pl.col("market_code").is_not_null()
        )
        .cast(pl.Int8)
        .alias("is_listed_info_valid")
    )

    # As-of設定: prepare_snapshot_plを使用
    # 当日情報: Date @ 09:00 JST
    # 翌日情報: today 17:30 JST（T+1 09:00まで使用不可）
    # build_dateを基準に"tomorrow"を判定
    if build_date is not None:
        from datetime import datetime, timedelta

        build_date_obj = datetime.strptime(build_date, "%Y-%m-%d").date()
        tomorrow_date = build_date_obj + timedelta(days=1)
        df = df.with_columns(
            (pl.col("date") > pl.lit(tomorrow_date, dtype=pl.Date)).cast(pl.Int8).alias("_is_tomorrow")
        )
    else:
        # build_dateがない場合は、dateが最大の日付より大きいものを"tomorrow"と判定
        max_date = df["date"].max()
        if max_date is not None:
            df = df.with_columns((pl.col("date") > pl.lit(max_date, dtype=pl.Date)).cast(pl.Int8).alias("_is_tomorrow"))
        else:
            df = df.with_columns(pl.lit(0).cast(pl.Int8).alias("_is_tomorrow"))

    # available_tsを計算
    # 当日情報: Date @ 09:00 JST
    # 翌日情報: today 17:30 JST（T+1 09:00まで使用不可）
    df = df.with_columns(
        pl.when(pl.col("_is_tomorrow") == 1)
        .then(
            # 翌日情報: 当日17:30（T+1 09:00まで使用不可）
            # 実際には、当日の日付で17:30を設定
            pl.datetime(
                pl.col("date").dt.year(),
                pl.col("date").dt.month(),
                pl.col("date").dt.day(),
                17,
                30,
                0,
            ).dt.replace_time_zone("Asia/Tokyo")
        )
        .otherwise(
            # 当日情報: Date @ 09:00 JST
            pl.datetime(
                pl.col("date").dt.year(),
                pl.col("date").dt.month(),
                pl.col("date").dt.day(),
                9,
                0,
                0,
            ).dt.replace_time_zone("Asia/Tokyo")
        )
        .alias("available_ts")
    )

    # 一時列を削除
    if "_is_tomorrow" in df.columns:
        df = df.drop("_is_tomorrow")

    # 重複除去（Code, Dateの組み合わせで最新を保持）
    df = df.sort(["code", "date", "available_ts"]).unique(subset=["code", "date"], keep="last")

    return df.sort(["code", "date", "available_ts"])
