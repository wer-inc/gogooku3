"""オプション特徴量生成モジュール（P0: 最小構成）"""

from __future__ import annotations

import logging

import polars as pl

from ..utils.asof_join import prepare_snapshot_pl

LOGGER = logging.getLogger(__name__)


def load_options(
    df: pl.DataFrame,
    categories: list[str] | None = None,
) -> pl.DataFrame:
    """
    オプションデータをロードし、正規化する。

    Args:
        df: J-Quantsオプションデータ（get_index_optionで取得）
        categories: 商品区分リスト（デフォルト: ["TOPIXE", "NK225E"]）

    Returns:
        正規化されたオプションDataFrame
    """
    if df.is_empty():
        return df

    if categories is None:
        categories = ["TOPIXE", "NK225E"]

    # カテゴリでフィルタ（DerivativesProductCategory または ProductCategory）
    category_col = "DerivativesProductCategory" if "DerivativesProductCategory" in df.columns else "ProductCategory"
    if category_col in df.columns:
        df = df.filter(pl.col(category_col).is_in(categories))
    else:
        LOGGER.warning("No category column found in options data")
        return pl.DataFrame()

    # 日付列の正規化
    date_cols = ["Date", "date"]
    date_expr = None
    for col in date_cols:
        if col in df.columns:
            date_expr = pl.col(col).cast(pl.Date, strict=False)
            break

    if date_expr is None:
        LOGGER.warning("No date column found in options data")
        return pl.DataFrame()

    df = df.with_columns(date_expr.alias("Date"))

    # 数値列の正規化（空文字→Null）
    numeric_cols = [
        "ImpliedVolatility",
        "TheoreticalPrice",
        "UnderlyingPrice",
        "StrikePrice",
        "OpenInterest",
        "Volume",
        "WholeDayOpen",
        "WholeDayHigh",
        "WholeDayLow",
        "WholeDayClose",
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
        "PutCallDivision",
        "LastTradingDay",
        "SpecialQuotationDay",
        "EmergencyMarginTriggerDivision",
    ]
    for col in string_cols:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Utf8, strict=False))

    return df


def select_near_next_contracts(df: pl.DataFrame) -> pl.DataFrame:
    """
    同日・同カテゴリ内で近月/次月限月を選定する。

    ルール:
    1. CentralContractMonthFlag=="1" を近月（30D）
    2. 残りで残存日数が30-90日の範囲で最小を次月（90D）
    3. 残存日数が10-45日の範囲でフィルタ

    Args:
        df: 同日・同カテゴリのオプションデータ

    Returns:
        近月/次月を横持ちにしたDataFrame
    """
    if df.is_empty():
        return pl.DataFrame()

    # 日付を取得
    date_val = df["Date"].head(1).item() if "Date" in df.columns else None
    if date_val is None:
        return pl.DataFrame()

    category_val = None
    category_col = "DerivativesProductCategory" if "DerivativesProductCategory" in df.columns else "ProductCategory"
    if category_col in df.columns:
        category_val = df[category_col].head(1).item()

    # 残存日数を計算
    if "SpecialQuotationDay" in df.columns and "Date" in df.columns:
        df = df.with_columns(
            (pl.col("SpecialQuotationDay").cast(pl.Date, strict=False) - pl.col("Date"))
            .dt.total_days()
            .alias("days_to_sq")
        )
        # 10-45日の範囲でフィルタ
        df = df.filter((pl.col("days_to_sq") >= 10) & (pl.col("days_to_sq") <= 45))
    else:
        df = df.with_columns(pl.lit(None).cast(pl.Int64).alias("days_to_sq"))

    # 近月の選定（CentralContractMonthFlag=="1"）
    near_candidates = df.filter(pl.col("CentralContractMonthFlag") == "1")
    if near_candidates.is_empty():
        # フォールバック: 残存日数が最小
        if "days_to_sq" in df.columns:
            near_candidates = df.sort("days_to_sq").head(1)
        else:
            near_candidates = df.head(1)

    # 次月の選定（残存日数30-90日の範囲で最小、または近月以外で最小）
    if "days_to_sq" in df.columns:
        next_candidates = df.filter((pl.col("days_to_sq") >= 30) & (pl.col("days_to_sq") <= 90))
        if next_candidates.is_empty():
            # フォールバック: 近月以外で残存日数が最小
            near_contract = near_candidates["ContractMonth"].head(1).item() if not near_candidates.is_empty() else None
            if near_contract:
                next_candidates = df.filter(pl.col("ContractMonth") != near_contract).sort("days_to_sq").head(1)
            else:
                next_candidates = df.sort("days_to_sq").head(1)
    else:
        next_candidates = pl.DataFrame()

    # 結果を構築
    result_data = {
        "date": [date_val],
        "category": [category_val],
    }

    if not near_candidates.is_empty():
        near_row = near_candidates.head(1)
        result_data["near_days_to_sq"] = [
            near_row["days_to_sq"].head(1).item() if "days_to_sq" in near_row.columns else None
        ]
    else:
        result_data["near_days_to_sq"] = [None]

    if not next_candidates.is_empty():
        next_row = next_candidates.head(1)
        result_data["next_days_to_sq"] = [
            next_row["days_to_sq"].head(1).item() if "days_to_sq" in next_row.columns else None
        ]
    else:
        result_data["next_days_to_sq"] = [None]

    return pl.DataFrame(result_data)


def build_option_signals(
    opt: pl.DataFrame,
    topix_df: pl.DataFrame | None = None,
    nk225_df: pl.DataFrame | None = None,
    trading_calendar: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """
    オプションのP0特徴量を生成。

    Args:
        opt: 正規化済みオプションデータ
        topix_df: TOPIX現物データ（Date, Close列が必要、VRP計算用）
        nk225_df: 日経225現物データ（Date, Close列が必要、VRP計算用）
        trading_calendar: 営業日カレンダー（date列が必要）

    Returns:
        特徴量DataFrame（date, available_ts, macro_opt_* 列）
    """
    if opt.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")})

    # カテゴリ列の確認
    category_col = "DerivativesProductCategory" if "DerivativesProductCategory" in opt.columns else "ProductCategory"
    if category_col not in opt.columns:
        LOGGER.warning("No category column found in options data")
        return pl.DataFrame(schema={"date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")})

    # 残存日数の計算
    if "SpecialQuotationDay" in opt.columns and "Date" in opt.columns:
        opt = opt.with_columns(
            (pl.col("SpecialQuotationDay").cast(pl.Date, strict=False) - pl.col("Date"))
            .dt.total_days()
            .alias("days_to_sq")
        )
        # 10-90日の範囲でフィルタ（近月10-45日、次月30-90日）
        opt = opt.filter((pl.col("days_to_sq") >= 10) & (pl.col("days_to_sq") <= 90))
    else:
        opt = opt.with_columns(pl.lit(None).cast(pl.Int64).alias("days_to_sq"))

    # Moneynessの計算（|ln(K/S)|）
    if "StrikePrice" in opt.columns and "UnderlyingPrice" in opt.columns:
        opt = opt.with_columns(((pl.col("StrikePrice") / pl.col("UnderlyingPrice")).log().abs()).alias("moneyness_abs"))
        # ATM近傍（5%）でフィルタ
        opt = opt.filter(pl.col("moneyness_abs") <= 0.05)
    else:
        opt = opt.with_columns(pl.lit(None).cast(pl.Float64).alias("moneyness_abs"))

    # 日次・カテゴリ別に集約（Pythonで処理）
    feature_list = []

    for (date_val, cat_val), group in opt.group_by(["Date", category_col], maintain_order=True):
        if group.is_empty():
            continue

        # カテゴリ名の正規化
        under_name = "topix" if cat_val in ["TOPIXE", "TOPIX"] else "nk225" if cat_val in ["NK225E", "NK225"] else None
        if under_name is None:
            continue

        # 近月の識別（CentralContractMonthFlag=="1"またはdays_to_sqが最小）
        central_flags = group.filter(pl.col("CentralContractMonthFlag") == "1")
        if not central_flags.is_empty():
            near_group = central_flags
        else:
            # 残存日数が最小のものを近月とする
            min_days = group.filter(pl.col("days_to_sq").is_not_null()).select(pl.col("days_to_sq").min()).item()
            if min_days is not None:
                near_group = group.filter(pl.col("days_to_sq") == min_days)
            else:
                near_group = group.head(1)

        # 次月の識別（30-90日の範囲）
        if "days_to_sq" in group.columns:
            next_group = group.filter(
                (pl.col("days_to_sq") >= 30)
                & (pl.col("days_to_sq") <= 90)
                & (pl.col("CentralContractMonthFlag") != "1")
            )
        else:
            next_group = pl.DataFrame()

        # 近月（30D）のIV（OI加重平均）
        near_iv = None
        if (
            not near_group.is_empty()
            and "ImpliedVolatility" in near_group.columns
            and "OpenInterest" in near_group.columns
        ):
            near_valid = near_group.filter(
                pl.col("ImpliedVolatility").is_not_null() & pl.col("OpenInterest").is_not_null()
            )
            if not near_valid.is_empty():
                iv_sum = (near_valid["ImpliedVolatility"] * near_valid["OpenInterest"]).sum()
                oi_sum = near_valid["OpenInterest"].sum()
                if oi_sum is not None and oi_sum > 0:
                    near_iv = iv_sum / oi_sum

        # 次月（90D）のIV（同様にOI加重平均）
        next_iv = None
        if (
            not next_group.is_empty()
            and "ImpliedVolatility" in next_group.columns
            and "OpenInterest" in next_group.columns
        ):
            next_valid = next_group.filter(
                pl.col("ImpliedVolatility").is_not_null() & pl.col("OpenInterest").is_not_null()
            )
            if not next_valid.is_empty():
                iv_sum = (next_valid["ImpliedVolatility"] * next_valid["OpenInterest"]).sum()
                oi_sum = next_valid["OpenInterest"].sum()
                if oi_sum is not None and oi_sum > 0:
                    next_iv = iv_sum / oi_sum

        # IVターム構造
        iv_tslope = (next_iv - near_iv) if (next_iv is not None and near_iv is not None) else None

        # IVスキュー（±5% モネネス）
        skew25 = None
        if (
            not near_group.is_empty()
            and "UnderlyingPrice" in near_group.columns
            and "StrikePrice" in near_group.columns
        ):
            # Put (K < S * 0.95)
            put_group = near_group.filter((pl.col("StrikePrice") / pl.col("UnderlyingPrice")) <= 0.95)
            # Call (K > S * 1.05)
            call_group = near_group.filter((pl.col("StrikePrice") / pl.col("UnderlyingPrice")) >= 1.05)

            put_iv_val = (
                put_group.filter(pl.col("ImpliedVolatility").is_not_null())
                .select(pl.col("ImpliedVolatility").mean())
                .item()
                if not put_group.is_empty()
                else None
            )
            call_iv_val = (
                call_group.filter(pl.col("ImpliedVolatility").is_not_null())
                .select(pl.col("ImpliedVolatility").mean())
                .item()
                if not call_group.is_empty()
                else None
            )
            skew25 = (put_iv_val - call_iv_val) if (put_iv_val is not None and call_iv_val is not None) else None

        # プット・コール比（建玉）
        pc_oi = None
        if (
            not near_group.is_empty()
            and "OpenInterest" in near_group.columns
            and "PutCallDivision" in near_group.columns
        ):
            put_oi = near_group.filter(pl.col("PutCallDivision") == "1").select(pl.col("OpenInterest").sum()).item()
            call_oi = near_group.filter(pl.col("PutCallDivision") == "2").select(pl.col("OpenInterest").sum()).item()
            if call_oi is not None and call_oi > 0 and put_oi is not None:
                pc_oi = put_oi / call_oi

        # プット・コール比（出来高）
        pc_vol = None
        if not near_group.is_empty() and "Volume" in near_group.columns and "PutCallDivision" in near_group.columns:
            put_vol = near_group.filter(pl.col("PutCallDivision") == "1").select(pl.col("Volume").sum()).item()
            call_vol = near_group.filter(pl.col("PutCallDivision") == "2").select(pl.col("Volume").sum()).item()
            if call_vol is not None and call_vol > 0 and put_vol is not None:
                pc_vol = put_vol / call_vol

        feature_list.append(
            {
                "date": date_val,
                "category": cat_val,
                "under_name": under_name,
                f"macro_opt_{under_name}_iv30": near_iv,
                f"macro_opt_{under_name}_iv90": next_iv,
                f"macro_opt_{under_name}_iv_tslope": iv_tslope,
                f"macro_opt_{under_name}_skew25": skew25,
                f"macro_opt_{under_name}_pc_oi": pc_oi,
                f"macro_opt_{under_name}_pc_vol": pc_vol,
            }
        )

    if not feature_list:
        return pl.DataFrame(schema={"date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")})

    # DataFrameに変換してwide形式に展開
    df_features = pl.DataFrame(feature_list)

    # Wide形式に展開（各日付ごとに全カテゴリの特徴量を横持ち）
    wide_features = []
    for (date_val,), group in df_features.group_by(["date"], maintain_order=True):
        row_data = {"date": date_val}
        for under_name in ["topix", "nk225"]:
            under_data = group.filter(pl.col("under_name") == under_name)
            if not under_data.is_empty():
                for col in under_data.columns:
                    if col.startswith("macro_opt_"):
                        row_data[col] = under_data[col].head(1).item()
        wide_features.append(row_data)

    if not wide_features:
        return pl.DataFrame(schema={"date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")})

    result_df = pl.DataFrame(wide_features).sort("date")

    # VRP計算（30日実現分散が必要）
    # これは既存の指数データから計算する必要があるため、ここではスキップ
    # 後で別途実装

    # as-of スナップショット化（T+1 09:00 JST）
    # Date列をPublishedDateとして扱う
    result_df = result_df.with_columns(pl.col("date").alias("PublishedDate"))
    result_df = prepare_snapshot_pl(
        result_df,
        published_date_col="PublishedDate",
        trading_calendar=trading_calendar,
        availability_hour=9,
        availability_minute=0,
    )

    LOGGER.info(
        "Generated option features: %d rows, %d features",
        len(result_df),
        len([c for c in result_df.columns if c.startswith("macro_opt_")]),
    )

    return result_df
