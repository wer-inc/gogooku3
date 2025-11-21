"""日経225オプション特徴量生成モジュール（P0: 最小構成）"""

from __future__ import annotations

import logging
from typing import Optional

import polars as pl

from ..utils.asof_join import prepare_snapshot_pl

LOGGER = logging.getLogger(__name__)


def load_index_option_225(df: pl.DataFrame) -> pl.DataFrame:
    """
    日経225オプションデータをロードし、正規化する。

    Args:
        df: J-Quants日経225オプションデータ（/option/index_option）

    Returns:
        正規化されたオプションDataFrame
    """
    if df.is_empty():
        return df

    # 日付列の正規化
    date_cols = ["Date", "date"]
    date_expr = None
    for col in date_cols:
        if col in df.columns:
            date_expr = pl.col(col).cast(pl.Date, strict=False)
            break

    if date_expr is None:
        LOGGER.warning("No date column found in index option data")
        return pl.DataFrame()

    df = df.with_columns(date_expr.alias("Date"))

    # 空文字→Null変換（特にナイトセッション項目）
    # 数値列の正規化
    numeric_cols = [
        "ImpliedVolatility",
        "BaseVolatility",
        "TheoreticalPrice",
        "UnderlyingPrice",
        "StrikePrice",
        "OpenInterest",
        "Volume",
        "WholeDayOpen",
        "WholeDayHigh",
        "WholeDayLow",
        "WholeDayClose",
        "DaySessionOpen",
        "DaySessionHigh",
        "DaySessionLow",
        "DaySessionClose",
        "NightSessionOpen",
        "NightSessionHigh",
        "NightSessionLow",
        "NightSessionClose",
        "InterestRate",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.when(
                    (pl.col(col).cast(pl.Utf8, strict=False) == "")
                    | (pl.col(col).cast(pl.Utf8, strict=False).is_in(["-", "*", "null", "NULL", "None"]))
                )
                .then(None)
                .otherwise(pl.col(col).cast(pl.Float64, strict=False))
                .alias(col)
            )

    # 文字列列の正規化
    string_cols = [
        "Code",
        "ContractMonth",
        "PutCallDivision",
        "LastTradingDay",
        "SpecialQuotationDay",
        "EmergencyMarginTriggerDivision",
    ]
    for col in string_cols:
        if col in df.columns:
            df = df.with_columns(
                pl.when(pl.col(col).cast(pl.Utf8, strict=False) == "")
                .then(None)
                .otherwise(pl.col(col).cast(pl.Utf8, strict=False))
                .alias(col)
            )

    # 一意キー：(Date, Code, EmergencyMarginTriggerDivision)
    # EmergencyMarginTriggerDivisionがNullの場合は空文字に正規化
    if "EmergencyMarginTriggerDivision" in df.columns:
        df = df.with_columns(
            pl.col("EmergencyMarginTriggerDivision").fill_null("").alias("EmergencyMarginTriggerDivision")
        )

    return df


def build_index_option_225_features(
    opt: pl.DataFrame,
    topix_df: Optional[pl.DataFrame] = None,
    trading_calendar: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    日経225オプションのP0特徴量を生成。

    P0フィーチャー:
    1. ATM IV（近月・30D合成）
    2. VRP（IV-RV, IV/RV）
    3. IV Term Structure（近月 vs 次近月）
    4. Put/Call センチメント（OI比、出来高比）
    5. Skew（25Δ近似）
    6. 満期までの日数
    7. ナイト→日中のIVジャンプ

    Args:
        opt: 正規化済み日経225オプションデータ
        topix_df: TOPIX現物データ（Date, Close列が必要、VRP計算用）
        trading_calendar: 営業日カレンダー（date列が必要、days_to_sq計算用）

    Returns:
        特徴量DataFrame（date, available_ts, idxopt_* 列）
    """
    if opt.is_empty():
        return pl.DataFrame(schema={"date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")})

    # 残存日数の計算（SpecialQuotationDay - Date）
    if "SpecialQuotationDay" in opt.columns and "Date" in opt.columns:
        opt = opt.with_columns(
            (pl.col("SpecialQuotationDay").cast(pl.Date, strict=False) - pl.col("Date"))
            .dt.total_days()
            .alias("days_to_sq_raw")
        )
        # 営業日換算が必要な場合は後で計算
        opt = opt.with_columns(pl.col("days_to_sq_raw").alias("days_to_sq"))
    else:
        opt = opt.with_columns(pl.lit(None).cast(pl.Int64).alias("days_to_sq"))

    # Moneynessの計算（|ln(K/S)|）
    if "StrikePrice" in opt.columns and "UnderlyingPrice" in opt.columns:
        opt = opt.with_columns(
            ((pl.col("StrikePrice") / (pl.col("UnderlyingPrice") + 1e-9)).log().abs()).alias("moneyness_abs")
        )
    else:
        opt = opt.with_columns(pl.lit(None).cast(pl.Float64).alias("moneyness_abs"))

    # 日次集約（Pythonで処理）
    feature_list = []

    for (date_val,), group in opt.group_by("Date", maintain_order=True):
        if group.is_empty():
            continue

        # 基本情報
        underlying_price = group["UnderlyingPrice"].median() if "UnderlyingPrice" in group.columns else None
        if underlying_price is None or underlying_price <= 0:
            continue

        # 1. ATM IV（近月・30D合成）
        # BaseVolatilityが存在する場合は優先、なければATM近傍のImpliedVolatilityを使用
        # 近月（CentralContractMonthFlag=="1"またはdays_to_sqが最小）
        near_days = group.filter(pl.col("days_to_sq").is_not_null()).select(pl.col("days_to_sq").min())
        near_days_val = near_days.item() if not near_days.is_empty() else None

        if near_days_val is not None:
            near_group = group.filter(
                (pl.col("days_to_sq") >= near_days_val - 5) & (pl.col("days_to_sq") <= near_days_val + 5)
            )
        else:
            near_group = group

        # ATM近傍（moneyness_abs <= 0.05）でフィルタ
        near_atm = near_group.filter((pl.col("moneyness_abs").is_not_null()) & (pl.col("moneyness_abs") <= 0.05))

        # BaseVolatilityから近月IVを取得
        iv_atm_near = None
        if not near_atm.is_empty() and "BaseVolatility" in near_atm.columns:
            base_vol = near_atm.filter(pl.col("BaseVolatility").is_not_null())["BaseVolatility"]
            if not base_vol.is_empty():
                iv_atm_near = base_vol.median()
        elif not near_atm.is_empty() and "ImpliedVolatility" in near_atm.columns:
            # ImpliedVolatilityからOI加重平均
            iv_valid = near_atm.filter(pl.col("ImpliedVolatility").is_not_null() & pl.col("OpenInterest").is_not_null())
            if not iv_valid.is_empty():
                iv_sum = (iv_valid["ImpliedVolatility"] * iv_valid["OpenInterest"]).sum()
                oi_sum = iv_valid["OpenInterest"].sum()
                if oi_sum is not None and oi_sum > 0:
                    iv_atm_near = iv_sum / oi_sum

        # 30D合成IV（近月/次近月を平方根時間補正で内挿）
        iv_atm_30d = None
        if iv_atm_near is not None and near_days_val is not None:
            # 次近月を探す（days_to_sqが30-90日の範囲）
            next_group = group.filter(
                (pl.col("days_to_sq").is_not_null())
                & (pl.col("days_to_sq") >= 30)
                & (pl.col("days_to_sq") <= 90)
                & (pl.col("days_to_sq") > near_days_val)
            )
            if not next_group.is_empty():
                next_atm = next_group.filter(
                    (pl.col("moneyness_abs").is_not_null()) & (pl.col("moneyness_abs") <= 0.05)
                )
                if not next_atm.is_empty():
                    # BaseVolatilityまたはImpliedVolatilityから次近月IVを取得
                    iv_atm_2nd = None
                    if "BaseVolatility" in next_atm.columns:
                        base_vol_2nd = next_atm.filter(pl.col("BaseVolatility").is_not_null())["BaseVolatility"]
                        if not base_vol_2nd.is_empty():
                            iv_atm_2nd = base_vol_2nd.median()
                    elif "ImpliedVolatility" in next_atm.columns:
                        iv_valid_2nd = next_atm.filter(
                            pl.col("ImpliedVolatility").is_not_null() & pl.col("OpenInterest").is_not_null()
                        )
                        if not iv_valid_2nd.is_empty():
                            iv_sum_2nd = (iv_valid_2nd["ImpliedVolatility"] * iv_valid_2nd["OpenInterest"]).sum()
                            oi_sum_2nd = iv_valid_2nd["OpenInterest"].sum()
                            if oi_sum_2nd is not None and oi_sum_2nd > 0:
                                iv_atm_2nd = iv_sum_2nd / oi_sum_2nd

                    if iv_atm_2nd is not None:
                        # 平方根時間補正で30日へ内挿
                        days_2nd = next_group["days_to_sq"].median()
                        if days_2nd is not None and days_2nd > near_days_val:
                            # IV * sqrt(T) = const の関係から
                            iv_near_scaled = iv_atm_near * (near_days_val / 30.0) ** 0.5
                            iv_2nd_scaled = iv_atm_2nd * (days_2nd / 30.0) ** 0.5
                            # 線形補間
                            weight = (30.0 - near_days_val) / (days_2nd - near_days_val)
                            iv_atm_30d = iv_near_scaled * (1 - weight) + iv_2nd_scaled * weight

        if iv_atm_30d is None and iv_atm_near is not None:
            # フォールバック: 近月IVをそのまま使用（時間補正）
            if near_days_val is not None:
                iv_atm_30d = iv_atm_near * (near_days_val / 30.0) ** 0.5

        # 2. IV Term Structure（次近月 - 近月）
        iv_ts_slope = None
        if iv_atm_near is not None:
            next_group_ts = (
                group.filter(
                    (pl.col("days_to_sq").is_not_null())
                    & (pl.col("days_to_sq") >= 30)
                    & (pl.col("days_to_sq") <= 90)
                    & (pl.col("days_to_sq") > near_days_val)
                )
                if near_days_val is not None
                else pl.DataFrame()
            )

            if not next_group_ts.is_empty():
                next_atm_ts = next_group_ts.filter(
                    (pl.col("moneyness_abs").is_not_null()) & (pl.col("moneyness_abs") <= 0.05)
                )
                if not next_atm_ts.is_empty():
                    iv_atm_2nd_ts = None
                    if "BaseVolatility" in next_atm_ts.columns:
                        base_vol_2nd_ts = next_atm_ts.filter(pl.col("BaseVolatility").is_not_null())["BaseVolatility"]
                        if not base_vol_2nd_ts.is_empty():
                            iv_atm_2nd_ts = base_vol_2nd_ts.median()
                    elif "ImpliedVolatility" in next_atm_ts.columns:
                        iv_valid_2nd_ts = next_atm_ts.filter(
                            pl.col("ImpliedVolatility").is_not_null() & pl.col("OpenInterest").is_not_null()
                        )
                        if not iv_valid_2nd_ts.is_empty():
                            iv_sum_2nd_ts = (
                                iv_valid_2nd_ts["ImpliedVolatility"] * iv_valid_2nd_ts["OpenInterest"]
                            ).sum()
                            oi_sum_2nd_ts = iv_valid_2nd_ts["OpenInterest"].sum()
                            if oi_sum_2nd_ts is not None and oi_sum_2nd_ts > 0:
                                iv_atm_2nd_ts = iv_sum_2nd_ts / oi_sum_2nd_ts

                    if iv_atm_2nd_ts is not None:
                        iv_ts_slope = iv_atm_2nd_ts - iv_atm_near

        # 3. Put/Call センチメント（OI比、出来高比）
        # 同一日・同一限月・同一ストライクでペアリング
        pc_oi_ratio = None
        pc_vol_ratio = None

        # PutCallDivision: 1=Put, 2=Call
        if "PutCallDivision" in group.columns:
            puts = group.filter(pl.col("PutCallDivision") == "1")
            calls = group.filter(pl.col("PutCallDivision") == "2")

            if not puts.is_empty() and not calls.is_empty():
                # OI比
                if "OpenInterest" in puts.columns and "OpenInterest" in calls.columns:
                    oi_put_sum = puts["OpenInterest"].sum()
                    oi_call_sum = calls["OpenInterest"].sum()
                    if oi_call_sum is not None and oi_call_sum > 0:
                        pc_oi_ratio = oi_put_sum / oi_call_sum

                # 出来高比
                if "Volume" in puts.columns and "Volume" in calls.columns:
                    vol_put_sum = puts["Volume"].sum()
                    vol_call_sum = calls["Volume"].sum()
                    if vol_call_sum is not None and vol_call_sum > 0:
                        pc_vol_ratio = vol_put_sum / vol_call_sum

        # 4. Skew（25Δ近似、±10% OTMのIV差で代用）
        skew_25 = None
        rr_25 = None
        if "ImpliedVolatility" in group.columns and "StrikePrice" in group.columns:
            # OTM Put（10% OTM、K < S * 0.9）
            otm_puts = group.filter(
                (pl.col("PutCallDivision") == "1")
                & (pl.col("StrikePrice") < underlying_price * 0.9)
                & (pl.col("StrikePrice") >= underlying_price * 0.85)
                & (pl.col("ImpliedVolatility").is_not_null())
            )
            # OTM Call（10% OTM、K > S * 1.1）
            otm_calls = group.filter(
                (pl.col("PutCallDivision") == "2")
                & (pl.col("StrikePrice") > underlying_price * 1.1)
                & (pl.col("StrikePrice") <= underlying_price * 1.15)
                & (pl.col("ImpliedVolatility").is_not_null())
            )

            if not otm_puts.is_empty() and not otm_calls.is_empty():
                iv_put_otm = otm_puts["ImpliedVolatility"].median()
                iv_call_otm = otm_calls["ImpliedVolatility"].median()
                if iv_put_otm is not None and iv_call_otm is not None:
                    skew_25 = iv_put_otm - iv_call_otm
                    # Risk Reversal 25Δ: call wing IV minus put wing IV（一般定義に合わせ符号反転）
                    rr_25 = iv_call_otm - iv_put_otm

        # 5. 満期までの日数（営業日換算）
        days_to_sq = None
        if "days_to_sq" in group.columns:
            days_to_sq = group["days_to_sq"].min()

        # 6. ナイト→日中のIVジャンプ
        iv_night_jump = None
        if "NightSessionClose" in group.columns and "DaySessionClose" in group.columns:
            # ナイト終値のIV（BaseVolatilityまたはImpliedVolatility）
            night_data = group.filter(pl.col("NightSessionClose").is_not_null())
            if not night_data.is_empty():
                iv_night = None
                if "BaseVolatility" in night_data.columns:
                    base_vol_night = night_data.filter(pl.col("BaseVolatility").is_not_null())["BaseVolatility"]
                    if not base_vol_night.is_empty():
                        iv_night = base_vol_night.median()
                elif "ImpliedVolatility" in night_data.columns:
                    iv_night_data = night_data.filter(
                        (pl.col("ImpliedVolatility").is_not_null())
                        & (pl.col("moneyness_abs").is_not_null())
                        & (pl.col("moneyness_abs") <= 0.05)
                    )
                    if not iv_night_data.is_empty():
                        iv_night = iv_night_data["ImpliedVolatility"].median()

                # 日中終値のIV
                day_data = group.filter(pl.col("DaySessionClose").is_not_null())
                if not day_data.is_empty():
                    iv_day = None
                    if "BaseVolatility" in day_data.columns:
                        base_vol_day = day_data.filter(pl.col("BaseVolatility").is_not_null())["BaseVolatility"]
                        if not base_vol_day.is_empty():
                            iv_day = base_vol_day.median()
                    elif "ImpliedVolatility" in day_data.columns:
                        iv_day_data = day_data.filter(
                            (pl.col("ImpliedVolatility").is_not_null())
                            & (pl.col("moneyness_abs").is_not_null())
                            & (pl.col("moneyness_abs") <= 0.05)
                        )
                        if not iv_day_data.is_empty():
                            iv_day = iv_day_data["ImpliedVolatility"].median()

                    if iv_night is not None and iv_day is not None:
                        iv_night_jump = iv_day - iv_night

        # 7. VRP（IV - RV, IV / RV）は後で計算（TOPIXデータが必要）

        feature_list.append(
            {
                "date": date_val,
                "idxopt_iv_atm_near": iv_atm_near,
                "idxopt_iv_atm_30d": iv_atm_30d,
                "idxopt_iv_ts_slope": iv_ts_slope,
                "idxopt_pc_oi_ratio": pc_oi_ratio,
                "idxopt_pc_vol_ratio": pc_vol_ratio,
                "idxopt_skew_25": skew_25,
                "idxopt_rr_25": rr_25,
                "idxopt_days_to_sq": days_to_sq,
                "idxopt_iv_night_jump": iv_night_jump,
                "idxopt_underlying_price": underlying_price,
            }
        )

    if not feature_list:
        return pl.DataFrame(schema={"date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")})

    features_df = pl.DataFrame(feature_list)

    # VRPの計算（TOPIXデータが必要）
    if topix_df is not None and "idxopt_iv_atm_30d" in features_df.columns:
        # TOPIXの実現ボラティリティを計算（20営業日）
        # 既存の実装を参照して、topix_dfから実現ボラティリティを計算
        # ここでは簡易的に既存の特徴量（topix_realized_vol_20d）が存在することを前提とする
        # 実際の実装では、topix_dfから直接計算する必要がある
        pass  # VRPは後で統合時に計算

    # available_tsの設定
    # 日中終値ベース: T+0 15:10 JST
    # ナイトセッション含む特徴: T+0 06:00 JST（別フラグ管理）
    features_df = prepare_snapshot_pl(
        features_df,
        published_date_col="date",
        availability_hour=15,
        availability_minute=10,
    )

    return features_df
