"""投資部門別売買状況特徴量生成モジュール（MVP: 最小構成）"""
from __future__ import annotations

import logging
from typing import Optional

import polars as pl

from ..utils.asof_join import prepare_snapshot_pl
from ..utils.rolling import roll_mean_safe, roll_std_safe

LOGGER = logging.getLogger(__name__)

# MarketCode → Section マッピング（2022年4月4日以降の市場区分）
MARKET_CODE_TO_SECTION = {
    # 新市場区分（2022年4月4日以降）
    "0111": "TSEPrime",  # プライム
    "0112": "TSEStandard",  # スタンダード
    "0113": "TSEGrowth",  # グロース
    # 旧市場区分（2022年4月3日まで）
    "0101": "TSE1st",  # 東証一部（→TSEPrime相当）
    "0102": "TSE2nd",  # 東証二部（→TSEStandard相当）
    "0104": "TSEMothers",  # マザーズ（→TSEGrowth相当）
    "0105": "TSEPro",  # TOKYO PRO MARKET
    "0106": "JASDAQStandard",  # JASDAQ スタンダード（→TSEStandard相当）
    "0107": "JASDAQGrowth",  # JASDAQ グロース（→TSEGrowth相当）
    "0109": "Other",  # その他
    # 地方市場
    "0301": "NSEPremier",  # 名証プレミア
    "0302": "NSEMain",  # 名証メイン
    "0303": "NSENext",  # 名証ネクスト
    "0304": "NSEOther",  # 名証その他
    "0501": "SSEMain",  # 札証本則
    "0502": "SSEAmbitious",  # 札証アンビシャス
    "0503": "SSEOther",  # 札証その他
    "0701": "FSEMain",  # 福証本則
    "0702": "FSEQBoard",  # 福証Q-Board
    "0703": "FSEOther",  # 福証その他
}


def map_market_code_to_section(df: pl.DataFrame, market_code_col: str = "market_code") -> pl.DataFrame:
    """
    MarketCodeをSectionに変換する。

    Args:
        df: MarketCode列を含むDataFrame
        market_code_col: MarketCode列名（デフォルト: "market_code"）

    Returns:
        Section列を追加したDataFrame
    """
    if market_code_col not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Utf8).alias("section"))

    # MarketCode → Section マッピング
    mapping_expr = pl.col(market_code_col).map_elements(
        lambda x: MARKET_CODE_TO_SECTION.get(str(x), None) if x is not None else None,
        return_dtype=pl.Utf8,
    )

    return df.with_columns(mapping_expr.alias("section"))


def load_trades_spec(df: pl.DataFrame) -> pl.DataFrame:
    """
    投資部門別売買状況データをロードし、正規化する。

    Args:
        df: J-Quants投資部門別売買状況データ（/markets/trades_spec）

    Returns:
        正規化されたDataFrame
    """
    if df.is_empty():
        return df

    # 列名の正規化
    col_map = {
        "PublishedDate": "published_date",
        "StartDate": "start_date",
        "EndDate": "end_date",
        "Section": "section",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename({old: new})

    # 日付列の正規化
    for col in ["published_date", "start_date", "end_date"]:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Date, strict=False).alias(col))

    # Sectionの正規化
    if "section" in df.columns:
        df = df.with_columns(pl.col("section").cast(pl.Utf8, strict=False))

    # 数値列の正規化（空文字→Null）
    # 投資主体別の売買金額列（Valueサフィックス付き）
    numeric_candidates = [
        "ForeignersPurchaseValue",
        "ForeignersSalesValue",
        "IndividualPurchaseValue",
        "IndividualSalesValue",
        "InvestmentTrustsPurchaseValue",
        "InvestmentTrustsSalesValue",
        "TrustBanksPurchaseValue",
        "TrustBanksSalesValue",
        "SecuritiesCompaniesPurchaseValue",
        "SecuritiesCompaniesSalesValue",
        "ProprietaryPurchaseValue",
        "ProprietarySalesValue",
        "BusinessCorporationsPurchaseValue",
        "BusinessCorporationsSalesValue",
        "OtherFinancialInstitutionsPurchaseValue",
        "OtherFinancialInstitutionsSalesValue",
        "TotalPurchaseValue",
        "TotalSalesValue",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df = df.with_columns(
                pl.when(
                    (pl.col(col).cast(pl.Utf8, strict=False).str.strip() == "")
                    | (pl.col(col).cast(pl.Utf8, strict=False).str.strip().is_in(["-", "*", "null", "NULL", "None"]))
                )
                .then(None)
                .otherwise(pl.col(col).cast(pl.Float64, strict=False))
                .alias(col)
            )

    return df


def build_trades_spec_features(
    trades_spec_df: pl.DataFrame,
    trading_calendar: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """
    投資部門別売買状況のMVP特徴量を生成。

    MVPフィーチャー:
    - 主体別ネット: flow_foreigners_net, flow_individuals_net, flow_trust_banks_net, flow_investment_trusts_net, flow_total_net
    - ネット比率: flow_foreigners_net_ratio, etc.
    - z-score: *_z13 (13週), *_z52 (52週)
    - モメンタム: *_wow, *_turn_flag
    - ダイバージェンス: flow_divergence_foreigners_vs_individuals
    - 観測性/鮮度: is_trades_spec_valid, trades_spec_staleness_bd

    Args:
        trades_spec_df: 正規化済み投資部門別売買状況データ
        trading_calendar: 営業日カレンダー（date列が必要、as-of計算用）

    Returns:
        特徴量DataFrame（section, date, available_ts, mkt_flow_* 列）
    """
    if trades_spec_df.is_empty():
        return pl.DataFrame(
            schema={"section": pl.Utf8, "date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")}
        )

    # 必須列の確認
    required = ["published_date", "section"]
    if not all(col in trades_spec_df.columns for col in required):
        LOGGER.warning("Required columns missing in trades_spec data")
        return pl.DataFrame(
            schema={"section": pl.Utf8, "date": pl.Date, "available_ts": pl.Datetime("us", "Asia/Tokyo")}
        )

    # 日付の決定（EndDateを代表日として使用）
    if "end_date" in trades_spec_df.columns:
        trades_spec_df = trades_spec_df.with_columns(pl.col("end_date").alias("date"))
    elif "start_date" in trades_spec_df.columns:
        trades_spec_df = trades_spec_df.with_columns(pl.col("start_date").alias("date"))
    else:
        trades_spec_df = trades_spec_df.with_columns(pl.col("published_date").alias("date"))

    # SectionとDateでソート
    trades_spec_df = trades_spec_df.sort(["section", "date"])

    # 投資主体別の列名解決（Valueサフィックス付きを優先）
    def _resolve_col(candidates: list[str]) -> Optional[str]:
        for candidate in candidates:
            if candidate in trades_spec_df.columns:
                return candidate
        return None

    # 主体別の購入・売却列
    foreigners_purchase = _resolve_col(["ForeignersPurchaseValue", "ForeignersPurchases"])
    foreigners_sales = _resolve_col(["ForeignersSalesValue", "ForeignersSales"])
    individuals_purchase = _resolve_col(["IndividualPurchaseValue", "IndividualsPurchases", "IndividualPurchases"])
    individuals_sales = _resolve_col(["IndividualSalesValue", "IndividualsSales"])
    trust_banks_purchase = _resolve_col(["TrustBanksPurchaseValue", "TrustBanksPurchases"])
    trust_banks_sales = _resolve_col(["TrustBanksSalesValue", "TrustBanksSales"])
    investment_trusts_purchase = _resolve_col(["InvestmentTrustsPurchaseValue", "InvestmentTrustsPurchases"])
    investment_trusts_sales = _resolve_col(["InvestmentTrustsSalesValue", "InvestmentTrustsSales"])
    total_purchase = _resolve_col(["TotalPurchaseValue", "TotalPurchases"])
    total_sales = _resolve_col(["TotalSalesValue", "TotalSales"])

    # 欠損列を0で埋める
    missing_cols = []
    for col_name, col in [
        ("foreigners_purchase", foreigners_purchase),
        ("foreigners_sales", foreigners_sales),
        ("individuals_purchase", individuals_purchase),
        ("individuals_sales", individuals_sales),
        ("trust_banks_purchase", trust_banks_purchase),
        ("trust_banks_sales", trust_banks_sales),
        ("investment_trusts_purchase", investment_trusts_purchase),
        ("investment_trusts_sales", investment_trusts_sales),
        ("total_purchase", total_purchase),
        ("total_sales", total_sales),
    ]:
        if col is None:
            missing_cols.append(pl.lit(0.0).cast(pl.Float64).alias(col_name))
        else:
            missing_cols.append(pl.col(col).fill_null(0.0).alias(col_name))

    trades_spec_df = trades_spec_df.with_columns(missing_cols)

    eps = 1e-9

    # 1. 主体別ネット（購入 - 売却）
    trades_spec_df = trades_spec_df.with_columns(
        [
            (pl.col("foreigners_purchase") - pl.col("foreigners_sales")).alias("flow_foreigners_net"),
            (pl.col("individuals_purchase") - pl.col("individuals_sales")).alias("flow_individuals_net"),
            (pl.col("trust_banks_purchase") - pl.col("trust_banks_sales")).alias("flow_trust_banks_net"),
            (pl.col("investment_trusts_purchase") - pl.col("investment_trusts_sales")).alias(
                "flow_investment_trusts_net"
            ),
            (pl.col("total_purchase") - pl.col("total_sales")).alias("flow_total_net"),
            (pl.col("total_purchase") + pl.col("total_sales")).alias("flow_total_volume"),
        ]
    )

    # 2. ネット比率（総売買に対するシェア）
    trades_spec_df = trades_spec_df.with_columns(
        [
            (pl.col("flow_foreigners_net") / (pl.col("flow_total_volume") + eps)).alias("flow_foreigners_net_ratio"),
            (pl.col("flow_individuals_net") / (pl.col("flow_total_volume") + eps)).alias("flow_individuals_net_ratio"),
            (pl.col("flow_trust_banks_net") / (pl.col("flow_total_volume") + eps)).alias("flow_trust_banks_net_ratio"),
            (pl.col("flow_investment_trusts_net") / (pl.col("flow_total_volume") + eps)).alias(
                "flow_investment_trusts_net_ratio"
            ),
        ]
    )

    # 3. z-score（13週、52週、shift(1)でリーク防止）
    # Sectionごとにグループ化
    for col_base in [
        "flow_foreigners_net_ratio",
        "flow_individuals_net_ratio",
        "flow_trust_banks_net_ratio",
        "flow_investment_trusts_net_ratio",
    ]:
        # 13週（約65営業日）
        trades_spec_df = trades_spec_df.with_columns(
            [
                roll_mean_safe(pl.col(col_base), 65, min_periods=33, by="section").alias(f"{col_base}_ma13w"),
                roll_std_safe(pl.col(col_base), 65, min_periods=33, by="section").alias(f"{col_base}_std13w"),
            ]
        )
        # z-score計算（roll_mean_safe/roll_std_safeは既にshift(1)を含むため、分子はshift(1)不要）
        trades_spec_df = trades_spec_df.with_columns(
            (
                pl.when(pl.col(f"{col_base}_std13w").abs() > eps)
                .then(
                    (pl.col(col_base).shift(1).over("section") - pl.col(f"{col_base}_ma13w"))
                    / (pl.col(f"{col_base}_std13w") + eps)
                )
                .otherwise(None)
            ).alias(f"{col_base}_z13")
        )

        # 52週（約260営業日）
        trades_spec_df = trades_spec_df.with_columns(
            [
                roll_mean_safe(pl.col(col_base), 260, min_periods=130, by="section").alias(f"{col_base}_ma52w"),
                roll_std_safe(pl.col(col_base), 260, min_periods=130, by="section").alias(f"{col_base}_std52w"),
            ]
        )
        trades_spec_df = trades_spec_df.with_columns(
            (
                pl.when(pl.col(f"{col_base}_std52w").abs() > eps)
                .then(
                    (pl.col(col_base).shift(1).over("section") - pl.col(f"{col_base}_ma52w"))
                    / (pl.col(f"{col_base}_std52w") + eps)
                )
                .otherwise(None)
            ).alias(f"{col_base}_z52")
        )

        # 一時列を削除
        for temp_col in [f"{col_base}_ma13w", f"{col_base}_std13w", f"{col_base}_ma52w", f"{col_base}_std52w"]:
            if temp_col in trades_spec_df.columns:
                trades_spec_df = trades_spec_df.drop(temp_col)

    # 4. モメンタム/変化率（WoW: Week over Week）
    # 先週との差分を13週平均で正規化
    for col_base in ["flow_foreigners_net_ratio", "flow_individuals_net_ratio"]:
        trades_spec_df = trades_spec_df.with_columns(
            [
                roll_mean_safe(pl.col(col_base), 65, min_periods=33, by="section").alias(f"{col_base}_ma13w_for_wow"),
            ]
        )
        trades_spec_df = trades_spec_df.with_columns(
            [
                (
                    pl.when(pl.col(f"{col_base}_ma13w_for_wow").abs() > eps)
                    .then(
                        (pl.col(col_base).shift(1).over("section") - pl.col(col_base).shift(8).over("section"))
                        / (pl.col(f"{col_base}_ma13w_for_wow").abs() + eps)
                    )
                    .otherwise(None)
                ).alias(f"{col_base}_wow"),
                # 符号変化フラグ
                (
                    ((pl.col(col_base).shift(1).over("section") >= 0) & (pl.col(col_base).shift(8).over("section") < 0))
                    | (
                        (pl.col(col_base).shift(1).over("section") < 0)
                        & (pl.col(col_base).shift(8).over("section") >= 0)
                    )
                )
                .cast(pl.Int8)
                .alias(f"{col_base}_turn_flag"),
            ]
        )
        if f"{col_base}_ma13w_for_wow" in trades_spec_df.columns:
            trades_spec_df = trades_spec_df.drop(f"{col_base}_ma13w_for_wow")

    # 5. ダイバージェンス
    trades_spec_df = trades_spec_df.with_columns(
        [
            (pl.col("flow_foreigners_net_ratio") - pl.col("flow_individuals_net_ratio")).alias(
                "flow_divergence_foreigners_vs_individuals"
            ),
        ]
    )

    # 6. 観測性/鮮度
    # 最新PublishedDateからの営業日差を計算
    # 簡易実装: 日付差を使用
    trades_spec_df = trades_spec_df.with_columns(
        [
            pl.col("date").cast(pl.Int64).alias("_date_int"),
            pl.col("published_date").cast(pl.Int64).alias("_published_date_int"),
        ]
    )

    # 各Sectionごとに最新PublishedDateを特定
    trades_spec_df = trades_spec_df.with_columns(
        [
            pl.col("_published_date_int").max().over("section").alias("_latest_published_date_int"),
        ]
    )

    trades_spec_df = trades_spec_df.with_columns(
        [
            (
                (pl.col("_latest_published_date_int") - pl.col("_published_date_int"))
                .cast(pl.Int32)
                .alias("trades_spec_staleness_bd")
            ),
            # is_trades_spec_valid: 主要列が有効な場合1
            pl.when(pl.col("flow_total_volume").is_not_null() & (pl.col("flow_total_volume").abs() > eps))
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("is_trades_spec_valid"),
        ]
    )

    # 一時列を削除
    cleanup_cols = [
        "foreigners_purchase",
        "foreigners_sales",
        "individuals_purchase",
        "individuals_sales",
        "trust_banks_purchase",
        "trust_banks_sales",
        "investment_trusts_purchase",
        "investment_trusts_sales",
        "total_purchase",
        "total_sales",
        "_date_int",
        "_published_date_int",
        "_latest_published_date_int",
    ]
    for col in cleanup_cols:
        if col in trades_spec_df.columns:
            trades_spec_df = trades_spec_df.drop(col)

    # available_tsを設定（PublishedDateの翌営業日09:00 JST）
    trades_spec_df = prepare_snapshot_pl(
        trades_spec_df,
        published_date_col="published_date",
        trading_calendar=trading_calendar,
        availability_hour=9,
        availability_minute=0,
    )

    # 列名にmkt_flow_プレフィックスを追加
    rename_map = {}
    for col in trades_spec_df.columns:
        if col.startswith("flow_") and col not in ["section", "date", "published_date", "available_ts"]:
            rename_map[col] = f"mkt_flow_{col}"
        elif col in ["is_trades_spec_valid", "trades_spec_staleness_bd"]:
            rename_map[col] = col  # そのまま保持
        # section, date, available_tsはそのまま

    if rename_map:
        trades_spec_df = trades_spec_df.rename(rename_map)

    # 選択列
    feature_cols = [col for col in ["section", "date", "available_ts"] if col in trades_spec_df.columns] + [
        col
        for col in trades_spec_df.columns
        if col.startswith("mkt_flow_") or col in ["is_trades_spec_valid", "trades_spec_staleness_bd"]
    ]
    result = trades_spec_df.select(feature_cols)

    LOGGER.info(
        "Generated trades_spec features: %d rows, %d sections, %d features",
        result.height,
        result["section"].n_unique() if "section" in result.columns else 0,
        len(feature_cols) - 3,  # section, date, available_tsを除く
    )

    return result
