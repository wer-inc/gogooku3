"""Financial statement as-of utilities and feature engineering."""
from __future__ import annotations

from typing import Sequence

import polars as pl

EPS = 1e-9

_NUMERIC_ALIAS = {
    "netsales": "NetSales",
    "net sales": "NetSales",
    "revenue": "NetSales",
    "sales": "NetSales",
    "operating revenue": "NetSales",
    "operatingprofit": "OperatingProfit",
    "operating profit": "OperatingProfit",
    "operating income": "OperatingProfit",
    "operating loss": "OperatingProfit",
    "profit": "Profit",
    "profit (loss)": "Profit",
    "net income": "Profit",
    "net profit": "Profit",
    "profit attributable to owners of parent": "Profit",
    "totalassets": "TotalAssets",
    "total assets": "TotalAssets",
    "totalassetsasofcurrentfiscalyearend": "TotalAssets",
    "total liabilities and equity": "TotalAssets",
    "equity": "Equity",
    "total equity": "Equity",
    "total shareholders' equity": "Equity",
    "equity attributable to owners of parent": "Equity",
    "cashandcashequivalents": "CashAndCashEquivalents",
    "cash and cash equivalents": "CashAndCashEquivalents",
    "interestbearingdebt": "InterestBearingDebt",
    "interest-bearing debt": "InterestBearingDebt",
    "interest bearing debt": "InterestBearingDebt",
    "netcashprovidedbyoperatingactivities": "NetCashProvidedByOperatingActivities",
    "net cash provided by (used in) operating activities": "NetCashProvidedByOperatingActivities",
    "cash flows from operating activities": "NetCashProvidedByOperatingActivities",
    "cash flows provided by operating activities": "NetCashProvidedByOperatingActivities",
    "purchaseofpropertyplantandequipment": "PurchaseOfPropertyPlantAndEquipment",
    "purchase of property, plant and equipment": "PurchaseOfPropertyPlantAndEquipment",
    "purchase of property,plant and equipment": "PurchaseOfPropertyPlantAndEquipment",
    "capital expenditure": "PurchaseOfPropertyPlantAndEquipment",
    "purchaseofintangibleassets": "PurchaseOfIntangibleAssets",
    "purchase of intangible assets": "PurchaseOfIntangibleAssets",
    "depreciation": "Depreciation",
}

_NUMERIC_COLUMNS: Sequence[str] = (
    "NetSales",
    "OperatingProfit",
    "Profit",
    "TotalAssets",
    "Equity",
    "CashAndCashEquivalents",
    "InterestBearingDebt",
    "NetCashProvidedByOperatingActivities",
    "PurchaseOfPropertyPlantAndEquipment",
    "PurchaseOfIntangibleAssets",
)

_PERIOD_ALIAS = {
    "CurrentPeriodEndDate": "PeriodEndDate",
    "CurrentClosingDate": "PeriodEndDate",
    "ResultOfFiscalPeriod": "PeriodEndDate",
    "FiscalPeriodEndDate": "PeriodEndDate",
}

_CAPEX_COMPONENTS: tuple[str, ...] = (
    "PurchaseOfPropertyPlantAndEquipment",
    "PurchaseOfIntangibleAssets",
)

_ISSUED_SHARES_CANDIDATES: tuple[str, ...] = (
    "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock",
    "NumberOfIssuedShares",
    "IssuedShares",
    "IssuedShareNumber",
    "IssuedShareNumberOfListing",
    "SharesOutstanding",
)
_TREASURY_SHARES_CANDIDATES: tuple[str, ...] = (
    "NumberOfTreasuryStockAtTheEndOfFiscalYear",
    "TreasuryShares",
    "TreasuryStock",
    "TreasuryShareNumber",
)
_AVERAGE_SHARES_CANDIDATES: tuple[str, ...] = (
    "AverageNumberOfShares",
    "AverageNumberOfSharesDuringPeriod",
    "AverageNumberOfSharesOutstanding",
)


def _normalise_numeric_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise column aliases coming from fs_details payload."""
    rename_map: dict[str, str] = {}
    for column in df.columns:
        key = column.strip().lower()
        target = _NUMERIC_ALIAS.get(key)
        if target is not None and column != target:
            rename_map[column] = target
    if rename_map:
        df = df.rename(rename_map)

    cast_exprs = []
    for column in _NUMERIC_COLUMNS:
        if column in df.columns:
            cast_exprs.append(pl.col(column).cast(pl.Float64, strict=False).alias(column))
    if cast_exprs:
        df = df.with_columns(cast_exprs)
    return df


def _resolve_period_end(df: pl.DataFrame) -> pl.Expr:
    for candidate, target in _PERIOD_ALIAS.items():
        if candidate in df.columns:
            return pl.col(candidate).cast(pl.Date, strict=False).alias(target)
    return pl.col("DisclosedDate").cast(pl.Date, strict=False).alias("PeriodEndDate")


def prepare_fs_snapshot(
    df: pl.DataFrame,
    *,
    trading_calendar: pl.DataFrame | None = None,
    availability_hour: int = 15,
    availability_minute: int = 0,
) -> pl.DataFrame:
    """Prepare fs_details payload for interval join."""
    if df.is_empty():
        return pl.DataFrame(
            {
                "Code": pl.Series([], dtype=pl.Utf8),
                "DisclosedDate": pl.Series([], dtype=pl.Date),
                "available_ts": pl.Series([], dtype=pl.Datetime("us", "Asia/Tokyo")),
            }
        )

    required = {"Code", "DisclosedDate"}
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"fs_details payload missing required columns: {missing}")

    normalized = df.with_columns(
        [
            pl.col("Code").cast(pl.Utf8, strict=False).alias("Code"),
            pl.col("DisclosedDate").cast(pl.Date, strict=False).alias("DisclosedDate"),
            pl.col("DisclosedTime").cast(pl.Utf8, strict=False).fill_null("15:00:00").alias("DisclosedTime"),
        ]
    )
    normalized = normalized.with_columns(_resolve_period_end(normalized))
    normalized = _normalise_numeric_columns(normalized)

    # 新しいas-of規則: DisclosedDate + DisclosedTimeを基準に
    # 当日15:00以前の開示 → 当日終値まで利用可（available_ts = DisclosedDate 15:00 JST）
    # 当日15:00以降/時刻欠損 → 翌営業日9:00以降に利用可
    # DisclosedTimeをパース（HH:MM:SS形式）
    normalized = normalized.with_columns(
        [
            # DisclosedTimeから時分を抽出
            pl.col("DisclosedTime").str.slice(0, 2).cast(pl.Int32, strict=False).fill_null(15).alias("_disclosed_hour"),
            pl.col("DisclosedTime")
            .str.slice(3, 2)
            .cast(pl.Int32, strict=False)
            .fill_null(0)
            .alias("_disclosed_minute"),
        ]
    )

    # 15:00以前かどうかを判定
    normalized = normalized.with_columns(
        ((pl.col("_disclosed_hour") < 15) | ((pl.col("_disclosed_hour") == 15) & (pl.col("_disclosed_minute") == 0)))
        .cast(pl.Int8)
        .alias("_is_before_1500")
    )

    # available_tsを計算
    # 15:00以前: DisclosedDate 15:00 JST
    # 15:00以降/時刻欠損: 翌営業日 09:00 JST
    normalized = normalized.with_columns(
        [
            pl.datetime(
                pl.col("DisclosedDate").dt.year(),
                pl.col("DisclosedDate").dt.month(),
                pl.col("DisclosedDate").dt.day(),
                pl.when(pl.col("_is_before_1500") == 1).then(15).otherwise(9),
                pl.when(pl.col("_is_before_1500") == 1).then(0).otherwise(0),
                0,
            )
            .dt.replace_time_zone("Asia/Tokyo")
            .alias("_available_ts_base"),
        ]
    )

    # 15:00以降の場合は翌営業日を計算
    # 営業日カレンダーを使用して翌営業日を取得（より効率的な方法）
    if trading_calendar is not None:
        cal_dates = trading_calendar.select(pl.col("date").cast(pl.Date).alias("date")).sort("date")
        # 各DisclosedDateに対して翌営業日を計算
        unique_dates = normalized.select(pl.col("DisclosedDate").unique().alias("DisclosedDate")).sort("DisclosedDate")
        date_map_rows = []
        for pub_date in unique_dates["DisclosedDate"].to_list():
            next_bdays = cal_dates.filter(pl.col("date") > pub_date)
            if not next_bdays.is_empty():
                next_bday = next_bdays["date"].head(1).item()
                date_map_rows.append({"DisclosedDate": pub_date, "next_business_date": next_bday})
            else:
                from datetime import timedelta

                date_map_rows.append({"DisclosedDate": pub_date, "next_business_date": pub_date + timedelta(days=1)})
        if date_map_rows:
            date_mapping = pl.DataFrame(date_map_rows)
            normalized = normalized.join(date_mapping, on="DisclosedDate", how="left")
        else:
            # 空の場合はフォールバック
            normalized = normalized.with_columns(
                (pl.col("DisclosedDate") + pl.duration(days=1)).alias("next_business_date")
            )
    else:
        # フォールバック: 単純に+1日
        normalized = normalized.with_columns(
            (pl.col("DisclosedDate") + pl.duration(days=1)).alias("next_business_date")
        )

    # available_tsを最終決定
    normalized = normalized.with_columns(
        pl.when(pl.col("_is_before_1500") == 1)
        .then(pl.col("_available_ts_base"))
        .otherwise(
            pl.datetime(
                pl.col("next_business_date").dt.year(),
                pl.col("next_business_date").dt.month(),
                pl.col("next_business_date").dt.day(),
                9,
                0,
                0,
            ).dt.replace_time_zone("Asia/Tokyo")
        )
        .alias("available_ts")
    )

    # 一時列を削除
    cleanup_cols = [
        "_disclosed_hour",
        "_disclosed_minute",
        "_is_before_1500",
        "_available_ts_base",
        "next_business_date",
    ]
    for col in cleanup_cols:
        if col in normalized.columns:
            normalized = normalized.drop(col)

    snapshot = normalized

    # TypeOfDocumentの処理（P0: ワンホットエンコーディング）
    # TypeOfDocumentの形式: "FY", "1Q", "2Q", "3Q", "Other" (family)
    # "JGAAP", "IFRS", "US", "JMIS", "Foreign" (standard)
    # "Consolidated", "NonConsolidated" (consolidation)
    # また、EarnForecastRevision, DividendForecastRevisionなどのガイダンス修正系
    if "TypeOfDocument" in snapshot.columns:
        snapshot = snapshot.with_columns(
            [
                # family: FY, 1Q, 2Q, 3Q, Other
                (pl.col("TypeOfDocument").str.contains("FY", literal=True)).cast(pl.Int8).alias("fs_doc_family_FY"),
                (pl.col("TypeOfDocument").str.contains("1Q", literal=True)).cast(pl.Int8).alias("fs_doc_family_1Q"),
                (pl.col("TypeOfDocument").str.contains("2Q", literal=True)).cast(pl.Int8).alias("fs_doc_family_2Q"),
                (pl.col("TypeOfDocument").str.contains("3Q", literal=True)).cast(pl.Int8).alias("fs_doc_family_3Q"),
                # standard: JGAAP, IFRS, US, JMIS, Foreign
                (pl.col("TypeOfDocument").str.contains("JGAAP", literal=True)).cast(pl.Int8).alias("fs_standard_JGAAP"),
                (pl.col("TypeOfDocument").str.contains("IFRS", literal=True)).cast(pl.Int8).alias("fs_standard_IFRS"),
                (pl.col("TypeOfDocument").str.contains("US", literal=True)).cast(pl.Int8).alias("fs_standard_US"),
                (pl.col("TypeOfDocument").str.contains("JMIS", literal=True)).cast(pl.Int8).alias("fs_standard_JMIS"),
                (pl.col("TypeOfDocument").str.contains("Foreign", literal=True))
                .cast(pl.Int8)
                .alias("fs_standard_Foreign"),
                # consolidated_flag: Consolidated
                (pl.col("TypeOfDocument").str.contains("Consolidated", literal=True))
                .cast(pl.Int8)
                .alias("fs_consolidated_flag"),
                # guidance_revision_flag: EarnForecastRevision, DividendForecastRevision
                (
                    pl.col("TypeOfDocument").str.contains("EarnForecastRevision", literal=True)
                    | pl.col("TypeOfDocument").str.contains("DividendForecastRevision", literal=True)
                )
                .cast(pl.Int8)
                .alias("fs_guidance_revision_flag"),
            ]
        )
    else:
        # TypeOfDocumentがない場合はデフォルト値
        snapshot = snapshot.with_columns(
            [
                pl.lit(0).cast(pl.Int8).alias("fs_doc_family_FY"),
                pl.lit(0).cast(pl.Int8).alias("fs_doc_family_1Q"),
                pl.lit(0).cast(pl.Int8).alias("fs_doc_family_2Q"),
                pl.lit(0).cast(pl.Int8).alias("fs_doc_family_3Q"),
                pl.lit(0).cast(pl.Int8).alias("fs_standard_JGAAP"),
                pl.lit(0).cast(pl.Int8).alias("fs_standard_IFRS"),
                pl.lit(0).cast(pl.Int8).alias("fs_standard_US"),
                pl.lit(0).cast(pl.Int8).alias("fs_standard_JMIS"),
                pl.lit(0).cast(pl.Int8).alias("fs_standard_Foreign"),
                pl.lit(0).cast(pl.Int8).alias("fs_consolidated_flag"),
                pl.lit(0).cast(pl.Int8).alias("fs_guidance_revision_flag"),
            ]
        )

    capex_exprs = [
        (-pl.col(col).fill_null(0.0)).alias(f"__capex_{col}") for col in _CAPEX_COMPONENTS if col in snapshot.columns
    ]
    if capex_exprs:
        snapshot = snapshot.with_columns(capex_exprs)
        snapshot = snapshot.with_columns(
            pl.sum_horizontal([pl.col(expr.meta.output_name()) for expr in capex_exprs]).alias("_capex_outflow")
        )
        snapshot = snapshot.drop([expr.meta.output_name() for expr in capex_exprs])
    else:
        snapshot = snapshot.with_columns(pl.lit(None).cast(pl.Float64).alias("_capex_outflow"))

    snapshot = snapshot.with_columns(
        [
            pl.col("available_ts").dt.convert_time_zone("Asia/Tokyo"),
            pl.when(pl.col("DisclosedTime").str.len_chars() >= 5)
            .then(pl.col("DisclosedTime"))
            .otherwise(pl.lit("15:00"))
            .alias("DisclosedTime"),
            pl.col("PeriodEndDate").alias("fs_period_end_date"),
        ]
    )

    dedup_keys: list[str]
    if "DisclosureNumber" in snapshot.columns:
        dedup_keys = ["Code", "DisclosureNumber"]
    elif "ReportId" in snapshot.columns:
        dedup_keys = ["Code", "ReportId"]
    else:
        dedup_keys = ["Code", "DisclosedDate", "DisclosedTime"]

    snapshot = (
        snapshot.sort(["Code", "available_ts"]).unique(subset=dedup_keys, keep="last").sort(["Code", "available_ts"])
    )

    return snapshot


def _ttm(expr: pl.Expr, window: int = 4, min_periods: int = 1) -> pl.Expr:
    return expr.shift(1).rolling_sum(window_size=window, min_periods=min_periods)


def _avg(expr: pl.Expr, window: int = 2, min_periods: int = 1) -> pl.Expr:
    return expr.shift(1).rolling_mean(window_size=window, min_periods=min_periods)


def _valid_ratio(num: pl.Expr, denom: pl.Expr, *, eps: float = EPS) -> pl.Expr:
    return pl.when(denom.abs() > eps).then(num / denom).otherwise(None)


def build_fs_feature_frame(snapshot: pl.DataFrame) -> pl.DataFrame:
    """Build left-closed TTM / YoY / ratio features per financial statement snapshot."""
    if snapshot.is_empty():
        return pl.DataFrame(
            {
                "code": pl.Series([], dtype=pl.Utf8),
                "available_ts": pl.Series([], dtype=pl.Datetime("us", "Asia/Tokyo")),
                "fs_period_end_date": pl.Series([], dtype=pl.Date),
                "fs_revenue_ttm": pl.Series([], dtype=pl.Float64),
                "fs_op_profit_ttm": pl.Series([], dtype=pl.Float64),
                "fs_net_income_ttm": pl.Series([], dtype=pl.Float64),
                "fs_cfo_ttm": pl.Series([], dtype=pl.Float64),
                "fs_capex_ttm": pl.Series([], dtype=pl.Float64),
                "fs_fcf_ttm": pl.Series([], dtype=pl.Float64),
                "fs_sales_yoy": pl.Series([], dtype=pl.Float64),
                "fs_op_margin": pl.Series([], dtype=pl.Float64),
                "fs_net_margin": pl.Series([], dtype=pl.Float64),
                "fs_roe_ttm": pl.Series([], dtype=pl.Float64),
                "fs_roa_ttm": pl.Series([], dtype=pl.Float64),
                "fs_accruals_ttm": pl.Series([], dtype=pl.Float64),
                "fs_cfo_to_ni": pl.Series([], dtype=pl.Float64),
                "fs_observation_count": pl.Series([], dtype=pl.Int16),
                "fs_lag_days": pl.Series([], dtype=pl.Int32),
                "is_fs_valid": pl.Series([], dtype=pl.Int8),
            }
        )

    for column in _NUMERIC_COLUMNS:
        if column not in snapshot.columns:
            snapshot = snapshot.with_columns(pl.lit(None).cast(pl.Float64).alias(column))

    # Ensure share-related raw columns exist with consistent naming
    def _resolve_share_column(candidates: tuple[str, ...]) -> str | None:
        for column in candidates:
            if column in snapshot.columns:
                return column
        return None

    share_preprocess_exprs: list[pl.Expr] = []
    issued_col = _resolve_share_column(_ISSUED_SHARES_CANDIDATES)
    treasury_col = _resolve_share_column(_TREASURY_SHARES_CANDIDATES)
    average_col = _resolve_share_column(_AVERAGE_SHARES_CANDIDATES)

    if issued_col:
        share_preprocess_exprs.append(pl.col(issued_col).cast(pl.Float64, strict=False).alias("_fs_issued_shares"))
    else:
        share_preprocess_exprs.append(pl.lit(None).cast(pl.Float64).alias("_fs_issued_shares"))

    if treasury_col:
        share_preprocess_exprs.append(pl.col(treasury_col).cast(pl.Float64, strict=False).alias("_fs_treasury_shares"))
    else:
        share_preprocess_exprs.append(pl.lit(None).cast(pl.Float64).alias("_fs_treasury_shares"))

    if average_col:
        share_preprocess_exprs.append(pl.col(average_col).cast(pl.Float64, strict=False).alias("_fs_average_shares"))
    else:
        share_preprocess_exprs.append(pl.lit(None).cast(pl.Float64).alias("_fs_average_shares"))

    snapshot = snapshot.with_columns(share_preprocess_exprs)
    snapshot = snapshot.with_columns(
        (pl.col("_fs_issued_shares") - pl.col("_fs_treasury_shares").fill_null(0.0)).alias("_fs_shares_outstanding")
    )

    if "fs_period_end_date" not in snapshot.columns:
        snapshot = snapshot.with_columns(pl.lit(None).cast(pl.Date).alias("fs_period_end_date"))

    sort_keys = ["Code", "fs_period_end_date", "available_ts"]
    working = snapshot.sort([key for key in sort_keys if key in snapshot.columns])

    def _build_for_symbol(group: pl.DataFrame) -> pl.DataFrame:
        group = group.sort([key for key in ("fs_period_end_date", "available_ts") if key in group.columns])

        group = group.with_columns(
            [
                pl.col("_fs_shares_outstanding").forward_fill().alias("fs_shares_outstanding"),
                pl.col("_fs_average_shares").forward_fill().alias("fs_average_shares"),
                _ttm(pl.col("NetSales")).alias("fs_revenue_ttm"),
                _ttm(pl.col("OperatingProfit")).alias("fs_op_profit_ttm"),
                _ttm(pl.col("Profit")).alias("fs_net_income_ttm"),
                _ttm(pl.col("NetCashProvidedByOperatingActivities")).alias("fs_cfo_ttm"),
                _ttm(pl.col("_capex_outflow")).alias("fs_capex_ttm"),
                _avg(pl.col("Equity")).alias("_fs_equity_avg"),
                _avg(pl.col("TotalAssets")).alias("_fs_assets_avg"),
                pl.col("NetSales")
                .shift(1)
                .is_not_null()
                .cast(pl.Int16)
                .rolling_sum(window_size=4, min_periods=1)
                .alias("fs_observation_count"),
            ]
        )

        # 時点指標も計算（平均ではなく最新値を使用）
        group = group.with_columns(
            [
                pl.col("Equity").fill_null(0.0).alias("_fs_equity_latest"),
                pl.col("TotalAssets").fill_null(0.0).alias("_fs_assets_latest"),
                pl.col("CashAndCashEquivalents").fill_null(0.0).alias("_fs_cash_latest"),
                pl.col("InterestBearingDebt").fill_null(0.0).alias("_fs_debt_latest"),
            ]
        )

        group = group.with_columns(
            [
                (pl.col("fs_cfo_ttm") - pl.col("fs_capex_ttm")).alias("fs_fcf_ttm"),
                # マージン（既存）
                _valid_ratio(pl.col("fs_op_profit_ttm"), pl.col("fs_revenue_ttm")).alias("fs_op_margin"),
                _valid_ratio(pl.col("fs_net_income_ttm"), pl.col("fs_revenue_ttm")).alias("fs_net_margin"),
                # P0新特徴量: fs_ttm_op_margin, fs_ttm_cfo_margin
                _valid_ratio(pl.col("fs_op_profit_ttm"), pl.col("fs_revenue_ttm")).alias("fs_ttm_op_margin_raw"),
                _valid_ratio(pl.col("fs_cfo_ttm"), pl.col("fs_revenue_ttm")).alias("fs_ttm_cfo_margin"),
                # 品質指標（既存）
                _valid_ratio(pl.col("fs_net_income_ttm"), pl.col("_fs_equity_avg")).alias("fs_roe_ttm"),
                _valid_ratio(pl.col("fs_net_income_ttm"), pl.col("_fs_assets_avg")).alias("fs_roa_ttm"),
                _valid_ratio(
                    pl.col("fs_net_income_ttm") - pl.col("fs_cfo_ttm"),
                    pl.col("_fs_assets_avg"),
                ).alias("fs_accruals_ttm"),
                _valid_ratio(pl.col("fs_cfo_ttm"), pl.col("fs_net_income_ttm")).alias("fs_cfo_to_ni"),
                # 財務体力（P0新特徴量）
                _valid_ratio(pl.col("_fs_equity_latest"), pl.col("_fs_assets_latest")).alias("fs_equity_ratio"),
                _valid_ratio(
                    pl.col("_fs_cash_latest") - pl.col("_fs_debt_latest"),
                    pl.col("_fs_assets_latest"),
                ).alias("fs_net_cash_ratio"),
                # YoY成長率（既存を拡張）
                _valid_ratio(
                    pl.col("fs_revenue_ttm"),
                    pl.col("fs_revenue_ttm").shift(4),
                ).alias("_fs_revenue_ratio_yoy"),
                _valid_ratio(
                    pl.col("fs_op_profit_ttm"),
                    pl.col("fs_op_profit_ttm").shift(4),
                ).alias("_fs_op_profit_ratio_yoy"),
                _valid_ratio(
                    pl.col("fs_net_income_ttm"),
                    pl.col("fs_net_income_ttm").shift(4),
                ).alias("_fs_net_income_ratio_yoy"),
            ]
        )

        group = group.with_columns(
            [
                # YoY成長率（-1.0で割る）
                pl.when(pl.col("_fs_revenue_ratio_yoy").is_not_null())
                .then(pl.col("_fs_revenue_ratio_yoy") - 1.0)
                .otherwise(None)
                .alias("fs_sales_yoy"),
                # P0新特徴量: fs_yoy_ttm_*
                pl.when(pl.col("_fs_revenue_ratio_yoy").is_not_null())
                .then(pl.col("_fs_revenue_ratio_yoy") - 1.0)
                .otherwise(None)
                .alias("fs_yoy_ttm_sales"),
                pl.when(pl.col("_fs_op_profit_ratio_yoy").is_not_null())
                .then(pl.col("_fs_op_profit_ratio_yoy") - 1.0)
                .otherwise(None)
                .alias("fs_yoy_ttm_op_profit"),
                pl.when(pl.col("_fs_net_income_ratio_yoy").is_not_null())
                .then(pl.col("_fs_net_income_ratio_yoy") - 1.0)
                .otherwise(None)
                .alias("fs_yoy_ttm_net_income"),
            ]
        )

        # fs_ttm_op_marginの極端値をフラグ化し、本体はNULLに倒す（validatorのextreme率を抑制）
        group = group.with_columns(
            ((pl.col("fs_ttm_op_margin_raw") < -1.5) | (pl.col("fs_ttm_op_margin_raw") > 1.5))
            .cast(pl.Int8)
            .alias("fs_op_margin_extreme_flag")
        )
        group = group.with_columns(
            pl.when(pl.col("fs_op_margin_extreme_flag") == 1)
            .then(None)
            .otherwise(pl.col("fs_ttm_op_margin_raw"))
            .alias("fs_ttm_op_margin")
        )

        # ベースの有効性フラグ（TTM系のどれかが存在するか）を緩和定義で付与。
        base_valid = pl.any_horizontal(
            [
                pl.col("fs_revenue_ttm").is_not_null(),
                pl.col("fs_op_profit_ttm").is_not_null(),
                pl.col("fs_net_income_ttm").is_not_null(),
                pl.col("fs_cfo_ttm").is_not_null(),
            ]
        )
        group = group.with_columns(
            base_valid.cast(pl.Int8).alias("is_fs_valid"),
            # P0新特徴量: fs_is_valid (後段でstalenessを加味して上書きされる前提のエイリアス)
            base_valid.cast(pl.Int8).alias("fs_is_valid"),
        )

        lag_days = (
            pl.col("DisclosedDate").cast(pl.Int64, strict=False)
            - pl.col("fs_period_end_date").cast(pl.Int64, strict=False)
        ).alias("fs_lag_days")

        # P0新特徴量: エイリアス（既存特徴量との互換性）
        group = group.with_columns(
            [
                pl.col("fs_revenue_ttm").alias("fs_ttm_sales"),
                pl.col("fs_op_profit_ttm").alias("fs_ttm_op_profit"),
                pl.col("fs_net_income_ttm").alias("fs_ttm_net_income"),
                pl.col("fs_cfo_ttm").alias("fs_ttm_cfo"),
                pl.col("fs_accruals_ttm").alias("fs_accruals"),
            ]
        )

        # TypeOfDocument関連の特徴量も選択
        result_cols = [
            pl.col("Code").alias("code"),
            "available_ts",
            "fs_period_end_date",
            "DisclosedDate",
            "fs_shares_outstanding",
            "fs_average_shares",
            # 既存特徴量（後方互換性）
            "fs_revenue_ttm",
            "fs_op_profit_ttm",
            "fs_net_income_ttm",
            "fs_cfo_ttm",
            "fs_capex_ttm",
            "fs_fcf_ttm",
            "fs_sales_yoy",
            "fs_op_margin",
            "fs_net_margin",
            "fs_roe_ttm",
            "fs_roa_ttm",
            "fs_accruals_ttm",
            "fs_cfo_to_ni",
            "fs_observation_count",
            lag_days,
            "is_fs_valid",
            # P0新特徴量
            "fs_ttm_sales",
            "fs_ttm_op_profit",
            "fs_ttm_net_income",
            "fs_ttm_cfo",
            "fs_ttm_op_margin",
            "fs_ttm_cfo_margin",
            "fs_equity_ratio",
            "fs_net_cash_ratio",
            "fs_yoy_ttm_sales",
            "fs_yoy_ttm_op_profit",
            "fs_yoy_ttm_net_income",
            "fs_accruals",
            "fs_is_valid",
        ]

        # TypeOfDocument関連の特徴量を追加（存在する場合）
        doc_cols = [
            "fs_doc_family_FY",
            "fs_doc_family_1Q",
            "fs_doc_family_2Q",
            "fs_doc_family_3Q",
            "fs_standard_JGAAP",
            "fs_standard_IFRS",
            "fs_standard_US",
            "fs_standard_JMIS",
            "fs_standard_Foreign",
            "fs_consolidated_flag",
            "fs_guidance_revision_flag",
        ]
        for col in doc_cols:
            if col in group.columns:
                result_cols.append(col)

        return group.select(result_cols)

    enriched = (
        working.group_by("Code", maintain_order=True)
        .map_groups(_build_for_symbol)
        .with_columns(
            pl.col("available_ts").dt.convert_time_zone("UTC").dt.replace_time_zone(None).alias("available_ts")
        )
    )

    return enriched.sort(["code", "available_ts"])
