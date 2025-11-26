"""Normalize /markets/short_selling_positions response."""

from __future__ import annotations

import polars as pl

_FIELD_MAP = {
    "DisclosedDate": "disclosed_date",
    "CalculatedDate": "calculated_date",
    "Code": "code",
    "ShortSellerName": "short_seller_name",
    "ShortSellerAddress": "short_seller_address",
    "DiscretionaryInvestmentContractorName": "discretionary_investment_contractor_name",
    "DiscretionaryInvestmentContractorAddress": "discretionary_investment_contractor_address",
    "InvestmentFundName": "investment_fund_name",
    "ShortPositionsToSharesOutstandingRatio": "short_positions_to_shares_outstanding_ratio",
    "ShortPositionsInSharesNumber": "short_positions_in_shares_number",
    "ShortPositionsInTradingUnitsNumber": "short_positions_in_trading_units_number",
    "CalculationInPreviousReportingDate": "calculation_in_previous_reporting_date",
    "ShortPositionsInPreviousReportingRatio": "short_positions_in_previous_reporting_ratio",
    "Notes": "notes",
}

_DATE_FIELDS = {"disclosed_date", "calculated_date", "calculation_in_previous_reporting_date"}


def normalize_short_selling_positions(records: list[dict]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame()
    mapped = [{v: rec.get(k) for k, v in _FIELD_MAP.items()} for rec in records]
    df = pl.DataFrame(mapped)
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col))
    for col in _DATE_FIELDS:
        if col in df.columns and df.select(pl.col(col).is_not_null().any()).item():
            df = df.with_columns(pl.col(col).str.strptime(pl.Date, strict=False, exact=False))
    return df
