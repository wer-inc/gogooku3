"""Normalize /fins/statements response."""

from __future__ import annotations

import polars as pl

# Mapping from API field -> snake_case column
_FIELD_MAP = {
    "DisclosedDate": "disclosed_date",
    "DisclosedTime": "disclosed_time",
    "LocalCode": "local_code",
    "DisclosureNumber": "disclosure_number",
    "TypeOfDocument": "type_of_document",
    "TypeOfCurrentPeriod": "type_of_current_period",
    "CurrentPeriodStartDate": "current_period_start_date",
    "CurrentPeriodEndDate": "current_period_end_date",
    "CurrentFiscalYearStartDate": "current_fiscal_year_start_date",
    "CurrentFiscalYearEndDate": "current_fiscal_year_end_date",
    "NextFiscalYearStartDate": "next_fiscal_year_start_date",
    "NextFiscalYearEndDate": "next_fiscal_year_end_date",
    "NetSales": "net_sales",
    "OperatingProfit": "operating_profit",
    "OrdinaryProfit": "ordinary_profit",
    "Profit": "profit",
    "EarningsPerShare": "earnings_per_share",
    "DilutedEarningsPerShare": "diluted_earnings_per_share",
    "TotalAssets": "total_assets",
    "Equity": "equity",
    "EquityToAssetRatio": "equity_to_asset_ratio",
    "BookValuePerShare": "book_value_per_share",
    "CashFlowsFromOperatingActivities": "cash_flows_from_operating_activities",
    "CashFlowsFromInvestingActivities": "cash_flows_from_investing_activities",
    "CashFlowsFromFinancingActivities": "cash_flows_from_financing_activities",
    "CashAndEquivalents": "cash_and_equivalents",
    "ResultDividendPerShare1stQuarter": "result_dividend_per_share_1st_quarter",
    "ResultDividendPerShare2ndQuarter": "result_dividend_per_share_2nd_quarter",
    "ResultDividendPerShare3rdQuarter": "result_dividend_per_share_3rd_quarter",
    "ResultDividendPerShareFiscalYearEnd": "result_dividend_per_share_fiscal_year_end",
    "ResultDividendPerShareAnnual": "result_dividend_per_share_annual",
    "DistributionsPerUnit(REIT)": "distributions_per_unit_reit",
    "ResultTotalDividendPaidAnnual": "result_total_dividend_paid_annual",
    "ResultPayoutRatioAnnual": "result_payout_ratio_annual",
    "ForecastDividendPerShare1stQuarter": "forecast_dividend_per_share_1st_quarter",
    "ForecastDividendPerShare2ndQuarter": "forecast_dividend_per_share_2nd_quarter",
    "ForecastDividendPerShare3rdQuarter": "forecast_dividend_per_share_3rd_quarter",
    "ForecastDividendPerShareFiscalYearEnd": "forecast_dividend_per_share_fiscal_year_end",
    "ForecastDividendPerShareAnnual": "forecast_dividend_per_share_annual",
    "ForecastDistributionsPerUnit(REIT)": "forecast_distributions_per_unit_reit",
    "ForecastTotalDividendPaidAnnual": "forecast_total_dividend_paid_annual",
    "ForecastPayoutRatioAnnual": "forecast_payout_ratio_annual",
    "NextYearForecastDividendPerShare1stQuarter": "next_year_forecast_dividend_per_share_1st_quarter",
    "NextYearForecastDividendPerShare2ndQuarter": "next_year_forecast_dividend_per_share_2nd_quarter",
    "NextYearForecastDividendPerShare3rdQuarter": "next_year_forecast_dividend_per_share_3rd_quarter",
    "NextYearForecastDividendPerShareFiscalYearEnd": "next_year_forecast_dividend_per_share_fiscal_year_end",
    "NextYearForecastDividendPerShareAnnual": "next_year_forecast_dividend_per_share_annual",
    "NextYearForecastDistributionsPerUnit(REIT)": "next_year_forecast_distributions_per_unit_reit",
    "NextYearForecastPayoutRatioAnnual": "next_year_forecast_payout_ratio_annual",
    "ForecastNetSales2ndQuarter": "forecast_net_sales_2nd_quarter",
    "ForecastOperatingProfit2ndQuarter": "forecast_operating_profit_2nd_quarter",
    "ForecastOrdinaryProfit2ndQuarter": "forecast_ordinary_profit_2nd_quarter",
    "ForecastProfit2ndQuarter": "forecast_profit_2nd_quarter",
    "ForecastEarningsPerShare2ndQuarter": "forecast_earnings_per_share_2nd_quarter",
    "NextYearForecastNetSales2ndQuarter": "next_year_forecast_net_sales_2nd_quarter",
    "NextYearForecastOperatingProfit2ndQuarter": "next_year_forecast_operating_profit_2nd_quarter",
    "NextYearForecastOrdinaryProfit2ndQuarter": "next_year_forecast_ordinary_profit_2nd_quarter",
    "NextYearForecastProfit2ndQuarter": "next_year_forecast_profit_2nd_quarter",
    "NextYearForecastEarningsPerShare2ndQuarter": "next_year_forecast_earnings_per_share_2nd_quarter",
    "ForecastNetSales": "forecast_net_sales",
    "ForecastOperatingProfit": "forecast_operating_profit",
    "ForecastOrdinaryProfit": "forecast_ordinary_profit",
    "ForecastProfit": "forecast_profit",
    "ForecastEarningsPerShare": "forecast_earnings_per_share",
    "NextYearForecastNetSales": "next_year_forecast_net_sales",
    "NextYearForecastOperatingProfit": "next_year_forecast_operating_profit",
    "NextYearForecastOrdinaryProfit": "next_year_forecast_ordinary_profit",
    "NextYearForecastProfit": "next_year_forecast_profit",
    "NextYearForecastEarningsPerShare": "next_year_forecast_earnings_per_share",
    "MaterialChangesInSubsidiaries": "material_changes_in_subsidiaries",
    "SignificantChangesInTheScopeOfConsolidation": "significant_changes_in_the_scope_of_consolidation",
    "ChangesBasedOnRevisionsOfAccountingStandard": "changes_based_on_revisions_of_accounting_standard",
    "ChangesOtherThanOnesBasedOnRevisionsOfAccountingStandard": "changes_other_than_ones_based_on_revisions_of_accounting_standard",
    "ChangesInAccountingEstimates": "changes_in_accounting_estimates",
    "RetrospectiveRestatement": "retrospective_restatement",
    "NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock": "number_of_issued_and_outstanding_shares_at_the_end_of_fiscal_year_including_treasury_stock",
    "NumberOfTreasuryStockAtTheEndOfFiscalYear": "number_of_treasury_stock_at_the_end_of_fiscal_year",
    "AverageNumberOfShares": "average_number_of_shares",
    "NonConsolidatedNetSales": "non_consolidated_net_sales",
    "NonConsolidatedOperatingProfit": "non_consolidated_operating_profit",
    "NonConsolidatedOrdinaryProfit": "non_consolidated_ordinary_profit",
    "NonConsolidatedProfit": "non_consolidated_profit",
    "NonConsolidatedEarningsPerShare": "non_consolidated_earnings_per_share",
    "NonConsolidatedTotalAssets": "non_consolidated_total_assets",
    "NonConsolidatedEquity": "non_consolidated_equity",
    "NonConsolidatedEquityToAssetRatio": "non_consolidated_equity_to_asset_ratio",
    "NonConsolidatedBookValuePerShare": "non_consolidated_book_value_per_share",
    "ForecastNonConsolidatedNetSales2ndQuarter": "forecast_non_consolidated_net_sales_2nd_quarter",
    "ForecastNonConsolidatedOperatingProfit2ndQuarter": "forecast_non_consolidated_operating_profit_2nd_quarter",
    "ForecastNonConsolidatedOrdinaryProfit2ndQuarter": "forecast_non_consolidated_ordinary_profit_2nd_quarter",
    "ForecastNonConsolidatedProfit2ndQuarter": "forecast_non_consolidated_profit_2nd_quarter",
    "ForecastNonConsolidatedEarningsPerShare2ndQuarter": "forecast_non_consolidated_earnings_per_share_2nd_quarter",
    "NextYearForecastNonConsolidatedNetSales2ndQuarter": "next_year_forecast_non_consolidated_net_sales_2nd_quarter",
    "NextYearForecastNonConsolidatedOperatingProfit2ndQuarter": "next_year_forecast_non_consolidated_operating_profit_2nd_quarter",
    "NextYearForecastNonConsolidatedOrdinaryProfit2ndQuarter": "next_year_forecast_non_consolidated_ordinary_profit_2nd_quarter",
    "NextYearForecastNonConsolidatedProfit2ndQuarter": "next_year_forecast_non_consolidated_profit_2nd_quarter",
    "NextYearForecastNonConsolidatedEarningsPerShare2ndQuarter": "next_year_forecast_non_consolidated_earnings_per_share_2nd_quarter",
    "ForecastNonConsolidatedNetSales": "forecast_non_consolidated_net_sales",
    "ForecastNonConsolidatedOperatingProfit": "forecast_non_consolidated_operating_profit",
    "ForecastNonConsolidatedOrdinaryProfit": "forecast_non_consolidated_ordinary_profit",
    "ForecastNonConsolidatedProfit": "forecast_non_consolidated_profit",
    "ForecastNonConsolidatedEarningsPerShare": "forecast_non_consolidated_earnings_per_share",
    "NextYearForecastNonConsolidatedNetSales": "next_year_forecast_non_consolidated_net_sales",
    "NextYearForecastNonConsolidatedOperatingProfit": "next_year_forecast_non_consolidated_operating_profit",
    "NextYearForecastNonConsolidatedOrdinaryProfit": "next_year_forecast_non_consolidated_ordinary_profit",
    "NextYearForecastNonConsolidatedProfit": "next_year_forecast_non_consolidated_profit",
    "NextYearForecastNonConsolidatedEarningsPerShare": "next_year_forecast_non_consolidated_earnings_per_share",
}

_DATE_FIELDS = {
    "disclosed_date",
    "current_period_start_date",
    "current_period_end_date",
    "current_fiscal_year_start_date",
    "current_fiscal_year_end_date",
    "next_fiscal_year_start_date",
    "next_fiscal_year_end_date",
}


def normalize_statements(records: list[dict]) -> pl.DataFrame:
    if not records:
        return pl.DataFrame()

    mapped = []
    for rec in records:
        mapped.append({v: rec.get(k, None) for k, v in _FIELD_MAP.items()})

    df = pl.DataFrame(mapped)
    # Cast date fields and normalize empty strings to null
    for col in df.columns:
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.when(pl.col(col) == "").then(None).otherwise(pl.col(col)).alias(col))
    for col in _DATE_FIELDS:
        if col in df.columns:
            # Skip parsing if the entire column is null/empty to avoid format errors.
            if df.select(pl.col(col).is_not_null().any()).item():
                df = df.with_columns(pl.col(col).str.strptime(pl.Date, strict=False, exact=False))
    return df
