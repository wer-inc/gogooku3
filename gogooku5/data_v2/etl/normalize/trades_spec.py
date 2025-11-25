"""Normalize markets/trades_spec payload."""

from __future__ import annotations

import polars as pl


def normalize_trades_spec(records: list[dict]) -> pl.DataFrame:
    """Normalize trades_spec JSON into a typed Polars DataFrame."""

    if not records:
        return pl.DataFrame(
            schema={
                "published_date": pl.Date,
                "start_date": pl.Date,
                "end_date": pl.Date,
                "section": pl.Utf8,
            }
        )

    df = pl.DataFrame(records)

    rename_map = {
        "PublishedDate": "published_date",
        "StartDate": "start_date",
        "EndDate": "end_date",
        "Section": "section",
        "ProprietarySales": "proprietary_sales",
        "ProprietaryPurchases": "proprietary_purchases",
        "ProprietaryTotal": "proprietary_total",
        "ProprietaryBalance": "proprietary_balance",
        "BrokerageSales": "brokerage_sales",
        "BrokeragePurchases": "brokerage_purchases",
        "BrokerageTotal": "brokerage_total",
        "BrokerageBalance": "brokerage_balance",
        "TotalSales": "total_sales",
        "TotalPurchases": "total_purchases",
        "TotalTotal": "total_total",
        "TotalBalance": "total_balance",
        "IndividualsSales": "individuals_sales",
        "IndividualsPurchases": "individuals_purchases",
        "IndividualsTotal": "individuals_total",
        "IndividualsBalance": "individuals_balance",
        "ForeignersSales": "foreigners_sales",
        "ForeignersPurchases": "foreigners_purchases",
        "ForeignersTotal": "foreigners_total",
        "ForeignersBalance": "foreigners_balance",
        "SecuritiesCosSales": "securities_cos_sales",
        "SecuritiesCosPurchases": "securities_cos_purchases",
        "SecuritiesCosTotal": "securities_cos_total",
        "SecuritiesCosBalance": "securities_cos_balance",
        "InvestmentTrustsSales": "investment_trusts_sales",
        "InvestmentTrustsPurchases": "investment_trusts_purchases",
        "InvestmentTrustsTotal": "investment_trusts_total",
        "InvestmentTrustsBalance": "investment_trusts_balance",
        "BusinessCosSales": "business_cos_sales",
        "BusinessCosPurchases": "business_cos_purchases",
        "BusinessCosTotal": "business_cos_total",
        "BusinessCosBalance": "business_cos_balance",
        "OtherCosSales": "other_cos_sales",
        "OtherCosPurchases": "other_cos_purchases",
        "OtherCosTotal": "other_cos_total",
        "OtherCosBalance": "other_cos_balance",
        "InsuranceCosSales": "insurance_cos_sales",
        "InsuranceCosPurchases": "insurance_cos_purchases",
        "InsuranceCosTotal": "insurance_cos_total",
        "InsuranceCosBalance": "insurance_cos_balance",
        "CityBKsRegionalBKsEtcSales": "city_bks_regional_bks_etc_sales",
        "CityBKsRegionalBKsEtcPurchases": "city_bks_regional_bks_etc_purchases",
        "CityBKsRegionalBKsEtcTotal": "city_bks_regional_bks_etc_total",
        "CityBKsRegionalBKsEtcBalance": "city_bks_regional_bks_etc_balance",
        "TrustBanksSales": "trust_banks_sales",
        "TrustBanksPurchases": "trust_banks_purchases",
        "TrustBanksTotal": "trust_banks_total",
        "TrustBanksBalance": "trust_banks_balance",
        "OtherFinancialInstitutionsSales": "other_financial_institutions_sales",
        "OtherFinancialInstitutionsPurchases": "other_financial_institutions_purchases",
        "OtherFinancialInstitutionsTotal": "other_financial_institutions_total",
        "OtherFinancialInstitutionsBalance": "other_financial_institutions_balance",
    }
    df = df.rename(rename_map)

    num_cols = [c for c in rename_map.values() if c not in {"published_date", "start_date", "end_date", "section"}]
    df = df.with_columns(
        [
            pl.col("published_date").str.strptime(pl.Date, strict=False),
            pl.col("start_date").str.strptime(pl.Date, strict=False),
            pl.col("end_date").str.strptime(pl.Date, strict=False),
        ]
    )
    df = df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num_cols if c in df.columns])

    return df.select(["published_date", "start_date", "end_date", "section"] + num_cols)
