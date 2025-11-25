from __future__ import annotations

import polars as pl
from builder.features.macro.futures_topix import load_futures


def test_load_futures_prefers_derivatives_category():
    df = pl.DataFrame(
        {
            "ProductCategory": ["FUTURES"] * 2,
            "DerivativesProductCategory": ["TOPIXF", "NK225F"],
            "Date": ["2020-01-01", "2020-01-01"],
            "SettlementPrice": [2000.0, 2500.0],
        }
    )

    filtered = load_futures(df, category="TOPIXF")

    assert filtered.height == 1
    assert filtered["SettlementPrice"].item() == 2000.0
