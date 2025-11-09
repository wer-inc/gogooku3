from __future__ import annotations

import polars as pl

from builder.features.macro.futures_topix import load_futures


def test_load_futures_normalizes_numeric_strings() -> None:
    df = pl.DataFrame(
        {
            "ProductCategory": ["TOPIXF", "TOPIXF"],
            "Date": ["2024-01-05", "2024-01-06"],
            "SettlementPrice": ["", " 12345 "],
            "Volume": ["  ", "987"],
        }
    )

    normalized = load_futures(df)

    assert normalized["SettlementPrice"].to_list()[0] is None
    assert normalized["SettlementPrice"].to_list()[1] == 12345.0
    assert normalized["Volume"].to_list()[0] is None
    assert normalized["Volume"].to_list()[1] == 987.0
