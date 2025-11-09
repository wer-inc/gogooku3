from __future__ import annotations

import polars as pl

from builder.features.macro.options_asof import load_options


def test_load_options_normalizes_numeric_strings() -> None:
    df = pl.DataFrame(
        {
            "DerivativesProductCategory": ["TOPIXE"],
            "Date": ["2024-02-01"],
            "ImpliedVolatility": ["  "],
            "WholeDayClose": [" 105.5 "],
        }
    )

    normalized = load_options(df)

    assert normalized["ImpliedVolatility"].to_list()[0] is None
    assert normalized["WholeDayClose"].to_list()[0] == 105.5
