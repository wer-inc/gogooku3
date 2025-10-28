from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl

import datetime as dt

import polars as pl

from scripts.data.ml_dataset_builder import MLDatasetBuilder
from src.features.macro.vix import shift_to_next_business_day


def _make_builder() -> MLDatasetBuilder:
    return MLDatasetBuilder(output_dir=Path("output/tests"))


def test_breakdown_features_use_callable_next_expr() -> None:
    builder = _make_builder()
    base_df = pl.DataFrame(
        {
            "Code": ["1001", "1001"],
            "Date": [
                dt.date(2024, 1, 2),
                dt.date(2024, 1, 3),
            ],
        }
    )

    breakdown_df = pl.DataFrame(
        {
            "Code": ["1001", "1001"],
            "Date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "LongBuyValue": [1000.0, 1200.0],
            "LongSellValue": [800.0, 700.0],
        }
    )

    result = builder.add_breakdown_features(
        base_df,
        breakdown_df,
        business_days=["2024-01-01", "2024-01-02", "2024-01-03"],
    )

    assert "bd_total" in result.columns
    assert "is_bd_valid" in result.columns
    row = result.filter(
        (pl.col("Code") == "1001") & (pl.col("Date") == dt.date(2024, 1, 2))
    )
    assert row["bd_total"][0] > 0
    assert row["is_bd_valid"][0] == 1


def test_fs_quality_falls_back_when_columns_missing() -> None:
    builder = _make_builder()
    base_df = pl.DataFrame(
        {"Code": ["1001"], "Date": [dt.date(2024, 1, 2)], "stmt_yoy_sales": [0.1]}
    )

    fs_df = pl.DataFrame(
        {
            "Code": ["1001"],
            "DisclosedDate": [dt.date(2024, 1, 1)],
            "DisclosedTime": ["15:30:00"],
            "NetSales": [1_000.0],
        }
    )

    result = builder.add_fs_quality_features(
        base_df,
        fs_df,
        business_days=["2024-01-01", "2024-01-02"],
    )

    assert "fs_yoy_sales" in result.columns
    assert result["fs_yoy_sales"][0] == 0.1
    assert "is_fs_valid" in result.columns
    assert result["is_fs_valid"][0] == 1


def test_fx_features_attach_with_callable_next_expr() -> None:
    builder = _make_builder()
    base_df = pl.DataFrame(
        {"Date": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)], "base": [1.0, 2.0]}
    )
    fx_df = pl.DataFrame(
        {
            "Date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "macro_fx_usdjpy_close": [110.0, 111.0],
        }
    )
    fx_shifted = shift_to_next_business_day(
        fx_df,
        business_days=["2024-01-01", "2024-01-02", "2024-01-03"],
    )
    result = builder.add_fx_features(
        base_df, fx_shifted, business_days=["2024-01-01", "2024-01-02", "2024-01-03"]
    )
    assert "macro_fx_usdjpy_close" in result.columns
    assert result["macro_fx_usdjpy_close"][0] == 110.0


def test_btc_features_attach_with_callable_next_expr() -> None:
    builder = _make_builder()
    base_df = pl.DataFrame(
        {"Date": [dt.date(2024, 1, 2), dt.date(2024, 1, 3)], "base": [1.0, 2.0]}
    )
    btc_df = pl.DataFrame(
        {
            "Date": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)],
            "macro_btc_close": [40000.0, 40100.0],
        }
    )
    btc_shifted = shift_to_next_business_day(
        btc_df,
        business_days=["2024-01-01", "2024-01-02", "2024-01-03"],
    )
    result = builder.add_btc_features(
        base_df, btc_shifted, business_days=["2024-01-01", "2024-01-02", "2024-01-03"]
    )
    assert "macro_btc_close" in result.columns
    assert result["macro_btc_close"][0] == 40000.0
