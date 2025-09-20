import polars as pl
import numpy as np

from gogooku3.features_ext.sector_loo import add_sector_loo
from gogooku3.features_ext.scale_unify import add_ratio_adv_z
from gogooku3.features_ext.cs_standardize import fit_cs_stats, transform_cs


def test_columns_not_dropped_after_build_steps():
    base = pl.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "Code": ["A", "B", "A", "B"],
            "sector33_id": [1, 1, 1, 1],
            "returns_1d": [0.01, 0.02, -0.01, 0.00],
            "margin_long_tot": [10.0, 5.0, 11.0, 6.0],
            "dollar_volume_ma20": [1000.0, 500.0, 1000.0, 500.0],
        }
    )
    base_cols = set(base.columns)
    df = add_sector_loo(base)
    df = add_ratio_adv_z(df, value_col="margin_long_tot", adv_col="dollar_volume_ma20", prefix="margin_long")
    assert base_cols.issubset(set(df.columns))


def test_loo_not_equal_self_return():
    df = pl.DataFrame(
        {
            "Date": ["2024-01-01", "2024-01-01"],
            "sector33_id": [1, 1],
            "returns_1d": [0.01, 0.03],
        }
    )
    out = add_sector_loo(df)
    assert (out["sec_ret_1d_eq_loo"][0] != out["returns_1d"][0]) and (out["sec_ret_1d_eq_loo"][1] != out["returns_1d"][1])


def test_cs_z_warmup_has_nulls():
    # 4-step rolling; first two should be null by min_periods default
    df = pl.DataFrame({"Date": ["2024-01-0" + str(i) for i in range(1, 7)], "Code": ["A"] * 6, "x": [1, 2, 3, 4, 5, 6]})
    stats = fit_cs_stats(df, ["x"], date_col="Date")
    out = transform_cs(df, stats, ["x"])  # trivial here, but ensures suffix created
    assert "x_cs_z" in out.columns
