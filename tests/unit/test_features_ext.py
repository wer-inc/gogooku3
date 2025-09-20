import numpy as np
import polars as pl
import pytest

from gogooku3.features_ext.cs_standardize import fit_cs_stats, transform_cs
from gogooku3.features_ext.interactions import add_interactions
from gogooku3.features_ext.outliers import winsorize
from gogooku3.features_ext.outliers import fit_winsor_stats, transform_winsor
from gogooku3.features_ext.scale_unify import add_ratio_adv_z
from gogooku3.features_ext.sector_loo import add_sector_loo


def test_sector_loo_mean_basic():
    df = pl.DataFrame(
        {
            "Date": ["2024-01-01"] * 4,
            "sector33_id": [1, 1, 2, 2],
            "returns_1d": [0.01, 0.03, -0.02, 0.00],
        }
    )
    out = add_sector_loo(df)
    # sector 1: other mean = (0.03)/1 for row 0; (0.01)/1 for row 1
    assert np.isclose(out["sec_ret_1d_eq_loo"][0], 0.03)
    assert np.isclose(out["sec_ret_1d_eq_loo"][1], 0.01)


def test_add_ratio_adv_z_small_window():
    df = pl.DataFrame(
        {
            "Code": ["A"] * 5,
            "val": [1, 2, 3, 4, 5],
            "adv": [10, 10, 10, 10, 10],
        }
    )
    out = add_ratio_adv_z(df, value_col="val", adv_col="adv", z_win=3, prefix="v")
    assert "v_to_adv20" in out.columns and "v_z3" in out.columns
    # center row has non-null z when window=3
    assert out["v_z3"][2] is not None


def test_winsorize_clips_extremes():
    df = pl.DataFrame({"x": [0.0, 0.0, 0.0, 1000.0]})
    out = winsorize(df, ["x"], k=1.0)
    assert out["x"].max() < 1000.0


def test_fold_safe_winsorize_fit_transform():
    train = pl.DataFrame({"x": [0.0, 0.0, 0.0, 10.0]})
    val = pl.DataFrame({"x": [1000.0, -999.0]})
    st = fit_winsor_stats(train, ["x"], k=1.0)
    val_t = transform_winsor(val, st)
    # should be clipped to train-based thresholds
    assert val_t["x"].max() <= 10.0


def test_add_interactions_all_columns_present():
    # Create minimal frame with required columns present
    n = 3
    cols = {
        "ma_gap_5_20": [0.1] * n,
        "mkt_gap_5_20": [0.2] * n,
        "rel_to_sec_5d": [0.0, 0.1, -0.1],
        "sec_mom_20": [0.05] * n,
        "returns_5d": [0.02, 0.0, -0.02],
        "volatility_20d": [0.01] * n,
        "volume_ratio_5": [1.0, 0.5, 2.0],
        "returns_1d": [0.01, -0.02, 0.0],
        "dmi_short_to_adv20": [0.3] * n,
        "rel_strength_5d": [0.2, -0.1, 0.0],
        "dmi_credit_ratio": [1.1, 0.9, 1.0],
        "z_close_20": [0.5, -1.2, 0.0],
        "stmt_rev_fore_op": [0.0, 0.0, 0.0],
        "stmt_progress_op": [0.0, 0.0, 0.0],
        "stmt_days_since_statement": [1, 2, 3],
        "mkt_high_vol": [1, 0, 1],
        "alpha_1d": [0.01, -0.02, 0.0],
        "beta_stability_60d": [0.8, 1.2, 1.0],
        "flow_smart_idx": [0.1, 0.2, 0.3],
        "Code": ["A", "A", "A"],
    }
    df = pl.DataFrame(cols)
    out = add_interactions(df)
    assert all(c in out.columns for c in [
        "x_trend_intensity",
        "x_rel_sec_mom",
        "x_mom_sh_5",
        "x_rvol5_dir",
        "x_squeeze_pressure",
        "x_credit_rev_bias",
        "x_pead_effect",
        "x_rev_gate",
        "x_alpha_meanrev_stable",
        "x_flow_smart_rel",
    ])


def test_cs_standardize_fit_transform():
    df = pl.DataFrame(
        {
            "Date": ["2024-01-01"] * 3 + ["2024-01-02"] * 3,
            "sector33_id": [1, 1, 1, 1, 1, 1],
            "f1": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
        }
    )
    train = df.filter(pl.col("Date") == "2024-01-01")
    test = df.filter(pl.col("Date") == "2024-01-02")
    stats = fit_cs_stats(train, ["f1"], date_col="Date")
    out = transform_cs(test, stats, ["f1"])
    # Standardization reflects train-day statistics; values likely far from 0, which is desired
    assert "f1_cs_z" in out.columns
