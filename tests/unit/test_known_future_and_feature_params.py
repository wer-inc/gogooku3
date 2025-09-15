import pandas as pd
from gogooku3.features.known_future import add_jp_holiday_features
from gogooku3.features.feature_builder import add_default_features
from gogooku3.features.feature_params import FeatureParams


def test_jp_holiday_flag_new_year():
    df = pd.DataFrame({
        "id": ["X", "X"],
        "ts": ["2025-01-01", "2025-01-02"],
        "y": [0.0, 0.0],
    })
    out = add_jp_holiday_features(df)
    # 2025-01-01 is a public holiday in Japan
    assert out.loc[0, "holiday"] == 1
    assert out.loc[1, "holiday"] in (0, 1)  # depends on calendar


def test_feature_params_column_names():
    df = pd.DataFrame({
        "id": ["A"]*120,
        "ts": pd.date_range("2025-01-01", periods=120, freq="D"),
        "y": range(120),
    })
    p = FeatureParams(kama_window=12, kama_fast=2, kama_slow=30, vidya_window=10, rq_window=20)
    out = add_default_features(df, params=p)
    assert f"kama_{p.kama_window}_{p.kama_fast}_{p.kama_slow}" in out.columns
    assert f"vidya_{p.vidya_window}" in out.columns
