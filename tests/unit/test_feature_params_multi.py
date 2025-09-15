import pandas as pd
from gogooku3.features.feature_builder import add_default_features
from gogooku3.features.feature_params import FeatureParams


def test_multiple_kama_vidya_from_config():
    df = pd.DataFrame({
        "id": ["A"]*120,
        "ts": pd.date_range("2025-01-01", periods=120, freq="D"),
        "y": range(120),
    })
    p = FeatureParams(
        kama_set=[(10, 2, 30), (20, 2, 30)],
        vidya_windows=[10, 20]
    )
    out = add_default_features(df, params=p)
    for (kw, kf, ks) in p.kama_set:
        assert f"kama_{kw}_{kf}_{ks}" in out.columns
    for vw in p.vidya_windows:
        assert f"vidya_{vw}" in out.columns

