import pandas as pd
import numpy as np
import pytest

from gogooku3.forecast import timesfm_predict


@pytest.mark.unit
def test_timesfm_predict_baseline_shape():
    ids = ["A", "B"]
    ts = pd.date_range("2025-01-01", periods=10, freq="D")
    rows = []
    for i in ids:
        for t in ts:
            rows.append({"id": i, "ts": t, "y": np.sin((t.dayofyear % 7) / 7)})
    df = pd.DataFrame(rows)
    fc = timesfm_predict(df, horizons=[1, 5, 30], context=8)
    assert set(["id", "ts", "h", "y_hat"]).issubset(fc.columns)
    assert fc["h"].nunique() == 3
    assert fc.groupby("id").size().min() >= 3

