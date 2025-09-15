import pandas as pd
import numpy as np
from gogooku3.forecast import TFTAdapter


def test_tft_adapter_fit_predict_shapes():
    # create simple linear + noise features
    ids = ["A", "B"]
    ts = pd.date_range("2025-01-01", periods=30, freq="D")
    rows = []
    rng = np.random.default_rng(0)
    for i in ids:
        base = rng.normal(0, 0.1, size=len(ts)).cumsum()
        for t, k in zip(ts, range(len(ts))):
            rows.append({
                "id": i, "ts": t, "y": base[k] + 0.1 * k,
                "feat1": k, "feat2": np.sin(k/7)
            })
    df = pd.DataFrame(rows)
    model = TFTAdapter(horizons=[1, 5, 10])
    model.fit(df)
    out = model.predict(df)
    assert set(["id", "ts", "h", "y_hat"]).issubset(out.columns)
    assert out["h"].nunique() == 3
    # per id one origin row per horizon
    assert out.groupby("id").size().min() >= 3

