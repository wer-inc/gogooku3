import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from gogooku3.features.feature_selector import SelectionConfig, select_features, save_selected


@pytest.mark.unit
def test_mutual_info_selection_and_save(tmp_path: Path):
    # Small synthetic dataset
    n = 200
    rng = np.random.default_rng(42)
    f1 = rng.normal(size=n)
    f2 = rng.normal(size=n)
    f3 = rng.normal(size=n)
    target = 0.7 * f1 + 0.3 * rng.normal(size=n)
    from datetime import datetime, timedelta
    dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    df = pl.DataFrame({
        "Date": dates,
        "Code": [1000 + i for i in range(n)],
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "target_1d": target,
    })

    cfg = SelectionConfig(method="mutual_info", top_k=2, target_column="target_1d")
    selected = select_features(df, cfg)
    assert len(selected) == 2
    # f1 should be preferred most of the time
    assert "f1" in selected

    out = tmp_path / "selected_features.json"
    save_selected(selected, out)
    loaded = json.loads(out.read_text())
    assert loaded["selected_features"] == selected
