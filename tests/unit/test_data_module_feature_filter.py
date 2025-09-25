import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from omegaconf import OmegaConf

from gogooku3.training.atft.data_module import ProductionDataModuleV2


def _minimal_cfg(data_dir: Path):
    return OmegaConf.create({
        "data": {
            "schema": {
                "date_column": "Date",
                "code_column": "Code",
                "target_column": "target",
                "feature_columns": [],
            },
            "source": {"data_dir": str(data_dir)},
            "time_series": {"sequence_length": 4, "prediction_horizons": [1, 5]},
        },
        "train": {"batch": {"train_batch_size": 8, "val_batch_size": 8}},
        "normalization": {"online_normalization": {"enabled": False}},
    })


@pytest.mark.unit
def test_feature_columns_auto_and_selected(tmp_path: Path, monkeypatch):
    # Create a tiny parquet file
    n = 16
    from datetime import datetime, timedelta
    dates = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(n)]
    df = pl.DataFrame({
        "Date": dates,
        "Code": [1000 + i for i in range(n)],
        "f1": np.arange(n),
        "f2": np.arange(n) * 2,
        "f3": np.arange(n) * 3,
        "target": np.linspace(-1, 1, n),
    })
    data_dir = tmp_path / "atft_data" / "train"
    data_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = data_dir / "part-000.parquet"
    df.write_parquet(parquet_path)

    cfg = _minimal_cfg(tmp_path / "atft_data")
    dm = ProductionDataModuleV2(config=cfg)

    # Auto-detect should find f1,f2,f3
    cols = dm._get_feature_columns()
    assert set(cols) == {"f1", "f2", "f3"}

    # Apply external selection JSON (subset)
    sel = tmp_path / "selected_features.json"
    sel.write_text(json.dumps({"selected_features": ["f1", "f2", "nonexist"]}))
    monkeypatch.setenv("SELECTED_FEATURES_JSON", str(sel))
    cols2 = dm._get_feature_columns()
    assert set(cols2) == {"f1", "f2"}
