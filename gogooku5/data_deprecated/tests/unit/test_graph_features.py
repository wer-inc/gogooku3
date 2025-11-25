from __future__ import annotations

from datetime import date

import polars as pl
from builder.features.core.graph.features import (
    GraphFeatureConfig,
    GraphFeatureEngineer,
)


def test_graph_feature_engineer_add_features() -> None:
    df = pl.DataFrame(
        {
            "code": ["A", "B", "C", "A", "B", "C"],
            "date": [
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 1),
                date(2024, 1, 2),
                date(2024, 1, 2),
                date(2024, 1, 2),
            ],
            "returns_1d": [0.01, 0.012, 0.009, 0.008, 0.011, 0.010],
        }
    )

    engineer = GraphFeatureEngineer(
        config=GraphFeatureConfig(
            window_days=2,
            min_observations=1,
            correlation_threshold=0.0,
            shift_to_next_day=False,
        )
    )
    out = engineer.add_features(df)

    assert "graph_degree" in out.columns
