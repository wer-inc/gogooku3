from __future__ import annotations

import polars as pl
import pytest

from builder.pipelines.dataset_builder import DatasetBuilder


def test_short_positions_delta_synthesized_when_missing() -> None:
    df = pl.DataFrame(
        {
            "shortpositionstosharesoutstandingratio": ["1.5", "0.8"],
            "shortpositionsinpreviousreportingratio": ["1.2", None],
        }
    )

    result = DatasetBuilder._ensure_short_positions_delta_column(
        df,
        ratio_col="shortpositionstosharesoutstandingratio",
        prev_col="shortpositionsinpreviousreportingratio",
    )

    deltas = result["differenceinshortpositionsratiofrompreviousreport"].to_list()
    assert deltas[0] == pytest.approx(0.3)
    assert deltas[1] is None
