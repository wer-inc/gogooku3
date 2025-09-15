import pandas as pd
import numpy as np
import pytest

from gogooku3.detect import score_to_ranges, evaluate_vus_pr, RangeLabel


@pytest.mark.unit
def test_score_to_ranges_and_vus_pr_simple():
    # Build a simple time series with a clear anomalous block
    ts = pd.date_range("2025-01-01", periods=30, freq="D")
    score = np.zeros(30)
    score[10:15] = 0.99  # anomaly 5 days
    df = pd.DataFrame({"id": ["AAA"] * 30, "ts": ts, "score": score})

    ranges = score_to_ranges(df, threshold=0.9, min_len=2)
    assert len(ranges) == 1
    r = ranges[0]
    assert r.start == ts[10]
    assert r.end == ts[14]

    labels = [RangeLabel(id="AAA", start=ts[10], end=ts[14], type="event")]
    res = evaluate_vus_pr(ranges, labels, thresholds=[0.1, 0.5, 0.9])
    assert 0.5 <= res["vus_pr"] <= 1.0  # reasonably high

