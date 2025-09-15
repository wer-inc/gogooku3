import numpy as np
import pandas as pd
from gogooku3.features.tech_indicators import kama, vidya, fractional_diff, rolling_quantiles


def test_kama_vidya_shapes():
    s = pd.Series(np.linspace(0, 1, 120))
    k = kama(s, window=10)
    v = vidya(s, window=14)
    assert len(k) == len(s) and len(v) == len(s)
    # early NaNs expected
    assert k.isna().sum() > 0
    assert v.isna().sum() > 0


def test_fractional_diff_basic():
    s = pd.Series(np.arange(50, dtype=float))
    fd = fractional_diff(s, d=0.4, window=10)
    assert len(fd) == len(s)
    # should not be all NaN
    assert fd.notna().sum() > 10


def test_rolling_quantiles_cols():
    s = pd.Series(np.random.randn(200))
    rq = rolling_quantiles(s, window=63, quants=(0.1, 0.5, 0.9))
    assert set(rq.columns) == {"rq_10", "rq_50", "rq_90"}
    assert rq.shape[0] == len(s)

