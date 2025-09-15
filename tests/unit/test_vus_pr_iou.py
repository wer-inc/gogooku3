import pandas as pd
from gogooku3.detect.ranges import Range, RangeLabel, evaluate_vus_pr_iou


def test_vus_pr_iou_basic():
    # One predicted range with good overlap should score well
    pred = [Range(id="X", start=pd.Timestamp("2025-01-05"), end=pd.Timestamp("2025-01-10"), score=0.9)]
    gold = [RangeLabel(id="X", start=pd.Timestamp("2025-01-07"), end=pd.Timestamp("2025-01-12"))]
    res = evaluate_vus_pr_iou(pred, gold, thresholds=[0.1, 0.5, 0.9], min_iou=0.2)
    assert 0.3 <= res["vus_pr"] <= 1.0

