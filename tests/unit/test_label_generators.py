import pandas as pd
from gogooku3.detect.label_generators import events_to_ranges


def test_events_to_ranges_broadcast_and_pre_post():
    ids = ["A", "B"]
    events = pd.DataFrame({
        "id": ["*"],
        "ts": ["2025-04-01"],
        "etype": ["earnings"],
    })
    labels = events_to_ranges(events, ids, pre_days=1, post_days=2, type_col="etype")
    assert len(labels) == 2
    assert labels[0].start <= labels[0].end

