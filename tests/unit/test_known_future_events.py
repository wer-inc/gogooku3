import pandas as pd
from gogooku3.features.known_future import add_event_flags


def test_add_event_flags_broadcast_and_specific():
    obs = pd.DataFrame({
        "id": ["X", "Y"],
        "ts": ["2025-03-28", "2025-03-28"],
        "y": [0.0, 0.0],
    })
    events = pd.DataFrame({
        "id": ["*", "Y"],
        "ts": ["2025-03-28", "2025-03-28"],
        "event_earnings": [1, 1],
    })
    out = add_event_flags(obs, events)
    assert out.loc[0, "event_earnings"] == 1
    assert out.loc[1, "event_earnings"] == 1

