from __future__ import annotations

"""Utilities to generate RangeLabel from event points.

Turns event dates (point labels) into range labels with optional pre/post windows.
Supports id='*' broadcasting to all ids in the provided universe.
"""

from collections.abc import Iterable

import pandas as pd

from .ranges import RangeLabel


def events_to_ranges(
    df_events: pd.DataFrame,
    ids: Iterable[str],
    pre_days: int = 0,
    post_days: int = 0,
    type_col: str | None = None,
) -> list[RangeLabel]:
    if df_events is None or df_events.empty:
        return []
    ev = df_events.copy()
    ev["ts"] = pd.to_datetime(ev["ts"])  # type: ignore[assignment]
    out: list[RangeLabel] = []
    id_set = list(dict.fromkeys([str(x) for x in ids]))
    for _, row in ev.iterrows():
        rid = str(row.get("id", "*"))
        ts = pd.to_datetime(row["ts"])  # type: ignore[arg-type]
        tpe = str(row.get(type_col, "event")) if type_col else "event"
        start = ts - pd.Timedelta(days=pre_days)
        end = ts + pd.Timedelta(days=post_days)
        targets = id_set if rid == "*" else [rid]
        for tid in targets:
            out.append(RangeLabel(id=tid, start=start, end=end, type=tpe))
    return out

