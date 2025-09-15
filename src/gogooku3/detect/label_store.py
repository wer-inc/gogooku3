from __future__ import annotations

"""Range Label Store utilities.

Schema (flat): id, start, end, type, source?, notes?
"""

from dataclasses import dataclass, asdict
from typing import Iterable, List, Optional

import pandas as pd


@dataclass(frozen=True)
class LabelRecord:
    id: str
    start: pd.Timestamp
    end: pd.Timestamp
    type: str = "generic"
    source: str | None = None
    notes: str | None = None


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # normalize columns
    cols = {c.lower(): c for c in d.columns}
    for need in ("id", "start", "end"):
        if need not in cols:
            raise ValueError(f"missing required column: {need}")
    # coerce
    d = d.rename(columns={cols["id"]: "id", cols["start"]: "start", cols["end"]: "end"})
    d["id"] = d["id"].astype(str)
    d["start"] = pd.to_datetime(d["start"])  # type: ignore[assignment]
    d["end"] = pd.to_datetime(d["end"])  # type: ignore[assignment]
    if "type" not in d.columns:
        d["type"] = "generic"
    if "source" not in d.columns:
        d["source"] = None
    if "notes" not in d.columns:
        d["notes"] = None
    return d[["id", "start", "end", "type", "source", "notes"]]


def load_labels(path: str) -> list[LabelRecord]:
    if path.endswith(".json") or path.endswith(".jsonl"):
        df = pd.read_json(path)
    elif path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    d = _normalize_df(df)
    return [
        LabelRecord(
            id=str(r.id),
            start=pd.to_datetime(r.start),
            end=pd.to_datetime(r.end),
            type=str(r.type) if pd.notna(r.type) else "generic",
            source=None if pd.isna(r.source) else str(r.source),
            notes=None if pd.isna(r.notes) else str(r.notes),
        )
        for r in d.itertuples(index=False)
    ]


def save_labels(path: str, labels: Iterable[LabelRecord]) -> None:
    df = pd.DataFrame([asdict(l) for l in labels])
    df = _normalize_df(df)
    if path.endswith(".json") or path.endswith(".jsonl"):
        df.to_json(path, orient="records", date_format="iso")
    elif path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def merge_labels(a: Iterable[LabelRecord], b: Iterable[LabelRecord]) -> list[LabelRecord]:
    # Deduplicate by (id,start,end,type)
    key = set()
    out: list[LabelRecord] = []
    for rec in list(a) + list(b):
        k = (rec.id, pd.Timestamp(rec.start), pd.Timestamp(rec.end), rec.type)
        if k in key:
            continue
        key.add(k)
        out.append(rec)
    # sort for stability
    out.sort(key=lambda r: (r.id, r.start, r.end, r.type))
    return out

