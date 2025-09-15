import pandas as pd
import tempfile
from gogooku3.detect.label_store import LabelRecord, save_labels, load_labels, merge_labels


def test_label_store_roundtrip(tmp_path):
    labels = [
        LabelRecord(id="A", start=pd.Timestamp("2025-01-01"), end=pd.Timestamp("2025-01-03"), type="event"),
        LabelRecord(id="B", start=pd.Timestamp("2025-02-01"), end=pd.Timestamp("2025-02-02")),
    ]
    p = tmp_path / "labels.json"
    save_labels(str(p), labels)
    back = load_labels(str(p))
    assert len(back) == 2
    assert back[0].id == "A"


def test_merge_labels_dedup():
    a = [LabelRecord(id="A", start=pd.Timestamp("2025-01-01"), end=pd.Timestamp("2025-01-02"), type="x")]
    b = [LabelRecord(id="A", start=pd.Timestamp("2025-01-01"), end=pd.Timestamp("2025-01-02"), type="x")]
    m = merge_labels(a, b)
    assert len(m) == 1

