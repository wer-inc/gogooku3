from __future__ import annotations

"""Range conversion and VUS-PR style evaluation utilities."""

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Range:
    id: str
    start: pd.Timestamp
    end: pd.Timestamp
    score: float
    type: str = "unknown"


@dataclass(frozen=True)
class RangeLabel:
    id: str
    start: pd.Timestamp
    end: pd.Timestamp
    type: str = "generic"


def _find_ranges_above_threshold(ts: np.ndarray, score: np.ndarray, thr: float, min_len: int = 1) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    n = score.size
    ranges: list[tuple[pd.Timestamp, pd.Timestamp, float]] = []
    i = 0
    while i < n:
        if score[i] >= thr:
            j = i
            smax = score[i]
            while j + 1 < n and score[j + 1] >= thr:
                j += 1
                smax = max(smax, score[j])
            length = j - i + 1
            if length >= min_len:
                ranges.append((pd.Timestamp(ts[i]), pd.Timestamp(ts[j]), float(smax)))
            i = j + 1
        else:
            i += 1
    return ranges


def score_to_ranges(
    df_score: pd.DataFrame,
    id_col: str = "id",
    ts_col: str = "ts",
    score_col: str = "score",
    threshold: float | None = None,
    min_len: int = 1,
    perc: float = 0.95,
) -> list[Range]:
    """Convert pointwise anomaly scores to contiguous ranges.

    - If threshold is None, uses per-id percentile threshold (perc).
    - Returns a list of Range objects.
    """
    d = df_score.copy()
    d[ts_col] = pd.to_datetime(d[ts_col])
    d.sort_values([id_col, ts_col], inplace=True)
    out: list[Range] = []
    for gid, g in d.groupby(id_col, sort=False):
        scores = g[score_col].astype(float).to_numpy()
        ts = g[ts_col].to_numpy()
        thr = float(np.quantile(scores, perc)) if threshold is None else float(threshold)
        rr = _find_ranges_above_threshold(ts, scores, thr, min_len=min_len)
        for s, e, sc in rr:
            out.append(Range(id=str(gid), start=s, end=e, score=sc))
    return out


def _overlap_len(a: tuple[pd.Timestamp, pd.Timestamp], b: tuple[pd.Timestamp, pd.Timestamp]) -> int:
    """Number of overlapping timesteps between two [start, end] closed intervals.

    Assumes both intervals align to the same underlying discrete timeline.
    """
    lo = max(a[0], b[0])
    hi = min(a[1], b[1])
    if hi < lo:
        return 0
    # Closed intervals: add 1
    return int((hi - lo).days) + 1


def _greedy_match(pred: list[Range], labels: list[RangeLabel]) -> tuple[int, int]:
    """Greedy 1-1 matching by maximum overlap length per id.

    Returns (#matched_pred, #matched_labels)
    """
    matched_pred = 0
    matched_labels = 0
    by_id: dict[str, list[RangeLabel]] = {}
    for l in labels:
        by_id.setdefault(l.id, []).append(l)
    for pid in by_id.keys():
        by_id[pid].sort(key=lambda r: (r.start, r.end))

    for r in pred:
        cand = by_id.get(r.id, [])
        best_idx = -1
        best_ov = 0
        for i, l in enumerate(cand):
            ov = _overlap_len((r.start, r.end), (l.start, l.end))
            if ov > best_ov:
                best_ov = ov
                best_idx = i
        if best_idx >= 0 and best_ov > 0:
            matched_pred += 1
            matched_labels += 1
            # Remove matched label to enforce 1-1 matching
            cand.pop(best_idx)
    return matched_pred, matched_labels


def evaluate_vus_pr(
    predicted_ranges: list[Range],
    label_ranges: list[RangeLabel],
    thresholds: Iterable[float] | None = None,
) -> dict:
    """Approximate VUS-PR for range detection via threshold sweep.

    This implementation treats VUS-PR as AUC-PR on range events where a
    predicted range is correct if it overlaps any labeled range of the same id
    (greedy 1-1). It is threshold-agnostic by integrating over thresholds.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    if not predicted_ranges:
        return {"vus_pr": 0.0, "curve": []}
    if not label_ranges:
        return {"vus_pr": 0.0, "curve": []}

    # Pre-group predictions per id sorted by score desc for stability
    preds = sorted(predicted_ranges, key=lambda r: (r.id, -r.score))
    total_labels = len(label_ranges)

    pr: list[tuple[float, float]] = []  # (recall, precision)
    for thr in thresholds:
        sel = [r for r in preds if r.score >= thr]
        if not sel:
            pr.append((0.0, 1.0))
            continue
        mp, ml = _greedy_match(sel, label_ranges.copy())
        precision = mp / max(len(sel), 1)
        recall = ml / max(total_labels, 1)
        pr.append((recall, precision))

    # Sort by recall and integrate precision via trapezoid
    pr_sorted = sorted(pr, key=lambda x: x[0])
    recalls = np.array([p[0] for p in pr_sorted])
    precisions = np.array([p[1] for p in pr_sorted])
    # Ensure starting at recall=0
    if recalls[0] > 0:
        recalls = np.insert(recalls, 0, 0.0)
        precisions = np.insert(precisions, 0, precisions[0])
    # AUC under PR curve
    vus = float(np.trapz(precisions, recalls))
    return {"vus_pr": vus, "curve": [(float(r), float(p)) for r, p in zip(recalls, precisions)]}


def _iou(a: tuple[pd.Timestamp, pd.Timestamp], b: tuple[pd.Timestamp, pd.Timestamp]) -> float:
    ov = _overlap_len(a, b)
    if ov == 0:
        return 0.0
    alen = _overlap_len(a, a)
    blen = _overlap_len(b, b)
    union = alen + blen - ov
    return float(ov / max(union, 1))


def evaluate_vus_pr_iou(
    predicted_ranges: list[Range],
    label_ranges: list[RangeLabel],
    thresholds: Iterable[float] | None = None,
    min_iou: float = 0.25,
) -> dict:
    """VUS-PR variant that uses IoU threshold to validate a match.

    A predicted range is considered correct if it has IoU>=min_iou with any
    labeled range for the same id (greedy 1–1 matching by highest IoU).
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    if not predicted_ranges or not label_ranges:
        return {"vus_pr": 0.0, "curve": []}

    preds = sorted(predicted_ranges, key=lambda r: (r.id, -r.score))
    total_labels = len(label_ranges)

    pr: list[tuple[float, float]] = []
    for thr in thresholds:
        sel = [r for r in preds if r.score >= thr]
        if not sel:
            pr.append((0.0, 1.0))
            continue
        # Greedy matching by IoU
        used = set()
        matched = 0
        for r in sel:
            best_iou = 0.0
            best_j = -1
            for j, l in enumerate(label_ranges):
                if j in used or l.id != r.id:
                    continue
                iou = _iou((r.start, r.end), (l.start, l.end))
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= min_iou:
                used.add(best_j)
                matched += 1
        precision = matched / max(len(sel), 1)
        recall = len(used) / max(total_labels, 1)
        pr.append((recall, precision))

    pr_sorted = sorted(pr, key=lambda x: x[0])
    recalls = np.array([p[0] for p in pr_sorted])
    precisions = np.array([p[1] for p in pr_sorted])
    if recalls[0] > 0:
        recalls = np.insert(recalls, 0, 0.0)
        precisions = np.insert(precisions, 0, precisions[0])
    vus = float(np.trapz(precisions, recalls))
    return {"vus_pr": vus, "curve": [(float(r), float(p)) for r, p in zip(recalls, precisions)]}


class VUS_PR:
    """VUS-PR evaluation class for range-based anomaly detection."""

    def __init__(self, min_iou: float = 0.25, thresholds: list[float] | None = None):
        """Initialize VUS-PR evaluator.

        Args:
            min_iou: Minimum IoU threshold for range matching
            thresholds: List of score thresholds to evaluate
        """
        self.min_iou = min_iou
        self.thresholds = thresholds or list(np.linspace(0.05, 0.95, 19))

    def evaluate(self, predicted_ranges: list[Range], label_ranges: list[RangeLabel]) -> dict:
        """Evaluate VUS-PR score.

        Args:
            predicted_ranges: List of predicted anomaly ranges
            label_ranges: List of ground truth label ranges

        Returns:
            Dictionary with VUS-PR score and curve data
        """
        return evaluate_vus_pr_iou(
            predicted_ranges=predicted_ranges,
            label_ranges=label_ranges,
            thresholds=self.thresholds,
            min_iou=self.min_iou
        )

    def score(self, predicted_ranges: list[Range], label_ranges: list[RangeLabel]) -> float:
        """Get VUS-PR score as single float.

        Args:
            predicted_ranges: List of predicted anomaly ranges
            label_ranges: List of ground truth label ranges

        Returns:
            VUS-PR score (0.0 to 1.0)
        """
        result = self.evaluate(predicted_ranges, label_ranges)
        return result["vus_pr"]
