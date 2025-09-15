from __future__ import annotations

"""Simple regression gate utilities.

Compares baseline vs candidate metrics with allowed degradation thresholds.
"""

import json
from dataclasses import dataclass
from typing import Dict


@dataclass
class GateRule:
    key: str
    direction: str  # "min" or "max"
    max_regress: float  # allowed regression (positive number)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_gate(baseline: Dict, candidate: Dict, rule: GateRule) -> dict:
    # Fetch metric value by key (top-level key expected)
    bval = float(baseline.get(rule.key))
    cval = float(candidate.get(rule.key))
    passed = True
    regress = 0.0
    if rule.direction == "min":
        # smaller is better; regression if cval - bval > max_regress
        regress = cval - bval
        passed = regress <= rule.max_regress
    else:
        # larger is better; regression if bval - cval > max_regress
        regress = bval - cval
        passed = regress <= rule.max_regress
    return {
        "key": rule.key,
        "direction": rule.direction,
        "baseline": bval,
        "candidate": cval,
        "regress": regress,
        "max_regress": rule.max_regress,
        "passed": passed,
    }

