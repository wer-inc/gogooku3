from __future__ import annotations

"""
Macro-level feature utilities (e.g., VIX, interest rates).

This package provides lightweight helpers that can be plugged into the dataset
generation pipeline without adding heavy dependencies to downstream modules.
"""

from .btc import load_btc_history, prepare_btc_features
from .fx import load_fx_history, prepare_fx_features
from .vix import load_vix_history, prepare_vix_features, shift_to_next_business_day

__all__ = [
    "load_vix_history",
    "prepare_vix_features",
    "shift_to_next_business_day",
    "load_fx_history",
    "prepare_fx_features",
    "load_btc_history",
    "prepare_btc_features",
]
