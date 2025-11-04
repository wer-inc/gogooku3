from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class FeatureParams:
    # KAMA (single) for backward-compat
    kama_window: int = 10
    kama_fast: int = 2
    kama_slow: int = 30
    # KAMA multiple configs: list of [window, fast, slow]
    kama_set: list[tuple[int, int, int]] = field(default_factory=list)
    # VIDYA
    vidya_window: int = 14
    # VIDYA multiple windows
    vidya_windows: list[int] = field(default_factory=list)
    # Fractional differencing
    fd_d: float = 0.4
    fd_window: int = 100
    # Rolling quantiles
    rq_window: int = 63
    rq_windows: list[int] = field(default_factory=list)
    rq_quantiles: Sequence[float] = (0.1, 0.5, 0.9)
    # Cross-sectional features to compute per date
    cs_features: Sequence[str] = ("y",)
    # Rolling std windows (adds columns roll_std_{w})
    roll_std_windows: list[int] = field(default_factory=lambda: [20])
