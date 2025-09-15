"""Detection layer: residual scoring, change-points, SR, calibration, and ranges.

Public utilities:
- robust_scale_qmad: robust rolling normalization for residuals
- residual_q_score: residual-based anomaly score
- change_point_score: simple mean-shift score via windowed difference
- spectral_residual_score: SR saliency map score
- calibrate_sigmoid: light-weight Platt scaling (optional)
- stack_and_score: combine component scores into a single probability
- score_to_ranges: threshold → contiguous range binarization
- evaluate_vus_pr: range-level PR curve area (threshold sweep)
"""

from .residual import robust_scale_qmad, residual_q_score
from .change_point import change_point_score
from .spectral_residual import spectral_residual_score
from .ensemble import calibrate_sigmoid, stack_and_score, DetectionEnsemble
from .ranges import score_to_ranges, evaluate_vus_pr, Range, RangeLabel

__all__ = [
    "robust_scale_qmad",
    "residual_q_score",
    "change_point_score",
    "spectral_residual_score",
    "calibrate_sigmoid",
    "stack_and_score",
    "DetectionEnsemble",
    "score_to_ranges",
    "evaluate_vus_pr",
    "Range",
    "RangeLabel",
]

