"""Portfolio optimisation helpers for APEX-Ranker backtests."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .costs import CostCalculator


@dataclass(slots=True)
class OptimizationConfig:
    """Hyper-parameters controlling cost-aware portfolio optimisation."""

    target_top_k: int = 35
    candidate_multiplier: float = 2.0
    min_weight: float = 0.02
    turnover_limit: float = 0.35
    cost_penalty: float = 1.0
    min_alpha: float = 0.1
    epsilon: float = 1e-8

    def candidate_count(self, available: int) -> int:
        """Return the candidate pool size given ``available`` predictions."""
        if available <= 0:
            return 0
        multiplier = max(self.candidate_multiplier, 1.0)
        desired = max(1, int(round(self.target_top_k * multiplier)))
        return min(available, max(self.target_top_k, desired))


@dataclass(slots=True)
class OptimizationResult:
    """Diagnostic information from ``generate_target_weights``."""

    candidate_codes: list[str] = field(default_factory=list)
    selected_codes: list[str] = field(default_factory=list)
    applied_alpha: float = 1.0
    unconstrained_turnover: float = 0.0
    constrained_turnover: float = 0.0
    dropped_for_min_weight: list[str] = field(default_factory=list)
    dropped_for_turnover: list[str] = field(default_factory=list)
    score_penalties: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert result to a JSON-serialisable dictionary."""
        return {
            "candidate_codes": list(self.candidate_codes),
            "selected_codes": list(self.selected_codes),
            "applied_alpha": float(self.applied_alpha),
            "unconstrained_turnover": float(self.unconstrained_turnover),
            "constrained_turnover": float(self.constrained_turnover),
            "dropped_for_min_weight": list(self.dropped_for_min_weight),
            "dropped_for_turnover": list(self.dropped_for_turnover),
            "score_penalties": {k: float(v) for k, v in self.score_penalties.items()},
            "notes": list(self.notes),
        }


def _compute_turnover(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
) -> float:
    """Return turnover implied by moving from ``current`` to ``target`` weights."""
    codes = set(current_weights.keys()) | set(target_weights.keys())
    total_change = sum(
        abs(target_weights.get(code, 0.0) - current_weights.get(code, 0.0))
        for code in codes
    )
    return total_change / 2.0


def _apply_alpha_step(
    current_weights: Mapping[str, float],
    target_weights: Mapping[str, float],
    alpha: float,
    *,
    epsilon: float,
) -> dict[str, float]:
    """Interpolate between current and target weights using factor ``alpha``."""
    alpha = max(0.0, min(1.0, alpha))
    weights: dict[str, float] = {}
    codes = set(current_weights.keys()) | set(target_weights.keys())
    for code in codes:
        current = current_weights.get(code, 0.0)
        target = target_weights.get(code, 0.0)
        interpolated = current + alpha * (target - current)
        if interpolated > epsilon:
            weights[code] = float(interpolated)

    total = sum(weights.values())
    if total > epsilon:
        for code in list(weights.keys()):
            weights[code] /= total
    else:
        weights.clear()
    return weights


def _estimate_round_trip_penalty(
    code: str,
    *,
    portfolio_value: float,
    weight: float,
    volumes: Mapping[str, float] | None,
    cost_calculator: CostCalculator | None,
    cost_penalty: float,
) -> float:
    """Estimate fractional score penalty from round-trip transaction cost."""
    if (
        cost_calculator is None
        or portfolio_value <= 0.0
        or weight <= 0.0
        or cost_penalty <= 0.0
    ):
        return 0.0

    trade_value = portfolio_value * weight
    if trade_value <= 0.0:
        return 0.0

    volume_value = 0.0
    if volumes is not None:
        volume_value = float(volumes.get(code, 0.0) or 0.0)
    if volume_value <= 0.0:
        volume_value = trade_value * 100.0

    buy_cost = cost_calculator.calculate_trade_cost(
        trade_value,
        volume_value,
        "buy",
    )
    sell_cost = cost_calculator.calculate_trade_cost(
        trade_value,
        volume_value,
        "sell",
    )
    round_trip_cost = buy_cost["total_cost"] + sell_cost["total_cost"]
    if round_trip_cost <= 0.0:
        return 0.0
    fraction = round_trip_cost / trade_value
    return cost_penalty * fraction


def _normalise_weights(
    weights: Mapping[str, float],
    *,
    epsilon: float,
) -> dict[str, float]:
    """Ensure weights sum to one and drop negligible allocations."""
    filtered = {code: float(max(weight, 0.0)) for code, weight in weights.items()}
    filtered = {code: weight for code, weight in filtered.items() if weight > epsilon}
    total = sum(filtered.values())
    if total <= epsilon:
        return {}
    return {code: weight / total for code, weight in filtered.items()}


def generate_target_weights(
    predictions: Mapping[str, float],
    current_weights: Mapping[str, float],
    *,
    portfolio_value: float,
    config: OptimizationConfig,
    cost_calculator: CostCalculator | None = None,
    volumes: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], OptimizationResult]:
    """Construct cost-aware target weights from model predictions."""
    result = OptimizationResult()

    if not predictions:
        result.notes.append("no_predictions_available")
        return {}, result

    sorted_items = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    candidate_count = config.candidate_count(len(sorted_items))
    if candidate_count == 0:
        result.notes.append("candidate_count_zero")
        return {}, result

    candidate_items = sorted_items[:candidate_count]
    result.candidate_codes = [code for code, _ in candidate_items]

    base_weight = 1.0 / max(config.target_top_k, 1)
    adjusted_scores: dict[str, float] = {}
    for code, score in candidate_items:
        is_existing = code in current_weights
        penalty = 0.0
        if not is_existing:
            penalty = _estimate_round_trip_penalty(
                code,
                portfolio_value=portfolio_value,
                weight=base_weight,
                volumes=volumes,
                cost_calculator=cost_calculator,
                cost_penalty=config.cost_penalty,
            )
        adjusted_scores[code] = score - penalty
        if penalty:
            result.score_penalties[code] = penalty

    candidate_sorted = sorted(
        candidate_items,
        key=lambda x: adjusted_scores.get(x[0], x[1]),
        reverse=True,
    )

    selected_codes = [
        code for code, _ in candidate_sorted[: config.target_top_k] if code
    ]

    dropped_for_min_weight: list[str] = []
    dropped_for_turnover: list[str] = []

    while selected_codes:
        num_positions = len(selected_codes)
        weight_per_position = 1.0 / num_positions

        if weight_per_position + config.epsilon < config.min_weight:
            removed = selected_codes.pop()
            dropped_for_min_weight.append(removed)
            continue

        target_weights = dict.fromkeys(selected_codes, weight_per_position)
        unconstrained_turnover = _compute_turnover(current_weights, target_weights)

        if (
            config.turnover_limit <= 0.0
            or unconstrained_turnover <= config.turnover_limit + config.epsilon
        ):
            applied_alpha = 1.0
            final_weights = target_weights
        else:
            raw_alpha = unconstrained_turnover
            alpha = config.turnover_limit / max(raw_alpha, config.epsilon)
            alpha = max(config.min_alpha, min(1.0, alpha))
            final_weights = _apply_alpha_step(
                current_weights,
                target_weights,
                alpha,
                epsilon=config.epsilon,
            )
            applied_alpha = alpha

            # Drop new positions that become too small after scaling
            too_small = [
                code
                for code in final_weights
                if code not in current_weights
                and final_weights[code] + config.epsilon < config.min_weight
            ]
            if too_small and alpha < 1.0:
                # Remove the weakest adjusted score among the too-small newcomers
                weakest = min(
                    too_small,
                    key=lambda code: adjusted_scores.get(code, float("-inf")),
                )
                if weakest in selected_codes:
                    selected_codes.remove(weakest)
                    dropped_for_turnover.append(weakest)
                    continue

        final_weights = _normalise_weights(final_weights, epsilon=config.epsilon)
        if not final_weights:
            removed = selected_codes.pop()
            dropped_for_turnover.append(removed)
            continue

        constrained_turnover = _compute_turnover(current_weights, final_weights)

        result.selected_codes = list(final_weights.keys())
        result.applied_alpha = applied_alpha
        result.unconstrained_turnover = unconstrained_turnover
        result.constrained_turnover = constrained_turnover
        result.dropped_for_min_weight = dropped_for_min_weight
        result.dropped_for_turnover = dropped_for_turnover

        return final_weights, result

    result.notes.append("unable_to_select_positions")
    result.dropped_for_min_weight = dropped_for_min_weight
    result.dropped_for_turnover = dropped_for_turnover
    return {}, result
