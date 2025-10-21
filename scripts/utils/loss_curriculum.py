#!/usr/bin/env python3
"""
Loss Schedule Curriculum for ATFT-GAT-FAN Training

Implements dynamic loss weight scheduling that transitions from balanced
multi-objective optimization to pure Sharpe ratio optimization.

Based on experiment_design.md (Experiment 1.2) from training_20251021_012605.

Usage:
    from scripts.utils.loss_curriculum import LossCurriculum

    curriculum = LossCurriculum(strategy='progressive_sharpe')
    weights = curriculum.get_weights(current_phase='Phase 2', epoch_in_phase=3)
    # {'sharpe': 0.7, 'rankic': 0.2, 'cs_ic': 0.1}
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)


PhaseType = Literal["Phase 0", "Phase 1", "Phase 2", "Phase 3"]
StrategyType = Literal["progressive_sharpe", "aggressive_sharpe", "balanced"]


class LossCurriculum:
    """
    Dynamic loss weight scheduler for phase-based training.

    Strategies:
    -----------
    progressive_sharpe (RECOMMENDED):
        Phase 0-1: Balanced (Sharpe 0.5, RankIC 0.3, IC 0.2)
        Phase 2: Sharpe-focused (0.7, 0.2, 0.1)
        Phase 3: Sharpe-only (1.0, 0.0, 0.0)

    aggressive_sharpe:
        Phase 0: Balanced (0.5, 0.3, 0.2)
        Phase 1: Sharpe-leaning (0.7, 0.2, 0.1)
        Phase 2: Sharpe-dominant (0.9, 0.05, 0.05)
        Phase 3: Sharpe-only (1.0, 0.0, 0.0)

    balanced:
        All phases: (0.6, 0.25, 0.15) - current best baseline
    """

    STRATEGIES = {
        "progressive_sharpe": {
            "Phase 0": {"sharpe": 0.5, "rankic": 0.3, "cs_ic": 0.2},
            "Phase 1": {"sharpe": 0.5, "rankic": 0.3, "cs_ic": 0.2},
            "Phase 2": {"sharpe": 0.7, "rankic": 0.2, "cs_ic": 0.1},
            "Phase 3": {"sharpe": 1.0, "rankic": 0.0, "cs_ic": 0.0},
        },
        "aggressive_sharpe": {
            "Phase 0": {"sharpe": 0.5, "rankic": 0.3, "cs_ic": 0.2},
            "Phase 1": {"sharpe": 0.7, "rankic": 0.2, "cs_ic": 0.1},
            "Phase 2": {"sharpe": 0.9, "rankic": 0.05, "cs_ic": 0.05},
            "Phase 3": {"sharpe": 1.0, "rankic": 0.0, "cs_ic": 0.0},
        },
        "balanced": {
            "Phase 0": {"sharpe": 0.6, "rankic": 0.25, "cs_ic": 0.15},
            "Phase 1": {"sharpe": 0.6, "rankic": 0.25, "cs_ic": 0.15},
            "Phase 2": {"sharpe": 0.6, "rankic": 0.25, "cs_ic": 0.15},
            "Phase 3": {"sharpe": 0.6, "rankic": 0.25, "cs_ic": 0.15},
        },
    }

    def __init__(self, strategy: StrategyType = "progressive_sharpe"):
        """
        Initialize loss curriculum.

        Args:
            strategy: One of 'progressive_sharpe', 'aggressive_sharpe', 'balanced'
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Must be one of: {list(self.STRATEGIES.keys())}"
            )

        self.strategy = strategy
        self.schedule = self.STRATEGIES[strategy]
        logger.info(f"[LossCurriculum] Initialized with strategy: {strategy}")

    def get_weights(
        self,
        current_phase: PhaseType,
        epoch_in_phase: int = 0,
    ) -> dict[str, float]:
        """
        Get loss weights for the current training phase and epoch.

        Args:
            current_phase: Current training phase ('Phase 0', 'Phase 1', etc.)
            epoch_in_phase: Epoch number within the current phase (unused for now)

        Returns:
            Dictionary with keys 'sharpe', 'rankic', 'cs_ic' and their weights

        Example:
            >>> curriculum = LossCurriculum('progressive_sharpe')
            >>> weights = curriculum.get_weights('Phase 2', epoch_in_phase=3)
            >>> weights
            {'sharpe': 0.7, 'rankic': 0.2, 'cs_ic': 0.1}
        """
        if current_phase not in self.schedule:
            raise ValueError(
                f"Unknown phase '{current_phase}'. "
                f"Must be one of: {list(self.schedule.keys())}"
            )

        weights = self.schedule[current_phase]

        # Validate weights sum to 1.0
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):
            logger.warning(
                f"[LossCurriculum] Weights don't sum to 1.0: {weights} (sum={total:.4f})"
            )

        logger.debug(
            f"[LossCurriculum] {current_phase} Epoch {epoch_in_phase}: {weights}"
        )

        return weights

    def get_env_string(self, current_phase: PhaseType) -> str:
        """
        Get loss weights as environment variable string.

        Useful for compatibility with existing training scripts that
        expect PHASE_LOSS_WEIGHTS environment variable.

        Args:
            current_phase: Current training phase

        Returns:
            String in format "sharpe=0.7,rankic=0.2,cs_ic=0.1"

        Example:
            >>> curriculum = LossCurriculum('progressive_sharpe')
            >>> curriculum.get_env_string('Phase 2')
            'sharpe=0.7,rankic=0.2,cs_ic=0.1'
        """
        weights = self.get_weights(current_phase)
        parts = [f"{k}={v}" for k, v in weights.items()]
        return ",".join(parts)

    def print_schedule(self) -> None:
        """Print the complete loss schedule for all phases."""
        print("=" * 80)
        print(f"Loss Curriculum Schedule: {self.strategy}")
        print("=" * 80)

        for phase in ["Phase 0", "Phase 1", "Phase 2", "Phase 3"]:
            weights = self.schedule[phase]
            sharpe = weights["sharpe"]
            rankic = weights["rankic"]
            cs_ic = weights["cs_ic"]

            # Visual bar representation
            bar_sharpe = "█" * int(sharpe * 40)
            bar_rankic = "▓" * int(rankic * 40)
            bar_cs_ic = "▒" * int(cs_ic * 40)

            print(f"\n{phase}:")
            print(f"  Sharpe:  {sharpe:.2f} {bar_sharpe}")
            print(f"  RankIC:  {rankic:.2f} {bar_rankic}")
            print(f"  CS-IC:   {cs_ic:.2f} {bar_cs_ic}")

        print("\n" + "=" * 80)


def apply_curriculum_to_phase_training(
    current_phase: PhaseType,
    strategy: StrategyType = "progressive_sharpe",
) -> dict[str, float]:
    """
    Convenience function for use in phase-based training loops.

    Args:
        current_phase: Current training phase
        strategy: Loss curriculum strategy

    Returns:
        Loss weights dictionary

    Example:
        In train_atft.py:

        for phase in ['Phase 0', 'Phase 1', 'Phase 2', 'Phase 3']:
            weights = apply_curriculum_to_phase_training(phase)
            # Update loss function with new weights
            loss_fn.update_weights(**weights)
    """
    curriculum = LossCurriculum(strategy=strategy)
    return curriculum.get_weights(current_phase)


# ============================================================================
# CLI Interface (for testing and visualization)
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Loss Curriculum Scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show progressive_sharpe schedule
  python scripts/utils/loss_curriculum.py --strategy progressive_sharpe

  # Show aggressive_sharpe schedule
  python scripts/utils/loss_curriculum.py --strategy aggressive_sharpe

  # Get weights for specific phase
  python scripts/utils/loss_curriculum.py --phase "Phase 2" --strategy progressive_sharpe
        """,
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="progressive_sharpe",
        choices=["progressive_sharpe", "aggressive_sharpe", "balanced"],
        help="Loss curriculum strategy",
    )

    parser.add_argument(
        "--phase",
        type=str,
        default=None,
        choices=["Phase 0", "Phase 1", "Phase 2", "Phase 3"],
        help="Show weights for specific phase only",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="table",
        choices=["table", "env"],
        help="Output format (table or environment variable string)",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    curriculum = LossCurriculum(strategy=args.strategy)

    if args.phase:
        # Show specific phase
        weights = curriculum.get_weights(args.phase)

        if args.format == "table":
            print(f"\n{args.phase} ({args.strategy}):")
            print(f"  Sharpe Weight:  {weights['sharpe']:.2f}")
            print(f"  RankIC Weight:  {weights['rankic']:.2f}")
            print(f"  CS-IC Weight:   {weights['cs_ic']:.2f}")
        elif args.format == "env":
            env_str = curriculum.get_env_string(args.phase)
            print(env_str)
    else:
        # Show full schedule
        if args.format == "table":
            curriculum.print_schedule()
        elif args.format == "env":
            print("# Environment variable format for each phase:")
            for phase in ["Phase 0", "Phase 1", "Phase 2", "Phase 3"]:
                env_str = curriculum.get_env_string(phase)  # type: ignore[arg-type]
                print(f"{phase}: {env_str}")
