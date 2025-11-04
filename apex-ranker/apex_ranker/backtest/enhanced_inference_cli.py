"""Enhanced Inference CLI - Unified argument registration to avoid conflicts.

This module provides a single source of truth for all enhanced inference CLI
arguments (A.3, A.4, etc.) to prevent argparse conflicts when multiple scripts
or modules try to register the same flags.

Usage:
    from apex_ranker.backtest.enhanced_inference_cli import register_ei_cli

    parser = argparse.ArgumentParser()
    register_ei_cli(parser)  # Call once, adds all EI arguments
    args = parser.parse_args()
"""
import argparse
from typing import Any


def _already_registered(parser: argparse.ArgumentParser, flag: str) -> bool:
    """Check if a flag is already registered in the parser."""
    for action in parser._actions:
        if flag in action.option_strings:
            return True
    return False


def register_ei_cli(parser: argparse.ArgumentParser) -> None:
    """Register all Enhanced Inference CLI arguments (once).

    Prevents duplicate registration by checking if flags already exist.
    Supports multiple aliases for the same argument (e.g., --use-enhanced-inference
    and --ei-neutralize-risk both map to the same dest).

    Args:
        parser: argparse.ArgumentParser to add arguments to
    """
    # Prevent double registration
    if getattr(parser, "_ei_cli_registered", False):
        return

    # A.3: Hysteresis selection
    if not _already_registered(parser, "--enable-a3"):
        parser.add_argument(
            "--enable-a3",
            dest="enable_a3",
            action="store_true",
            default=False,
            help="Enable A.3 hysteresis logic (different entry/exit thresholds)",
        )

    if not _already_registered(parser, "--ei-hysteresis-entry-k"):
        parser.add_argument(
            "--ei-hysteresis-entry-k",
            type=int,
            default=35,
            help="A.3: Hysteresis entry threshold (default: 35)",
        )

    if not _already_registered(parser, "--ei-hysteresis-exit-k"):
        parser.add_argument(
            "--ei-hysteresis-exit-k",
            type=int,
            default=60,
            help="A.3: Hysteresis exit threshold (default: 60, must be >= entry_k)",
        )

    # A.4: Risk neutralization - Main switch (with aliases)
    if not (
        _already_registered(parser, "--enable-neutralize")
        or _already_registered(parser, "--use-enhanced-inference")
        or _already_registered(parser, "--ei-neutralize-risk")
    ):
        parser.add_argument(
            "--enable-neutralize",
            "--use-enhanced-inference",
            "--ei-neutralize-risk",
            dest="enable_neutralize",
            action="store_true",
            default=False,
            help="Enable A.4 risk neutralization. Aliases: --use-enhanced-inference, --ei-neutralize-risk",
        )

    # A.4: Risk factors (with aliases)
    if not (
        _already_registered(parser, "--ei-factors")
        or _already_registered(parser, "--ei-risk-factors")
    ):
        parser.add_argument(
            "--ei-factors",
            "--ei-risk-factors",
            dest="ei_factors",
            nargs="+",
            default=None,
            help="A.4: Risk factors for neutralization (space or comma-separated). Example: Sector33Code volatility_60d",
        )

    # A.4: Gamma (partial neutralization coefficient)
    if not (
        _already_registered(parser, "--ei-neutralize-gamma")
        or _already_registered(parser, "--neutralize-gamma")
    ):
        parser.add_argument(
            "--ei-neutralize-gamma",
            "--neutralize-gamma",
            dest="ei_neutralize_gamma",
            type=float,
            default=0.3,
            help="A.4: Partial neutralization factor (default: 0.3, range: [0.2, 0.5]). 0.0=none, 1.0=full",
        )

    # A.4: Ridge alpha (regularization)
    if not (
        _already_registered(parser, "--ei-ridge-alpha")
        or _already_registered(parser, "--ridge-alpha")
    ):
        parser.add_argument(
            "--ei-ridge-alpha",
            "--ridge-alpha",
            dest="ei_ridge_alpha",
            type=float,
            default=10.0,
            help="A.4: Ridge regression regularization (default: 10.0, higher=more conservative)",
        )

    # A.4: Re-scale flag
    if not (
        _already_registered(parser, "--ei-rescale")
        or _already_registered(parser, "--neutralize-rescale")
    ):
        parser.add_argument(
            "--ei-rescale",
            "--neutralize-rescale",
            dest="ei_rescale",
            action="store_true",
            default=True,
            help="A.4: Re-scale residuals to original std (default: True, recommended)",
        )

    # A.4: Clip multiplier (safety guard)
    if not (
        _already_registered(parser, "--ei-clip-mult")
        or _already_registered(parser, "--neutralize-clip-mult")
    ):
        parser.add_argument(
            "--ei-clip-mult",
            "--neutralize-clip-mult",
            dest="ei_clip_mult",
            type=float,
            default=0.25,
            help="A.4: Max correction as fraction of std (default: 0.25, clips extreme adjustments)",
        )

    # Supply guard: Minimum selection count
    if not _already_registered(parser, "--k-min"):
        parser.add_argument(
            "--k-min",
            dest="k_min",
            type=int,
            default=53,
            help="Minimum selected count (supply guard). Fallback fills to this if needed.",
        )

    # Mark as registered
    parser._ei_cli_registered = True


def parse_ei_factors(factors_arg: Any) -> list[str] | None:
    """Parse risk factors from CLI argument (space or comma-separated).

    Args:
        factors_arg: Can be:
            - None: Return None
            - str: Split by comma/space
            - list[str]: Join then split (handles both "A B" and "A,B")

    Returns:
        List of factor names, or None if no factors

    Examples:
        >>> parse_ei_factors("Sector33Code,volatility_60d")
        ['Sector33Code', 'volatility_60d']

        >>> parse_ei_factors(["Sector33Code", "volatility_60d"])
        ['Sector33Code', 'volatility_60d']

        >>> parse_ei_factors("Sector33Code volatility_60d")
        ['Sector33Code', 'volatility_60d']
    """
    if factors_arg is None:
        return None

    import re

    if isinstance(factors_arg, str):
        text = factors_arg
    elif isinstance(factors_arg, list):
        text = " ".join(factors_arg)
    else:
        return None

    # Split by comma or whitespace
    tokens = re.split(r"[,\s]+", text.strip())
    factors = [t for t in tokens if t]

    return factors if factors else None


__all__ = [
    "register_ei_cli",
    "parse_ei_factors",
]
