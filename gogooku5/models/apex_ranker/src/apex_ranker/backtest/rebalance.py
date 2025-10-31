"""Helpers for controlling portfolio rebalance frequency."""
from __future__ import annotations

from datetime import date

# Supported frequency labels
_VALID_FREQUENCIES = {"daily", "weekly", "monthly"}


def normalise_frequency(freq: str) -> str:
    """
    Normalise and validate rebalance frequency string.

    Args:
        freq: Frequency string provided by user

    Returns:
        Lower-case frequency label

    Raises:
        ValueError: If the frequency is unsupported
    """
    normalised = freq.lower()
    if normalised not in _VALID_FREQUENCIES:
        options = ", ".join(sorted(_VALID_FREQUENCIES))
        raise ValueError(
            f"Unsupported rebalance frequency '{freq}'. Options: {options}"
        )
    return normalised


def should_rebalance(
    current_date: date,
    last_rebalance: date | None,
    freq: str,
) -> bool:
    """
    Determine whether a portfolio rebalance should occur on ``current_date``.

    Rules:
        - ``daily``: rebalance every trading day
        - ``weekly``: rebalance on Fridays (first call always rebalances)
        - ``monthly``: rebalance on the first trading day of a new month

    Args:
        current_date: Date under consideration
        last_rebalance: Date of the previous rebalance (``None`` if none)
        freq: Desired rebalance frequency

    Returns:
        True if a rebalance should be executed
    """
    mode = normalise_frequency(freq)

    # Always rebalance if we have not invested yet
    if last_rebalance is None:
        return True

    if mode == "daily":
        return True

    if mode == "weekly":
        return current_date.weekday() == 4  # Friday

    # Monthly: first trading day of a new month
    if mode == "monthly":
        return (
            current_date.year != last_rebalance.year
            or current_date.month != last_rebalance.month
        )

    return False
