"""
Phase 2 Patch B: Forward return labels generation.

IMPORTANT: These are LABELS for training, NOT features.
They contain future information and must NEVER be used as model inputs.
"""

from __future__ import annotations

import polars as pl

from ..utils import configure_logger

LOGGER = configure_logger("builder.targets.labels")


def generate_forward_returns(df: pl.DataFrame, horizons: list[int] | None = None) -> pl.DataFrame:
    """
    Generate forward returns (labels) for multi-horizon prediction.

    Args:
        df: DataFrame with columns ['code', 'date', 'close', 'adjustmentclose']
        horizons: List of forward horizons in days (default: [1, 5, 10, 20])

    Returns:
        DataFrame with columns: ['code', 'date', 'ret_fwd_1d', 'ret_fwd_5d', ...]

    IMPORTANT: The output contains FUTURE information and must be used ONLY as labels,
    never as features during training.

    Example:
        >>> labels = generate_forward_returns(prices_df, horizons=[1, 5, 10, 20])
        >>> # Save separately from features!
        >>> labels.write_parquet("labels.parquet")
    """
    if df.is_empty():
        LOGGER.warning("Empty DataFrame provided to generate_forward_returns")
        return df.select(["code", "date"]) if "code" in df.columns and "date" in df.columns else df

    if "close" not in df.columns:
        LOGGER.error("Missing 'close' column - cannot generate forward returns")
        return df.select(["code", "date"]) if "code" in df.columns and "date" in df.columns else df

    if horizons is None:
        horizons = [1, 5, 10, 20]

    # Use adjusted close if available
    if "adjustmentclose" in df.columns:
        base_price = pl.col("adjustmentclose").fill_null(pl.col("close"))
    else:
        base_price = pl.col("close")

    # Generate forward returns (shift NEGATIVE = look ahead)
    exprs = []
    for horizon in horizons:
        # shift(-horizon) = look forward (LABELS ONLY!)
        future = base_price.shift(-horizon).over("code")
        col_name = f"ret_fwd_{horizon}d"
        exprs.append(((future / (base_price + 1e-12)) - 1.0).alias(col_name))

    # Sort by code and date for proper shift operation
    labels_df = df.select(["code", "date"]).sort(["code", "date"]).with_columns(exprs)

    # Log statistics
    for horizon in horizons:
        col_name = f"ret_fwd_{horizon}d"
        if col_name in labels_df.columns:
            non_null = labels_df[col_name].is_not_null().sum()
            total = labels_df.height
            LOGGER.info(
                "[LABELS] %s: %d/%d non-null (%.1f%%)",
                col_name,
                non_null,
                total,
                100.0 * non_null / total if total > 0 else 0.0,
            )

    return labels_df
