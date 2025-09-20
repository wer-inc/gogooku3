from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import polars as pl


@dataclass(frozen=True)
class CSStats:
    """Cross-sectional statistics keyed by date (and optionally sector)."""

    stats: pl.DataFrame
    date_col: str
    key_cols: tuple[str, ...]


def fit_cs_stats(
    df_train: pl.DataFrame,
    cols: Iterable[str],
    *,
    date_col: str = "Date",
    by_cols: Iterable[str] | None = None,
) -> Dict[str, CSStats]:
    """Fit cross-sectional mean/std on the training split only.

    Parameters
    ----------
    df_train : pl.DataFrame
        Training slice only.
    cols : Iterable[str]
        Numeric columns to standardize.
    date_col : str
        Date column name.
    by_cols : Iterable[str] | None
        Additional keys (e.g., ["sector33_id"]) for in-sector standardization.
    """
    keys = [date_col, *(by_cols or [])]
    out: Dict[str, CSStats] = {}
    for c in cols:
        s = (
            df_train.group_by(keys)
            .agg(pl.col(c).mean().alias(f"{c}__mu"), pl.col(c).std().alias(f"{c}__sd"))
            .sort(keys)
        )
        out[c] = CSStats(stats=s, date_col=date_col, key_cols=tuple(keys))
    return out


def transform_cs(
    df: pl.DataFrame, stats_map: Mapping[str, CSStats], cols: Iterable[str]
) -> pl.DataFrame:
    """Apply cross-sectional standardization using pre-fit statistics.

    Produces new columns suffixed with `_cs_z`. Missing stats rows result in
    nulls; callers may choose to fill later, but we keep nulls to avoid
    accidental leakage via forward filling.
    """
    out = df
    for c in cols:
        st = stats_map[c]
        out = (
            out.join(st.stats, on=list(st.key_cols), how="left")
            .with_columns(
                ((pl.col(c) - pl.col(f"{c}__mu")) / (pl.col(f"{c}__sd") + 1e-12)).alias(
                    f"{c}_cs_z"
                )
            )
            .drop([f"{c}__mu", f"{c}__sd"])
        )
    return out


__all__ = ["fit_cs_stats", "transform_cs", "CSStats"]

