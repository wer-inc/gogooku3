from __future__ import annotations

from typing import Iterable, List

import polars as pl


class AuditError(Exception):
    pass


def audit_stmt_nonnegative(df: pl.DataFrame, col: str = "stmt_days_since_statement") -> None:
    if col in df.columns:
        neg = df.filter(pl.col(col) < 0)
        if neg.height > 0:
            raise AuditError(f"{col} contains negative values (n={neg.height}) – as-of violation")


def audit_loo_present(df: pl.DataFrame, col: str = "sec_ret_1d_eq_loo") -> None:
    if "returns_1d" in df.columns and col in df.columns:
        # spot-check that not identical to self-return on dates with >1 name in sector
        same = df.filter(pl.col(col) == pl.col("returns_1d"))
        if same.height > 0:
            # allow potential degenerate sectors; keep as warning via exception to be handled by caller if needed
            raise AuditError("LOO column equals self return for some rows – check sector group sizes")


def run_basic_audits(df: pl.DataFrame) -> List[str]:
    """Run lightweight, non-destructive audits; return list of warnings.

    Audits raise only for critical as-of violations; less strict checks return warnings.
    """
    notes: List[str] = []
    try:
        audit_stmt_nonnegative(df)
    except AuditError as e:
        raise
    try:
        audit_loo_present(df)
    except AuditError as e:
        notes.append(str(e))
    return notes


__all__ = ["run_basic_audits", "audit_stmt_nonnegative", "audit_loo_present", "AuditError"]

