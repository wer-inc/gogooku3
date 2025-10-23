from __future__ import annotations

import polars as pl


def _hinge_pos(col: str) -> pl.Expr:
    return pl.col(col).clip(lower_bound=0.0)


def _hinge_neg(col: str) -> pl.Expr:
    return (-pl.col(col)).clip(lower_bound=0.0)


def add_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """Add a compact set of engineered interaction features.

    The function assumes the referenced base columns exist; if not, Polars
    will raise at evaluation time. Keep this behavior to surface data issues
    early in development and CI.
    """
    cols = set(df.columns)
    exprs: list[pl.Expr] = []

    if {"ma_gap_5_20", "mkt_gap_5_20"}.issubset(cols):
        exprs.append(
            (pl.col("ma_gap_5_20") * pl.col("mkt_gap_5_20")).alias("x_trend_intensity")
        )
    if {"rel_to_sec_5d", "sec_mom_20"}.issubset(cols):
        exprs.append(
            (pl.col("rel_to_sec_5d") * pl.col("sec_mom_20")).alias("x_rel_sec_mom")
        )
    if {"returns_5d", "volatility_20d"}.issubset(cols):
        exprs.append(
            (pl.col("returns_5d") / (pl.col("volatility_20d") + 1e-12)).alias(
                "x_mom_sh_5"
            )
        )
    if {"volume_ratio_5", "returns_1d"}.issubset(cols):
        exprs.append(
            (pl.col("volume_ratio_5") * pl.col("returns_1d").sign()).alias(
                "x_rvol5_dir"
            )
        )
    if {"dmi_short_to_adv20", "rel_strength_5d"}.issubset(cols):
        exprs.append(
            (pl.col("dmi_short_to_adv20") * _hinge_pos("rel_strength_5d")).alias(
                "x_squeeze_pressure"
            )
        )
    if {"dmi_credit_ratio", "z_close_20", "Code"}.issubset(cols):
        exprs.append(
            (
                (pl.col("dmi_credit_ratio") - 1.0)
                .rolling_mean(26)
                .over("Code")
                .fill_null(0)
                * _hinge_neg("z_close_20")
            ).alias("x_credit_rev_bias")
        )
    if {"stmt_rev_fore_op", "stmt_progress_op", "stmt_days_since_statement"}.issubset(
        cols
    ):
        exprs.append(
            (
                (
                    pl.col("stmt_rev_fore_op").fill_null(0)
                    + pl.col("stmt_progress_op").fill_null(0)
                )
                * ((-pl.col("stmt_days_since_statement") / 5.0).exp())
            ).alias("x_pead_effect")
        )
    if {"mkt_high_vol", "z_close_20"}.issubset(cols):
        exprs.append(
            (pl.col("mkt_high_vol").cast(pl.Float64) * _hinge_neg("z_close_20")).alias(
                "x_rev_gate"
            )
        )
    if {"alpha_1d", "beta_stability_60d"}.issubset(cols):
        exprs.append(
            ((-pl.col("alpha_1d")) * pl.col("beta_stability_60d")).alias(
                "x_alpha_meanrev_stable"
            )
        )
    if {"flow_smart_idx", "rel_strength_5d"}.issubset(cols):
        exprs.append(
            (pl.col("flow_smart_idx") * pl.col("rel_strength_5d")).alias(
                "x_flow_smart_rel"
            )
        )

    if not exprs:
        return df
    return df.with_columns(exprs)


__all__ = ["add_interactions"]
