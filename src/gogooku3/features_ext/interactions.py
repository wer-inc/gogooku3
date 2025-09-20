from __future__ import annotations

import polars as pl


def _hinge_pos(col: str) -> pl.Expr:
    return pl.col(col).clip_min(0.0)


def _hinge_neg(col: str) -> pl.Expr:
    return (-pl.col(col)).clip_min(0.0)


def add_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """Add a compact set of engineered interaction features.

    The function assumes the referenced base columns exist; if not, Polars
    will raise at evaluation time. Keep this behavior to surface data issues
    early in development and CI.
    """
    return df.with_columns(
        [
            (pl.col("ma_gap_5_20") * pl.col("mkt_gap_5_20")).alias("x_trend_intensity"),
            (pl.col("rel_to_sec_5d") * pl.col("sec_mom_20")).alias("x_rel_sec_mom"),
            (pl.col("returns_5d") / (pl.col("volatility_20d") + 1e-12)).alias("x_mom_sh_5"),
            (pl.col("volume_ratio_5") * pl.col("returns_1d").sign()).alias("x_rvol5_dir"),
            (pl.col("dmi_short_to_adv20") * _hinge_pos("rel_strength_5d")).alias("x_squeeze_pressure"),
            (
                (pl.col("dmi_credit_ratio") - 1.0)
                .rolling_mean(26)
                .over("Code")
                .fill_null(0)
                * _hinge_neg("z_close_20")
            ).alias("x_credit_rev_bias"),
            (
                (pl.col("stmt_rev_fore_op").fill_null(0) + pl.col("stmt_progress_op").fill_null(0))
                * ((-pl.col("stmt_days_since_statement") / 5.0).exp())
            ).alias("x_pead_effect"),
            (pl.col("mkt_high_vol").cast(pl.Float64) * _hinge_neg("z_close_20")).alias("x_rev_gate"),
            ((-pl.col("alpha_1d")) * pl.col("beta_stability_60d")).alias("x_alpha_meanrev_stable"),
            (pl.col("flow_smart_idx") * pl.col("rel_strength_5d")).alias("x_flow_smart_rel"),
        ]
    )


__all__ = ["add_interactions"]

