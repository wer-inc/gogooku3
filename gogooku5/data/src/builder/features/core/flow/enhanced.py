"""Enhanced investor flow features adapted for gogooku5."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl

EPS = 1e-10


def _ensure_date(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    if column not in frame.columns:
        return frame
    dtype = frame.schema.get(column)
    if dtype == pl.Date:
        return frame
    if dtype == pl.Datetime:
        return frame.with_columns(pl.col(column).dt.date().alias(column))
    return frame.with_columns(pl.col(column).cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias(column))


def _sum_expr(exprs: list[pl.Expr], *, default: float = 0.0) -> pl.Expr:
    if not exprs:
        return pl.lit(default)
    result = exprs[0]
    for expr in exprs[1:]:
        result = result + expr
    return result


@dataclass
class FlowFeatureConfig:
    code_column: str = "code"
    date_column: str = "date"


class FlowFeatureEngineer:
    """Generate advanced institutional flow metrics from trades_spec data."""

    def __init__(self, config: FlowFeatureConfig | None = None) -> None:
        self.config = config or FlowFeatureConfig()

    def add_features(self, base: pl.DataFrame, flows: pl.DataFrame) -> pl.DataFrame:
        """Attach enhanced flow metrics to the base dataset."""

        if base.is_empty():
            return base

        cfg = self.config
        df_base = _ensure_date(base, cfg.date_column)

        if flows.is_empty():
            return self._add_null_columns(df_base)

        df_flows = self._prepare_flows(flows)
        if df_flows.is_empty():
            return self._add_null_columns(df_base)

        merged = self._merge(base=df_base, flows=df_flows)
        merged = self._add_divergence(merged)
        merged = self._add_persistence(merged)
        merged = self._add_concentration(merged)

        drop_cols = [c for c in merged.columns if c.startswith("_flow_tmp_") or c.endswith("_flow_sign")]
        if drop_cols:
            merged = merged.drop(drop_cols)
        return merged

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    def _prepare_flows(self, flows: pl.DataFrame) -> pl.DataFrame:
        rename_map = {
            "Code": "code",
            "PublishedDate": "date",
            "StartDate": "week_start",
            "EndDate": "week_end",
            "Section": "section",
        }
        present = {k: v for k, v in rename_map.items() if k in flows.columns}
        df = flows.rename(present) if present else flows
        df = _ensure_date(df, "date")

        investor_types = self._identify_investors(df)
        if not investor_types:
            return df

        exprs: list[pl.Expr] = []
        for inv in investor_types:
            buy = f"{inv}PurchaseValue"
            sell = f"{inv}SalesValue"
            if buy in df.columns and sell in df.columns:
                exprs.append((pl.col(buy) - pl.col(sell)).alias(f"{inv}_net_flow"))
                exprs.append(
                    ((pl.col(buy) - pl.col(sell)) / (pl.col(buy) + pl.col(sell) + EPS)).alias(f"{inv}_flow_ratio")
                )
        if exprs:
            df = df.with_columns(exprs)
        return df

    def _identify_investors(self, df: pl.DataFrame) -> list[str]:
        candidates = [
            "Proprietary",
            "Investment",
            "Business",
            "Individual",
            "Foreigners",
            "Securities",
            "Other",
        ]
        return [inv for inv in candidates if f"{inv}PurchaseValue" in df.columns]

    def _merge(self, base: pl.DataFrame, flows: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        merged = base.join(flows, on=[cfg.code_column, cfg.date_column], how="left")
        merged = merged.sort([cfg.code_column, cfg.date_column])

        investor_types = self._identify_investors(flows)
        fill_cols: list[str] = []
        for inv in investor_types:
            fill_cols.extend([f"{inv}_net_flow", f"{inv}_flow_ratio"])

        for col in fill_cols:
            if col in merged.columns:
                merged = merged.with_columns(pl.col(col).forward_fill().over(cfg.code_column))

        institutional_cols = [
            f"{t}_net_flow" for t in investor_types if t != "Individual" and f"{t}_net_flow" in merged.columns
        ]
        if institutional_cols:
            merged = merged.with_columns(
                _sum_expr([pl.col(c) for c in institutional_cols]).alias("institutional_accumulation")
            )

        if "Foreigners_flow_ratio" in merged.columns:
            merged = merged.with_columns(pl.col("Foreigners_flow_ratio").alias("foreign_sentiment"))

        return merged

    # ------------------------------------------------------------------
    # Feature blocks
    # ------------------------------------------------------------------
    def _add_divergence(self, df: pl.DataFrame) -> pl.DataFrame:
        if {"Individual_flow_ratio", "institutional_accumulation"}.issubset(df.columns):
            df = df.with_columns(
                (pl.col("Individual_flow_ratio") * -1.0 * pl.col("institutional_accumulation").sign()).alias(
                    "retail_institutional_divergence"
                )
            )

        if "Foreigners_flow_ratio" in df.columns:
            dom_cols = [
                col
                for col in ["Individual_flow_ratio", "Business_flow_ratio", "Investment_flow_ratio"]
                if col in df.columns
            ]
            if dom_cols:
                df = df.with_columns(_sum_expr([pl.col(c) for c in dom_cols]).alias("_flow_tmp_domestic"))
                df = df.with_columns(
                    (pl.col("Foreigners_flow_ratio") - pl.col("_flow_tmp_domestic")).alias(
                        "foreign_domestic_divergence"
                    )
                )
        return df

    def _add_persistence(self, df: pl.DataFrame) -> pl.DataFrame:
        if "institutional_accumulation" in df.columns:
            df = df.with_columns(pl.col("institutional_accumulation").sign().alias("_flow_inst_sign"))
            df = df.with_columns(
                pl.col("_flow_inst_sign")
                .rolling_mean(window_size=10)
                .over(self.config.code_column)
                .abs()
                .alias("institutional_persistence")
            )

        if "foreign_sentiment" in df.columns:
            df = df.with_columns(pl.col("foreign_sentiment").sign().alias("_flow_foreign_sign"))
            df = df.with_columns(
                pl.col("_flow_foreign_sign")
                .rolling_mean(window_size=10)
                .over(self.config.code_column)
                .abs()
                .alias("foreign_persistence")
            )

        if {"institutional_persistence", "institutional_accumulation"}.issubset(df.columns):
            df = df.with_columns(
                (pl.col("institutional_persistence") * pl.col("institutional_accumulation").sign()).alias(
                    "smart_flow_indicator"
                )
            )
        return df

    def _add_concentration(self, df: pl.DataFrame) -> pl.DataFrame:
        flow_cols = [c for c in df.columns if c.endswith("_net_flow")]
        if len(flow_cols) < 2:
            return df

        df = df.with_columns(_sum_expr([pl.col(c).abs() for c in flow_cols]).alias("_flow_tmp_total_abs"))

        concentration_terms = [(pl.col(c).abs() / (pl.col("_flow_tmp_total_abs") + EPS)) ** 2 for c in flow_cols]
        df = df.with_columns(_sum_expr(concentration_terms).alias("flow_concentration"))

        if "institutional_accumulation" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("flow_concentration") > 0.7)
                .then(pl.col("institutional_accumulation").sign())
                .otherwise(0)
                .alias("concentrated_flow_signal")
            )
        return df

    # ------------------------------------------------------------------
    def _add_null_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        null_exprs = [
            pl.lit(None).cast(pl.Float32).alias("institutional_accumulation"),
            pl.lit(None).cast(pl.Float32).alias("foreign_sentiment"),
            pl.lit(None).cast(pl.Float32).alias("retail_institutional_divergence"),
            pl.lit(None).cast(pl.Float32).alias("foreign_domestic_divergence"),
            pl.lit(None).cast(pl.Float32).alias("institutional_persistence"),
            pl.lit(None).cast(pl.Float32).alias("foreign_persistence"),
            pl.lit(None).cast(pl.Float32).alias("smart_flow_indicator"),
            pl.lit(None).cast(pl.Float32).alias("flow_concentration"),
            pl.lit(None).cast(pl.Float32).alias("concentrated_flow_signal"),
        ]
        return df.with_columns(null_exprs)
