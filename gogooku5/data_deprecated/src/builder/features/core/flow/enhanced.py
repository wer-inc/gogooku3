"""Enhanced investor flow features adapted for gogooku5."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

EPS = 1e-10
LOGGER = logging.getLogger(__name__)


def _ensure_date(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    if column not in frame.columns:
        return frame
    dtype = frame.schema.get(column)
    if dtype == pl.Date:
        return frame
    if dtype == pl.Datetime:
        return frame.with_columns(pl.col(column).dt.date().alias(column))
    return frame.with_columns(pl.col(column).cast(pl.Utf8).str.strptime(pl.Date, strict=False).alias(column))


def _safe_float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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
        available_dates = {
            d.isoformat() if hasattr(d, "isoformat") else str(d)
            for d in df_base.select(cfg.date_column).drop_nulls().to_series().to_list()
        }

        if flows.is_empty():
            LOGGER.warning("Flow features skipped: trades_spec dataframe is empty for requested window")
            return self._add_null_columns(df_base)

        df_flows = self._prepare_flows(flows)
        if df_flows.is_empty():
            LOGGER.warning("Flow features skipped: trades_spec normalization produced empty dataframe")
            return self._add_null_columns(df_base)

        market_features = self._build_market_level_features(df_flows)
        if market_features.is_empty():
            LOGGER.warning("Flow features skipped: market-level aggregation returned empty dataframe")
            return self._add_null_columns(df_base)

        if available_dates:
            market_features = market_features.filter(
                pl.col(self.config.date_column).cast(pl.Utf8).is_in(list(available_dates))
            )

        # Join on both Code and date to preserve cross-sectional information
        # Fall back to date-only join if Code not present in flow features
        join_keys = (
            [self.config.code_column, self.config.date_column]
            if self.config.code_column in market_features.columns
            else self.config.date_column
        )
        merged = df_base.join(market_features, on=join_keys, how="left")
        merged = self._ensure_expected_columns(merged)
        return merged

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    def _prepare_flows(self, flows: pl.DataFrame) -> pl.DataFrame:
        rename_map = {
            "Code": "code",
            "PublishedDate": "published_date",
            "StartDate": "week_start",
            "EndDate": "week_end",
            "Section": "section",
        }
        present = {k: v for k, v in rename_map.items() if k in flows.columns}
        df = flows.rename(present) if present else flows

        ensure_exprs: list[pl.Expr] = []
        if "week_start" not in df.columns:
            ensure_exprs.append(pl.lit(None).cast(pl.Date).alias("week_start"))
        if "week_end" not in df.columns:
            ensure_exprs.append(pl.lit(None).cast(pl.Date).alias("week_end"))
        if "section" not in df.columns:
            ensure_exprs.append(pl.lit(None).cast(pl.Utf8).alias("section"))
        if ensure_exprs:
            df = df.with_columns(ensure_exprs)

        for column in ("week_start", "week_end"):
            if column in df.columns:
                df = _ensure_date(df, column)

        if "date" in df.columns:
            df = _ensure_date(df, "date")
        else:
            df = df.with_columns(pl.lit(None).cast(pl.Date).alias("date"))
        if "published_date" in df.columns:
            df = df.with_columns(
                pl.col("published_date")
                .cast(pl.Utf8, strict=False)
                .str.strptime(pl.Date, strict=False)
                .alias("published_date")
            )
            dedup_keys = [col for col in ("section", "week_start", "week_end", "date") if col in df.columns]
            df = df.sort(dedup_keys + ["published_date"]).unique(subset=dedup_keys, keep="last")
        df = df.with_columns(
            pl.when(pl.col("published_date").is_not_null())
            .then(pl.col("published_date"))
            .when(pl.col("week_end").is_not_null())
            .then(pl.col("week_end"))
            .otherwise(pl.col("date"))
            .alias("release_date")
        )
        df = _ensure_date(df, "release_date")
        df = df.filter(pl.col("release_date").is_not_null())
        return df

    # ------------------------------------------------------------------
    def _build_market_level_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aggregate trades_spec data to market-level features keyed by date."""

        if df.is_empty():
            return pl.DataFrame()

        # API field names corrected to match J-Quants API response
        # (all fields end with "Value" suffix, not just purchase/sales)
        candidate_columns: dict[str, tuple[list[str], list[str]]] = {
            "foreigners": (
                ["ForeignersPurchaseValue", "ForeignersPurchases"],
                ["ForeignersSalesValue", "ForeignersSales"],
            ),
            "individuals": (
                ["IndividualPurchaseValue", "IndividualsPurchases"],
                ["IndividualSalesValue", "IndividualsSales"],
            ),
            "investment_trusts": (
                ["InvestmentTrustsPurchaseValue", "InvestmentTrustsPurchases"],
                ["InvestmentTrustsSalesValue", "InvestmentTrustsSales"],
            ),
            "trust_banks": (
                ["TrustBanksPurchaseValue", "TrustBanksPurchases"],
                ["TrustBanksSalesValue", "TrustBanksSales"],
            ),
            "securities": (
                ["SecuritiesCompaniesPurchaseValue", "SecuritiesCosPurchases"],
                ["SecuritiesCompaniesSalesValue", "SecuritiesCosSales"],
            ),
            "proprietary": (
                ["ProprietaryPurchaseValue", "ProprietaryPurchases"],
                ["ProprietarySalesValue", "ProprietarySales"],
            ),
            "business": (
                ["BusinessCorporationsPurchaseValue", "BusinessCosPurchases"],
                ["BusinessCorporationsSalesValue", "BusinessCosSales"],
            ),
            "other_fin": (
                ["OtherFinancialInstitutionsPurchaseValue", "OtherFinancialInstitutionsPurchases"],
                ["OtherFinancialInstitutionsSalesValue", "OtherFinancialInstitutionsSales"],
            ),
        }

        resolved_columns: dict[str, tuple[str, str]] = {}
        missing_exprs: list[pl.Expr] = []

        for label, (purchase_candidates, sales_candidates) in candidate_columns.items():
            purchase_col = next((name for name in purchase_candidates if name in df.columns), None)
            sales_col = next((name for name in sales_candidates if name in df.columns), None)

            if purchase_col is None:
                purchase_col = purchase_candidates[0]
                missing_exprs.append(pl.lit(0.0).alias(purchase_col))
            if sales_col is None:
                sales_col = sales_candidates[0]
                missing_exprs.append(pl.lit(0.0).alias(sales_col))

            resolved_columns[label] = (purchase_col, sales_col)

        if missing_exprs:
            df = df.with_columns(missing_exprs)

        cast_exprs = []
        for purchase, sales in resolved_columns.values():
            cast_exprs.extend(
                [
                    pl.col(purchase).cast(pl.Float64).alias(purchase),
                    pl.col(sales).cast(pl.Float64).alias(sales),
                ]
            )
        df = df.with_columns(cast_exprs)

        # Preserve Code dimension to maintain cross-sectional information
        # Group by both Code and release_date instead of just release_date
        group_cols = ["Code", "release_date"] if "Code" in df.columns else ["release_date"]
        aggregated = df.group_by(group_cols).agg(
            [pl.col(purchase).sum().alias(purchase) for purchase, _ in resolved_columns.values()]
            + [pl.col(sales).sum().alias(sales) for _, sales in resolved_columns.values()]
        )

        if aggregated.is_empty():
            return pl.DataFrame()

        # Rename to match expected column names (lowercase)
        rename_map = {"release_date": "date"}
        if "Code" in aggregated.columns:
            rename_map["Code"] = "code"
        aggregated = aggregated.rename(rename_map)

        for label, (purchase, sales) in resolved_columns.items():
            aggregated = aggregated.rename(
                {
                    purchase: f"{label}_purchases",
                    sales: f"{label}_sales",
                }
            )

        for label in resolved_columns.keys():
            aggregated = aggregated.with_columns(
                (pl.col(f"{label}_purchases") - pl.col(f"{label}_sales")).alias(f"{label}_net")
            )

        institutional_purchases = _sum_expr(
            [pl.col(f"{label}_purchases") for label in resolved_columns.keys() if label != "individuals"],
            default=0.0,
        )
        institutional_sales = _sum_expr(
            [pl.col(f"{label}_sales") for label in resolved_columns.keys() if label != "individuals"],
            default=0.0,
        )

        aggregated = aggregated.with_columns(
            [
                (institutional_purchases - institutional_sales).alias("institutional_accumulation"),
                (
                    (pl.col("foreigners_net")) / (pl.col("foreigners_purchases") + pl.col("foreigners_sales") + EPS)
                ).alias("foreign_sentiment"),
                (
                    (pl.col("individuals_net")) / (pl.col("individuals_purchases") + pl.col("individuals_sales") + EPS)
                ).alias("_individuals_sentiment"),
            ]
        )

        aggregated = aggregated.with_columns(
            [
                (pl.col("_individuals_sentiment") * -1.0 * pl.col("institutional_accumulation").sign()).alias(
                    "retail_institutional_divergence"
                ),
                (pl.col("foreign_sentiment") - pl.col("_individuals_sentiment")).alias("foreign_domestic_divergence"),
            ]
        )

        flow_cols = [f"{label}_net" for label in resolved_columns.keys()]
        aggregated = aggregated.with_columns(
            _sum_expr([pl.col(col).abs() for col in flow_cols]).alias("_flow_total_abs")
        )
        concentration_terms = [(pl.col(col).abs() / (pl.col("_flow_total_abs") + EPS)) ** 2 for col in flow_cols]
        aggregated = aggregated.with_columns(_sum_expr(concentration_terms).alias("flow_concentration"))
        aggregated = aggregated.with_columns(
            pl.when(pl.col("flow_concentration") > 0.7)
            .then(pl.col("institutional_accumulation").sign())
            .otherwise(0.0)
            .alias("concentrated_flow_signal")
        ).drop("_flow_total_abs")

        aggregated = aggregated.sort("date").with_columns(
            [
                pl.col("institutional_accumulation")
                .sign()
                .rolling_mean(window_size=10, min_periods=1)
                .alias("institutional_persistence"),
                pl.col("foreign_sentiment")
                .sign()
                .rolling_mean(window_size=10, min_periods=1)
                .alias("foreign_persistence"),
            ]
        )
        aggregated = aggregated.with_columns(
            (pl.col("institutional_persistence") * pl.col("institutional_accumulation").sign()).alias(
                "smart_flow_indicator"
            )
        )

        return aggregated.select(
            [
                "date",
                pl.col("institutional_accumulation").cast(pl.Float64),
                pl.col("foreign_sentiment").cast(pl.Float64),
                pl.col("retail_institutional_divergence").cast(pl.Float64),
                pl.col("foreign_domestic_divergence").cast(pl.Float64),
                pl.col("institutional_persistence").cast(pl.Float64),
                pl.col("foreign_persistence").cast(pl.Float64),
                pl.col("smart_flow_indicator").cast(pl.Float64),
                pl.col("flow_concentration").cast(pl.Float64),
                pl.col("concentrated_flow_signal").cast(pl.Float64),
            ]
        )

    def _add_null_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        null_exprs = [
            pl.lit(None).cast(pl.Float64).alias("institutional_accumulation"),
            pl.lit(None).cast(pl.Float64).alias("foreign_sentiment"),
            pl.lit(None).cast(pl.Float64).alias("retail_institutional_divergence"),
            pl.lit(None).cast(pl.Float64).alias("foreign_domestic_divergence"),
            pl.lit(None).cast(pl.Float64).alias("institutional_persistence"),
            pl.lit(None).cast(pl.Float64).alias("foreign_persistence"),
            pl.lit(None).cast(pl.Float64).alias("smart_flow_indicator"),
            pl.lit(None).cast(pl.Float64).alias("flow_concentration"),
            pl.lit(None).cast(pl.Float64).alias("concentrated_flow_signal"),
        ]
        return df.with_columns(null_exprs)

    def _ensure_expected_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ensure all expected flow feature columns exist after join."""

        required = [
            "institutional_accumulation",
            "foreign_sentiment",
            "retail_institutional_divergence",
            "foreign_domestic_divergence",
            "institutional_persistence",
            "foreign_persistence",
            "smart_flow_indicator",
            "flow_concentration",
            "concentrated_flow_signal",
        ]
        missing = [col for col in required if col not in df.columns]
        if not missing:
            return df

        exprs = [pl.lit(None).cast(pl.Float32).alias(name) for name in missing]
        return df.with_columns(exprs)
