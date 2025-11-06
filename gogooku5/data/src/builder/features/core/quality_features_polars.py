"""Polars-native quality feature generator migrated from gogooku3."""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Union

import polars as pl

from ..utils.rolling import roll_mean_safe, roll_std_safe

LOGGER = logging.getLogger(__name__)


class QualityFinancialFeaturesGeneratorPolars:
    """Generate quality-related features using Polars transformations."""

    def __init__(
        self,
        *,
        use_cross_sectional_quantiles: bool = True,
        sigma_threshold: float = 2.0,
        quantile_bins: int = 5,
        rolling_window: int = 20,
        date_column: str = "date",
        code_column: str = "code",
    ) -> None:
        self.use_cross_sectional_quantiles = use_cross_sectional_quantiles
        self.sigma_threshold = sigma_threshold
        self.quantile_bins = quantile_bins
        self.rolling_window = rolling_window
        self.date_column = date_column
        self.code_column = code_column

        self.numeric_features: List[str] = []
        self.generated_features: List[str] = []
        self._zscore_features: List[str] = []

    def generate_quality_features(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        *,
        target_column: Optional[str] = None,
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        is_lazy = isinstance(df, pl.LazyFrame)
        working_df: pl.LazyFrame

        if is_lazy:
            working_df = df
            schema = working_df.head(1).collect().schema
        else:
            working_df = df.lazy()
            schema = df.schema

        exclude_prefixes = (
            "fs_",
            "div_",
            "bd_",
            "dmi_",
            "wmi_",
            "gap_",
            "macro_",
            "fut_",
            "opt_",
        )
        self.numeric_features = [
            col
            for col, dtype in schema.items()
            if dtype in {pl.Float32, pl.Float64, pl.Int32, pl.Int64}
            and col not in {self.date_column, self.code_column}
            and not any(col.startswith(prefix) for prefix in exclude_prefixes)
        ]
        self._zscore_features = []

        if self.use_cross_sectional_quantiles and self.numeric_features:
            working_df = self._add_cross_sectional_quantiles(working_df)

        if self.numeric_features:
            working_df = self._add_rolling_statistics(working_df)
            working_df = self._add_volatility_indicators(working_df)
            working_df = self._add_outlier_flags(working_df)

        if self.numeric_features and "sector_code" in schema:
            working_df = self._add_peer_relative_features(working_df)

        if target_column and target_column in schema:
            working_df = self._add_market_regime_features(working_df, target_column)

        LOGGER.info("Generated %d quality features", len(self.generated_features))

        if is_lazy:
            return working_df
        return working_df.collect()

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    def _add_cross_sectional_quantiles(self, df: pl.LazyFrame) -> pl.LazyFrame:
        for feature in self._limit(self.numeric_features, 25):
            rank_col = f"{feature}_cs_rank"
            count_col = f"_{feature}_cs_count"
            pct_col = f"{feature}_cs_pct"
            top_flag = f"{feature}_cs_top20_flag"
            bottom_flag = f"{feature}_cs_bottom20_flag"

            df = df.with_columns(
                [
                    pl.col(feature).rank(method="ordinal").over(self.date_column).alias(rank_col),
                    pl.count().over(self.date_column).alias(count_col),
                ]
            )

            df = df.with_columns(
                pl.when(pl.col(count_col) > 1)
                .then((pl.col(rank_col) - 1.0) / (pl.col(count_col) - 1.0))
                .otherwise(0.5)
                .alias(pct_col)
            )

            df = df.with_columns(
                [
                    (pl.col(pct_col) >= 0.8).cast(pl.Int8).alias(top_flag),
                    (pl.col(pct_col) <= 0.2).cast(pl.Int8).alias(bottom_flag),
                ]
            )

            df = df.drop(count_col)

            self.generated_features.extend([rank_col, pct_col, top_flag, bottom_flag])

        return df

    def _add_rolling_statistics(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Phase 2 Patch C: All rolling operations exclude current day (left-closed).

        Uses safe rolling utilities to prevent look-ahead bias.
        """
        window = self.rolling_window
        min_p = max(window // 2, 5)  # Require at least half window or 5 observations

        for feature in self.numeric_features:
            mean_col = f"{feature}_roll_mean_{window}d"
            std_col = f"{feature}_roll_std_{window}d"
            z_col = f"{feature}_zscore_{window}d"

            # Phase 2 Patch C: Use safe rolling (shift(1) before rolling)
            df = df.with_columns(
                roll_mean_safe(pl.col(feature), window, min_periods=min_p, by=self.code_column).alias(mean_col)
            )
            df = df.with_columns(
                roll_std_safe(pl.col(feature), window, min_periods=min_p, by=self.code_column).alias(std_col)
            )
            df = df.with_columns(((pl.col(feature) - pl.col(mean_col)) / (pl.col(std_col) + 1e-8)).alias(z_col))

            self.generated_features.extend([mean_col, std_col, z_col])
            self._zscore_features.append(feature)

        return df

    def _add_volatility_indicators(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Phase 2 Patch C: Volatility indicators exclude current day.

        Uses safe rolling utilities to prevent look-ahead bias.
        """
        # Exclude feat_ret_* columns (forward-looking) from z-score/volatility processing
        return_cols = [
            col
            for col in self.numeric_features
            if ("return" in col.lower() or "ret" in col.lower()) and not col.startswith("feat_ret_")
        ]

        for col in return_cols:
            vol_col = f"{col}_hist_vol"
            sharpe_col = f"{col}_rolling_sharpe"

            # Phase 2 Patch C: Use safe rolling (shift(1) before rolling)
            df = df.with_columns(roll_std_safe(pl.col(col), 20, min_periods=10, by=self.code_column).alias(vol_col))
            df = df.with_columns(
                (roll_mean_safe(pl.col(col), 20, min_periods=10, by=self.code_column) / (pl.col(vol_col) + 1e-8)).alias(
                    sharpe_col
                )
            )

            self.generated_features.extend([vol_col, sharpe_col])

        return df

    def _add_outlier_flags(self, df: pl.LazyFrame) -> pl.LazyFrame:
        sigma = self.sigma_threshold

        for feature in self._zscore_features:
            z_col = f"{feature}_zscore_{self.rolling_window}d"
            flag_col = f"{feature}_outlier_flag"

            df = df.with_columns((pl.col(z_col).abs() > sigma).cast(pl.Int8).alias(flag_col))
            self.generated_features.append(flag_col)

        return df

    def _add_market_regime_features(self, df: pl.LazyFrame, target_column: str) -> pl.LazyFrame:
        """
        Phase 2 Patch C: Market regime momentum excludes current day.

        Note: This rolls over date_column (market-wide), not code_column.
        """
        momentum_col = f"{target_column}_momentum_{self.rolling_window}d"
        window = self.rolling_window
        min_p = max(window // 2, 5)

        # Phase 2 Patch C: Use safe rolling over date_column
        df = df.with_columns(
            roll_mean_safe(pl.col(target_column), window, min_periods=min_p, by=self.date_column).alias(momentum_col)
        )
        self.generated_features.append(momentum_col)
        return df

    def _add_peer_relative_features(self, df: pl.LazyFrame) -> pl.LazyFrame:
        for feature in self.numeric_features:
            mean_col = f"{feature}_sector_mean"
            rel_col = f"{feature}_sector_rel"

            df = df.with_columns(pl.col(feature).mean().over([self.date_column, "sector_code"]).alias(mean_col))
            df = df.with_columns((pl.col(feature) - pl.col(mean_col)).alias(rel_col))

            self.generated_features.extend([mean_col, rel_col])

        return df

    @staticmethod
    def _limit(seq: Sequence[str], limit: int) -> Iterable[str]:
        return seq[:limit] if isinstance(seq, list) else list(seq)[:limit]
