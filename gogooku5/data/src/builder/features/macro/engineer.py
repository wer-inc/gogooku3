"""Combine macro feature tables with the dataset."""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class MacroFeatureEngineer:
    date_column: str = "date"

    def add_vix(self, df: pl.DataFrame, vix_features: pl.DataFrame) -> pl.DataFrame:
        if vix_features.is_empty() or self.date_column not in df.columns:
            return df

        merge_df = vix_features
        if "Date" in merge_df.columns and self.date_column != "Date":
            merge_df = merge_df.rename({"Date": self.date_column})
        if self.date_column not in merge_df.columns:
            return df

        dtype = merge_df.schema.get(self.date_column)
        if dtype != pl.Date:
            merge_df = merge_df.with_columns(
                pl.col(self.date_column)
                .cast(pl.Utf8, strict=False)
                .str.strptime(pl.Date, strict=False)
                .alias(self.date_column)
            )

        return df.join(merge_df, on=self.date_column, how="left")

    def add_global_regime(self, df: pl.DataFrame, regime_features: pl.DataFrame) -> pl.DataFrame:
        """Join VVMD global regime features to dataset.

        Broadcasts date-level macro features to all stocks on that date.
        """
        if regime_features.is_empty() or self.date_column not in df.columns:
            return df

        merge_df = regime_features
        if "Date" in merge_df.columns and self.date_column != "Date":
            merge_df = merge_df.rename({"Date": self.date_column})
        if self.date_column not in merge_df.columns:
            return df

        dtype = merge_df.schema.get(self.date_column)
        if dtype != pl.Date:
            merge_df = merge_df.with_columns(
                pl.col(self.date_column)
                .cast(pl.Utf8, strict=False)
                .str.strptime(pl.Date, strict=False)
                .alias(self.date_column)
            )

        return df.join(merge_df, on=self.date_column, how="left")

    def add_futures_topix(self, df: pl.DataFrame, futures_features: pl.DataFrame) -> pl.DataFrame:
        """Join TOPIX futures features to dataset using as-of join.

        Broadcasts date-level futures features to all stocks on that date.
        Uses as-of join to ensure T+1 availability compliance.
        """
        if futures_features.is_empty() or self.date_column not in df.columns:
            return df

        # Futures features have 'date' and 'available_ts' columns
        # We need to join using as-of join if 'asof_ts' exists in df
        merge_df = futures_features
        if "date" in merge_df.columns and self.date_column != "date":
            merge_df = merge_df.rename({"date": self.date_column})

        if self.date_column not in merge_df.columns:
            return df

        # Check if we have asof_ts for as-of join
        if "asof_ts" in df.columns and "available_ts" in merge_df.columns:
            # Use as-of join for temporal alignment

            # Ensure both dataframes are sorted
            df_sorted = df.sort([self.date_column])
            merge_sorted = merge_df.sort([self.date_column, "available_ts"])

            # As-of join (backward: use latest available past data)
            result = df_sorted.join_asof(
                merge_sorted,
                left_on="asof_ts",
                right_on="available_ts",
                by=None,  # No code-based join (broadcast to all stocks)
                strategy="backward",
                suffix="_fut",  # Explicit suffix to avoid collision with other joins
            )
            # Remove available_ts from result (keep only feature columns)
            feature_cols = [col for col in merge_df.columns if col not in [self.date_column, "available_ts"]]
            result = result.drop([col for col in feature_cols if col + "_fut" in result.columns])
            result = result.drop(["available_ts_fut"], strict=False)  # Also drop the timestamp column
            result = result.drop([self.date_column + "_fut"], strict=False)  # Also drop date column with suffix
            return result
        else:
            # Fallback: simple date-based join
            dtype = merge_df.schema.get(self.date_column)
            if dtype != pl.Date:
                merge_df = merge_df.with_columns(
                    pl.col(self.date_column)
                    .cast(pl.Utf8, strict=False)
                    .str.strptime(pl.Date, strict=False)
                    .alias(self.date_column)
                )

            # Select only feature columns (exclude available_ts)
            feature_cols = [c for c in merge_df.columns if c.startswith("fut_") or c == self.date_column]
            merge_select = merge_df.select(feature_cols)
            return df.join(merge_select, on=self.date_column, how="left")

    def add_options_features(self, df: pl.DataFrame, options_features: pl.DataFrame) -> pl.DataFrame:
        """Join index option features to dataset using as-of join.

        Broadcasts date-level option features to all stocks on that date.
        Uses as-of join to ensure T+1 availability compliance.
        """
        if options_features.is_empty() or self.date_column not in df.columns:
            return df

        # Options features have 'date' and 'available_ts' columns
        merge_df = options_features
        if "date" in merge_df.columns and self.date_column != "date":
            merge_df = merge_df.rename({"date": self.date_column})

        if self.date_column not in merge_df.columns:
            return df

        # Check if we have asof_ts for as-of join
        if "asof_ts" in df.columns and "available_ts" in merge_df.columns:
            # Use as-of join for temporal alignment

            # Ensure both dataframes are sorted
            df_sorted = df.sort([self.date_column])
            merge_sorted = merge_df.sort([self.date_column, "available_ts"])

            # As-of join (backward: use latest available past data)
            result = df_sorted.join_asof(
                merge_sorted,
                left_on="asof_ts",
                right_on="available_ts",
                by=None,  # No code-based join (broadcast to all stocks)
                strategy="backward",
                suffix="_opt",  # Explicit suffix to avoid collision with other joins
            )
            # Remove available_ts from result (keep only feature columns)
            feature_cols = [col for col in merge_df.columns if col not in [self.date_column, "available_ts"]]
            result = result.drop([col for col in feature_cols if col + "_opt" in result.columns])
            result = result.drop(["available_ts_opt"], strict=False)  # Also drop the timestamp column
            result = result.drop([self.date_column + "_opt"], strict=False)  # Also drop date column with suffix
            return result
        else:
            # Fallback: simple date-based join
            dtype = merge_df.schema.get(self.date_column)
            if dtype != pl.Date:
                merge_df = merge_df.with_columns(
                    pl.col(self.date_column)
                    .cast(pl.Utf8, strict=False)
                    .str.strptime(pl.Date, strict=False)
                    .alias(self.date_column)
                )

            # Select only feature columns (exclude available_ts)
            feature_cols = [c for c in merge_df.columns if c.startswith("macro_opt_") or c == self.date_column]
            merge_select = merge_df.select(feature_cols)
            return df.join(merge_select, on=self.date_column, how="left")
