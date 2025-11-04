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
