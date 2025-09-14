"""
Cross-sectional normalization for financial data.
Prevents data leakage by fitting on train data only.
"""
from __future__ import annotations

from collections.abc import Iterable

import polars as pl


class CrossSectionalNormalizer:
    """
    日次クロスセクショナル正規化（train統計のみでtestを変換）
    - by='date' の zscore
    - データリーク防止: fit(train) → transform(train/test)
    """
    def __init__(self, date_col: str = "date", code_col: str = "code",
                 features: Iterable[str] | None = None):
        self.date_col = date_col
        self.code_col = code_col
        self.features = list(features) if features else None
        self._stats: pl.DataFrame | None = None

    def _infer_features(self, df: pl.DataFrame) -> list[str]:
        """Infer numeric feature columns automatically."""
        ignore = {self.date_col, self.code_col}
        numeric_types = {
            pl.Float64, pl.Float32,
            pl.Int64, pl.Int32, pl.Int16, pl.Int8,
            pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8,
        }
        return [c for c, dt in zip(df.columns, df.dtypes, strict=False)
                if c not in ignore and dt in numeric_types]

    def fit(self, train: pl.DataFrame) -> CrossSectionalNormalizer:
        """Fit normalizer on training data only."""
        feats = self.features or self._infer_features(train)

        agg = train.lazy().group_by(self.date_col).agg(
            [pl.col(feats).mean().suffix("__mu"),
             pl.col(feats).std(ddof=1).suffix("__sigma")]
        ).collect(streaming=True)

        self._stats = agg
        self.features = feats
        return self

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform data using fitted statistics."""
        assert self._stats is not None and self.features is not None, "Must call fit() first"

        out = (df.lazy()
                 .join(self._stats.lazy(), on=self.date_col, how="left")
                 .with_columns([
                     ((pl.col(f) - pl.col(f"{f}__mu")) / pl.col(f"{f}__sigma")).alias(f)
                     for f in self.features
                 ])
                 .drop([f"{f}__mu" for f in self.features] +
                       [f"{f}__sigma" for f in self.features])
               ).collect(streaming=True)
        return out

    def fit_transform(self, train: pl.DataFrame) -> pl.DataFrame:
        """Fit and transform training data."""
        return self.fit(train).transform(train)
