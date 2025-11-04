from __future__ import annotations

"""Enhanced margin trading features extraction from J-Quants margin data.

Extracts advanced margin trading signals:
- margin_balance_ratio: Net long-short ratio of outstanding margins
- margin_velocity: Rate of change in net margin positions
- margin_divergence: Divergence between long and short velocities
- margin_momentum: Trend strength in long margin outstanding
- margin_stress_indicator: Composite risk metric
- institutional_margin_ratio: Cross-sectional rank of weekly long margin volume
"""

import logging
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


class EnhancedMarginTradingExtractor:
    """Extract enhanced margin trading features from J-Quants margin data."""

    def __init__(
        self,
        velocity_window: int = 5,
        momentum_window: int = 20,
        stress_lookback: int = 60,
    ):
        """Initialize enhanced margin trading extractor.

        Args:
            velocity_window: Window for margin velocity calculation
            momentum_window: Window for margin momentum calculation
            stress_lookback: Lookback period for stress indicator
        """
        self.velocity_window = velocity_window
        self.momentum_window = momentum_window
        self.stress_lookback = stress_lookback

    def extract_features(
        self,
        df_base: pl.DataFrame,
        df_margin_daily: pl.DataFrame | None = None,
        df_margin_weekly: pl.DataFrame | None = None,
        df_listed: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Extract enhanced margin trading features.

        Args:
            df_base: Base dataset with columns [code, date]
            df_margin_daily: Daily margin data from J-Quants
            df_margin_weekly: Weekly margin data from J-Quants
            df_listed: Listed info for market cap normalization

        Returns:
            DataFrame with enhanced margin features added
        """
        if df_margin_daily is None and df_margin_weekly is None:
            logger.warning("No margin data provided, adding null features")
            return self._add_null_features(df_base)

        # Ensure date columns are datetime
        if "date" in df_base.columns and df_base["date"].dtype == pl.Utf8:
            df_base = df_base.with_columns(pl.col("date").str.to_date())

        # Process daily margin data if available
        if df_margin_daily is not None and not df_margin_daily.is_empty():
            df_features = self._process_daily_margin(
                df_base, df_margin_daily, df_listed
            )
        else:
            df_features = df_base

        # Add weekly margin features if available
        if df_margin_weekly is not None and not df_margin_weekly.is_empty():
            df_features = self._add_weekly_margin_features(
                df_features, df_margin_weekly
            )

        # Calculate composite indicators
        df_features = self._calculate_composite_indicators(df_features)

        return df_features

    def _process_daily_margin(
        self,
        df_base: pl.DataFrame,
        df_margin: pl.DataFrame,
        df_listed: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Process daily margin data for feature extraction.

        Args:
            df_base: Base dataset
            df_margin: Daily margin data
            df_listed: Listed info for normalization

        Returns:
            DataFrame with daily margin features
        """
        # Ensure consistent column names
        if "Date" in df_margin.columns:
            df_margin = df_margin.rename({"Date": "date"})
        if "ApplicationDate" in df_margin.columns:
            df_margin = df_margin.rename({"ApplicationDate": "date"})
        if "Code" in df_margin.columns:
            df_margin = df_margin.rename({"Code": "code"})

        # Join margin data with base
        df_merged = df_base.join(df_margin, on=["code", "date"], how="left")

        # Column names align with J-Quants daily margin interest normalization
        long_col = "LongMarginOutstanding"
        short_col = "ShortMarginOutstanding"

        # Calculate net outstanding and margin balance ratio
        if long_col in df_merged.columns and short_col in df_merged.columns:
            df_merged = df_merged.with_columns(
                [
                    (pl.col(long_col) - pl.col(short_col)).alias(
                        "margin_net_outstanding"
                    ),
                    (
                        (pl.col(long_col) - pl.col(short_col))
                        / (pl.col(long_col) + pl.col(short_col) + 1e-10)
                    ).alias("margin_balance_ratio"),
                ]
            )

        # Calculate long/short velocities (rate of change)
        if long_col in df_merged.columns:
            df_merged = df_merged.sort(["code", "date"])
            df_merged = df_merged.with_columns(
                pl.col(long_col)
                .pct_change(self.velocity_window)
                .over("code")
                .alias("margin_buy_velocity")
            )

        if short_col in df_merged.columns:
            df_merged = df_merged.with_columns(
                pl.col(short_col)
                .pct_change(self.velocity_window)
                .over("code")
                .alias("margin_sell_velocity")
            )

        # Calculate margin divergence
        if (
            "margin_buy_velocity" in df_merged.columns
            and "margin_sell_velocity" in df_merged.columns
        ):
            df_merged = df_merged.with_columns(
                (pl.col("margin_buy_velocity") - pl.col("margin_sell_velocity")).alias(
                    "margin_divergence"
                )
            )

        # Net margin velocity
        if "margin_net_outstanding" in df_merged.columns:
            df_merged = df_merged.with_columns(
                pl.col("margin_net_outstanding")
                .pct_change(self.velocity_window)
                .over("code")
                .alias("margin_velocity")
            )

        # Calculate margin momentum
        if long_col in df_merged.columns:
            df_merged = df_merged.with_columns(
                pl.col(long_col)
                .rolling_mean(window_size=self.momentum_window)
                .over("code")
                .alias("margin_buy_ma")
            )

            df_merged = df_merged.with_columns(
                (
                    (pl.col(long_col) - pl.col("margin_buy_ma"))
                    / (pl.col("margin_buy_ma") + 1e-10)
                ).alias("margin_momentum")
            )

        # Normalize by market cap if available
        if df_listed is not None and "market_cap" in df_listed.columns:
            df_merged = self._normalize_by_market_cap(df_merged, df_listed)

        return df_merged

    def _add_weekly_margin_features(
        self, df_features: pl.DataFrame, df_weekly: pl.DataFrame
    ) -> pl.DataFrame:
        """Add weekly margin features to the dataset.

        Args:
            df_features: DataFrame with existing features
            df_weekly: Weekly margin data

        Returns:
            DataFrame with weekly features added
        """
        # Ensure consistent column names
        if "Date" in df_weekly.columns:
            df_weekly = df_weekly.rename({"Date": "date"})
        if "Code" in df_weekly.columns:
            df_weekly = df_weekly.rename({"Code": "code"})

        # Forward fill weekly data to daily
        df_weekly = df_weekly.sort(["code", "date"])

        # Join and forward fill
        df_merged = df_features.join(
            df_weekly.select(
                ["code", "date", "LongMarginTradeVolume", "ShortMarginTradeVolume"]
            ),
            on=["code", "date"],
            how="left",
        )

        # Forward fill weekly values
        df_merged = df_merged.with_columns(
            [
                pl.col("LongMarginTradeVolume").forward_fill().over("code"),
                pl.col("ShortMarginTradeVolume").forward_fill().over("code"),
            ]
        )

        # Calculate institutional ratio (assuming high volume = institutional)
        if "LongMarginTradeVolume" in df_merged.columns:
            df_merged = df_merged.with_columns(
                pl.col("LongMarginTradeVolume")
                .rank(method="average")
                .over("date")
                .alias("institutional_margin_rank")
            )

            # Normalize to 0-1
            df_merged = df_merged.with_columns(
                (
                    pl.col("institutional_margin_rank")
                    / pl.col("institutional_margin_rank").max()
                ).alias("institutional_margin_ratio")
            )

        return df_merged

    def _calculate_composite_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate composite margin stress indicator.

        Args:
            df: DataFrame with margin features

        Returns:
            DataFrame with composite indicators added
        """
        # Calculate margin stress indicator
        stress_components = []

        # Component 1: High margin balance ratio
        if "margin_balance_ratio" in df.columns:
            df = df.with_columns(
                pl.col("margin_balance_ratio").abs().alias("margin_imbalance")
            )
            stress_components.append("margin_imbalance")

        # Component 2: High velocity
        if "margin_buy_velocity" in df.columns:
            df = df.with_columns(
                pl.col("margin_buy_velocity").abs().alias("margin_velocity_stress")
            )
            stress_components.append("margin_velocity_stress")

        # Component 3: Divergence
        if "margin_divergence" in df.columns:
            df = df.with_columns(
                pl.col("margin_divergence").abs().alias("margin_divergence_stress")
            )
            stress_components.append("margin_divergence_stress")

        # Calculate composite stress indicator
        if stress_components:
            # Normalize each component to 0-1 using percentile rank
            for comp in stress_components:
                df = df.with_columns(
                    pl.col(comp).rank(method="average").alias(f"{comp}_rank")
                )
                df = df.with_columns(
                    (pl.col(f"{comp}_rank") / pl.col(f"{comp}_rank").max()).alias(
                        f"{comp}_norm"
                    )
                )

            # Weighted average of normalized components
            stress_expr = pl.lit(0.0)
            for comp in stress_components:
                stress_expr = stress_expr + pl.col(f"{comp}_norm")

            df = df.with_columns(
                (stress_expr / len(stress_components)).alias("margin_stress_indicator")
            )

            # Clean up intermediate columns
            cols_to_drop = []
            for comp in stress_components:
                cols_to_drop.extend([comp, f"{comp}_rank", f"{comp}_norm"])
            df = df.drop(cols_to_drop)

        return df

    def _normalize_by_market_cap(
        self, df: pl.DataFrame, df_listed: pl.DataFrame
    ) -> pl.DataFrame:
        """Normalize margin values by market capitalization.

        Args:
            df: DataFrame with margin data
            df_listed: Listed info with market cap

        Returns:
            DataFrame with normalized values
        """
        # Join market cap data
        if "Code" in df_listed.columns:
            df_listed = df_listed.rename({"Code": "code"})

        df = df.join(df_listed.select(["code", "market_cap"]), on="code", how="left")

        # Normalize margin balances by market cap
        if "LongMarginOutstanding" in df.columns and "market_cap" in df.columns:
            df = df.with_columns(
                (
                    pl.col("LongMarginOutstanding") / (pl.col("market_cap") + 1e-10)
                ).alias("margin_buy_to_mcap")
            )

        if "ShortMarginOutstanding" in df.columns and "market_cap" in df.columns:
            df = df.with_columns(
                (
                    pl.col("ShortMarginOutstanding") / (pl.col("market_cap") + 1e-10)
                ).alias("margin_sell_to_mcap")
            )

        return df

    def _add_null_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add null margin features when no data available.

        Args:
            df: Input dataframe

        Returns:
            DataFrame with null features
        """
        return df.with_columns(
            [
                pl.lit(None).cast(pl.Float32).alias("margin_balance_ratio"),
                pl.lit(None).cast(pl.Float32).alias("margin_buy_velocity"),
                pl.lit(None).cast(pl.Float32).alias("margin_sell_velocity"),
                pl.lit(None).cast(pl.Float32).alias("margin_divergence"),
                pl.lit(None).cast(pl.Float32).alias("margin_velocity"),
                pl.lit(None).cast(pl.Float32).alias("margin_momentum"),
                pl.lit(None).cast(pl.Float32).alias("institutional_margin_ratio"),
                pl.lit(None).cast(pl.Float32).alias("margin_stress_indicator"),
                pl.lit(None).cast(pl.Float32).alias("margin_buy_to_mcap"),
                pl.lit(None).cast(pl.Float32).alias("margin_sell_to_mcap"),
            ]
        )


def add_enhanced_margin_features(
    df_base: pl.DataFrame,
    fetcher: Any,
    start_date: str,
    end_date: str,
    use_weekly: bool = True,
) -> pl.DataFrame:
    """Convenience function to add enhanced margin trading features.

    Args:
        df_base: Base dataset with [code, date] columns
        fetcher: JQuantsAsyncFetcher instance
        start_date: Start date for margin data
        end_date: End date for margin data
        use_weekly: Whether to include weekly margin data

    Returns:
        DataFrame with enhanced margin features added
    """
    import asyncio

    import aiohttp

    async def fetch_margin_data():
        async with aiohttp.ClientSession() as session:
            # Authenticate first
            await fetcher.authenticate(session)

            # Fetch daily margin data
            df_daily = await fetcher.get_daily_margin_interest(
                session, from_date=start_date, to_date=end_date
            )

            # Fetch weekly margin data if requested
            df_weekly = None
            if use_weekly:
                df_weekly = await fetcher.get_weekly_margin_interest(
                    session, from_date=start_date, to_date=end_date
                )

            # Fetch listed info for market cap
            df_listed = await fetcher.get_listed_info(session)

            return df_daily, df_weekly, df_listed

    # Run async fetch
    try:
        asyncio.get_running_loop()
        try:
            import nest_asyncio  # type: ignore

            nest_asyncio.apply()
        except Exception:
            pass
        df_daily, df_weekly, df_listed = asyncio.run(fetch_margin_data())
    except RuntimeError:
        df_daily, df_weekly, df_listed = asyncio.run(fetch_margin_data())

    # Extract features
    extractor = EnhancedMarginTradingExtractor()

    return extractor.extract_features(df_base, df_daily, df_weekly, df_listed)
