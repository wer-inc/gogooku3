from __future__ import annotations

"""Build a pseudo-VIX style 30-day forward volatility index from forecasts.

Input forecasts (flat DataFrame):
  id, ts, h, y_hat[, p10, p90]

Assumptions:
  - y_hat at h=30 represents estimated daily volatility (std) over the next 30 days
    expressed in daily units (e.g., ~percent/100). This function annualizes as
    vol_annual = y_hat * sqrt(252) and scales by 100 for index points.
  - If quantiles are provided, p10/p90 are propagated through the same weighting.

Weights:
  - Optional `weights` DataFrame with columns [id, weight] (e.g., market-cap).
  - If omitted, uses equal weight per id.

Output (flat DataFrame):
  ts, index, level[, p10, p90]
"""

from typing import Optional

import numpy as np
import pandas as pd


def build_pseudo_vix(
    df_fcst: pd.DataFrame,
    index_name: str = "ESVI_JP",
    horizon: int = 30,
    weights: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    required_cols = {"id", "ts", "h", "y_hat"}
    if not required_cols.issubset(df_fcst.columns):
        raise ValueError(f"df_fcst must contain {required_cols}")
    df = df_fcst[df_fcst["h"] == horizon].copy()
    if df.empty:
        return pd.DataFrame(columns=["ts", "index", "level"])  # nothing to do

    df["ts"] = pd.to_datetime(df["ts"])  # type: ignore[assignment]
    df.sort_values(["ts", "id"], inplace=True)

    # Prepare weights
    if weights is not None and {"id", "weight"}.issubset(weights.columns):
        w = weights[["id", "weight"]].copy()
    else:
        w = df[["id"]].drop_duplicates().copy()
        w["weight"] = 1.0
    wsum = w["weight"].sum()
    if wsum <= 0:
        w["weight"] = 1.0
        wsum = float(len(w))
    w["weight"] = w["weight"].astype(float) / wsum

    df = df.merge(w, on="id", how="left")
    df["weight"].fillna(0.0, inplace=True)

    def _agg(group: pd.DataFrame) -> pd.Series:
        y = group["y_hat"].astype(float).to_numpy()
        wt = group["weight"].astype(float).to_numpy()
        # Weighted average daily vol
        vol_daily = float(np.sum(y * wt))
        # Annualize and scale to points
        level = vol_daily * np.sqrt(252.0) * 100.0
        row = {"level": level}
        # Propagate quantiles if present
        for q in ("p10", "p90"):
            if q in group.columns:
                row[q] = float(np.sum(group[q].astype(float).to_numpy() * wt) * np.sqrt(252.0) * 100.0)
        return pd.Series(row)

    out = df.groupby("ts", sort=True).apply(_agg).reset_index()
    out.insert(1, "index", index_name)
    return out


class ESVICalculator:
    """Enhanced Stochastic Volatility Index Calculator."""

    def __init__(self, index_name: str = "ESVI_JP", horizon: int = 30):
        """Initialize ESVI calculator.

        Args:
            index_name: Name of the index
            horizon: Volatility forecast horizon in days
        """
        self.index_name = index_name
        self.horizon = horizon

    def calculate(self, df_fcst: pd.DataFrame, weights: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate ESVI index from forecasts.

        Args:
            df_fcst: DataFrame with forecast data
            weights: Optional weights DataFrame

        Returns:
            DataFrame with index values
        """
        return build_pseudo_vix(
            df_fcst=df_fcst,
            index_name=self.index_name,
            horizon=self.horizon,
            weights=weights
        )

    def get_latest_value(self, df_fcst: pd.DataFrame, weights: Optional[pd.DataFrame] = None) -> float:
        """Get the latest ESVI value.

        Args:
            df_fcst: DataFrame with forecast data
            weights: Optional weights DataFrame

        Returns:
            Latest ESVI value
        """
        if df_fcst.empty:
            return 0.0

        # Calculate full ESVI index
        index_df = self.calculate(df_fcst, weights)

        if index_df.empty:
            return 0.0

        # Return the most recent value
        return float(index_df.iloc[-1]["level"])

    def calculate_with_forecasts(
        self,
        forecast_predictions: dict,
        symbols: list[str],
        weights: Optional[pd.DataFrame] = None
    ) -> float:
        """Calculate ESVI from API forecast predictions format.

        Args:
            forecast_predictions: Dict with forecast results from TimesFM/TFT
            symbols: List of symbol identifiers
            weights: Optional market cap or other weights

        Returns:
            Current ESVI value
        """
        # Convert API predictions to forecast DataFrame format
        fcst_rows = []
        current_ts = pd.Timestamp.now()

        for symbol in symbols:
            # Mock volatility forecasts (replace with actual prediction extraction)
            # In real implementation, extract volatility estimates from forecast_predictions
            for h in [self.horizon]:
                # Extract volatility forecast at horizon h
                # This would come from forecast model predictions
                vol_forecast = 0.02 * (1 + np.random.normal(0, 0.1))  # Mock daily vol ~2%

                fcst_rows.append({
                    "id": symbol,
                    "ts": current_ts,
                    "h": h,
                    "y_hat": vol_forecast
                })

        if not fcst_rows:
            return 0.0

        df_fcst = pd.DataFrame(fcst_rows)
        return self.get_latest_value(df_fcst, weights)

