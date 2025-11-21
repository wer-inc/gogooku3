"""Simplified graph-based peer features."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Deque, Dict, Iterable, Sequence

import numpy as np
import polars as pl


@dataclass
class GraphFeatureConfig:
    code_column: str = "Code"  # Fixed: Use uppercase to match dataset schema
    date_column: str = "Date"  # Fixed: Use uppercase to match dataset schema
    return_column: str = "ret_prev_1d"  # Phase 2: changed from returns_1d
    window_days: int = 60
    min_observations: int = 20
    correlation_threshold: float = 0.3
    shift_to_next_day: bool = True
    block_size: int = 512


@dataclass
class _WindowSlice:
    date: date
    code_indices: np.ndarray
    returns: np.ndarray


class GraphFeatureEngineer:
    """Build simple correlation-based peer graph features per day.

    The generated columns:

    - ``graph_degree``
    - ``graph_peer_corr_mean``
    - ``graph_peer_corr_max``

    are included in the final ML dataset when
    ``ENABLE_GRAPH_FEATURES=1`` (see ``DatasetBuilderSettings``).
    Downstream models such as apex-ranker and ATFT-GAT-FAN can opt-in
    by adding these feature names to their feature group configs. This
    repository treats them as optional enhancements that can be toggled
    on/off in experiments without changing the core dataset schema.
    """

    def __init__(self, config: GraphFeatureConfig | None = None) -> None:
        self.config = config or GraphFeatureConfig()

    def add_features(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config

        return_col = self._resolve_return_column(df, cfg)
        required = {cfg.code_column, cfg.date_column}
        if not return_col or df.is_empty() or not required.issubset(df.columns):
            return df

        working = (
            df.select([cfg.code_column, cfg.date_column, return_col])
            .rename({cfg.code_column: "code", cfg.date_column: "date", return_col: "ret"})
            .with_columns(pl.col("date").cast(pl.Date))
            .drop_nulls()
            .sort("date")
        )
        if working.is_empty():
            return df

        codes = working.select("code").unique().sort("code")["code"].to_list()
        if not codes:
            return df
        code_to_idx = {code: idx for idx, code in enumerate(codes)}

        grouped = (
            working.group_by("date", maintain_order=True)
            .agg([pl.col("code").alias("_codes"), pl.col("ret").alias("_rets")])
            .sort("date")
        )

        window: Deque[_WindowSlice] = deque()
        max_age = timedelta(days=cfg.window_days)
        feature_frames: list[pl.DataFrame] = []

        for row in grouped.iter_rows(named=True):
            current_date: date = row["date"]
            mapped = self._map_codes_to_indices(row["_codes"], row["_rets"], code_to_idx)
            if mapped is None:
                continue
            window.append(_WindowSlice(current_date, mapped[0], mapped[1]))
            while window and (current_date - window[0].date) > max_age:
                window.popleft()

            if len(window) < cfg.min_observations:
                continue

            window_stats = self._compute_window_graph_statistics(window, len(codes))
            if window_stats is None:
                continue

            global_indices, degree, mean_corr, max_corr = window_stats
            codes_for_date = [codes[idx] for idx in global_indices]
            if not codes_for_date:
                continue

            feature_frames.append(
                pl.DataFrame(
                    {
                        cfg.code_column: codes_for_date,
                        cfg.date_column: [current_date] * len(codes_for_date),
                        "graph_degree": degree.astype(np.int32).tolist(),
                        "graph_peer_corr_mean": mean_corr.astype(np.float32, copy=False).tolist(),
                        "graph_peer_corr_max": max_corr.astype(np.float32, copy=False).tolist(),
                    }
                )
            )

        if not feature_frames:
            return df

        graph_df = pl.concat(feature_frames, how="vertical")
        graph_df = graph_df.with_columns(
            [
                pl.col(cfg.date_column).cast(pl.Date),
                pl.col("graph_degree").cast(pl.Int32),
                pl.col("graph_peer_corr_mean").cast(pl.Float32),
                pl.col("graph_peer_corr_max").cast(pl.Float32),
            ]
        )

        if cfg.shift_to_next_day:
            graph_df = self._shift_to_next_trading_day(df, graph_df)
            if graph_df.is_empty():
                return df

        out = df.join(
            graph_df,
            left_on=[cfg.code_column, cfg.date_column],
            right_on=[cfg.code_column, cfg.date_column],
            how="left",
        )
        return out

    def _resolve_return_column(self, df: pl.DataFrame, cfg: GraphFeatureConfig) -> str | None:
        candidates = [cfg.return_column, "returns_1d", "ret_prev_1d"]
        for name in candidates:
            if name in df.columns:
                return name
        return None

    def _map_codes_to_indices(
        self,
        codes: Sequence[str],
        returns: Sequence[float],
        code_to_idx: Dict[str, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        mapped_idx: list[int] = []
        mapped_ret: list[float] = []
        for code, value in zip(codes, returns):
            if value is None:
                continue
            idx = code_to_idx.get(code)
            if idx is None:
                continue
            try:
                float_value = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(float_value):
                continue
            mapped_idx.append(idx)
            mapped_ret.append(float_value)
        if not mapped_idx:
            return None
        return np.asarray(mapped_idx, dtype=np.int32), np.asarray(mapped_ret, dtype=np.float32)

    def _compute_window_graph_statistics(
        self,
        slices: Deque[_WindowSlice],
        universe_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        active_indices = self._active_indices(slices)
        if active_indices.size == 0:
            return None
        local_lookup = np.full(universe_size, -1, dtype=np.int32)
        local_lookup[active_indices] = np.arange(active_indices.size, dtype=np.int32)
        width = len(slices)
        matrix = np.full((active_indices.size, width), np.nan, dtype=np.float32)
        for col_idx, entry in enumerate(slices):
            local_idx = local_lookup[entry.code_indices]
            matrix[local_idx, col_idx] = entry.returns

        stats = self._pairwise_graph_metrics(matrix)
        if stats is None:
            return None
        eligible_local_idx, degree, mean_corr, max_corr = stats
        if eligible_local_idx.size == 0:
            return None
        global_indices = active_indices[eligible_local_idx]
        return global_indices, degree, mean_corr, max_corr

    def _active_indices(self, slices: Iterable[_WindowSlice]) -> np.ndarray:
        if not slices:
            return np.array([], dtype=np.int32)
        concatenated = np.concatenate([entry.code_indices for entry in slices])
        if concatenated.size == 0:
            return np.array([], dtype=np.int32)
        return np.unique(concatenated)

    def _pairwise_graph_metrics(
        self,
        matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        cfg = self.config
        mask = ~np.isnan(matrix)
        obs_per_code = mask.sum(axis=1)
        eligible_mask = obs_per_code >= cfg.min_observations
        eligible_idx = np.where(eligible_mask)[0]
        if eligible_idx.size == 0:
            return None

        sub_matrix = matrix[eligible_idx]
        sub_mask = mask[eligible_idx]
        obs = obs_per_code[eligible_idx].astype(np.float32)

        sums = np.sum(np.where(sub_mask, sub_matrix, 0.0), axis=1)
        means = np.divide(sums, obs, out=np.zeros_like(obs), where=obs > 0)
        centered = np.where(sub_mask, sub_matrix - means[:, None], 0.0).astype(np.float32)

        variance = np.divide(
            np.sum(centered**2, axis=1),
            np.maximum(obs - 1.0, 1.0),
            out=np.zeros_like(obs),
        )
        std = np.sqrt(variance, dtype=np.float32)
        std_valid = np.isfinite(std) & (std > 0)
        if not np.any(std_valid):
            return None

        sub_matrix = sub_matrix[std_valid]
        centered = centered[std_valid]
        sub_mask = sub_mask[std_valid]
        std = std[std_valid]
        eligible_idx = eligible_idx[std_valid]

        mask_uint8 = sub_mask.astype(np.uint8)
        degree = np.zeros(len(eligible_idx), dtype=np.int32)
        sum_abs = np.zeros(len(eligible_idx), dtype=np.float32)
        max_abs = np.zeros(len(eligible_idx), dtype=np.float32)

        block_size = max(1, cfg.block_size)
        for i_start in range(0, len(eligible_idx), block_size):
            i_end = min(i_start + block_size, len(eligible_idx))
            block_centered = centered[i_start:i_end]
            block_mask = mask_uint8[i_start:i_end]
            block_std = std[i_start:i_end]

            for j_start in range(i_start, len(eligible_idx), block_size):
                j_end = min(j_start + block_size, len(eligible_idx))
                other_centered = centered[j_start:j_end]
                other_mask = mask_uint8[j_start:j_end]
                other_std = std[j_start:j_end]

                counts = block_mask @ other_mask.T
                if not counts.any():
                    continue
                cov = block_centered @ other_centered.T
                denom = counts.astype(np.float32) - 1.0
                std_outer = block_std[:, None] * other_std[None, :]
                with np.errstate(invalid="ignore", divide="ignore"):
                    corr_block = cov / (denom * std_outer)
                corr_block[counts < cfg.min_observations] = np.nan

                abs_corr = np.abs(corr_block).astype(np.float32, copy=False)
                strong_mask = (abs_corr >= cfg.correlation_threshold) & np.isfinite(corr_block)

                if i_start == j_start:
                    strong_mask = np.triu(strong_mask, k=1)

                if not strong_mask.any():
                    continue

                weighted = np.where(strong_mask, abs_corr, 0.0)
                row_deg = strong_mask.sum(axis=1, dtype=np.int32)
                row_sum = weighted.sum(axis=1, dtype=np.float32)
                row_max = np.where(row_deg > 0, weighted.max(axis=1), 0.0)

                i_slice = slice(i_start, i_end)
                degree[i_slice] += row_deg
                sum_abs[i_slice] += row_sum
                max_abs[i_slice] = np.maximum(max_abs[i_slice], row_max)

                col_deg = strong_mask.sum(axis=0, dtype=np.int32)
                col_sum = weighted.sum(axis=0, dtype=np.float32)
                col_max = np.where(col_deg > 0, weighted.max(axis=0), 0.0)
                j_slice = slice(j_start, j_end)
                degree[j_slice] += col_deg
                sum_abs[j_slice] += col_sum
                max_abs[j_slice] = np.maximum(max_abs[j_slice], col_max)

        mean_abs = np.zeros_like(sum_abs)
        non_zero = degree > 0
        mean_abs[non_zero] = sum_abs[non_zero] / degree[non_zero]
        return eligible_idx, degree, mean_abs, max_abs

    def _shift_to_next_trading_day(self, base_df: pl.DataFrame, graph_df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config
        base_dates = (
            base_df.select(cfg.date_column).drop_nulls().unique().sort(cfg.date_column)[cfg.date_column].to_list()
        )
        if len(base_dates) < 2:
            return graph_df.head(0)
        # Build next_map using zip instead of range(len())
        next_map = dict(zip(base_dates[:-1], base_dates[1:], strict=False))
        shifted = graph_df.with_columns(
            pl.col(cfg.date_column).replace(next_map, default=None).cast(pl.Date).alias(cfg.date_column)
        ).drop_nulls(subset=[cfg.date_column])
        return shifted
