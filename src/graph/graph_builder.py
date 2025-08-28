"""
GraphBuilder: 3-channel external graph for GAT
- Channel 0: EWM residual correlation (0..1)
- Channel 1: Size similarity via RBF on log market cap (0..1)
- Channel 2: Same-sector indicator (0/1)

Notes
- Strictly uses only past information up to t-1 for a day t graph
- Caches per-day graph for reproducibility and speed
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import logging


@dataclass
class GBConfig:
    source_glob: str
    lookback: int = 60
    k: int = 15
    ewm_halflife: int = 20
    shrinkage_gamma: float = 0.05
    min_obs: int = 40
    size_tau: float = 1.0
    cache_dir: str = "graph_cache"
    # IMPORTANT: Avoid label_* here to prevent look-ahead leakage
    return_cols: Sequence[str] = (
        "return_1d",
        "feat_ret_1d",
    )
    sector_col: Optional[str] = "sector"
    log_mktcap_col: Optional[str] = "log_mktcap"
    method: str = "ewm_demean"  # "ewm_demean" | "simple"
    symmetric: bool = True


class GraphBuilder:
    def __init__(self, cfg: GBConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.df: Optional[pd.DataFrame] = None
        self._ret_col_selected: Optional[str] = None
        self._logger = logging.getLogger(__name__)
        self._last_asof_ts: Optional[pd.Timestamp] = None
        self._load_index()
        # Safety log
        try:
            self._logger.info(
                f"[GraphBuilder] using return_cols candidates={list(self.cfg.return_cols)} selected={self._ret_col_selected} sector_col={self.cfg.sector_col} log_mktcap_col={self.cfg.log_mktcap_col}"
            )
        except Exception:
            pass

    # ---------- Public API ----------
    def build_for_day(
        self, date: Any, codes: Sequence[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph for a given day using only past (t-1) data.

        Args:
            date: Current day (graph is built using window up to t-1)
            codes: Codes for the batch, in the order they appear in the batch

        Returns:
            edge_index: LongTensor [2, E]
            edge_attr:  FloatTensor [E, 3]
        """
        if self.df is None:
            raise RuntimeError("GraphBuilder index not loaded. Check source_glob.")
        day = pd.Timestamp(date).normalize()
        cache_path = self.cache_dir / f"{day:%Y%m%d}.npz"

        if cache_path.exists():
            npz = np.load(cache_path, allow_pickle=True)
            all_codes = list(npz["codes"])  # stored universe order
            ei_all = torch.from_numpy(npz["edge_index"])  # [2, E]
            ea_all = torch.from_numpy(npz["edge_attr"])  # [E, 3]
            # Read-side staleness guard using cached asof if present
            try:
                asof_int = int(npz["asof"]) if "asof" in npz.files else None
            except Exception:
                asof_int = None
            try:
                max_stale = int(os.getenv("EDGE_STALENESS_MAX_DAYS", "7"))
            except Exception:
                max_stale = 7
            if asof_int is not None:
                try:
                    asof_ts = pd.Timestamp(str(asof_int))  # YYYYMMDD
                    self._last_asof_ts = asof_ts
                    stale_days = int((pd.Timestamp(day) - asof_ts).days)
                    if stale_days > max_stale:
                        self._logger.warning(
                            f"[GraphBuilder] cached graph staleness {stale_days}d exceeds {max_stale}d; rebuilding"
                        )
                    else:
                        return self._subset_edges(
                            all_codes, ei_all, ea_all, list(codes)
                        )
                except Exception:
                    return self._subset_edges(all_codes, ei_all, ea_all, list(codes))
            else:
                # No asof in cache: backward-compat, use as is
                return self._subset_edges(all_codes, ei_all, ea_all, list(codes))

        # Build fresh for the day and cache
        dfw = self._get_window(day, self.cfg.lookback)
        if dfw.empty:
            raise RuntimeError(f"No window data available for {day.date()}")

        # limit to today's codes
        dfw = dfw[dfw["code"].isin(codes)]
        if dfw.empty:
            raise RuntimeError(
                f"Window data for requested codes is empty on {day.date()}"
            )

        # pivot into [L, N]
        pivot = dfw.pivot_table(index="date", columns="code", values="ret").sort_index()
        valid = pivot.count() >= self.cfg.min_obs
        pivot = pivot.loc[:, valid]
        if pivot.shape[1] < 2:
            raise RuntimeError("Effective universe < 2 after min_obs filtering")
        sel_codes = list(pivot.columns)

        X = pivot.to_numpy(dtype=np.float64)  # [L, N]
        L, N = X.shape
        w = self._ewm_weights(L, self.cfg.ewm_halflife)

        # Residualize by market and sector (based on last known sector before day)
        mkt = np.nanmean(X, axis=1, keepdims=True)
        Xc = X - mkt
        sec_map = self._last_sector_before(day, sel_codes)
        sec_labels = np.array([sec_map.get(c, None) for c in sel_codes], dtype=object)
        for s in pd.unique(sec_labels):
            if s is None:
                continue
            cols = np.where(sec_labels == s)[0]
            if len(cols) >= 2:
                sec_mean = np.nanmean(Xc[:, cols], axis=1, keepdims=True)
                Xc[:, cols] = Xc[:, cols] - sec_mean
        Xc = np.nan_to_num(Xc, nan=0.0)

        corr = self._weighted_corr(Xc, w)  # [-1, 1]
        np.fill_diagonal(corr, -1.0)
        if self.cfg.shrinkage_gamma > 0:
            mask = corr > -1.0
            grand = float(corr[mask].mean()) if np.any(mask) else 0.0
            corr = (
                1 - self.cfg.shrinkage_gamma
            ) * corr + self.cfg.shrinkage_gamma * grand

        K = min(self.cfg.k, max(1, N - 1))
        idx = np.argpartition(-corr, K, axis=1)[:, :K]
        rows = np.repeat(np.arange(N), K)
        cols = idx.reshape(-1)
        corr01 = (corr[rows, cols] + 1.0) * 0.5

        size_map = self._last_size_before(day, sel_codes)
        size = np.array([size_map.get(c, 0.0) for c in sel_codes], dtype=np.float64)
        tau = max(1e-6, float(self.cfg.size_tau))
        size_sim = np.exp(-np.abs(size[rows] - size[cols]) / tau)
        same_sector = (sec_labels[rows] == sec_labels[cols]).astype(np.float64)

        if self.cfg.symmetric:
            rows = np.concatenate([rows, cols])
            cols = np.concatenate([cols, rows[: len(cols)]])
            corr01 = np.concatenate([corr01, corr01])
            size_sim = np.concatenate([size_sim, size_sim])
            same_sector = np.concatenate([same_sector, same_sector])

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(
            np.stack([corr01, size_sim, same_sector], axis=1), dtype=torch.float32
        )

        # cache universal graph
        # cache universal graph with asof timestamp (= t-1)
        asof = pd.Timestamp(day) - pd.Timedelta(days=1)
        self._last_asof_ts = asof
        np.savez_compressed(
            self.cache_dir / f"{day:%Y%m%d}.npz",
            codes=np.array(sel_codes, dtype=object),
            edge_index=edge_index.numpy(),
            edge_attr=edge_attr.numpy(),
            asof=np.array(int(asof.strftime("%Y%m%d")), dtype=np.int64),
        )

        return self._subset_edges(sel_codes, edge_index, edge_attr, list(codes))

    # Expose last asof timestamp used/built
    def last_asof_ts(self) -> Optional[pd.Timestamp]:
        return self._last_asof_ts

    # ---------- Internal ----------
    def _load_index(self) -> None:
        paths = (
            sorted(Path().glob(self.cfg.source_glob))
            if ("*" in self.cfg.source_glob or "?" in self.cfg.source_glob)
            else [Path(self.cfg.source_glob)]
        )
        if not paths:
            raise FileNotFoundError(
                f"GraphBuilder: no files matched source_glob: {self.cfg.source_glob}"
            )
        # find available columns
        available_cols = set()
        for p in paths:
            try:
                df1 = pd.read_parquet(p)
                available_cols |= set(df1.columns)
            except Exception:
                continue
        if not available_cols:
            raise RuntimeError("GraphBuilder: failed to read any parquet columns")

        ret_col = None
        for c in self.cfg.return_cols:
            if c in available_cols:
                ret_col = c
                break
        if ret_col is None:
            raise RuntimeError(
                f"GraphBuilder: none of return_cols present: {self.cfg.return_cols}"
            )
        self._ret_col_selected = ret_col

        need_cols = ["date", "code", ret_col]
        if self.cfg.sector_col and self.cfg.sector_col in available_cols:
            need_cols.append(self.cfg.sector_col)
        if self.cfg.log_mktcap_col and self.cfg.log_mktcap_col in available_cols:
            need_cols.append(self.cfg.log_mktcap_col)

        dfs: List[pd.DataFrame] = []
        for p in paths:
            try:
                d = pd.read_parquet(
                    p, columns=[c for c in need_cols if c in available_cols]
                )
                dfs.append(d)
            except Exception:
                continue
        if not dfs:
            raise RuntimeError(
                "GraphBuilder: failed to load required columns from any file"
            )
        df = pd.concat(dfs, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df = df.dropna(subset=["date", "code"]).copy()
        df["ret"] = pd.to_numeric(df[ret_col], errors="coerce")
        if df["ret"].abs().max() > 100.0:
            df["ret"] = df["ret"] / 1e4
        if self.cfg.sector_col and self.cfg.sector_col in df.columns:
            df["sector"] = df[self.cfg.sector_col]
        else:
            df["sector"] = None
        if self.cfg.log_mktcap_col and self.cfg.log_mktcap_col in df.columns:
            df["log_mktcap"] = pd.to_numeric(
                df[self.cfg.log_mktcap_col], errors="coerce"
            )
        else:
            df["log_mktcap"] = np.nan
        self.df = df[["date", "code", "ret", "sector", "log_mktcap"]].copy()

    def _get_window(self, day: pd.Timestamp, lookback: int) -> pd.DataFrame:
        assert self.df is not None
        past = self.df[self.df["date"] < day]
        if past.empty:
            return past
        last_days = np.sort(past["date"].unique())[-int(lookback) :]
        return past[past["date"].isin(last_days)]

    @staticmethod
    def _ewm_weights(L: int, halflife: int) -> np.ndarray:
        if L <= 0:
            return np.array([1.0], dtype=np.float64)
        t = np.arange(L, dtype=np.float64)
        lam = math.log(2) / max(1, int(halflife))
        w = np.exp(lam * (t - (L - 1)))
        w = w / max(1e-12, w.sum())
        return w

    def _last_sector_before(
        self, day: pd.Timestamp, codes: List[str]
    ) -> Dict[str, Any]:
        assert self.df is not None
        df = self.df[(self.df["code"].isin(codes)) & (self.df["date"] < day)][
            ["date", "code", "sector"]
        ]
        df = df.sort_values(["code", "date"]).dropna(subset=["sector"])
        return df.groupby("code").tail(1).set_index("code")["sector"].to_dict()

    def _last_size_before(
        self, day: pd.Timestamp, codes: List[str]
    ) -> Dict[str, float]:
        assert self.df is not None
        df = self.df[(self.df["code"].isin(codes)) & (self.df["date"] < day)][
            ["date", "code", "log_mktcap"]
        ]
        df = df.sort_values(["code", "date"]).dropna(subset=["log_mktcap"])
        return df.groupby("code").tail(1).set_index("code")["log_mktcap"].to_dict()

    @staticmethod
    def _weighted_corr(X: np.ndarray, w: np.ndarray) -> np.ndarray:
        w = w.reshape(-1, 1)
        mu = (X * w).sum(axis=0, keepdims=True)
        Xc = X - mu
        var = (w * (Xc**2)).sum(axis=0)
        var[var <= 1e-12] = 1e-12
        cov = (Xc * w).T @ Xc
        std = np.sqrt(var)
        corr = cov / (std[:, None] * std[None, :])
        return np.clip(corr, -1.0, 1.0)

    @staticmethod
    def _subset_edges(
        all_codes: List[str],
        ei: torch.Tensor,
        ea: torch.Tensor,
        want_codes: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pos = {c: i for i, c in enumerate(all_codes)}
        idx = [pos.get(c, None) for c in want_codes]
        if any(i is None for i in idx):
            valid = [(c, i) for c, i in zip(want_codes, idx) if i is not None]
            if not valid:
                return (
                    torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros((0, 3), dtype=torch.float32),
                )
            want_codes, idx = zip(*valid)
            idx = list(idx)
        idx = torch.tensor(idx, dtype=torch.long)
        mask = (ei[0].unsqueeze(1) == idx.unsqueeze(0)).any(dim=1) & (
            ei[1].unsqueeze(1) == idx.unsqueeze(0)
        ).any(dim=1)
        if mask.sum() == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, 3), dtype=torch.float32),
            )
        remap = {int(i): j for j, i in enumerate(idx.tolist())}
        rows = torch.tensor(
            [remap[int(v)] for v in ei[0, mask].tolist()], dtype=torch.long
        )
        cols = torch.tensor(
            [remap[int(v)] for v in ei[1, mask].tolist()], dtype=torch.long
        )
        return torch.stack([rows, cols], dim=0), ea[mask]
