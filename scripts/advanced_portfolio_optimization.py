#!/usr/bin/env python3
"""
高度なポートフォリオ最適化（実データ対応）
- 予測と実リターンのファイルを読み込み、日次ポートフォリオSharpeを最適化
- デモモード（入力未指定）では疑似データで挙動確認
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class InputSpec:
    path: Optional[str] = None
    date_col: str = "date"
    code_col: str = "Code"
    pred_col: str = "predicted_return"
    ret_col: Optional[str] = None  # 例: returns_1d / target


def _infer_return_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["actual_return", "returns_1d", "ret_1d", "feat_ret_1d", "target", "label_excess_1_bps"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _daily_long_short_returns(
    df: pd.DataFrame,
    date_col: str,
    code_col: str,
    pred_col: str,
    ret_col: str,
    long_frac: float = 0.2,
    short_frac: float = 0.2,
    mode: str = "ls",  # ls|lo|so
    invert_sign: bool = False,
    cost_bps: float = 5.0,
) -> Tuple[pd.DataFrame, float]:
    """日次でロングショートの均等重みポートを構築し、日次リターンとSharpeを返す"""
    df = df.copy()
    if invert_sign:
        df[pred_col] = -df[pred_col]

    daily = []
    last_weights: Dict[str, float] = {}
    for d, g in df.groupby(date_col):
        g = g[[code_col, pred_col, ret_col]].dropna()
        if len(g) < 10:
            continue
        g = g.sort_values(pred_col)
        longs = pd.DataFrame()
        shorts = pd.DataFrame()
        if mode in ("ls", "lo") and long_frac > 0:
            n_long = max(1, int(len(g) * long_frac))
            longs = g.tail(n_long)
        if mode in ("ls", "so") and short_frac > 0:
            n_short = max(1, int(len(g) * short_frac))
            shorts = g.head(n_short)

        weights: Dict[str, float] = {}
        if len(longs) > 0:
            w = 1.0 / max(len(longs), 1)
            for code in longs[code_col].values:
                weights[str(code)] = weights.get(str(code), 0.0) + w
        if len(shorts) > 0:
            w = 1.0 / max(len(shorts), 1)
            for code in shorts[code_col].values:
                weights[str(code)] = weights.get(str(code), 0.0) - w

        # 当日リターン
        ret_map = dict(zip(g[code_col].astype(str).values, g[ret_col].values))
        gross = sum(weights.get(k, 0.0) * ret_map.get(k, 0.0) for k in weights.keys())

        # ターンオーバーコスト（単純化）：前日との差のL1ノルム×cost_bps
        turnover = sum(abs(weights.get(k, 0.0) - last_weights.get(k, 0.0)) for k in set(weights) | set(last_weights))
        net = gross - (cost_bps * 1e-4) * turnover
        daily.append({"date": d, "gross": gross, "turnover": turnover, "net": net})
        last_weights = weights

    daily_df = pd.DataFrame(daily)
    if daily_df.empty:
        return daily_df, 0.0
    sharpe = daily_df["net"].mean() / (daily_df["net"].std() + 1e-12)
    return daily_df, float(sharpe)


def run_with_input(spec: InputSpec, long_frac: float, short_frac: float, mode: str, invert_sign: bool, cost_bps: float) -> Dict:
    path = Path(spec.path) if spec.path else None
    if path is None or not path.exists():
        raise FileNotFoundError(f"Input not found: {spec.path}")

    if path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # 列の存在チェック
    for c in (spec.date_col, spec.code_col, spec.pred_col):
        if c not in df.columns:
            raise ValueError(f"Column missing in input: {c}")

    ret_col = spec.ret_col or _infer_return_column(df)
    if ret_col is None:
        raise ValueError("Return column not found. Provide --ret-col or include one of: actual_return, returns_1d, ret_1d, feat_ret_1d, target")

    daily_df, sharpe = _daily_long_short_returns(
        df,
        date_col=spec.date_col,
        code_col=spec.code_col,
        pred_col=spec.pred_col,
        ret_col=ret_col,
        long_frac=long_frac,
        short_frac=short_frac,
        mode=mode,
        invert_sign=invert_sign,
        cost_bps=cost_bps,
    )

    report = {
        "input": str(path),
        "rows": int(len(df)),
        "days": int(daily_df["date"].nunique()) if not daily_df.empty else 0,
        "mode": mode,
        "long_frac": long_frac,
        "short_frac": short_frac,
        "invert_sign": invert_sign,
        "cost_bps": cost_bps,
        "sharpe": sharpe,
    }

    out_dir = Path("output/portfolio")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    daily_path = out_dir / f"daily_returns_{ts}.parquet"
    daily_df.to_parquet(daily_path, index=False) if not daily_df.empty else None
    (out_dir / f"report_{ts}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info(f"Saved portfolio report: {out_dir}/report_{ts}.json")
    if not daily_df.empty:
        logger.info(f"Saved daily returns: {daily_path}")
    logger.info(f"Sharpe (net): {sharpe:.4f}")
    return report


def demo() -> Dict:
    logger.info("=== デモモード（擬似データ） ===")
    rng = np.random.default_rng(42)
    n_days, n_assets = 250, 300
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    codes = [f"S{i:04d}" for i in range(n_assets)]

    rows = []
    for d in dates:
        preds = rng.normal(0, 0.05, size=n_assets)
        rets = rng.normal(0, 0.01, size=n_assets)
        for c, p, r in zip(codes, preds, rets):
            rows.append({"date": d, "Code": c, "predicted_return": p, "returns_1d": r})
    df = pd.DataFrame(rows)

    spec = InputSpec(path=None)
    # ロングショート20/20、コスト5bps、符号反転ありで試す
    daily_df, sharpe = _daily_long_short_returns(df, "date", "Code", "predicted_return", "returns_1d", 0.2, 0.2, "ls", True, 5.0)
    logger.info(f"デモ Sharpe (net): {sharpe:.4f} | days={len(daily_df)}")
    return {"demo_sharpe": sharpe}


def main():
    p = argparse.ArgumentParser(description="Advanced Portfolio Optimization (with real inputs)")
    p.add_argument("--input", type=str, help="Predictions file (csv/parquet) including returns column")
    p.add_argument("--date-col", type=str, default="date")
    p.add_argument("--code-col", type=str, default="Code")
    p.add_argument("--pred-col", type=str, default="predicted_return")
    p.add_argument("--ret-col", type=str, default=None)
    p.add_argument("--mode", type=str, default="ls", choices=["ls", "lo", "so"])
    p.add_argument("--long-frac", type=float, default=0.2)
    p.add_argument("--short-frac", type=float, default=0.2)
    p.add_argument("--invert-sign", action="store_true")
    p.add_argument("--cost-bps", type=float, default=5.0)
    p.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    args = p.parse_args()

    if args.demo and not args.input:
        demo()
        return

    spec = InputSpec(
        path=args.input,
        date_col=args.date_col,
        code_col=args.code_col,
        pred_col=args.pred_col,
        ret_col=args.ret_col,
    )
    run_with_input(spec, args.long_frac, args.short_frac, args.mode, args.invert_sign, args.cost_bps)


if __name__ == "__main__":
    main()
