#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P0-4/6/7 Coefficient Auto-Tuner from RFI-5/6 Metrics

Automatically determines optimal coefficients and GAT hyperparameters
based on actual RFI-5/6 measurements from Quick Run.

Usage:
    python tools/tune_p0467_from_rfi.py rfi_56_metrics.txt

Output:
    - Recommended export commands for environment variables
    - Graph builder hints
    - Loss hints

Created: 2025-11-02
Status: Production Ready
"""
import re
import sys
import statistics as st
from pathlib import Path


def parse(path: str) -> tuple[dict[str, float | None], int]:
    """
    Parse RFI-5/6 metrics file and extract median values.

    Args:
        path: Path to rfi_56_metrics.txt

    Returns:
        (median_dict, num_lines)
    """
    if not Path(path).exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)

    lines = [ln.strip() for ln in open(path) if "RFI56 |" in ln]

    if len(lines) == 0:
        print(f"ERROR: No RFI56 lines found in {path}")
        sys.exit(1)

    # Metrics to extract
    keyz = [
        "gat_gate_mean", "gat_gate_std",
        "deg_avg", "isolates",
        "RankIC", "WQL", "CRPS", "qx_rate",
        "grad_ratio"
    ]

    vals = {k: [] for k in keyz}

    # Extract all values
    for ln in lines:
        for k in keyz:
            m = re.search(rf"{k}=([+-]?\d+(?:\.\d+)?)", ln)
            if m:
                vals[k].append(float(m.group(1)))

    # Compute medians
    med = {k: (st.median(v) if v else None) for k, v in vals.items()}

    return med, len(lines)


def decide(m: dict[str, float | None]) -> dict[str, str]:
    """
    Decide optimal coefficients and hyperparameters based on metrics.

    Args:
        m: Median metrics dict

    Returns:
        Recommended settings dict
    """
    rec = {}

    # === P0-6: Quantile crossing penalty ===
    if m["qx_rate"] is not None and m["qx_rate"] > 0.05:
        rec["LAMBDA_QC"] = "5e-3"
        rec["QC_REASON"] = f"qx_rate={m['qx_rate']:.4f} > 0.05 (high crossing rate)"
    else:
        rec["LAMBDA_QC"] = "2e-3"
        rec["QC_REASON"] = f"qx_rate={m['qx_rate']:.4f} <= 0.05 (low crossing rate)" if m["qx_rate"] else "default"

    # === P0-7: Sharpe EMA ===
    rec["SHARPE_EMA_DECAY"] = "0.95"  # Fixed initial value, adjust to 0.92-0.95 if needed

    # === GAT: Gate temperature and edge dropout ===
    tau = 1.25
    edge_do = 0.05
    gm, gs = m["gat_gate_mean"], m["gat_gate_std"]

    if gm is not None:
        if gm < 0.20:       # Gate too closed → Make it easier to open
            tau = 1.6
            rec["GAT_REASON"] = f"gate_mean={gm:.4f} < 0.20 (stuck at base, increasing tau)"
        elif gm > 0.70:     # Gate too open → Reduce dependence
            tau = 1.6
            edge_do = 0.10
            rec["GAT_REASON"] = f"gate_mean={gm:.4f} > 0.70 (stuck at GAT, increasing tau+dropout)"
        else:
            rec["GAT_REASON"] = f"gate_mean={gm:.4f} in healthy range [0.2, 0.7]"

    if gs is not None and (gs < 0.05 or gs > 0.30):
        tau = max(tau, 1.6)  # Insufficient/excessive variance → Flatten with tau
        if not rec.get("GAT_REASON"):
            rec["GAT_REASON"] = f"gate_std={gs:.4f} out of range [0.05, 0.30], increasing tau"

    rec["GAT_TAU"] = str(tau)
    rec["EDGE_DROPOUT"] = str(edge_do)

    # === Graph builder: Connectivity check ===
    if m["deg_avg"] is not None:
        if m["deg_avg"] < 10 or (m["isolates"] is not None and m["isolates"] > 0.02):
            rec["GRAPH_HINT"] = "increase_k_or_lower_threshold"
            rec["GRAPH_REASON"] = f"deg_avg={m['deg_avg']:.2f} < 10 or isolates={m['isolates']:.4f} > 0.02 (too sparse)"
        elif m["deg_avg"] > 40:
            rec["GRAPH_HINT"] = "lower_k_or_raise_threshold"
            rec["GRAPH_REASON"] = f"deg_avg={m['deg_avg']:.2f} > 40 (too dense)"
        else:
            rec["GRAPH_HINT"] = "ok"
            rec["GRAPH_REASON"] = f"deg_avg={m['deg_avg']:.2f} in healthy range [10, 40]"

    # === Gradient balance ===
    gr = m["grad_ratio"]
    if gr is not None:
        if gr < 0.5:    # GAT too strong
            rec["EDGE_DROPOUT"] = "0.10"
            rec["GRAD_REASON"] = f"grad_ratio={gr:.3f} < 0.5 (GAT too strong, increasing dropout)"
        elif gr > 2.0:  # GAT too weak
            rec["GAT_TAU"] = "1.6"  # Make it easier to open
            rec["GRAD_REASON"] = f"grad_ratio={gr:.3f} > 2.0 (GAT too weak, increasing tau)"
        else:
            rec["GRAD_REASON"] = f"grad_ratio={gr:.3f} in healthy range [0.5, 2.0]"

    # === RankIC negative ===
    if m["RankIC"] is not None and m["RankIC"] <= 0.0:
        rec["LOSS_HINT"] = "keep_rankic_weights_0.20_0.15_and_reduce_lr_to_0.7x"
        rec["LOSS_REASON"] = f"RankIC={m['RankIC']:.4f} <= 0 (negative, reduce LR temporarily)"
    else:
        rec["LOSS_HINT"] = "ok"
        rec["LOSS_REASON"] = f"RankIC={m['RankIC']:.4f} > 0 (positive correlation)" if m["RankIC"] else "ok"

    return rec


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/tune_p0467_from_rfi.py rfi_56_metrics.txt")
        sys.exit(1)

    path = sys.argv[1]

    print("=" * 80)
    print("P0-4/6/7 Coefficient Auto-Tuner")
    print("=" * 80)
    print()

    # Parse metrics
    med, n = parse(path)

    print(f"📊 RFI-5/6 Median Metrics (from {path}, {n} lines)")
    print("-" * 80)
    for k, v in med.items():
        if v is not None:
            print(f"  {k:20s}: {v:.6f}")
        else:
            print(f"  {k:20s}: None")
    print()

    # Decide coefficients
    rec = decide(med)

    print("=" * 80)
    print("🎛  Recommended Settings (Copy & Paste)")
    print("=" * 80)
    print()

    # P0-4: Loss Rebalancing (fixed)
    print("# P0-4: Loss Rebalancing (fixed initial values)")
    print("export QUANTILE_WEIGHT=1.0")
    print("export SHARPE_WEIGHT=0.30")
    print("export RANKIC_WEIGHT=0.20")
    print("export CS_IC_WEIGHT=0.15")
    print()

    # P0-6: Quantile Crossing
    print("# P0-6: Quantile Crossing Penalty")
    print(f"export LAMBDA_QC={rec['LAMBDA_QC']}")
    print(f"# Reason: {rec['QC_REASON']}")
    print()

    # P0-7: Sharpe EMA
    print("# P0-7: Sharpe EMA")
    print(f"export SHARPE_EMA_DECAY={rec['SHARPE_EMA_DECAY']}")
    print()

    # GAT: Temperature and dropout
    print("# GAT: Temperature and Edge Dropout")
    print(f"export GAT_TAU={rec['GAT_TAU']}")
    print(f"export EDGE_DROPOUT={rec['EDGE_DROPOUT']}")
    print(f"# Reason: {rec.get('GAT_REASON', 'default')}")
    if "GRAD_REASON" in rec:
        print(f"# Gradient: {rec['GRAD_REASON']}")
    print()

    # Hints
    print("=" * 80)
    print("💡 Additional Hints")
    print("=" * 80)
    print()

    print(f"Graph Builder: {rec.get('GRAPH_HINT', 'ok')}")
    if "GRAPH_REASON" in rec:
        print(f"  → {rec['GRAPH_REASON']}")
    print()

    print(f"Loss Weights: {rec.get('LOSS_HINT', 'ok')}")
    if "LOSS_REASON" in rec:
        print(f"  → {rec['LOSS_REASON']}")
    print()

    # Next steps
    print("=" * 80)
    print("🚀 Next Steps")
    print("=" * 80)
    print()
    print("1. Copy the export commands above")
    print("2. Run short WF:")
    print()
    print("   USE_GAT_SHIM=1 BATCH_SIZE=1024 \\")
    print("   python scripts/train_atft.py --max-epochs 30 \\")
    print("     --data-path output/ml_dataset_latest_full.parquet \\")
    print("     2>&1 | tee _logs/train_p0467_wf3.log")
    print()
    print("3. Check results:")
    print("   - RankIC avg > 0.05")
    print("   - Sharpe > 0.30")
    print("   - qx_rate < 0.05")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
