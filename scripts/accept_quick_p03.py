#!/usr/bin/env python
"""
P0-3 Quick Acceptance Test (Go/No-Go)

Usage:
    python scripts/accept_quick_p03.py rfi_56_metrics.txt

Validates:
- 3 epochs completed (3 RFI56 lines)
- GAT gate statistics (mean, std, deg_avg, isolates)
- Loss metrics (RankIC, WQL, CRPS, qx_rate)
- Gradient ratio (Base/GAT balance)

Exit codes:
    0: PASS - All checks passed
    1: FAIL - Critical failure (< 3 epochs, segfault, etc.)
    2: WARN - Borderline (review manually)
"""
import re
import statistics as st
import sys
from pathlib import Path


def parse_rfi56_file(filepath: str) -> dict[str, list[float]]:
    """Parse RFI56 metrics file and extract all values."""
    if not Path(filepath).exists():
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    text = Path(filepath).read_text().strip().splitlines()

    if len(text) < 3:
        print(f"FAIL: RFI56 lines < 3 (found {len(text)})")
        print("Expected 3 epochs with RFI56 logs")
        sys.exit(1)

    # Initialize value collections
    vals = {
        "gat_gate_mean": [],
        "gat_gate_std": [],
        "deg_avg": [],
        "isolates": [],
        "RankIC": [],
        "WQL": [],
        "CRPS": [],
        "qx_rate": [],
        "grad_ratio": []
    }

    # Extract all metrics
    for ln in text:
        for k in list(vals):
            m = re.search(fr"{k}=([+-]?\d+(\.\d+)?)", ln)
            if m:
                vals[k].append(float(m.group(1)))

    return vals


def median(vals: list[float]) -> float | None:
    """Safe median calculation."""
    return st.median(vals) if vals else None


def check_acceptance(vals: dict[str, list[float]]) -> tuple[bool, list[str]]:
    """
    Run acceptance checks.

    Returns:
        (passed, messages)
    """
    checks = []
    messages = []

    # 1. GAT Gate Statistics
    gate_mean = median(vals["gat_gate_mean"])
    if gate_mean is not None:
        if 0.2 <= gate_mean <= 0.7:
            checks.append(True)
            messages.append(f"✅ GAT gate_mean: {gate_mean:.4f} (healthy range)")
        else:
            checks.append(False)
            messages.append(f"❌ GAT gate_mean: {gate_mean:.4f} (expected 0.2-0.7)")
            if gate_mean < 0.1:
                messages.append("   → GAT being ignored (stuck at 0)")
            elif gate_mean > 0.9:
                messages.append("   → Base being ignored (stuck at 1)")
            messages.append("   → Fix: Increase tau (1.5-2.0)")
    else:
        checks.append(False)
        messages.append("❌ GAT gate_mean: missing")

    # 2. Graph Statistics
    deg_avg = median(vals["deg_avg"])
    isolates = median(vals["isolates"])

    if deg_avg is not None:
        if 10.0 <= deg_avg <= 40.0:
            checks.append(True)
            messages.append(f"✅ Graph deg_avg: {deg_avg:.2f} (healthy connectivity)")
        else:
            checks.append(False)
            messages.append(f"❌ Graph deg_avg: {deg_avg:.2f} (expected 10-40)")
            if deg_avg < 10:
                messages.append("   → Fix: Increase k-NN or lower correlation threshold")
    else:
        checks.append(False)
        messages.append("❌ Graph deg_avg: missing")

    if isolates is not None:
        if isolates < 0.02:
            checks.append(True)
            messages.append(f"✅ Graph isolates: {isolates:.4f} (minimal isolation)")
        else:
            checks.append(False)
            messages.append(f"❌ Graph isolates: {isolates:.4f} (expected < 0.02)")
            messages.append("   → Fix: Adjust GraphBuilder connectivity")
    else:
        checks.append(False)
        messages.append("❌ Graph isolates: missing")

    # 3. RankIC (direction check)
    rank_ic = median(vals["RankIC"])
    if rank_ic is not None:
        if rank_ic > 0.0:
            checks.append(True)
            messages.append(f"✅ RankIC: {rank_ic:.4f} (positive correlation)")
            if rank_ic < 0.02:
                messages.append("   ℹ️  Low but acceptable for initial epochs")
        else:
            checks.append(False)
            messages.append(f"❌ RankIC: {rank_ic:.4f} (expected > 0)")
            messages.append("   → Monitor: Should improve after 10+ epochs")
    else:
        checks.append(False)
        messages.append("❌ RankIC: missing")

    # 4. Gradient Ratio (Base/GAT balance)
    grad_ratio = median(vals["grad_ratio"])
    if grad_ratio is not None:
        if 0.5 <= grad_ratio <= 2.0:
            checks.append(True)
            messages.append(f"✅ Gradient ratio: {grad_ratio:.3f} (balanced)")
        else:
            checks.append(False)
            messages.append(f"❌ Gradient ratio: {grad_ratio:.3f} (expected 0.5-2.0)")
            if grad_ratio < 0.5:
                messages.append("   → GAT gradient too weak")
            else:
                messages.append("   → GAT gradient too strong")
            messages.append("   → Fix: Adjust tau and edge_dropout together")
    else:
        checks.append(False)
        messages.append("❌ Gradient ratio: missing")

    # 5. Quantile Crossing Rate (diagnostic)
    qx_rate = median(vals["qx_rate"])
    if qx_rate is not None:
        if qx_rate < 0.05:
            messages.append(f"✅ Quantile crossing: {qx_rate:.4f} (low)")
        else:
            messages.append(f"⚠️  Quantile crossing: {qx_rate:.4f} (expected < 0.05)")
            messages.append("   → Enable P0-6 penalty with λ=5e-3")

    # 6. Loss trends (diagnostic)
    wql = vals["WQL"]
    crps = vals["CRPS"]
    if len(wql) >= 2:
        if wql[-1] < wql[0]:
            messages.append(f"✅ WQL trend: {wql[0]:.6f} → {wql[-1]:.6f} (improving)")
        else:
            messages.append(f"ℹ️  WQL trend: {wql[0]:.6f} → {wql[-1]:.6f} (monitor)")

    if len(crps) >= 2:
        if crps[-1] < crps[0]:
            messages.append(f"✅ CRPS trend: {crps[0]:.6f} → {crps[-1]:.6f} (improving)")
        else:
            messages.append(f"ℹ️  CRPS trend: {crps[0]:.6f} → {crps[-1]:.6f} (monitor)")

    return all(checks), messages


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/accept_quick_p03.py rfi_56_metrics.txt")
        sys.exit(1)

    filepath = sys.argv[1]

    print("=" * 80)
    print("P0-3 Quick Acceptance Test (Go/No-Go)")
    print("=" * 80)
    print()

    # Parse metrics
    vals = parse_rfi56_file(filepath)

    print(f"📊 Parsed {len([v for vl in vals.values() for v in vl])} metrics from {len(vals['RankIC'])} epochs")
    print()

    # Run checks
    passed, messages = check_acceptance(vals)

    # Print results
    for msg in messages:
        print(msg)

    print()
    print("=" * 80)

    if passed:
        print("✅ PASS: P0-3 Quick Acceptance")
        print()
        print("Next steps:")
        print("1. Enable P0-4/6/7 coefficients")
        print("2. Run short WF validation (3 splits)")
        print("3. Monitor full training (120 epochs)")
        print("=" * 80)
        sys.exit(0)
    else:
        # Check if borderline
        num_failed = sum(1 for msg in messages if msg.startswith("❌"))
        if num_failed <= 2:
            print("⚠️  WARN: Borderline (review manually)")
            print()
            print("Some checks failed, but may be acceptable for initial epochs.")
            print("Review rfi_56_metrics.txt and apply fixes if needed.")
            print("=" * 80)
            sys.exit(2)
        else:
            print("❌ FAIL: Multiple critical checks failed")
            print()
            print("Apply fixes and re-run Quick Run (3 epochs).")
            print("=" * 80)
            sys.exit(1)


if __name__ == "__main__":
    main()
