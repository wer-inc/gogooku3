#!/usr/bin/env python3
"""
Evaluate a trained model by parsing training logs and generating a report.

Features
- Phase-wise aggregation (Baseline, Adaptive Norm, GAT, Fine-tuning)
- Epoch-level metric parsing (loss, LR, Sharpe, IC, RankIC, HitRate)
- Summary statistics (mean, std, min/max, best epochs)
- Improvement calculations and simple health checks
- Auto-generate docs/EVALUATION_REPORT.md

Usage
  python scripts/evaluate_trained_model.py \
      --log-file logs/production_ic_fixed_20251015_041033.log \
      --out-markdown docs/EVALUATION_REPORT.md

Notes
- Python 3.10+
- No external dependencies required
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

PHASE_PATTERN = re.compile(r"^.* - Phase (\d+):\s*(.+?)\s*$")
EPOCH_PATTERN = re.compile(
    r"^.* - Epoch\s+(\d+)/(\d+):\s+Train Loss=([\d.]+),\s+Val Loss=([\d.]+),\s+LR=([\d.eE+-]+)\s*$"
)
TRAIN_METRICS_PATTERN = re.compile(
    r"^.*Train Metrics\s+-\s+Sharpe:\s*([-\d.]+),\s*IC:\s*([-\d.eE]+),\s*RankIC:\s*([-\d.eE]+)\s*$"
)
VAL_METRICS_PATTERN = re.compile(
    r"^.*Val Metrics\s+-\s+Sharpe:\s*([-\d.]+),\s*IC:\s*([-\d.eE]+),\s*RankIC:\s*([-\d.eE]+),\s*HitRate\(h1\):\s*([-\d.eE]+)\s*$"
)
SAVED_BEST_PATTERN = re.compile(
    r"^.*Saved best model \(val_loss=([\d.]+),\s*val_loss=([\d.]+)\)\s*$"
)


@dataclass
class EpochMetrics:
    epoch: int
    epochs_total: int
    train_loss: float
    val_loss: float
    lr: float
    train_sharpe: float | None = None
    train_ic: float | None = None
    train_rank_ic: float | None = None
    val_sharpe: float | None = None
    val_ic: float | None = None
    val_rank_ic: float | None = None
    val_hitrate: float | None = None


@dataclass
class Phase:
    idx: int
    name: str
    epochs: list[EpochMetrics] = dataclasses.field(default_factory=list)

    def aggregate(self) -> dict[str, float | int | None]:
        def safe_mean(vals: list[float]) -> float | None:
            vals = [v for v in vals if v is not None and not math.isnan(v)]
            return float(statistics.fmean(vals)) if vals else None

        def safe_std(vals: list[float]) -> float | None:
            vals = [v for v in vals if v is not None and not math.isnan(v)]
            if len(vals) <= 1:
                return None
            return float(statistics.pstdev(vals))

        def safe_min(vals: list[float]) -> float | None:
            vals = [v for v in vals if v is not None and not math.isnan(v)]
            return float(min(vals)) if vals else None

        def safe_max(vals: list[float]) -> float | None:
            vals = [v for v in vals if v is not None and not math.isnan(v)]
            return float(max(vals)) if vals else None

        val_ic = [e.val_ic for e in self.epochs if e.val_ic is not None]
        val_rank_ic = [e.val_rank_ic for e in self.epochs if e.val_rank_ic is not None]
        val_sharpe = [e.val_sharpe for e in self.epochs if e.val_sharpe is not None]
        val_loss = [e.val_loss for e in self.epochs]

        return {
            "phase": self.name,
            "phase_idx": self.idx,
            "epochs": len(self.epochs),
            "val_ic_mean": safe_mean(val_ic),
            "val_ic_std": safe_std(val_ic),
            "val_ic_min": safe_min(val_ic),
            "val_ic_max": safe_max(val_ic),
            "val_rank_ic_mean": safe_mean(val_rank_ic),
            "val_rank_ic_std": safe_std(val_rank_ic),
            "val_sharpe_mean": safe_mean(val_sharpe),
            "val_sharpe_std": safe_std(val_sharpe),
            "val_loss_mean": safe_mean(val_loss),
            "val_loss_min": safe_min(val_loss),
        }


def parse_log(path: Path) -> tuple[dict[int, Phase], list[tuple[float, float]]]:
    phases: dict[int, Phase] = {}
    current_phase: Phase | None = None
    best_saves: list[tuple[float, float]] = []  # (val_loss reported 1, reported 2)

    # Temporary storage for epoch metrics lines (epoch line followed by metrics
    # lines). We attach metrics to the last seen epoch record.
    last_epoch: EpochMetrics | None = None

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            # Phase
            m = PHASE_PATTERN.match(line)
            if m:
                idx = int(m.group(1))
                name = m.group(2).strip()
                current_phase = phases.setdefault(idx, Phase(idx=idx, name=name))
                continue

            # Epoch line
            m = EPOCH_PATTERN.match(line)
            if m:
                epoch = int(m.group(1))
                epochs_total = int(m.group(2))
                train_loss = float(m.group(3))
                val_loss = float(m.group(4))
                lr = float(m.group(5))
                last_epoch = EpochMetrics(
                    epoch=epoch,
                    epochs_total=epochs_total,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    lr=lr,
                )
                if current_phase is None:
                    # If phase wasn't explicitly set yet, assign to -1 "Unknown"
                    current_phase = phases.setdefault(-1, Phase(idx=-1, name="Unknown"))
                current_phase.epochs.append(last_epoch)
                continue

            # Train metrics
            m = TRAIN_METRICS_PATTERN.match(line)
            if m and last_epoch is not None:
                last_epoch.train_sharpe = float(m.group(1))
                last_epoch.train_ic = float(m.group(2))
                last_epoch.train_rank_ic = float(m.group(3))
                continue

            # Val metrics
            m = VAL_METRICS_PATTERN.match(line)
            if m and last_epoch is not None:
                last_epoch.val_sharpe = float(m.group(1))
                last_epoch.val_ic = float(m.group(2))
                last_epoch.val_rank_ic = float(m.group(3))
                last_epoch.val_hitrate = float(m.group(4))
                continue

            # Best saves
            m = SAVED_BEST_PATTERN.match(line)
            if m:
                try:
                    best_saves.append((float(m.group(1)), float(m.group(2))))
                except Exception:
                    pass

    return phases, best_saves


def percent_improvement(old: float | None, new: float | None) -> float | None:
    if old is None or new is None:
        return None
    if old == 0:
        return None
    try:
        return (new - old) / abs(old) * 100.0
    except ZeroDivisionError:
        return None


def format_float(x: float | None, digits: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def write_markdown(
    out_path: Path,
    phases: dict[int, Phase],
    best_saves: list[tuple[float, float]],
    target_ic: float = 0.04,
    target_rank_ic: float = 0.05,
) -> None:
    # Order phases by idx
    ordered = [phases[k] for k in sorted(phases.keys())]

    # Aggregates
    aggs = [p.aggregate() for p in ordered]

    # Compute key deltas
    agg_by_idx = {a["phase_idx"]: a for a in aggs}
    ic_phase1 = agg_by_idx.get(1, {}).get("val_ic_mean")
    ic_phase3 = agg_by_idx.get(3, {}).get("val_ic_mean")
    ic_improve_p13 = percent_improvement(ic_phase1, ic_phase3)

    # Overall IC volatility (phase 0..3 combined std over epoch ICs)
    all_ics: list[float] = []
    for p in ordered:
        for e in p.epochs:
            if e.val_ic is not None and not math.isnan(e.val_ic):
                all_ics.append(e.val_ic)
    ic_volatility = float(statistics.pstdev(all_ics)) if len(all_ics) > 1 else None

    # Overall averages (simple across all epochs)
    def overall_mean(getter) -> float | None:
        vals = []
        for p in ordered:
            for e in p.epochs:
                v = getter(e)
                if v is not None and not math.isnan(v):
                    vals.append(v)
        return float(statistics.fmean(vals)) if vals else None

    overall_ic = overall_mean(lambda e: e.val_ic)
    overall_rank_ic = overall_mean(lambda e: e.val_rank_ic)
    overall_sharpe = overall_mean(lambda e: e.val_sharpe)

    # Best observed metrics across all epochs
    best_ic = max((e.val_ic for p in ordered for e in p.epochs if e.val_ic is not None), default=None)
    best_rank_ic = max(
        (e.val_rank_ic for p in ordered for e in p.epochs if e.val_rank_ic is not None),
        default=None,
    )
    best_sharpe = max(
        (e.val_sharpe for p in ordered for e in p.epochs if e.val_sharpe is not None),
        default=None,
    )

    # Determine zero-IC incidence (exact zeros or near-zero threshold)
    near_zero_threshold = 1e-6
    zero_like = sum(1 for v in all_ics if abs(v) <= near_zero_threshold)
    zero_like_ratio = zero_like / len(all_ics) if all_ics else 0.0

    lines: list[str] = []
    lines.append("# Model Evaluation Report")
    lines.append("")
    lines.append("Generated by scripts/evaluate_trained_model.py")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"- Overall IC (mean): {format_float(overall_ic, 4)}")
    lines.append(f"- Overall RankIC (mean): {format_float(overall_rank_ic, 4)}")
    lines.append(f"- Overall Sharpe (mean): {format_float(overall_sharpe, 4)}")
    lines.append(
        f"- IC Volatility (std): {format_float(ic_volatility, 4)}"
    )
    if best_ic is not None:
        achieve = (best_ic / target_ic * 100.0) if target_ic else None
        lines.append(
            f"- Best IC: {format_float(best_ic, 4)}"
            + (f" ({format_float(achieve,2)}% of target {target_ic})" if achieve is not None else "")
        )
    if best_rank_ic is not None:
        achieve = (best_rank_ic / target_rank_ic * 100.0) if target_rank_ic else None
        lines.append(
            f"- Best RankIC: {format_float(best_rank_ic, 4)}"
            + (f" ({format_float(achieve,2)}% of target {target_rank_ic})" if achieve is not None else "")
        )
    if best_sharpe is not None:
        lines.append(f"- Best Sharpe: {format_float(best_sharpe, 4)}")
    if ic_improve_p13 is not None:
        lines.append(
            f"- IC Improvement (Phase 1 -> 3): {format_float(ic_improve_p13, 2)}%"
        )
    lines.append(
        f"- IC=0 Incidence (near-zero ratio): {format_float(zero_like_ratio*100, 2)}%"
    )
    lines.append("")

    lines.append("## Phase-by-Phase Analysis")
    lines.append("")
    for a in aggs:
        lines.append(f"### Phase {a['phase_idx']}: {a['phase']}")
        lines.append("")
        lines.append(f"- Epochs: {a['epochs']}")
        lines.append(
            f"- Val IC mean: {format_float(a['val_ic_mean'])} (std: {format_float(a['val_ic_std'])})"
        )
        lines.append(
            f"- Val RankIC mean: {format_float(a['val_rank_ic_mean'])}"
            f" (std: {format_float(a['val_rank_ic_std'])})"
        )
        lines.append(
            f"- Val Sharpe mean: {format_float(a['val_sharpe_mean'])}"
            f" (std: {format_float(a['val_sharpe_std'])})"
        )
        lines.append(
            f"- Val Loss mean: {format_float(a['val_loss_mean'])},"
            f" min: {format_float(a['val_loss_min'])}"
        )
        lines.append("")

    lines.append("## Detected Issues")
    lines.append("")
    # Heuristics for issues
    if (overall_sharpe or 0.0) < 0:
        lines.append("- Sharpe remains negative across phases.")
    if ic_volatility and ic_volatility > 0.02:
        lines.append("- IC volatility is elevated (> 0.02).")
    if zero_like_ratio < 1e-3:
        lines.append("- Prior IC=0 issue appears resolved (near-zero ratio ~0%).")
    lines.append("")

    # Peak improvement (by best IC of phase)
    peak_phase1 = None
    peak_phase3 = None
    if 1 in phases:
        peak_phase1 = max((e.val_ic for e in phases[1].epochs if e.val_ic is not None), default=None)
    if 3 in phases:
        peak_phase3 = max((e.val_ic for e in phases[3].epochs if e.val_ic is not None), default=None)
    peak_improve = percent_improvement(peak_phase1, peak_phase3)
    if peak_improve is not None:
        lines.append("")
        lines.append("## Peak IC Comparison")
        lines.append(
            f"- Phase 1 peak IC: {format_float(peak_phase1)} -> Phase 3 peak IC: {format_float(peak_phase3)}"
        )
        lines.append(f"- Peak IC improvement (P1 -> P3): {format_float(peak_improve,2)}%")

    lines.append("")
    lines.append("## Recommendations")
    lines.append("")
    lines.append("- Prioritize Sharpe-oriented improvements (costs, loss shaping, sizing).")
    lines.append("- Consider Rank/IC-augmented objectives to align with targets.")
    lines.append("- Stabilize fine-tuning with lower LR and partial freezing.")
    lines.append("- Ensure feature config and model dims are aligned to avoid rebuilds.")
    lines.append("")

    if best_saves:
        lines.append("## Best Checkpoints")
        for i, (v1, _v2) in enumerate(best_saves, start=1):
            lines.append(f"- Save {i}: val_loss={format_float(v1)}")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/production_ic_fixed_20251015_041033.log"),
        help="Path to training log file",
    )
    parser.add_argument(
        "--out-markdown",
        type=Path,
        default=Path("docs/EVALUATION_REPORT.md"),
        help="Path to output markdown report",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print JSON summary to stdout",
    )
    args = parser.parse_args()

    phases, best_saves = parse_log(args.log_file)
    # Compose JSON summary
    summary = {
        "phases": {idx: ph.aggregate() for idx, ph in phases.items()},
        "best_saves": best_saves,
    }

    write_markdown(args.out_markdown, phases, best_saves)

    if args.print_json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
