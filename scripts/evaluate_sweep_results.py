#!/usr/bin/env python3
"""
Evaluate parallel sweep results and select top configurations.

Gate Criteria:
1. pred_std > 0.010 (no variance collapse)
2. Val Sharpe > -0.01 (not terrible)
3. Val RankIC > 0.02 (predictive power)

Usage:
    python scripts/evaluate_sweep_results.py --sweep-dir output/sweep_results
"""

import argparse
import re
from pathlib import Path

import pandas as pd


class SweepEvaluator:
    """Evaluate sweep results and rank configurations"""

    def __init__(self, sweep_dir: Path):
        self.sweep_dir = Path(sweep_dir)
        self.log_dir = self.sweep_dir / "logs"
        self.results = []

    def extract_metrics(self, log_file: Path) -> dict[str, float] | None:
        """Extract final metrics from log file"""
        try:
            with open(log_file) as f:
                content = f.read()

            # Check if training completed
            if "Training complete" not in content and "Final model" not in content:
                return None

            # Extract prediction std (multiple samples, take mean)
            pred_stds = re.findall(r"pred_std[=:\s]+([\d.]+)", content)
            if not pred_stds:
                return None
            avg_pred_std = sum(float(x) for x in pred_stds) / len(pred_stds)

            # Extract final validation metrics
            val_sharpe_pattern = r"Val Metrics.*?Sharpe:\s*([-\d.]+)"
            val_sharpes = re.findall(val_sharpe_pattern, content)
            if not val_sharpes:
                return None

            val_ic_pattern = r"Val Metrics.*?IC:\s*([-\d.]+)"
            val_ics = re.findall(val_ic_pattern, content)

            val_rankic_pattern = r"Val Metrics.*?RankIC:\s*([-\d.]+)"
            val_rankics = re.findall(val_rankic_pattern, content)

            # Take final epoch values
            final_sharpe = float(val_sharpes[-1])
            final_ic = float(val_ics[-1]) if val_ics else 0.0
            final_rankic = float(val_rankics[-1]) if val_rankics else 0.0

            # Average across all epochs for stability estimate
            avg_sharpe = sum(float(x) for x in val_sharpes) / len(val_sharpes)
            avg_ic = sum(float(x) for x in val_ics) / len(val_ics) if val_ics else 0.0
            avg_rankic = (
                sum(float(x) for x in val_rankics) / len(val_rankics)
                if val_rankics
                else 0.0
            )

            return {
                "pred_std": avg_pred_std,
                "final_sharpe": final_sharpe,
                "final_ic": final_ic,
                "final_rankic": final_rankic,
                "avg_sharpe": avg_sharpe,
                "avg_ic": avg_ic,
                "avg_rankic": avg_rankic,
            }

        except Exception as e:
            print(f"❌ Error parsing {log_file.name}: {e}")
            return None

    def load_config_meta(self, meta_file: Path) -> dict[str, str]:
        """Load configuration metadata"""
        config = {}
        if meta_file.exists():
            with open(meta_file) as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        config[key] = value
        return config

    def evaluate_all(self) -> pd.DataFrame:
        """Evaluate all sweep runs"""
        log_files = sorted(self.log_dir.glob("*.log"))

        print(f"Found {len(log_files)} log files")

        for log_file in log_files:
            config_id = log_file.stem
            meta_file = self.log_dir / f"{config_id}.meta"

            # Extract metrics
            metrics = self.extract_metrics(log_file)
            if metrics is None:
                print(f"⏭️  Skipping {config_id} (incomplete or failed)")
                continue

            # Load config
            config = self.load_config_meta(meta_file)

            # Apply gate criteria
            gate_pass = (
                metrics["pred_std"] > 0.010
                and metrics["final_sharpe"] > -0.01
                and metrics["final_rankic"] > 0.02
            )

            result = {
                "config_id": config_id,
                "gate_pass": gate_pass,
                **metrics,
                **config,
            }

            self.results.append(result)

            # Print status
            status = "✅" if gate_pass else "❌"
            print(
                f"{status} {config_id}: pred_std={metrics['pred_std']:.4f}, "
                f"sharpe={metrics['final_sharpe']:.4f}, rankic={metrics['final_rankic']:.4f}"
            )

        df = pd.DataFrame(self.results)
        return df

    def rank_configs(self, df: pd.DataFrame, top_k: int = 4) -> pd.DataFrame:
        """Rank configurations by composite score"""
        # Filter by gate criteria
        passed = df[df["gate_pass"]].copy()

        if len(passed) == 0:
            print("\n❌ No configurations passed gate criteria!")
            return pd.DataFrame()

        # Composite score (normalized and weighted)
        # Higher is better for all metrics after normalization
        passed["pred_std_norm"] = (
            passed["pred_std"] / passed["pred_std"].max()
        )  # Higher = better
        passed["sharpe_norm"] = (
            passed["avg_sharpe"] - passed["avg_sharpe"].min()
        ) / (passed["avg_sharpe"].max() - passed["avg_sharpe"].min() + 1e-8)
        passed["rankic_norm"] = (
            passed["avg_rankic"] - passed["avg_rankic"].min()
        ) / (passed["avg_rankic"].max() - passed["avg_rankic"].min() + 1e-8)

        # Composite score (weights can be adjusted)
        passed["score"] = (
            0.3 * passed["pred_std_norm"]  # Avoid collapse (30%)
            + 0.4 * passed["sharpe_norm"]  # Profitability (40%)
            + 0.3 * passed["rankic_norm"]  # Predictive power (30%)
        )

        # Sort by score
        ranked = passed.sort_values("score", ascending=False).head(top_k)

        return ranked

    def save_results(self, df: pd.DataFrame, ranked: pd.DataFrame):
        """Save evaluation results"""
        # Save full results
        df.to_csv(self.sweep_dir / "all_results.csv", index=False)
        print(f"\n✅ Saved all results to {self.sweep_dir}/all_results.csv")

        # Save top configs
        if len(ranked) > 0:
            ranked.to_csv(self.sweep_dir / "top_configs.csv", index=False)
            print(f"✅ Saved top configs to {self.sweep_dir}/top_configs.csv")

            # Save top config IDs for easy shell access
            with open(self.sweep_dir / "top_config_ids.txt", "w") as f:
                for config_id in ranked["config_id"]:
                    f.write(f"{config_id}\n")
            print(f"✅ Saved top config IDs to {self.sweep_dir}/top_config_ids.txt")

    def print_summary(self, df: pd.DataFrame, ranked: pd.DataFrame):
        """Print evaluation summary"""
        print("\n" + "=" * 80)
        print("SWEEP EVALUATION SUMMARY")
        print("=" * 80)

        total = len(df)
        passed = len(df[df["gate_pass"]])
        failed = total - passed

        print(f"Total configurations: {total}")
        print(f"Passed gate criteria: {passed}")
        print(f"Failed gate criteria: {failed}")

        if len(ranked) > 0:
            print("\n" + "=" * 80)
            print(f"TOP {len(ranked)} CONFIGURATIONS (by composite score)")
            print("=" * 80)

            for _idx, row in ranked.iterrows():
                print(f"\n{row['config_id']}:")
                print(f"  Score:       {row['score']:.4f}")
                print(f"  pred_std:    {row['pred_std']:.4f} (> 0.010 ✅)")
                print(f"  Final Sharpe: {row['final_sharpe']:.4f}")
                print(f"  Avg Sharpe:   {row['avg_sharpe']:.4f}")
                print(f"  Final RankIC: {row['final_rankic']:.4f}")
                print(f"  Avg RankIC:   {row['avg_rankic']:.4f}")
                print("  Config:")
                print(f"    TURNOVER_WEIGHT={row.get('TURNOVER_WEIGHT', 'N/A')}")
                print(f"    PRED_VAR_WEIGHT={row.get('PRED_VAR_WEIGHT', 'N/A')}")
                print(f"    OUTPUT_NOISE_STD={row.get('OUTPUT_NOISE_STD', 'N/A')}")
                print(f"    RANKIC_WEIGHT={row.get('RANKIC_WEIGHT', 'N/A')}")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate sweep results")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("output/sweep_results"),
        help="Sweep results directory",
    )
    parser.add_argument(
        "--top-k", type=int, default=4, help="Number of top configs to select"
    )
    args = parser.parse_args()

    evaluator = SweepEvaluator(args.sweep_dir)

    print("Evaluating sweep results...")
    df = evaluator.evaluate_all()

    if len(df) == 0:
        print("❌ No results found!")
        return 1

    print("\nRanking configurations...")
    ranked = evaluator.rank_configs(df, top_k=args.top_k)

    evaluator.save_results(df, ranked)
    evaluator.print_summary(df, ranked)

    if len(ranked) == 0:
        print("\n❌ No configurations passed gate criteria!")
        print("Suggestion: Relax gate thresholds or adjust sweep parameters")
        return 1

    print("\n✅ Evaluation complete!")
    print(
        "\nNext step: Run 10-epoch validation on top config(s) or proceed to full training"
    )
    print(f"  bash scripts/run_best_config.sh {args.sweep_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
