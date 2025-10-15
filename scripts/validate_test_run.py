#!/usr/bin/env python3
"""
Validate Sharpe optimization test run against gate criteria.

Usage:
    python scripts/validate_test_run.py --log-file /tmp/sharpe_test_run.log
"""

import argparse
import re
import sys
from pathlib import Path


class TestRunValidator:
    """Validate test run against acceptance criteria"""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_content = ""
        self.results = {}
        self.passed = True

    def load_log(self):
        """Load log file"""
        if not self.log_file.exists():
            print(f"❌ Log file not found: {self.log_file}")
            sys.exit(1)

        with open(self.log_file, "r") as f:
            self.log_content = f.read()

    def check_completion(self) -> bool:
        """Gate 1: Training completed successfully within 10 epochs"""
        print("\n" + "=" * 80)
        print("Gate 1: Training Completion")
        print("=" * 80)

        # Check for completion
        if "Training complete" in self.log_content or "Final model saved" in self.log_content:
            print("✅ Training completed successfully")
            return True
        elif "Error" in self.log_content or "Traceback" in self.log_content:
            print("❌ Training failed with errors")
            # Extract error
            error_match = re.search(r"(Error|Exception):.*", self.log_content)
            if error_match:
                print(f"   Error: {error_match.group(0)}")
            return False
        else:
            print("⏳ Training still in progress")
            return None  # Still running

    def check_metrics_logging(self) -> bool:
        """Gate 2: Required metrics are logged (adapted to actual logging patterns)"""
        print("\n" + "=" * 80)
        print("Gate 2: Metrics Logging")
        print("=" * 80)

        # Updated to match actual log patterns from train_atft.py
        required_patterns = {
            "Sharpe": r"Sharpe:\s*[-\d.]+",  # "Val Metrics - Sharpe: X.XXXX"
            "IC": r"IC:\s*[-\d.]+",          # "IC: X.XXXX"
            "RankIC": r"RankIC:\s*[-\d.]+",  # "RankIC: X.XXXX"
        }

        all_present = True
        for metric_name, pattern in required_patterns.items():
            matches = re.findall(pattern, self.log_content)
            if matches:
                print(f"✅ {metric_name} logged ({len(matches)} times)")
            else:
                print(f"❌ {metric_name} NOT logged")
                all_present = False

        return all_present

    def check_prediction_variance(self) -> bool:
        """Gate 3: No prediction collapse (variance maintained)"""
        print("\n" + "=" * 80)
        print("Gate 3: Prediction Variance")
        print("=" * 80)

        # Extract prediction std from logs
        pred_std_pattern = r"pred_std[=:\s]+([\d.]+)"
        pred_stds = [float(m) for m in re.findall(pred_std_pattern, self.log_content)]

        if not pred_stds:
            print("⚠️  No prediction std found in logs")
            return None

        avg_pred_std = sum(pred_stds) / len(pred_stds)
        print(f"   Average prediction std: {avg_pred_std:.4f}")

        # Target std should be > 0.01 (avoid collapse)
        if avg_pred_std > 0.01:
            print(f"✅ Prediction variance maintained (>{0.01:.4f})")
            return True
        else:
            print(f"❌ Prediction collapse detected (std={avg_pred_std:.4f} < 0.01)")
            return False

    def check_feature_dimension_consistency(self) -> bool:
        """Gate 4: Feature dimensions handled properly"""
        print("\n" + "=" * 80)
        print("Gate 4: Feature Dimension Handling")
        print("=" * 80)

        # Check if model handled dimension mismatch gracefully
        rebuild_pattern = r"Rebuilding variable selection network"
        if re.search(rebuild_pattern, self.log_content):
            print("✅ Feature dimension mismatch detected and rebuilt automatically")
            return True

        # Check for FATAL dimension errors (not warnings)
        fatal_patterns = [
            r"RuntimeError.*size mismatch",
            r"ValueError.*dimension mismatch",
        ]

        for pattern in fatal_patterns:
            if re.search(pattern, self.log_content, re.IGNORECASE):
                print("❌ FATAL dimension error detected")
                match = re.search(pattern, self.log_content, re.IGNORECASE)
                print(f"   {match.group(0)}")
                return False

        # Check for successful feature dimension detection
        feature_detect = re.search(r"detected (\d+) feature", self.log_content, re.IGNORECASE)
        if feature_detect:
            print(f"✅ Feature dimensions detected: {feature_detect.group(1)} features")
            return True

        print("⚠️  Cannot verify feature dimensions (no logs)")
        return None

    def check_ic_stability(self) -> bool:
        """Gate 5: IC/RankIC signs are stable"""
        print("\n" + "=" * 80)
        print("Gate 5: IC Stability")
        print("=" * 80)

        # Extract IC values
        ic_pattern = r"Val IC[=:\s]+([-\d.]+)"
        ics = [float(m) for m in re.findall(ic_pattern, self.log_content)]

        if len(ics) < 5:
            print("⚠️  Not enough IC samples for stability check")
            return None

        # Check for sign flips
        positive_ics = sum(1 for ic in ics if ic > 0)
        negative_ics = sum(1 for ic in ics if ic < 0)

        print(f"   Positive ICs: {positive_ics}/{len(ics)}")
        print(f"   Negative ICs: {negative_ics}/{len(ics)}")

        # At least 60% should have same sign
        if positive_ics / len(ics) >= 0.6 or negative_ics / len(ics) >= 0.6:
            print("✅ IC signs are stable")
            return True
        else:
            print("❌ IC signs are unstable (frequent flips)")
            return False

    def check_turnover_reduction(self) -> bool:
        """Gate 6: Turnover is trending downward"""
        print("\n" + "=" * 80)
        print("Gate 6: Turnover Reduction")
        print("=" * 80)

        # Extract turnover values
        turnover_pattern = r"turnover[=:\s]+([\d.]+)"
        turnovers = [float(m) for m in re.findall(turnover_pattern, self.log_content)]

        if len(turnovers) < 3:
            print("⚠️  Not enough turnover samples")
            return None

        # Compare first half vs second half
        mid = len(turnovers) // 2
        early_avg = sum(turnovers[:mid]) / mid
        late_avg = sum(turnovers[mid:]) / (len(turnovers) - mid)

        print(f"   Early turnover avg: {early_avg:.4f}")
        print(f"   Late turnover avg: {late_avg:.4f}")

        # Should see reduction (or at least not increase)
        if late_avg < early_avg:
            reduction = (early_avg - late_avg) / early_avg * 100
            print(f"✅ Turnover reduced by {reduction:.1f}%")
            return True
        elif late_avg <= early_avg * 1.1:  # Allow 10% increase
            print("⚠️  Turnover stable (no significant reduction)")
            return None
        else:
            print(f"❌ Turnover increased (early={early_avg:.4f} → late={late_avg:.4f})")
            return False

    def check_sharpe_trend(self) -> bool:
        """Extra check: Sharpe ratio trend"""
        print("\n" + "=" * 80)
        print("Bonus: Sharpe Ratio Trend")
        print("=" * 80)

        # Extract Sharpe values
        sharpe_pattern = r"val/sharpe[_ratio]*[=:\s]+([-\d.]+)"
        sharpes = [float(m) for m in re.findall(sharpe_pattern, self.log_content)]

        if len(sharpes) < 3:
            print("⚠️  Not enough Sharpe samples")
            return None

        first_sharpe = sharpes[0]
        last_sharpe = sharpes[-1]
        best_sharpe = max(sharpes)

        print(f"   First Sharpe: {first_sharpe:.4f}")
        print(f"   Last Sharpe: {last_sharpe:.4f}")
        print(f"   Best Sharpe: {best_sharpe:.4f}")

        if last_sharpe > first_sharpe:
            improvement = last_sharpe - first_sharpe
            print(f"✅ Sharpe improved by {improvement:+.4f}")
            return True
        else:
            print(f"⚠️  Sharpe declined or flat")
            return None

    def run_all_checks(self):
        """Run all validation checks"""
        print("\n" + "🔍" * 40)
        print("SHARPE OPTIMIZATION TEST RUN VALIDATION")
        print("🔍" * 40)

        self.load_log()

        # Run all gates
        gates = [
            ("Completion", self.check_completion),
            ("Metrics Logging", self.check_metrics_logging),
            ("Prediction Variance", self.check_prediction_variance),
            ("Feature Consistency", self.check_feature_dimension_consistency),
            ("IC Stability", self.check_ic_stability),
            ("Turnover Reduction", self.check_turnover_reduction),
        ]

        results = {}
        for gate_name, check_func in gates:
            results[gate_name] = check_func()

        # Bonus check
        results["Sharpe Trend"] = self.check_sharpe_trend()

        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        passed = 0
        failed = 0
        pending = 0

        for gate_name, result in results.items():
            if result is True:
                status = "✅ PASS"
                passed += 1
            elif result is False:
                status = "❌ FAIL"
                failed += 1
            else:
                status = "⏳ PENDING"
                pending += 1

            print(f"{status:12} {gate_name}")

        print("=" * 80)
        print(f"Total: {passed} passed, {failed} failed, {pending} pending")
        print("=" * 80)

        # Determine overall status
        if failed > 0:
            print("\n❌ TEST RUN FAILED - Fix issues before proceeding to Step B")
            return False
        elif pending > 0:
            print("\n⏳ TEST RUN IN PROGRESS - Wait for completion")
            return None
        else:
            print("\n✅ TEST RUN PASSED - Ready to proceed to Step B (Backtest)")
            return True


def main():
    parser = argparse.ArgumentParser(description="Validate Sharpe optimization test run")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("/tmp/sharpe_test_run.log"),
        help="Path to test run log file",
    )
    args = parser.parse_args()

    validator = TestRunValidator(args.log_file)
    result = validator.run_all_checks()

    if result is False:
        sys.exit(1)
    elif result is None:
        sys.exit(2)  # Still running
    else:
        sys.exit(0)  # Success


if __name__ == "__main__":
    main()
