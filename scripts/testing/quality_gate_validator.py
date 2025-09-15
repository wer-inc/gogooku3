#!/usr/bin/env python3
"""
Phase 3: Quality Gate Validator
Comprehensive quality gate validation for CI/CD integration
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse


class QualityGateValidator:
    """Quality gate validation for pipeline CI/CD integration"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.validation_results = {}
        self.quality_thresholds = {
            "missing_data_max": 0.05,  # Max 5% missing data
            "outlier_ratio_max": 0.02,  # Max 2% outliers
            "execution_time_max": 300,  # Max 5 minutes for test pipeline
            "memory_usage_max": 8.0,    # Max 8GB memory usage
            "vus_pr_min": 0.05,         # Min VUS-PR score
            "consistency_match_min": 0.98  # Min 98% data consistency
        }

    def validate_data_quality(self, output_dir: Path) -> Dict[str, Any]:
        """Validate data quality metrics against thresholds"""
        print("🔍 Validating data quality...")

        quality_file = output_dir / "data_quality_summary.json"
        if not quality_file.exists():
            return {
                "status": "FAIL",
                "reason": "Quality summary file not found",
                "quality_file_exists": False
            }

        with open(quality_file, 'r') as f:
            quality_data = json.load(f)

        validations = {}

        # Check missing data ratio
        missing_ratio = quality_data.get("missing_data", {}).get("overall_missing_ratio", 0)
        validations["missing_data"] = {
            "value": missing_ratio,
            "threshold": self.quality_thresholds["missing_data_max"],
            "pass": missing_ratio <= self.quality_thresholds["missing_data_max"]
        }

        # Check outlier ratio
        outlier_ratio = quality_data.get("outliers", {}).get("outlier_ratio", 0)
        validations["outlier_ratio"] = {
            "value": outlier_ratio,
            "threshold": self.quality_thresholds["outlier_ratio_max"],
            "pass": outlier_ratio <= self.quality_thresholds["outlier_ratio_max"]
        }

        # Overall data quality status
        all_quality_checks_pass = all(check["pass"] for check in validations.values())

        return {
            "status": "PASS" if all_quality_checks_pass else "FAIL",
            "validations": validations,
            "quality_file_exists": True,
            "overall_pass": all_quality_checks_pass
        }

    def validate_performance(self, output_dir: Path) -> Dict[str, Any]:
        """Validate performance metrics against thresholds"""
        print("⚡ Validating performance metrics...")

        profiling_file = output_dir / "profiling_report.json"
        if not profiling_file.exists():
            return {
                "status": "FAIL",
                "reason": "Profiling report not found",
                "profiling_file_exists": False
            }

        with open(profiling_file, 'r') as f:
            profiling_data = json.load(f)

        validations = {}

        # Check execution time
        total_time = sum(profiling_data.get("execution_times", {}).values())
        validations["execution_time"] = {
            "value": total_time,
            "threshold": self.quality_thresholds["execution_time_max"],
            "pass": total_time <= self.quality_thresholds["execution_time_max"]
        }

        # Check memory usage
        peak_memory = max(profiling_data.get("memory_usage", {}).values()) if profiling_data.get("memory_usage") else 0
        validations["memory_usage"] = {
            "value": peak_memory,
            "threshold": self.quality_thresholds["memory_usage_max"],
            "pass": peak_memory <= self.quality_thresholds["memory_usage_max"]
        }

        all_performance_checks_pass = all(check["pass"] for check in validations.values())

        return {
            "status": "PASS" if all_performance_checks_pass else "FAIL",
            "validations": validations,
            "profiling_file_exists": True,
            "overall_pass": all_performance_checks_pass
        }

    def validate_lineage_integrity(self, output_dir: Path) -> Dict[str, Any]:
        """Validate data lineage integrity"""
        print("📊 Validating lineage integrity...")

        lineage_file = output_dir / "lineage.jsonl"
        if not lineage_file.exists():
            return {
                "status": "FAIL",
                "reason": "Lineage file not found",
                "lineage_file_exists": False
            }

        # Read lineage entries
        lineage_entries = []
        with open(lineage_file, 'r') as f:
            for line in f:
                if line.strip():
                    lineage_entries.append(json.loads(line))

        # Expected transformation chain
        expected_transformations = ["base_pipeline_run", "enrich_and_save"]
        found_transformations = [entry["transformation"] for entry in lineage_entries]

        validations = {
            "lineage_entries_count": len(lineage_entries),
            "expected_transformations": expected_transformations,
            "found_transformations": found_transformations,
            "has_complete_chain": all(t in found_transformations for t in expected_transformations)
        }

        return {
            "status": "PASS" if validations["has_complete_chain"] else "FAIL",
            "validations": validations,
            "lineage_file_exists": True,
            "overall_pass": validations["has_complete_chain"]
        }

    def validate_output_artifacts(self, output_dir: Path) -> Dict[str, Any]:
        """Validate required output artifacts exist"""
        print("📦 Validating output artifacts...")

        required_artifacts = [
            "ml_dataset_latest_full.parquet",
            "ml_dataset_latest_full_metadata.json",
            "data_quality_summary.json",
            "data_quality_report.md",
            "profiling_report.json",
            "lineage.jsonl"
        ]

        artifact_status = {}
        for artifact in required_artifacts:
            artifact_path = output_dir / artifact
            artifact_status[artifact] = {
                "exists": artifact_path.exists(),
                "size_bytes": artifact_path.stat().st_size if artifact_path.exists() else 0
            }

        all_artifacts_exist = all(status["exists"] for status in artifact_status.values())

        return {
            "status": "PASS" if all_artifacts_exist else "FAIL",
            "artifact_status": artifact_status,
            "overall_pass": all_artifacts_exist
        }

    def validate_incremental_consistency(self, incremental_dir: Path, full_dir: Path) -> Dict[str, Any]:
        """Validate consistency between incremental and full pipeline results"""
        print("🔄 Validating incremental consistency...")

        inc_parquet = incremental_dir / "ml_dataset_latest_full.parquet"
        full_parquet = full_dir / "ml_dataset_latest_full.parquet"

        if not inc_parquet.exists() or not full_parquet.exists():
            return {
                "status": "FAIL",
                "reason": "Required parquet files missing for consistency check",
                "incremental_exists": inc_parquet.exists(),
                "full_exists": full_parquet.exists()
            }

        try:
            import polars as pl

            df_inc = pl.read_parquet(inc_parquet)
            df_full = pl.read_parquet(full_parquet)

            # Basic consistency checks
            record_count_match = len(df_inc) == len(df_full)
            date_range_inc = (df_inc["date"].min(), df_inc["date"].max())
            date_range_full = (df_full["date"].min(), df_full["date"].max())
            date_range_match = date_range_inc == date_range_full

            # Sample comparison (first 100 records)
            sample_size = min(100, len(df_inc), len(df_full))
            sample_inc = df_inc.head(sample_size).sort(["date", "code"])
            sample_full = df_full.head(sample_size).sort(["date", "code"])

            # Compare key columns
            key_columns = ["date", "code", "close", "volume"] if all(col in df_inc.columns for col in ["date", "code", "close", "volume"]) else ["date", "code"]
            sample_match = sample_inc.select(key_columns).equals(sample_full.select(key_columns))

            consistency_score = sum([record_count_match, date_range_match, sample_match]) / 3

            validations = {
                "record_count_match": record_count_match,
                "date_range_match": date_range_match,
                "sample_data_match": sample_match,
                "consistency_score": consistency_score,
                "incremental_records": len(df_inc),
                "full_records": len(df_full)
            }

            consistency_pass = consistency_score >= self.quality_thresholds["consistency_match_min"]

            return {
                "status": "PASS" if consistency_pass else "FAIL",
                "validations": validations,
                "overall_pass": consistency_pass
            }

        except Exception as e:
            return {
                "status": "FAIL",
                "reason": f"Consistency check failed: {str(e)}",
                "error": str(e)
            }

    def run_pipeline_for_validation(self, config_overrides: Dict[str, str] = None) -> Path:
        """Run pipeline for validation with specified configuration"""
        test_dir = self.base_dir / "quality_gate_test"
        test_dir.mkdir(exist_ok=True)

        cmd = [
            "python", "scripts/pipelines/run_full_dataset_hydra.py",
            f"pipeline.output_dir={test_dir}",
            "pipeline.start_date=2025-09-01",
            "pipeline.end_date=2025-09-05",  # Small date range for fast testing
            "pipeline.jquants=false",  # Use offline mode
            "pipeline.quality_checks.enabled=true",
            "pipeline.resilience.enabled=true",
            "pipeline.profiling.enabled=true"
        ]

        if config_overrides:
            for key, value in config_overrides.items():
                cmd.append(f"{key}={value}")

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)

        if result.returncode != 0:
            raise RuntimeError(f"Pipeline execution failed: {result.stderr}")

        return test_dir

    def run_comprehensive_quality_gate(self) -> Dict[str, Any]:
        """Run comprehensive quality gate validation"""
        print("🚀 Starting Phase 3 Quality Gate Validation...")

        # Step 1: Run pipeline for validation
        print("1️⃣ Running pipeline for validation...")
        try:
            test_output_dir = self.run_pipeline_for_validation()
            pipeline_execution = {"status": "SUCCESS", "output_dir": str(test_output_dir)}
        except Exception as e:
            pipeline_execution = {"status": "FAIL", "error": str(e)}
            return {
                "overall_status": "FAIL",
                "pipeline_execution": pipeline_execution,
                "timestamp": datetime.now().isoformat()
            }

        # Step 2: Validate individual components
        print("2️⃣ Running quality validations...")
        validations = {
            "data_quality": self.validate_data_quality(test_output_dir),
            "performance": self.validate_performance(test_output_dir),
            "lineage_integrity": self.validate_lineage_integrity(test_output_dir),
            "output_artifacts": self.validate_output_artifacts(test_output_dir)
        }

        # Step 3: Optional incremental consistency check
        print("3️⃣ Running incremental consistency check...")
        try:
            # Run incremental pipeline
            inc_test_dir = self.base_dir / "quality_gate_incremental"
            inc_test_dir.mkdir(exist_ok=True)

            # Create base
            base_cmd = [
                "python", "scripts/pipelines/run_full_dataset_hydra.py",
                f"pipeline.output_dir={inc_test_dir}",
                "pipeline.start_date=2025-09-01",
                "pipeline.end_date=2025-09-03",
                "pipeline.jquants=false"
            ]
            subprocess.run(base_cmd, capture_output=True, text=True, cwd=self.base_dir)

            # Run incremental
            inc_cmd = [
                "python", "scripts/pipelines/run_full_dataset_hydra.py",
                "pipeline.update_mode=incremental",
                "pipeline.since_date=2025-09-04",
                f"pipeline.output_dir={inc_test_dir}",
                "pipeline.jquants=false"
            ]
            subprocess.run(inc_cmd, capture_output=True, text=True, cwd=self.base_dir)

            # Run full for comparison
            full_test_dir = self.base_dir / "quality_gate_full_comparison"
            full_test_dir.mkdir(exist_ok=True)
            full_cmd = [
                "python", "scripts/pipelines/run_full_dataset_hydra.py",
                f"pipeline.output_dir={full_test_dir}",
                "pipeline.start_date=2025-09-01",
                "pipeline.end_date=2025-09-05",
                "pipeline.jquants=false"
            ]
            subprocess.run(full_cmd, capture_output=True, text=True, cwd=self.base_dir)

            validations["incremental_consistency"] = self.validate_incremental_consistency(
                inc_test_dir, full_test_dir
            )

        except Exception as e:
            validations["incremental_consistency"] = {
                "status": "FAIL",
                "error": str(e),
                "reason": "Failed to run incremental consistency check"
            }

        # Step 4: Determine overall status
        all_validations_pass = all(
            validation.get("overall_pass", validation.get("status") == "PASS")
            for validation in validations.values()
        )

        overall_status = "PASS" if all_validations_pass else "FAIL"

        return {
            "overall_status": overall_status,
            "pipeline_execution": pipeline_execution,
            "validations": validations,
            "quality_thresholds": self.quality_thresholds,
            "timestamp": datetime.now().isoformat(),
            "summary": self.generate_validation_summary(validations, overall_status)
        }

    def generate_validation_summary(self, validations: Dict[str, Any], overall_status: str) -> Dict[str, Any]:
        """Generate validation summary"""
        summary = {
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "check_details": []
        }

        for check_name, check_result in validations.items():
            summary["total_checks"] += 1
            check_passed = check_result.get("overall_pass", check_result.get("status") == "PASS")

            if check_passed:
                summary["passed_checks"] += 1
            else:
                summary["failed_checks"] += 1

            summary["check_details"].append({
                "check": check_name,
                "status": "PASS" if check_passed else "FAIL",
                "reason": check_result.get("reason", "")
            })

        summary["pass_rate"] = summary["passed_checks"] / summary["total_checks"] if summary["total_checks"] > 0 else 0

        return summary

    def generate_quality_gate_report(self, results: Dict[str, Any]) -> str:
        """Generate quality gate report"""
        report = []
        report.append("# Phase 3 Quality Gate Validation Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")

        # Overall status
        status_emoji = "✅" if results["overall_status"] == "PASS" else "❌"
        report.append(f"## Overall Status: {status_emoji} {results['overall_status']}")
        report.append("")

        # Summary
        if "summary" in results:
            summary = results["summary"]
            report.append("## Summary")
            report.append(f"- **Total Checks**: {summary['total_checks']}")
            report.append(f"- **Passed**: {summary['passed_checks']}")
            report.append(f"- **Failed**: {summary['failed_checks']}")
            report.append(f"- **Pass Rate**: {summary['pass_rate']:.1%}")
            report.append("")

        # Detailed results
        report.append("## Detailed Results")

        for check_name, check_result in results.get("validations", {}).items():
            status = check_result.get("overall_pass", check_result.get("status") == "PASS")
            status_emoji = "✅" if status else "❌"
            report.append(f"### {status_emoji} {check_name.replace('_', ' ').title()}")

            if "validations" in check_result:
                for key, value in check_result["validations"].items():
                    if isinstance(value, dict) and "pass" in value:
                        sub_status_emoji = "✅" if value["pass"] else "❌"
                        report.append(f"- {sub_status_emoji} {key}: {value['value']} (threshold: {value['threshold']})")
                    else:
                        report.append(f"- {key}: {value}")

            if check_result.get("reason"):
                report.append(f"- **Reason**: {check_result['reason']}")

            report.append("")

        return "\n".join(report)

    def save_results(self, results: Dict[str, Any], output_file: Path):
        """Save validation results"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save report
        report_file = output_file.with_suffix('.md')
        with open(report_file, 'w') as f:
            f.write(self.generate_quality_gate_report(results))


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Quality Gate Validation")
    parser.add_argument("--output", default="quality_gate_results.json", help="Output results file")
    parser.add_argument("--thresholds", help="JSON file with custom quality thresholds")

    args = parser.parse_args()

    base_dir = Path.cwd()
    validator = QualityGateValidator(base_dir)

    # Load custom thresholds if provided
    if args.thresholds and Path(args.thresholds).exists():
        with open(args.thresholds, 'r') as f:
            custom_thresholds = json.load(f)
            validator.quality_thresholds.update(custom_thresholds)

    try:
        results = validator.run_comprehensive_quality_gate()

        # Save results
        output_path = base_dir / args.output
        validator.save_results(results, output_path)

        print(f"\n🎉 Quality Gate Validation Complete!")
        print(f"📊 Results saved to: {output_path}")
        print(f"📝 Report saved to: {output_path.with_suffix('.md')}")

        # Print summary
        status_emoji = "✅" if results["overall_status"] == "PASS" else "❌"
        print(f"\n{status_emoji} Overall Status: {results['overall_status']}")

        if "summary" in results:
            summary = results["summary"]
            print(f"📊 Summary: {summary['passed_checks']}/{summary['total_checks']} checks passed ({summary['pass_rate']:.1%})")

        # Exit with appropriate code
        return 0 if results["overall_status"] == "PASS" else 1

    except Exception as e:
        print(f"❌ Quality Gate Validation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())