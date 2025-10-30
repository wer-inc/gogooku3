#!/usr/bin/env python3
"""
Phase 3: E2E Incremental Update Testing
Comprehensive end-to-end testing of incremental dataset update functionality
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl


class E2EIncrementalTester:
    """End-to-end testing for incremental dataset updates"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.test_output_base = base_dir / "test_output_base"
        self.test_output_full = base_dir / "test_output_full"
        self.results = {}

    def setup_test_environment(self):
        """Setup clean test environment"""
        print("🧹 Setting up test environment...")

        # Clean previous test results
        for test_dir in [self.test_output_base, self.test_output_full]:
            if test_dir.exists():
                shutil.rmtree(test_dir)
            test_dir.mkdir(parents=True, exist_ok=True)

        print("✅ Test environment ready")

    def run_base_dataset_creation(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Create base dataset for incremental testing"""
        print(f"📊 Creating base dataset ({start_date} to {end_date})...")

        start_time = time.time()

        cmd = [
            "python", "scripts/pipelines/run_full_dataset_hydra.py",
            f"pipeline.start_date={start_date}",
            f"pipeline.end_date={end_date}",
            f"pipeline.output_dir={self.test_output_base}",
            "pipeline.jquants=true"  # Use JQuants API for realistic testing
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
        execution_time = time.time() - start_time

        if result.returncode != 0:
            raise RuntimeError(f"Base dataset creation failed: {result.stderr}")

        # Verify base dataset
        base_parquet = self.test_output_base / "ml_dataset_latest_full.parquet"
        if not base_parquet.exists():
            raise RuntimeError("Base dataset parquet not created")

        df_base = pl.read_parquet(base_parquet)

        return {
            "execution_time": execution_time,
            "record_count": len(df_base),
            "date_range": (df_base["date"].min(), df_base["date"].max()),
            "columns": df_base.columns,
            "memory_usage_mb": df_base.estimated_size("mb")
        }

    def run_incremental_update(self, since_date: str) -> dict[str, Any]:
        """Run incremental update test"""
        print(f"🔄 Running incremental update (since {since_date})...")

        start_time = time.time()

        cmd = [
            "python", "scripts/pipelines/run_full_dataset_hydra.py",
            "pipeline.update_mode=incremental",
            f"pipeline.since_date={since_date}",
            f"pipeline.output_dir={self.test_output_base}",
            "pipeline.jquants=true"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
        execution_time = time.time() - start_time

        if result.returncode != 0:
            raise RuntimeError(f"Incremental update failed: {result.stderr}")

        # Verify incremental result
        updated_parquet = self.test_output_base / "ml_dataset_latest_full.parquet"
        df_updated = pl.read_parquet(updated_parquet)

        return {
            "execution_time": execution_time,
            "record_count": len(df_updated),
            "date_range": (df_updated["date"].min(), df_updated["date"].max()),
            "memory_usage_mb": df_updated.estimated_size("mb")
        }

    def run_full_pipeline_comparison(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Run full pipeline for comparison"""
        print(f"📈 Running full pipeline for comparison ({start_date} to {end_date})...")

        start_time = time.time()

        cmd = [
            "python", "scripts/pipelines/run_full_dataset_hydra.py",
            f"pipeline.start_date={start_date}",
            f"pipeline.end_date={end_date}",
            f"pipeline.output_dir={self.test_output_full}",
            "pipeline.jquants=true"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)
        execution_time = time.time() - start_time

        if result.returncode != 0:
            raise RuntimeError(f"Full pipeline failed: {result.stderr}")

        full_parquet = self.test_output_full / "ml_dataset_latest_full.parquet"
        df_full = pl.read_parquet(full_parquet)

        return {
            "execution_time": execution_time,
            "record_count": len(df_full),
            "date_range": (df_full["date"].min(), df_full["date"].max()),
            "memory_usage_mb": df_full.estimated_size("mb")
        }

    def verify_data_consistency(self) -> dict[str, Any]:
        """Verify incremental vs full pipeline data consistency"""
        print("🔍 Verifying data consistency...")

        incremental_parquet = self.test_output_base / "ml_dataset_latest_full.parquet"
        full_parquet = self.test_output_full / "ml_dataset_latest_full.parquet"

        if not incremental_parquet.exists() or not full_parquet.exists():
            return {"error": "Missing parquet files for comparison"}

        df_incremental = pl.read_parquet(incremental_parquet)
        df_full = pl.read_parquet(full_parquet)

        # Sort both dataframes for comparison
        df_incremental = df_incremental.sort(["date", "code"])
        df_full = df_full.sort(["date", "code"])

        # Basic consistency checks
        record_count_match = len(df_incremental) == len(df_full)
        date_range_match = (
            df_incremental["date"].min() == df_full["date"].min() and
            df_incremental["date"].max() == df_full["date"].max()
        )

        # Sample data comparison (check first 100 rows)
        sample_size = min(100, len(df_incremental), len(df_full))
        sample_incremental = df_incremental.head(sample_size).select(["date", "code", "close"])
        sample_full = df_full.head(sample_size).select(["date", "code", "close"])

        data_values_match = sample_incremental.equals(sample_full)

        return {
            "record_count_match": record_count_match,
            "date_range_match": date_range_match,
            "data_values_match": data_values_match,
            "incremental_records": len(df_incremental),
            "full_records": len(df_full),
            "incremental_date_range": (df_incremental["date"].min(), df_incremental["date"].max()),
            "full_date_range": (df_full["date"].min(), df_full["date"].max())
        }

    def check_lineage_tracking(self) -> dict[str, Any]:
        """Verify lineage tracking functionality"""
        print("📊 Checking lineage tracking...")

        lineage_file = self.test_output_base / "lineage.jsonl"
        if not lineage_file.exists():
            return {"error": "Lineage file not found"}

        lineage_entries = []
        with open(lineage_file) as f:
            for line in f:
                if line.strip():
                    lineage_entries.append(json.loads(line))

        # Expected transformations for incremental update
        expected_transformations = ["base_pipeline_run", "enrich_and_save", "incremental_merge"]
        found_transformations = [entry["transformation"] for entry in lineage_entries]

        return {
            "lineage_entries": len(lineage_entries),
            "found_transformations": found_transformations,
            "has_expected_chain": all(t in found_transformations for t in expected_transformations),
            "latest_entries": lineage_entries[-3:] if len(lineage_entries) >= 3 else lineage_entries
        }

    def check_quality_reports(self) -> dict[str, Any]:
        """Check data quality reports"""
        print("🔍 Checking quality reports...")

        quality_report = self.test_output_base / "data_quality_report.md"
        quality_summary = self.test_output_base / "data_quality_summary.json"
        profiling_report = self.test_output_base / "profiling_report.json"

        reports_exist = {
            "quality_report": quality_report.exists(),
            "quality_summary": quality_summary.exists(),
            "profiling_report": profiling_report.exists()
        }

        quality_data = {}
        if quality_summary.exists():
            with open(quality_summary) as f:
                quality_data = json.load(f)

        profiling_data = {}
        if profiling_report.exists():
            with open(profiling_report) as f:
                profiling_data = json.load(f)

        return {
            "reports_exist": reports_exist,
            "quality_summary": quality_data,
            "profiling_summary": profiling_data
        }

    def run_full_e2e_test(self, base_start: str, base_end: str, incremental_since: str, full_end: str):
        """Run complete E2E test suite"""
        print("🚀 Starting Phase 3 E2E Incremental Testing...")

        self.setup_test_environment()

        # Phase 1: Create base dataset
        base_results = self.run_base_dataset_creation(base_start, base_end)
        self.results["base_creation"] = base_results

        # Phase 2: Run incremental update
        incremental_results = self.run_incremental_update(incremental_since)
        self.results["incremental_update"] = incremental_results

        # Phase 3: Run full pipeline for comparison
        full_results = self.run_full_pipeline_comparison(base_start, full_end)
        self.results["full_pipeline"] = full_results

        # Phase 4: Verify consistency
        consistency_results = self.verify_data_consistency()
        self.results["consistency_check"] = consistency_results

        # Phase 5: Check lineage
        lineage_results = self.check_lineage_tracking()
        self.results["lineage_tracking"] = lineage_results

        # Phase 6: Check quality reports
        quality_results = self.check_quality_reports()
        self.results["quality_reports"] = quality_results

        # Calculate performance metrics
        self.calculate_performance_metrics()

        return self.results

    def calculate_performance_metrics(self):
        """Calculate performance improvement metrics"""
        base_time = self.results["base_creation"]["execution_time"]
        incremental_time = self.results["incremental_update"]["execution_time"]
        full_time = self.results["full_pipeline"]["execution_time"]

        total_incremental_time = base_time + incremental_time
        time_savings = full_time - total_incremental_time
        time_savings_percent = (time_savings / full_time) * 100 if full_time > 0 else 0

        self.results["performance_metrics"] = {
            "base_creation_time": base_time,
            "incremental_update_time": incremental_time,
            "total_incremental_time": total_incremental_time,
            "full_pipeline_time": full_time,
            "time_savings_seconds": time_savings,
            "time_savings_percent": time_savings_percent,
            "incremental_efficiency": incremental_time / full_time if full_time > 0 else 0
        }

    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("# Phase 3 E2E Incremental Update Test Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")

        # Performance Summary
        if "performance_metrics" in self.results:
            perf = self.results["performance_metrics"]
            report.append("## Performance Summary")
            report.append(f"- **Full Pipeline Time**: {perf['full_pipeline_time']:.2f}s")
            report.append(f"- **Base Creation Time**: {perf['base_creation_time']:.2f}s")
            report.append(f"- **Incremental Update Time**: {perf['incremental_update_time']:.2f}s")
            report.append(f"- **Total Incremental Time**: {perf['total_incremental_time']:.2f}s")
            report.append(f"- **Time Savings**: {perf['time_savings_seconds']:.2f}s ({perf['time_savings_percent']:.1f}%)")
            report.append(f"- **Incremental Efficiency**: {perf['incremental_efficiency']:.2%}")
            report.append("")

        # Data Consistency
        if "consistency_check" in self.results:
            consistency = self.results["consistency_check"]
            report.append("## Data Consistency")
            report.append(f"- **Record Count Match**: {'✅' if consistency['record_count_match'] else '❌'}")
            report.append(f"- **Date Range Match**: {'✅' if consistency['date_range_match'] else '❌'}")
            report.append(f"- **Data Values Match**: {'✅' if consistency['data_values_match'] else '❌'}")
            report.append("")

        # Lineage Tracking
        if "lineage_tracking" in self.results:
            lineage = self.results["lineage_tracking"]
            report.append("## Lineage Tracking")
            report.append(f"- **Lineage Entries**: {lineage['lineage_entries']}")
            report.append(f"- **Expected Chain Present**: {'✅' if lineage['has_expected_chain'] else '❌'}")
            report.append(f"- **Transformations**: {', '.join(lineage['found_transformations'])}")
            report.append("")

        # Quality Reports
        if "quality_reports" in self.results:
            quality = self.results["quality_reports"]
            report.append("## Quality Reports")
            for report_name, exists in quality["reports_exist"].items():
                report.append(f"- **{report_name}**: {'✅' if exists else '❌'}")
            report.append("")

        return "\n".join(report)

    def save_results(self, output_file: Path):
        """Save test results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Also save markdown report
        report_file = output_file.with_suffix('.md')
        with open(report_file, 'w') as f:
            f.write(self.generate_test_report())


def main():
    parser = argparse.ArgumentParser(description="Phase 3 E2E Incremental Testing")
    parser.add_argument("--base-start", default="2024-09-01", help="Base dataset start date")
    parser.add_argument("--base-end", default="2024-09-10", help="Base dataset end date")
    parser.add_argument("--incremental-since", default="2024-09-11", help="Incremental update since date")
    parser.add_argument("--full-end", default="2024-09-15", help="Full pipeline end date")
    parser.add_argument("--output", default="e2e_test_results.json", help="Output results file")

    args = parser.parse_args()

    base_dir = Path.cwd()
    tester = E2EIncrementalTester(base_dir)

    try:
        results = tester.run_full_e2e_test(
            args.base_start,
            args.base_end,
            args.incremental_since,
            args.full_end
        )

        # Save results
        output_path = base_dir / args.output
        tester.save_results(output_path)

        print("\n🎉 E2E Testing Complete!")
        print(f"📊 Results saved to: {output_path}")
        print(f"📝 Report saved to: {output_path.with_suffix('.md')}")

        # Print summary
        print("\n📈 Performance Summary:")
        if "performance_metrics" in results:
            perf = results["performance_metrics"]
            print(f"   Time Savings: {perf['time_savings_percent']:.1f}%")
            print(f"   Incremental Efficiency: {perf['incremental_efficiency']:.2%}")

        print("\n✅ Data Consistency:")
        if "consistency_check" in results:
            consistency = results["consistency_check"]
            print(f"   Records Match: {consistency['record_count_match']}")
            print(f"   Data Values Match: {consistency['data_values_match']}")

    except Exception as e:
        print(f"❌ E2E Testing failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
