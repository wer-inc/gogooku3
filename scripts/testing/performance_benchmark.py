#!/usr/bin/env python3
"""
Phase 3: Performance Benchmarking
Detailed performance analysis and benchmarking for pipeline components
"""

from __future__ import annotations

import json
import psutil
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import argparse

import matplotlib.pyplot as plt
import pandas as pd


class PipelinePerformanceBenchmark:
    """Comprehensive performance benchmarking for pipeline components"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.benchmark_results = {}
        self.test_output_dir = base_dir / "benchmark_output"

    def setup_benchmark_environment(self):
        """Setup benchmark environment"""
        print("🏁 Setting up benchmark environment...")

        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        # Record system specs
        self.benchmark_results["system_specs"] = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": subprocess.run(["python", "--version"], capture_output=True, text=True).stdout.strip(),
            "timestamp": datetime.now().isoformat()
        }

        print("✅ Benchmark environment ready")

    def benchmark_data_size_scaling(self, date_ranges: List[tuple]) -> Dict[str, Any]:
        """Benchmark performance across different data sizes"""
        print("📊 Benchmarking data size scaling...")

        scaling_results = []

        for i, (start_date, end_date) in enumerate(date_ranges):
            print(f"  Testing range {i+1}/{len(date_ranges)}: {start_date} to {end_date}")

            test_dir = self.test_output_dir / f"scaling_test_{i+1}"
            test_dir.mkdir(exist_ok=True)

            # Monitor system resources during execution
            start_time = time.time()
            process = psutil.Process()
            initial_memory = process.memory_info().rss / (1024**2)  # MB

            cmd = [
                "python", "scripts/pipelines/run_full_dataset_hydra.py",
                f"pipeline.start_date={start_date}",
                f"pipeline.end_date={end_date}",
                f"pipeline.output_dir={test_dir}",
                "pipeline.jquants=false"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)

            execution_time = time.time() - start_time
            final_memory = process.memory_info().rss / (1024**2)  # MB
            memory_delta = final_memory - initial_memory

            # Get dataset info
            dataset_info = {"record_count": 0, "file_size_mb": 0}
            parquet_file = test_dir / "ml_dataset_latest_full.parquet"
            if parquet_file.exists():
                dataset_info["file_size_mb"] = parquet_file.stat().st_size / (1024**2)
                try:
                    import polars as pl
                    df = pl.read_parquet(parquet_file)
                    dataset_info["record_count"] = len(df)
                except Exception:
                    pass

            scaling_results.append({
                "date_range": f"{start_date}_to_{end_date}",
                "days": (datetime.strptime(end_date, "%Y-%m-%d") -
                        datetime.strptime(start_date, "%Y-%m-%d")).days,
                "execution_time": execution_time,
                "memory_delta_mb": memory_delta,
                "success": result.returncode == 0,
                "record_count": dataset_info["record_count"],
                "file_size_mb": dataset_info["file_size_mb"]
            })

        return {"scaling_results": scaling_results}

    def benchmark_incremental_vs_full(self, base_period_days: int = 30, incremental_days: int = 7) -> Dict[str, Any]:
        """Benchmark incremental vs full pipeline performance"""
        print("⚡ Benchmarking incremental vs full pipeline...")

        # Calculate date ranges
        end_date = datetime.now().strftime("%Y-%m-%d")
        full_start = (datetime.now() - timedelta(days=base_period_days + incremental_days)).strftime("%Y-%m-%d")
        base_end = (datetime.now() - timedelta(days=incremental_days)).strftime("%Y-%m-%d")
        incremental_start = (datetime.now() - timedelta(days=incremental_days - 1)).strftime("%Y-%m-%d")

        results = {}

        # Test 1: Full pipeline
        print("  Running full pipeline test...")
        full_dir = self.test_output_dir / "full_comparison"
        full_dir.mkdir(exist_ok=True)

        start_time = time.time()
        cmd_full = [
            "python", "scripts/pipelines/run_full_dataset_hydra.py",
            f"pipeline.start_date={full_start}",
            f"pipeline.end_date={end_date}",
            f"pipeline.output_dir={full_dir}",
            "pipeline.jquants=false"
        ]
        result_full = subprocess.run(cmd_full, capture_output=True, text=True, cwd=self.base_dir)
        full_time = time.time() - start_time

        results["full_pipeline"] = {
            "execution_time": full_time,
            "success": result_full.returncode == 0,
            "date_range": f"{full_start}_to_{end_date}"
        }

        # Test 2: Incremental pipeline (base + increment)
        print("  Running incremental pipeline test...")
        inc_dir = self.test_output_dir / "incremental_comparison"
        inc_dir.mkdir(exist_ok=True)

        # 2a: Create base
        start_time = time.time()
        cmd_base = [
            "python", "scripts/pipelines/run_full_dataset_hydra.py",
            f"pipeline.start_date={full_start}",
            f"pipeline.end_date={base_end}",
            f"pipeline.output_dir={inc_dir}",
            "pipeline.jquants=false"
        ]
        result_base = subprocess.run(cmd_base, capture_output=True, text=True, cwd=self.base_dir)
        base_time = time.time() - start_time

        # 2b: Incremental update
        start_time = time.time()
        cmd_inc = [
            "python", "scripts/pipelines/run_full_dataset_hydra.py",
            "pipeline.update_mode=incremental",
            f"pipeline.since_date={incremental_start}",
            f"pipeline.output_dir={inc_dir}",
            "pipeline.jquants=false"
        ]
        result_inc = subprocess.run(cmd_inc, capture_output=True, text=True, cwd=self.base_dir)
        inc_time = time.time() - start_time

        total_incremental_time = base_time + inc_time

        results["incremental_pipeline"] = {
            "base_creation_time": base_time,
            "incremental_update_time": inc_time,
            "total_time": total_incremental_time,
            "base_success": result_base.returncode == 0,
            "incremental_success": result_inc.returncode == 0
        }

        # Calculate efficiency metrics
        if full_time > 0:
            time_savings = full_time - total_incremental_time
            efficiency_gain = (time_savings / full_time) * 100
            incremental_efficiency = inc_time / full_time

            results["efficiency_metrics"] = {
                "time_savings_seconds": time_savings,
                "efficiency_gain_percent": efficiency_gain,
                "incremental_only_efficiency": incremental_efficiency,
                "speedup_factor": full_time / total_incremental_time if total_incremental_time > 0 else 0
            }

        return results

    def benchmark_component_performance(self) -> Dict[str, Any]:
        """Benchmark individual component performance"""
        print("🔧 Benchmarking individual components...")

        test_dir = self.test_output_dir / "component_benchmark"
        test_dir.mkdir(exist_ok=True)

        # Run pipeline with profiling enabled
        cmd = [
            "python", "scripts/pipelines/run_full_dataset_hydra.py",
            "pipeline.start_date=2025-09-01",
            "pipeline.end_date=2025-09-10",
            f"pipeline.output_dir={test_dir}",
            "pipeline.profiling.enabled=true",
            "pipeline.jquants=false"
        ]

        subprocess.run(cmd, capture_output=True, text=True, cwd=self.base_dir)

        # Read profiling results
        profiling_file = test_dir / "profiling_report.json"
        component_results = {}

        if profiling_file.exists():
            with open(profiling_file, 'r') as f:
                profiling_data = json.load(f)
                component_results = profiling_data

        return {"component_performance": component_results}

    def benchmark_memory_usage(self, test_duration: int = 60) -> Dict[str, Any]:
        """Monitor memory usage during pipeline execution"""
        print("💾 Benchmarking memory usage...")

        memory_samples = []
        test_dir = self.test_output_dir / "memory_benchmark"
        test_dir.mkdir(exist_ok=True)

        # Start pipeline in background
        cmd = [
            "python", "scripts/pipelines/run_full_dataset_hydra.py",
            "pipeline.start_date=2025-09-01",
            "pipeline.end_date=2025-09-07",  # Shorter range for memory testing
            f"pipeline.output_dir={test_dir}",
            "pipeline.jquants=false"
        ]

        start_time = time.time()
        process = subprocess.Popen(cmd, cwd=self.base_dir)

        # Monitor memory usage
        try:
            while process.poll() is None and (time.time() - start_time) < test_duration:
                try:
                    memory_info = psutil.virtual_memory()
                    memory_samples.append({
                        "timestamp": time.time() - start_time,
                        "used_gb": (memory_info.total - memory_info.available) / (1024**3),
                        "available_gb": memory_info.available / (1024**3),
                        "percent": memory_info.percent
                    })
                    time.sleep(1)  # Sample every second
                except psutil.NoSuchProcess:
                    break
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()

        # Calculate memory statistics
        if memory_samples:
            used_memory = [sample["used_gb"] for sample in memory_samples]
            memory_stats = {
                "peak_memory_gb": max(used_memory),
                "avg_memory_gb": sum(used_memory) / len(used_memory),
                "min_memory_gb": min(used_memory),
                "samples_count": len(memory_samples),
                "memory_samples": memory_samples
            }
        else:
            memory_stats = {"error": "No memory samples collected"}

        return {"memory_usage": memory_stats}

    def generate_performance_plots(self):
        """Generate performance visualization plots"""
        print("📊 Generating performance plots...")

        plots_dir = self.test_output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Plot 1: Scaling performance
        if "data_size_scaling" in self.benchmark_results:
            scaling_data = self.benchmark_results["data_size_scaling"]["scaling_results"]
            if scaling_data:
                df = pd.DataFrame(scaling_data)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

                # Execution time vs data size
                ax1.plot(df["days"], df["execution_time"], 'bo-')
                ax1.set_xlabel("Days of Data")
                ax1.set_ylabel("Execution Time (seconds)")
                ax1.set_title("Pipeline Execution Time vs Data Size")
                ax1.grid(True)

                # Memory usage vs data size
                ax2.plot(df["days"], df["memory_delta_mb"], 'ro-')
                ax2.set_xlabel("Days of Data")
                ax2.set_ylabel("Memory Delta (MB)")
                ax2.set_title("Memory Usage vs Data Size")
                ax2.grid(True)

                plt.tight_layout()
                plt.savefig(plots_dir / "scaling_performance.png", dpi=300, bbox_inches='tight')
                plt.close()

        # Plot 2: Incremental vs Full comparison
        if "incremental_vs_full" in self.benchmark_results:
            comp_data = self.benchmark_results["incremental_vs_full"]

            categories = ["Full Pipeline", "Base Creation", "Incremental Update", "Total Incremental"]
            times = [
                comp_data["full_pipeline"]["execution_time"],
                comp_data["incremental_pipeline"]["base_creation_time"],
                comp_data["incremental_pipeline"]["incremental_update_time"],
                comp_data["incremental_pipeline"]["total_time"]
            ]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(categories, times, color=['red', 'lightblue', 'green', 'blue'])
            ax.set_ylabel("Execution Time (seconds)")
            ax.set_title("Pipeline Execution Time Comparison")
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{time_val:.1f}s', ha='center', va='bottom')

            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / "incremental_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

    def run_full_benchmark_suite(self):
        """Run complete benchmark suite"""
        print("🚀 Starting Phase 3 Performance Benchmark Suite...")

        self.setup_benchmark_environment()

        # Test different data size ranges
        date_ranges = [
            ("2025-09-01", "2025-09-03"),  # 2 days
            ("2025-09-01", "2025-09-08"),  # 1 week
            ("2025-09-01", "2025-09-15"),  # 2 weeks
            ("2025-09-01", "2025-09-30"),  # 1 month
        ]

        # Run benchmarks
        try:
            self.benchmark_results["data_size_scaling"] = self.benchmark_data_size_scaling(date_ranges)
            self.benchmark_results["incremental_vs_full"] = self.benchmark_incremental_vs_full()
            self.benchmark_results["component_performance"] = self.benchmark_component_performance()
            self.benchmark_results["memory_usage"] = self.benchmark_memory_usage()

            # Generate visualizations
            self.generate_performance_plots()

        except Exception as e:
            print(f"⚠️  Benchmark error: {e}")
            self.benchmark_results["error"] = str(e)

        return self.benchmark_results

    def save_benchmark_results(self, output_file: Path):
        """Save benchmark results to file"""
        with open(output_file, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2, default=str)

    def generate_benchmark_report(self) -> str:
        """Generate comprehensive benchmark report"""
        report = []
        report.append("# Phase 3 Performance Benchmark Report")
        report.append(f"Generated: {datetime.now().isoformat()}")
        report.append("")

        # System specifications
        if "system_specs" in self.benchmark_results:
            specs = self.benchmark_results["system_specs"]
            report.append("## System Specifications")
            report.append(f"- **CPU Cores**: {specs['cpu_count']}")
            report.append(f"- **Memory**: {specs['memory_gb']:.1f} GB")
            report.append(f"- **Python Version**: {specs['python_version']}")
            report.append("")

        # Incremental vs Full Performance
        if "incremental_vs_full" in self.benchmark_results:
            comp = self.benchmark_results["incremental_vs_full"]
            report.append("## Incremental vs Full Pipeline Performance")

            full_time = comp["full_pipeline"]["execution_time"]
            total_inc_time = comp["incremental_pipeline"]["total_time"]
            inc_only_time = comp["incremental_pipeline"]["incremental_update_time"]

            report.append(f"- **Full Pipeline**: {full_time:.2f}s")
            report.append(f"- **Incremental Base Creation**: {comp['incremental_pipeline']['base_creation_time']:.2f}s")
            report.append(f"- **Incremental Update Only**: {inc_only_time:.2f}s")
            report.append(f"- **Total Incremental**: {total_inc_time:.2f}s")

            if "efficiency_metrics" in comp:
                eff = comp["efficiency_metrics"]
                report.append(f"- **Time Savings**: {eff['time_savings_seconds']:.2f}s ({eff['efficiency_gain_percent']:.1f}%)")
                report.append(f"- **Speedup Factor**: {eff['speedup_factor']:.2f}x")
                report.append(f"- **Incremental Efficiency**: {eff['incremental_only_efficiency']:.2%}")
            report.append("")

        # Memory Usage
        if "memory_usage" in self.benchmark_results:
            mem = self.benchmark_results["memory_usage"]["memory_usage"]
            if "peak_memory_gb" in mem:
                report.append("## Memory Usage")
                report.append(f"- **Peak Memory**: {mem['peak_memory_gb']:.2f} GB")
                report.append(f"- **Average Memory**: {mem['avg_memory_gb']:.2f} GB")
                report.append(f"- **Minimum Memory**: {mem['min_memory_gb']:.2f} GB")
                report.append("")

        # Data Size Scaling
        if "data_size_scaling" in self.benchmark_results:
            scaling = self.benchmark_results["data_size_scaling"]["scaling_results"]
            if scaling:
                report.append("## Data Size Scaling Performance")
                report.append("| Days | Execution Time (s) | Memory Delta (MB) | Records | File Size (MB) |")
                report.append("|------|-------------------|------------------|---------|----------------|")
                for result in scaling:
                    report.append(f"| {result['days']} | {result['execution_time']:.2f} | "
                                f"{result['memory_delta_mb']:.1f} | {result['record_count']} | "
                                f"{result['file_size_mb']:.1f} |")
                report.append("")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Performance Benchmark")
    parser.add_argument("--output", default="benchmark_results.json", help="Output results file")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (reduced test cases)")

    args = parser.parse_args()

    base_dir = Path.cwd()
    benchmark = PipelinePerformanceBenchmark(base_dir)

    try:
        results = benchmark.run_full_benchmark_suite()

        # Save results
        output_path = base_dir / args.output
        benchmark.save_benchmark_results(output_path)

        # Save report
        report_path = output_path.with_suffix('.md')
        with open(report_path, 'w') as f:
            f.write(benchmark.generate_benchmark_report())

        print(f"\n🎉 Benchmark Complete!")
        print(f"📊 Results saved to: {output_path}")
        print(f"📝 Report saved to: {report_path}")
        print(f"📈 Plots saved to: {base_dir}/benchmark_output/plots/")

        # Print key metrics
        if "incremental_vs_full" in results and "efficiency_metrics" in results["incremental_vs_full"]:
            eff = results["incremental_vs_full"]["efficiency_metrics"]
            print(f"\n⚡ Key Performance Metrics:")
            print(f"   Efficiency Gain: {eff['efficiency_gain_percent']:.1f}%")
            print(f"   Speedup Factor: {eff['speedup_factor']:.2f}x")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())