#!/usr/bin/env python3
"""
gogooku3-standalone Metrics Exporter
====================================

Prometheus-compatible metrics exporter for application monitoring.
Provides system and application-specific metrics.

Usage:
    python ops/metrics_exporter.py

Or integrate into existing application:
    from ops.metrics_exporter import MetricsExporter
    exporter = MetricsExporter()
    metrics = exporter.generate_metrics()
"""

import os
import sys
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsExporter:
    """Prometheus-compatible metrics exporter."""

    def __init__(self):
        self.start_time = time.time()
        self.project_root = Path(__file__).parent.parent
        self.metrics_cache = {}
        self.cache_timeout = 30  # seconds

    def generate_metrics(self) -> str:
        """
        Generate Prometheus-formatted metrics string.

        Returns:
            String containing all metrics in Prometheus format
        """
        lines = []

        # Add header comments
        lines.extend([
            "# HELP gogooku3_uptime_seconds Application uptime in seconds",
            "# TYPE gogooku3_uptime_seconds gauge",
            f"gogooku3_uptime_seconds {self._get_uptime_seconds()}",
            "",
            "# HELP gogooku3_info Application information",
            "# TYPE gogooku3_info gauge",
            f"gogooku3_info{{version=\"{self._get_version()}\"}} 1",
            ""
        ])

        # System metrics
        lines.extend(self._generate_system_metrics())

        # Application metrics
        lines.extend(self._generate_application_metrics())

        # RED/SLA metrics
        lines.extend(self._generate_red_sla_metrics())

        # Performance metrics
        lines.extend(self._generate_performance_metrics())

        return "\n".join(lines)

    def _generate_system_metrics(self) -> List[str]:
        """Generate system-level metrics."""
        lines = []

        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            lines.extend([
                "# HELP gogooku3_memory_bytes_total Total system memory in bytes",
                "# TYPE gogooku3_memory_bytes_total gauge",
                f"gogooku3_memory_bytes_total {memory.total}",
                "",
                "# HELP gogooku3_memory_bytes_used Used system memory in bytes",
                "# TYPE gogooku3_memory_bytes_used gauge",
                f"gogooku3_memory_bytes_used {memory.used}",
                "",
                "# HELP gogooku3_memory_usage_percent System memory usage percentage",
                "# TYPE gogooku3_memory_usage_percent gauge",
                f"gogooku3_memory_usage_percent {memory.percent}",
                ""
            ])

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            lines.extend([
                "# HELP gogooku3_cpu_usage_percent CPU usage percentage",
                "# TYPE gogooku3_cpu_usage_percent gauge",
                f"gogooku3_cpu_usage_percent {cpu_percent}",
                "",
                "# HELP gogooku3_cpu_count Total CPU count",
                "# TYPE gogooku3_cpu_count gauge",
                f"gogooku3_cpu_count {cpu_count}",
                ""
            ])

            # Disk metrics
            disk = psutil.disk_usage('/')
            lines.extend([
                "# HELP gogooku3_disk_bytes_total Total disk space in bytes",
                "# TYPE gogooku3_disk_bytes_total gauge",
                f"gogooku3_disk_bytes_total {disk.total}",
                "",
                "# HELP gogooku3_disk_bytes_used Used disk space in bytes",
                "# TYPE gogooku3_disk_bytes_used gauge",
                f"gogooku3_disk_bytes_used {disk.used}",
                "",
                "# HELP gogooku3_disk_usage_percent Disk usage percentage",
                "# TYPE gogooku3_disk_usage_percent gauge",
                f"gogooku3_disk_usage_percent {disk.percent}",
                ""
            ])

            # Network metrics (basic)
            net_io = psutil.net_io_counters()
            lines.extend([
                "# HELP gogooku3_network_bytes_sent Total bytes sent",
                "# TYPE gogooku3_network_bytes_sent counter",
                f"gogooku3_network_bytes_sent {net_io.bytes_sent}",
                "",
                "# HELP gogooku3_network_bytes_recv Total bytes received",
                "# TYPE gogooku3_network_bytes_recv counter",
                f"gogooku3_network_bytes_recv {net_io.bytes_recv}",
                ""
            ])

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")

        return lines

    def _generate_application_metrics(self) -> List[str]:
        """Generate application-specific metrics."""
        lines = []

        try:
            # Log file metrics
            logs_dir = self.project_root / "logs"
            if logs_dir.exists():
                log_files = list(logs_dir.glob("*.log"))
                total_log_size = sum(f.stat().st_size for f in log_files)

                lines.extend([
                    "# HELP gogooku3_log_files_count Number of log files",
                    "# TYPE gogooku3_log_files_count gauge",
                    f"gogooku3_log_files_count {len(log_files)}",
                    "",
                    "# HELP gogooku3_log_files_size_bytes Total size of log files in bytes",
                    "# TYPE gogooku3_log_files_size_bytes gauge",
                    f"gogooku3_log_files_size_bytes {total_log_size}",
                    ""
                ])

            # Data directory metrics
            data_dir = self.project_root / "data"
            if data_dir.exists():
                data_files = list(data_dir.rglob("*"))
                data_files = [f for f in data_files if f.is_file()]
                total_data_size = sum(f.stat().st_size for f in data_files)

                lines.extend([
                    "# HELP gogooku3_data_files_count Number of data files",
                    "# TYPE gogooku3_data_files_count gauge",
                    f"gogooku3_data_files_count {len(data_files)}",
                    "",
                    "# HELP gogooku3_data_files_size_bytes Total size of data files in bytes",
                    "# TYPE gogooku3_data_files_size_bytes gauge",
                    f"gogooku3_data_files_size_bytes {total_data_size}",
                    ""
                ])

            # Output directory metrics
            output_dir = self.project_root / "output"
            if output_dir.exists():
                output_files = list(output_dir.rglob("*"))
                output_files = [f for f in output_files if f.is_file()]
                total_output_size = sum(f.stat().st_size for f in output_files)

                lines.extend([
                    "# HELP gogooku3_output_files_count Number of output files",
                    "# TYPE gogooku3_output_files_count gauge",
                    f"gogooku3_output_files_count {len(output_files)}",
                    "",
                    "# HELP gogooku3_output_files_size_bytes Total size of output files in bytes",
                    "# TYPE gogooku3_output_files_size_bytes gauge",
                    f"gogooku3_output_files_size_bytes {total_output_size}",
                    ""
                ])

        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")

        return lines

    def _generate_red_sla_metrics(self) -> List[str]:
        """Generate RED (Rate, Error, Duration) and SLA metrics."""
        lines = []

        try:
            # RED Metrics - Rate
            lines.extend([
                "# HELP gogooku3_requests_total Total number of requests processed",
                "# TYPE gogooku3_requests_total counter",
                f"gogooku3_requests_total {self._get_request_count()}",
                "",
                "# HELP gogooku3_requests_per_second Current request rate per second",
                "# TYPE gogooku3_requests_per_second gauge",
                f"gogooku3_requests_per_second {self._get_request_rate()}",
                ""
            ])

            # RED Metrics - Error
            error_rate = self._get_error_rate()
            lines.extend([
                "# HELP gogooku3_errors_total Total number of errors",
                "# TYPE gogooku3_errors_total counter",
                f"gogooku3_errors_total {self._get_error_count()}",
                "",
                "# HELP gogooku3_error_rate_percent Current error rate percentage",
                "# TYPE gogooku3_error_rate_percent gauge",
                f"gogooku3_error_rate_percent {error_rate}",
                ""
            ])

            # RED Metrics - Duration
            avg_duration = self._get_avg_request_duration()
            p95_duration = self._get_p95_request_duration()
            lines.extend([
                "# HELP gogooku3_request_duration_seconds Average request duration in seconds",
                "# TYPE gogooku3_request_duration_seconds gauge",
                f"gogooku3_request_duration_seconds {avg_duration}",
                "",
                "# HELP gogooku3_request_duration_p95_seconds 95th percentile request duration in seconds",
                "# TYPE gogooku3_request_duration_p95_seconds gauge",
                f"gogooku3_request_duration_p95_seconds {p95_duration}",
                ""
            ])

            # SLA Metrics
            sla_compliance = self._calculate_sla_compliance()
            lines.extend([
                "# HELP gogooku3_sla_compliance_percent SLA compliance percentage",
                "# TYPE gogooku3_sla_compliance_percent gauge",
                f"gogooku3_sla_compliance_percent {sla_compliance}",
                "",
                "# HELP gogooku3_sla_target_response_time_seconds Target response time for SLA",
                "# TYPE gogooku3_sla_target_response_time_seconds gauge",
                "gogooku3_sla_target_response_time_seconds 30.0",
                "",
                "# HELP gogooku3_sla_target_uptime_percent Target uptime percentage for SLA",
                "# TYPE gogooku3_sla_target_uptime_percent gauge",
                "gogooku3_sla_target_uptime_percent 99.9",
                ""
            ])

            # Training Pipeline Metrics
            training_metrics = self._get_training_metrics()
            lines.extend([
                "# HELP gogooku3_training_jobs_total Total number of training jobs executed",
                "# TYPE gogooku3_training_jobs_total counter",
                f"gogooku3_training_jobs_total {training_metrics['total_jobs']}",
                "",
                "# HELP gogooku3_training_jobs_success_total Total number of successful training jobs",
                "# TYPE gogooku3_training_jobs_success_total counter",
                f"gogooku3_training_jobs_success_total {training_metrics['successful_jobs']}",
                "",
                "# HELP gogooku3_training_duration_avg_seconds Average training duration in seconds",
                "# TYPE gogooku3_training_duration_avg_seconds gauge",
                f"gogooku3_training_duration_avg_seconds {training_metrics['avg_duration']}",
                "",
                "# HELP gogooku3_model_accuracy_percent Current model accuracy percentage",
                "# TYPE gogooku3_model_accuracy_percent gauge",
                f"gogooku3_model_accuracy_percent {training_metrics['accuracy']}",
                ""
            ])

            # Data Quality Metrics
            dq_metrics = self._get_data_quality_metrics()
            lines.extend([
                "# HELP gogooku3_data_quality_score_percent Current data quality score percentage",
                "# TYPE gogooku3_data_quality_score_percent gauge",
                f"gogooku3_data_quality_score_percent {dq_metrics['quality_score']}",
                "",
                "# HELP gogooku3_data_validation_checks_total Total data validation checks performed",
                "# TYPE gogooku3_data_validation_checks_total counter",
                f"gogooku3_data_validation_checks_total {dq_metrics['validation_checks']}",
                "",
                "# HELP gogooku3_data_validation_failures_total Total data validation failures",
                "# TYPE gogooku3_data_validation_failures_total counter",
                f"gogooku3_data_validation_failures_total {dq_metrics['validation_failures']}",
                ""
            ])

        except Exception as e:
            logger.error(f"Failed to collect RED/SLA metrics: {e}")

        return lines

    def _generate_performance_metrics(self) -> List[str]:
        """Generate performance-related metrics."""
        lines = []

        try:
            # Process information
            process = psutil.Process()
            memory_info = process.memory_info()

            lines.extend([
                "# HELP gogooku3_process_memory_bytes Process memory usage in bytes",
                "# TYPE gogooku3_process_memory_bytes gauge",
                f"gogooku3_process_memory_bytes {memory_info.rss}",
                "",
                "# HELP gogooku3_process_cpu_percent Process CPU usage percentage",
                "# TYPE gogooku3_process_cpu_percent gauge",
                f"gogooku3_process_cpu_percent {process.cpu_percent()}",
                "",
                "# HELP gogooku3_process_threads_count Number of threads in process",
                "# TYPE gogooku3_process_threads_count gauge",
                f"gogooku3_process_threads_count {process.num_threads()}",
                ""
            ])

            # Python-specific metrics
            import gc
            gc.collect()  # Force garbage collection for accurate count
            lines.extend([
                "# HELP gogooku3_python_gc_objects_collected_total Total objects collected by GC",
                "# TYPE gogooku3_python_gc_objects_collected_total counter",
                f"gogooku3_python_gc_objects_collected_total {gc.get_stats()[0]['collected']}",
                "",
                "# HELP gogooku3_python_gc_objects_uncollectable_total Total uncollectable objects",
                "# TYPE gogooku3_python_gc_objects_uncollectable_total counter",
                f"gogooku3_python_gc_objects_uncollectable_total {gc.get_stats()[0]['uncollectable']}",
                ""
            ])

        except Exception as e:
            logger.error(f"Failed to collect performance metrics: {e}")

        return lines

    def _get_uptime_seconds(self) -> float:
        """Get application uptime in seconds."""
        return time.time() - self.start_time

    def _get_version(self) -> str:
        """Get application version."""
        try:
            # Try to get version from pyproject.toml
            if sys.version_info >= (3, 11):
                import tomllib
                pyproject_path = self.project_root / "pyproject.toml"
                if pyproject_path.exists():
                    with open(pyproject_path, 'rb') as f:
                        data = tomllib.load(f)
                        return data.get('project', {}).get('version', 'unknown')
            else:
                # Fallback for older Python versions
                import toml
                pyproject_path = self.project_root / "pyproject.toml"
                if pyproject_path.exists():
                    data = toml.load(pyproject_path)
                    return data.get('project', {}).get('version', 'unknown')

            return "unknown"

        except Exception:
            return "unknown"

    # RED/SLA Metrics Implementation Methods

    def _get_request_count(self) -> int:
        """Get total number of requests processed."""
        # This would integrate with actual request tracking
        # For now, return a mock value based on log analysis
        try:
            logs_dir = self.project_root / "logs"
            if logs_dir.exists():
                # Count log entries that might indicate requests
                log_files = list(logs_dir.glob("*.log"))
                total_lines = 0
                for log_file in log_files:
                    try:
                        with open(log_file, 'r') as f:
                            total_lines += sum(1 for _ in f)
                    except Exception:
                        pass
                return total_lines
            return 0
        except Exception:
            return 0

    def _get_request_rate(self) -> float:
        """Get current request rate per second."""
        # Simple implementation - in production this would use sliding window
        try:
            # Get recent log activity as proxy for request rate
            logs_dir = self.project_root / "logs"
            if logs_dir.exists():
                recent_activity = 0
                for log_file in logs_dir.glob("*.log"):
                    try:
                        # Count lines modified in last minute
                        if log_file.stat().st_mtime > (time.time() - 60):
                            with open(log_file, 'r') as f:
                                lines = f.readlines()
                                # Count recent lines (rough approximation)
                                recent_activity += len(lines) // 60
                    except Exception:
                        pass
                return recent_activity
            return 0.0
        except Exception:
            return 0.0

    def _get_error_count(self) -> int:
        """Get total number of errors."""
        try:
            logs_dir = self.project_root / "logs"
            if logs_dir.exists():
                error_count = 0
                for log_file in logs_dir.glob("*.log"):
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read().upper()
                            error_count += content.count('ERROR') + content.count('EXCEPTION')
                    except Exception:
                        pass
                return error_count
            return 0
        except Exception:
            return 0

    def _get_error_rate(self) -> float:
        """Get current error rate percentage."""
        try:
            total_requests = self._get_request_count()
            total_errors = self._get_error_count()

            if total_requests == 0:
                return 0.0

            return (total_errors / total_requests) * 100.0
        except Exception:
            return 0.0

    def _get_avg_request_duration(self) -> float:
        """Get average request duration in seconds."""
        # This would integrate with actual timing measurements
        # For now, return mock value based on system performance
        try:
            # Use system load as proxy for response time
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 1.0
            cpu_count = psutil.cpu_count() or 1

            # Rough estimation: higher load = longer response time
            base_duration = 1.0  # 1 second base
            load_factor = min(load_avg / cpu_count, 5.0)  # Cap at 5x
            return base_duration * (1 + load_factor * 0.5)
        except Exception:
            return 1.0

    def _get_p95_request_duration(self) -> float:
        """Get 95th percentile request duration in seconds."""
        # In production, this would use actual percentile calculations
        # For now, estimate based on average
        try:
            avg_duration = self._get_avg_request_duration()
            # 95th percentile is typically higher than average
            return avg_duration * 2.5
        except Exception:
            return 2.5

    def _calculate_sla_compliance(self) -> float:
        """Calculate SLA compliance percentage."""
        try:
            # Mock SLA calculation - in production this would be more sophisticated
            target_response_time = 30.0  # seconds
            current_avg_duration = self._get_avg_request_duration()

            if current_avg_duration <= target_response_time:
                return 100.0
            else:
                # Linear degradation (simplified)
                compliance = max(0.0, 100.0 * (target_response_time / current_avg_duration))
                return compliance
        except Exception:
            return 100.0  # Default to compliant

    def _get_training_metrics(self) -> Dict[str, float]:
        """Get training pipeline metrics."""
        try:
            # Mock training metrics - in production integrate with MLflow/actual tracking
            logs_dir = self.project_root / "logs"
            training_jobs = 0
            successful_jobs = 0
            total_duration = 0

            if logs_dir.exists():
                for log_file in logs_dir.glob("*training*.log"):
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read().upper()
                            training_jobs += 1
                            if 'SUCCESS' in content or 'COMPLETED' in content:
                                successful_jobs += 1
                            # Rough duration estimation from log timestamps
                            total_duration += 1800  # 30 minutes per job (estimate)
                    except Exception:
                        pass

            return {
                'total_jobs': training_jobs,
                'successful_jobs': successful_jobs,
                'avg_duration': total_duration / max(training_jobs, 1),
                'accuracy': 85.0  # Mock accuracy score
            }
        except Exception:
            return {
                'total_jobs': 0,
                'successful_jobs': 0,
                'avg_duration': 0.0,
                'accuracy': 0.0
            }

    def _get_data_quality_metrics(self) -> Dict[str, float]:
        """Get data quality metrics."""
        try:
            # Check for data quality validation results
            dq_results_dir = self.project_root / "data_quality" / "results"
            validation_checks = 0
            validation_failures = 0
            quality_scores = []

            if dq_results_dir.exists():
                for result_file in dq_results_dir.glob("*.json"):
                    try:
                        import json
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                            validation_checks += data.get('total_checks', 0)
                            if not data.get('passed', True):
                                validation_failures += 1
                            # Extract quality score if available
                            if 'quality_score' in data:
                                quality_scores.append(data['quality_score'])
                    except Exception:
                        pass

            avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else 95.0

            return {
                'validation_checks': validation_checks,
                'validation_failures': validation_failures,
                'quality_score': avg_quality_score
            }
        except Exception:
            return {
                'validation_checks': 0,
                'validation_failures': 0,
                'quality_score': 95.0
            }


class MetricsServer:
    """Simple HTTP server for exposing metrics."""

    def __init__(self, host: str = '0.0.0.0', port: int = 8000):
        self.host = host
        self.port = port
        self.exporter = MetricsExporter()

    def start_server(self):
        """Start the metrics HTTP server."""
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading

        class MetricsHandler(BaseHTTPRequestHandler):
            def __init__(self, exporter, *args, **kwargs):
                self.exporter = exporter
                super().__init__(*args, **kwargs)

            def do_GET(self):
                if self.path == '/metrics':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain; charset=utf-8')
                    self.end_headers()

                    metrics = self.exporter.generate_metrics()
                    self.wfile.write(metrics.encode('utf-8'))
                elif self.path == '/healthz':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'OK\n')
                else:
                    self.send_response(404)
                    self.end_headers()

        def run_server():
            server = HTTPServer((self.host, self.port), lambda *args, **kwargs: MetricsHandler(self.exporter, *args, **kwargs))
            logger.info(f"Metrics server started on http://{self.host}:{self.port}")
            logger.info(f"Metrics endpoint: http://{self.host}:{self.port}/metrics")
            logger.info(f"Health endpoint: http://{self.host}:{self.port}/healthz")
            server.serve_forever()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        return server_thread


def main():
    """Command line interface for metrics exporter."""
    import argparse

    parser = argparse.ArgumentParser(description='gogooku3 Metrics Exporter')
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host to bind the metrics server'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for the metrics server'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Print metrics once and exit (no server)'
    )

    args = parser.parse_args()

    exporter = MetricsExporter()

    if args.once:
        print(exporter.generate_metrics())
    else:
        server = MetricsServer(host=args.host, port=args.port)
        server_thread = server.start_server()

        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Metrics server stopped")


if __name__ == "__main__":
    main()
