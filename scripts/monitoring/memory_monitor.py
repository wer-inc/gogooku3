#!/usr/bin/env python3
"""
Real-time memory monitoring system for dataset generation.

Monitors system memory usage and provides warnings/termination
when thresholds are exceeded to prevent OOM killer.

Usage:
    python scripts/monitoring/memory_monitor.py --pid <PID> [options]
    python scripts/monitoring/memory_monitor.py --command "make dataset-gpu" [options]

Features:
    - Real-time memory tracking
    - Configurable warning/stop thresholds
    - Graceful process termination
    - Detailed logging
    - GPU memory monitoring (optional)
"""

import argparse
import logging
import os
import psutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Setup logging
LOG_DIR = Path("_logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"memory_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Monitor system and process memory usage."""

    def __init__(
        self,
        pid: Optional[int] = None,
        warn_threshold: float = 80.0,
        stop_threshold: float = 90.0,
        check_interval: float = 5.0,
        gpu_monitor: bool = False,
    ):
        """Initialize memory monitor.

        Args:
            pid: Process ID to monitor (None = monitor system only)
            warn_threshold: Warning threshold (% of total memory)
            stop_threshold: Stop threshold (% of total memory)
            check_interval: Check interval in seconds
            gpu_monitor: Enable GPU memory monitoring
        """
        self.pid = pid
        self.process: Optional[psutil.Process] = None
        self.warn_threshold = warn_threshold
        self.stop_threshold = stop_threshold
        self.check_interval = check_interval
        self.gpu_monitor = gpu_monitor
        self.warned = False
        self.stopped = False

        if self.pid:
            try:
                self.process = psutil.Process(self.pid)
                logger.info(f"Monitoring process {self.pid}: {self.process.name()}")
            except psutil.NoSuchProcess:
                logger.error(f"Process {self.pid} not found")
                sys.exit(1)

        # Get total system memory
        self.total_memory = psutil.virtual_memory().total / (1024**3)  # GB
        logger.info(f"Total system memory: {self.total_memory:.1f} GB")
        logger.info(f"Warning threshold: {self.warn_threshold}% ({self.total_memory * self.warn_threshold / 100:.1f} GB)")
        logger.info(f"Stop threshold: {self.stop_threshold}% ({self.total_memory * self.stop_threshold / 100:.1f} GB)")

        # Initialize GPU monitoring if requested
        self.gpu_available = False
        if self.gpu_monitor:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_available = True
                logger.info("GPU monitoring enabled")
            except Exception as e:
                logger.warning(f"GPU monitoring unavailable: {e}")

    def get_memory_stats(self) -> dict:
        """Get current memory statistics."""
        vm = psutil.virtual_memory()
        stats = {
            "system_used_gb": vm.used / (1024**3),
            "system_used_pct": vm.percent,
            "system_available_gb": vm.available / (1024**3),
        }

        # Process-specific stats
        if self.process and self.process.is_running():
            try:
                mem_info = self.process.memory_info()
                stats["process_rss_gb"] = mem_info.rss / (1024**3)
                stats["process_vms_gb"] = mem_info.vms / (1024**3)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # GPU stats
        if self.gpu_available:
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                stats["gpu_used_gb"] = mem_info.used / (1024**3)
                stats["gpu_total_gb"] = mem_info.total / (1024**3)
                stats["gpu_used_pct"] = (mem_info.used / mem_info.total) * 100
            except Exception:
                pass

        return stats

    def check_thresholds(self, stats: dict) -> str:
        """Check if memory exceeds thresholds.

        Returns:
            "ok", "warn", or "stop"
        """
        used_pct = stats["system_used_pct"]

        if used_pct >= self.stop_threshold:
            return "stop"
        elif used_pct >= self.warn_threshold:
            return "warn"
        else:
            return "ok"

    def terminate_process(self):
        """Gracefully terminate the monitored process."""
        if not self.process or not self.process.is_running():
            logger.warning("No running process to terminate")
            return

        try:
            logger.warning(f"Sending SIGTERM to process {self.pid}")
            self.process.terminate()

            # Wait up to 30 seconds for graceful termination
            try:
                self.process.wait(timeout=30)
                logger.info("Process terminated gracefully")
            except psutil.TimeoutExpired:
                logger.error("Process did not terminate, sending SIGKILL")
                self.process.kill()
                logger.info("Process killed")
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.error(f"Failed to terminate process: {e}")

    def log_stats(self, stats: dict, status: str):
        """Log memory statistics."""
        msg_parts = [
            f"Memory: {stats['system_used_gb']:.1f}/{self.total_memory:.1f} GB ({stats['system_used_pct']:.1f}%)"
        ]

        if "process_rss_gb" in stats:
            msg_parts.append(f"Process RSS: {stats['process_rss_gb']:.2f} GB")

        if "gpu_used_gb" in stats:
            msg_parts.append(f"GPU: {stats['gpu_used_gb']:.1f}/{stats['gpu_total_gb']:.1f} GB ({stats['gpu_used_pct']:.1f}%)")

        msg = " | ".join(msg_parts)

        if status == "stop":
            logger.error(f"[CRITICAL] {msg}")
        elif status == "warn":
            logger.warning(f"[WARNING] {msg}")
        else:
            logger.info(f"[OK] {msg}")

    def run(self):
        """Main monitoring loop."""
        logger.info("Memory monitoring started")
        logger.info(f"Check interval: {self.check_interval}s")

        try:
            while True:
                # Check if process is still running
                if self.process and not self.process.is_running():
                    logger.info("Monitored process has terminated")
                    break

                # Get memory stats
                stats = self.get_memory_stats()
                status = self.check_thresholds(stats)

                # Log stats
                self.log_stats(stats, status)

                # Handle thresholds
                if status == "stop" and not self.stopped:
                    logger.critical(
                        f"Memory usage exceeded stop threshold ({self.stop_threshold}%)! "
                        "Terminating process to prevent OOM killer..."
                    )
                    self.terminate_process()
                    self.stopped = True
                    break
                elif status == "warn" and not self.warned:
                    logger.warning(
                        f"Memory usage exceeded warning threshold ({self.warn_threshold}%)! "
                        "Consider stopping the process or increasing available memory."
                    )
                    self.warned = True
                elif status == "ok" and self.warned:
                    # Reset warning flag if memory drops back below threshold
                    logger.info("Memory usage returned to normal levels")
                    self.warned = False

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        finally:
            logger.info(f"Monitoring stopped. Log saved to: {LOG_FILE}")
            if self.gpu_available:
                try:
                    import pynvml
                    pynvml.nvmlShutdown()
                except Exception:
                    pass


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor memory usage and prevent OOM killer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pid",
        type=int,
        help="Process ID to monitor (if not specified, monitors system only)",
    )
    parser.add_argument(
        "--command",
        type=str,
        help="Command to run and monitor (alternative to --pid)",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=float(os.getenv("MEMORY_THRESHOLD_WARN", "80")),
        help="Warning threshold (percent of total memory, default: 80)",
    )
    parser.add_argument(
        "--stop-threshold",
        type=float,
        default=float(os.getenv("MEMORY_THRESHOLD_STOP", "90")),
        help="Stop threshold (percent of total memory, default: 90)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Check interval in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable GPU memory monitoring",
    )

    args = parser.parse_args()

    # Handle command execution
    if args.command:
        import subprocess

        logger.info(f"Starting command: {args.command}")
        proc = subprocess.Popen(args.command, shell=True)
        args.pid = proc.pid
        logger.info(f"Command started with PID: {args.pid}")

    # Create and run monitor
    monitor = MemoryMonitor(
        pid=args.pid,
        warn_threshold=args.warn_threshold,
        stop_threshold=args.stop_threshold,
        check_interval=args.interval,
        gpu_monitor=args.gpu,
    )

    try:
        monitor.run()
    except Exception as e:
        logger.error(f"Monitor failed: {e}", exc_info=True)
        sys.exit(1)

    # Exit with appropriate code
    if monitor.stopped:
        logger.error("Process was terminated due to memory threshold")
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
