#!/usr/bin/env python3
"""
gogooku3-standalone Health Check Module
========================================

This module provides health check endpoints for the application.
All checks are designed to be lightweight and non-invasive.

Usage:
    python ops/health_check.py

Or integrate into existing application:
    from ops.health_check import HealthChecker
    checker = HealthChecker()
    result = checker.health_check()
"""

import os
import sys
import json
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthChecker:
    """Health checker for gogooku3-standalone application."""

    def __init__(self):
        self.start_time = datetime.now()
        self.project_root = Path(__file__).parent.parent

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Dict containing health status and details
        """
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "filesystem": self._check_filesystem(),
                "memory": self._check_memory(),
                "cpu": self._check_cpu(),
                "disk": self._check_disk(),
                "network": self._check_network(),
                "application": self._check_application(),
                "dependencies": self._check_dependencies()
            },
            "version": self._get_version(),
            "uptime": self._get_uptime()
        }

    def readiness_check(self) -> Dict[str, Any]:
        """
        Perform readiness check (can the application accept traffic?).

        Returns:
            Dict containing readiness status
        """
        checks = {
            "filesystem": self._check_filesystem(),
            "critical_paths": self._check_critical_paths(),
            "dependencies": self._check_dependencies()
        }

        all_healthy = all(check.get("status") == "healthy" for check in checks.values())

        return {
            "status": "ready" if all_healthy else "not ready",
            "timestamp": datetime.now().isoformat(),
            "checks": checks
        }

    def liveness_check(self) -> Dict[str, Any]:
        """
        Perform liveness check (is the application running properly?).

        Returns:
            Dict containing liveness status
        """
        return {
            "status": "alive",
            "timestamp": datetime.now().isoformat(),
            "pid": os.getpid(),
            "uptime": self._get_uptime()
        }

    def _check_filesystem(self) -> Dict[str, Any]:
        """Check filesystem accessibility."""
        try:
            # Check critical directories exist and are writable
            critical_paths = [
                self.project_root / "logs",
                self.project_root / "output",
                self.project_root / "data"
            ]

            missing_paths = []
            unwritable_paths = []

            for path in critical_paths:
                if not path.exists():
                    missing_paths.append(str(path))
                elif not os.access(path, os.W_OK):
                    unwritable_paths.append(str(path))

            if missing_paths or unwritable_paths:
                return {
                    "status": "unhealthy",
                    "message": f"Filesystem issues: missing={missing_paths}, unwritable={unwritable_paths}"
                }

            return {
                "status": "healthy",
                "message": "All critical paths accessible"
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Filesystem check failed: {str(e)}"
            }

    def _check_critical_paths(self) -> Dict[str, Any]:
        """Check critical file paths for readiness."""
        try:
            critical_files = [
                self.project_root / "main.py",
                self.project_root / "pyproject.toml",
                self.project_root / "requirements.txt"
            ]

            missing_files = [str(f) for f in critical_files if not f.exists()]

            if missing_files:
                return {
                    "status": "unhealthy",
                    "message": f"Missing critical files: {missing_files}"
                }

            return {
                "status": "healthy",
                "message": "All critical files present"
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Critical paths check failed: {str(e)}"
            }

    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent

            # Warning at 85%, critical at 95%
            if memory_usage_percent >= 95:
                status = "critical"
            elif memory_usage_percent >= 85:
                status = "warning"
            else:
                status = "healthy"

            return {
                "status": status,
                "message": f"Memory usage: {memory_usage_percent:.1f}%",
                "details": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percentage": memory_usage_percent
                }
            }

        except Exception as e:
            return {
                "status": "unknown",
                "message": f"Memory check failed: {str(e)}"
            }

    def _check_cpu(self) -> Dict[str, Any]:
        """Check CPU usage."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)

            # Warning at 80%, critical at 95%
            if cpu_percent >= 95:
                status = "critical"
            elif cpu_percent >= 80:
                status = "warning"
            else:
                status = "healthy"

            return {
                "status": status,
                "message": f"CPU usage: {cpu_percent:.1f}%",
                "details": {
                    "cpu_percent": cpu_percent,
                    "cpu_count": psutil.cpu_count()
                }
            }

        except Exception as e:
            return {
                "status": "unknown",
                "message": f"CPU check failed: {str(e)}"
            }

    def _check_disk(self) -> Dict[str, Any]:
        """Check disk usage."""
        try:
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent

            # Warning at 85%, critical at 90%
            if disk_usage_percent >= 90:
                status = "critical"
            elif disk_usage_percent >= 85:
                status = "warning"
            else:
                status = "healthy"

            return {
                "status": status,
                "message": f"Disk usage: {disk_usage_percent:.1f}%",
                "details": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percentage": disk_usage_percent
                }
            }

        except Exception as e:
            return {
                "status": "unknown",
                "message": f"Disk check failed: {str(e)}"
            }

    def _check_network(self) -> Dict[str, Any]:
        """Check network connectivity."""
        try:
            # Basic network connectivity check
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return {
                "status": "healthy",
                "message": "Network connectivity OK"
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Network check failed: {str(e)}"
            }

    def _check_application(self) -> Dict[str, Any]:
        """Check application-specific health."""
        try:
            # Check if main.py exists and is executable
            main_py = self.project_root / "main.py"
            if not main_py.exists():
                return {
                    "status": "unhealthy",
                    "message": "main.py not found"
                }

            # Check if logs directory has recent activity
            logs_dir = self.project_root / "logs"
            if logs_dir.exists():
                recent_logs = []
                for log_file in logs_dir.glob("*.log"):
                    if log_file.stat().st_mtime > (time.time() - 3600):  # Last hour
                        recent_logs.append(log_file.name)

                return {
                    "status": "healthy",
                    "message": f"Application files OK, recent logs: {recent_logs}"
                }
            else:
                return {
                    "status": "healthy",
                    "message": "Application files OK, no logs directory yet"
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Application check failed: {str(e)}"
            }

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check if critical dependencies are available."""
        try:
            critical_modules = [
                'polars', 'pandas', 'torch', 'numpy',
                'aiohttp', 'clickhouse_driver'
            ]

            missing_modules = []
            for module in critical_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_modules.append(module)

            if missing_modules:
                return {
                    "status": "unhealthy",
                    "message": f"Missing dependencies: {missing_modules}"
                }

            return {
                "status": "healthy",
                "message": "All critical dependencies available"
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "message": f"Dependency check failed: {str(e)}"
            }

    def _get_version(self) -> str:
        """Get application version."""
        try:
            # Try to get version from pyproject.toml
            import tomllib
            pyproject_path = self.project_root / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, 'rb') as f:
                    data = tomllib.load(f)
                    return data.get('project', {}).get('version', 'unknown')

            return "unknown"

        except Exception:
            return "unknown"

    def _get_uptime(self) -> str:
        """Get application uptime."""
        uptime = datetime.now() - self.start_time
        return str(uptime).split('.')[0]  # Remove microseconds


def main():
    """Command line interface for health checks."""
    import argparse

    parser = argparse.ArgumentParser(description='gogooku3 Health Checker')
    parser.add_argument(
        'check_type',
        choices=['health', 'ready', 'live'],
        default='health',
        nargs='?',
        help='Type of check to perform'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'pretty'],
        default='pretty',
        help='Output format'
    )

    args = parser.parse_args()

    checker = HealthChecker()

    if args.check_type == 'health':
        result = checker.health_check()
    elif args.check_type == 'ready':
        result = checker.readiness_check()
    elif args.check_type == 'live':
        result = checker.liveness_check()

    if args.format == 'json':
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Status: {result['status']}")
        print(f"Timestamp: {result['timestamp']}")

        if 'checks' in result:
            print("\nChecks:")
            for check_name, check_result in result['checks'].items():
                status = check_result.get('status', 'unknown')
                message = check_result.get('message', 'No message')
                print(f"  {check_name}: {status} - {message}")

        if 'uptime' in result:
            print(f"Uptime: {result['uptime']}")

    # Exit with appropriate code
    exit_code = 0 if result.get('status') in ['healthy', 'ready', 'alive'] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
