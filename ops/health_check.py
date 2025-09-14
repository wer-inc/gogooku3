from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

import psutil  # type: ignore


@dataclass
class CheckResult:
    status: str
    message: str
    details: Dict[str, Any] | None = None


class HealthChecker:
    """System health/readiness/liveness checks used by tests and CLI."""

    def __init__(self) -> None:
        self.start_time = time.time()
        # Project root: repo root inferred from this file location
        self.project_root = Path(__file__).resolve().parents[1]

    # ------- Internal helpers (unit-tested) -------
    def _check_memory(self) -> Dict[str, Any]:
        vm = psutil.virtual_memory()
        pct = float(vm.percent)
        status = "healthy" if pct < 80 else ("warning" if pct < 95 else "critical")
        msg = f"Memory usage: {pct:.1f}%"
        return {
            "status": status,
            "message": msg,
            "details": {
                "percentage": pct,
                "total": int(getattr(vm, "total", 0)),
                "available": int(getattr(vm, "available", 0)),
                "used": int(getattr(vm, "used", 0)),
            },
        }

    def _check_cpu(self) -> Dict[str, Any]:
        pct = float(psutil.cpu_percent(interval=0.0))
        status = "healthy" if pct < 85 else ("warning" if pct < 95 else "critical")
        msg = f"CPU usage: {pct:.1f}%"
        return {"status": status, "message": msg, "details": {"percentage": pct}}

    def _check_disk(self) -> Dict[str, Any]:
        du = psutil.disk_usage(str(self.project_root))
        pct = float(du.percent)
        status = "healthy" if pct < 85 else ("warning" if pct < 95 else "critical")
        msg = f"Disk usage: {pct:.1f}%"
        return {
            "status": status,
            "message": msg,
            "details": {"percentage": pct, "total": int(du.total), "used": int(du.used), "free": int(du.free)},
        }

    def _check_network(self) -> Dict[str, Any]:
        try:
            # Will be patched in tests
            with socket.create_connection(("8.8.8.8", 53), timeout=1.0):
                pass
            return {"status": "healthy", "message": "Network connectivity OK"}
        except OSError as e:
            return {"status": "unhealthy", "message": f"Network check failed: {e}"}

    def _check_filesystem(self) -> Dict[str, Any]:
        # Accept either "_logs" or "logs" as valid logs directory
        logs_candidates = [self.project_root / "_logs", self.project_root / "logs"]
        logs_dir = next((p for p in logs_candidates if p.exists()), logs_candidates[0])
        required = [logs_dir, self.project_root / "data", self.project_root / "output"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            return {"status": "unhealthy", "message": f"Critical paths missing: missing={','.join(missing)}"}
        # Verify write access
        for p in required:
            try:
                p.mkdir(parents=True, exist_ok=True)
                test = p / ".hc_write_test"
                with open(test, "w", encoding="utf-8") as f:
                    f.write("ok")
                test.unlink(missing_ok=True)
            except Exception as e:
                return {"status": "unhealthy", "message": f"Path not writable: {p} ({e})"}
        return {"status": "healthy", "message": "All critical paths accessible"}

    def _check_dependencies(self) -> Dict[str, Any]:
        required = ["polars", "pandas", "numpy"]
        missing: list[str] = []
        for mod in required:
            try:
                __import__(mod)
            except Exception:
                missing.append(mod)
        if missing:
            return {"status": "unhealthy", "message": f"Missing dependencies: {', '.join(missing)}"}
        return {"status": "healthy", "message": "All critical dependencies available"}

    def _get_version(self) -> str:
        # Parse project version from pyproject.toml in project_root
        pyproj = self.project_root / "pyproject.toml"
        if not pyproj.exists():
            return "unknown"
        txt = pyproj.read_text(encoding="utf-8", errors="ignore")
        # Simple regex for: version = "x.y.z"
        m = re.search(r"^version\s*=\s*\"([^\"]+)\"", txt, re.MULTILINE)
        return m.group(1) if m else "unknown"

    def _get_uptime(self) -> str:
        delta = timedelta(seconds=int(time.time() - self.start_time))
        # Format as HH:MM:SS
        total_seconds = int(delta.total_seconds())
        h = total_seconds // 3600
        m = (total_seconds % 3600) // 60
        s = total_seconds % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    # ------- Public API -------
    def health_check(self) -> Dict[str, Any]:
        checks = {
            "memory": self._check_memory(),
            "cpu": self._check_cpu(),
            "disk": self._check_disk(),
            "network": self._check_network(),
            "filesystem": self._check_filesystem(),
            "dependencies": self._check_dependencies(),
        }
        statuses = [c["status"] for c in checks.values()]
        if "critical" in statuses or "unhealthy" in statuses:
            overall = "unhealthy"
        elif "warning" in statuses:
            overall = "healthy"  # degrade but still healthy by default
        else:
            overall = "healthy"
        return {
            "status": overall,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "checks": checks,
            "version": self._get_version(),
            "uptime": self._get_uptime(),
        }

    def readiness_check(self) -> Dict[str, Any]:
        core = {
            "memory": self._check_memory(),
            "cpu": self._check_cpu(),
            "disk": self._check_disk(),
            "filesystem": self._check_filesystem(),
            "dependencies": self._check_dependencies(),
        }
        statuses = [c["status"] for c in core.values()]
        ready = all(s in ("healthy", "warning") for s in statuses)
        return {
            "status": "ready" if ready else "not ready",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "checks": core,
        }

    def liveness_check(self) -> Dict[str, Any]:
        return {
            "status": "alive",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pid": os.getpid(),
            "uptime": self._get_uptime(),
        }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Health check CLI")
    p.add_argument("command", choices=["health", "ready", "liveness", "healthz", "readiness"], help="Check type")
    p.add_argument("--format", choices=["text", "json"], default="text")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    parser = _build_parser()
    args = parser.parse_args(argv)

    checker = HealthChecker()
    if args.command in ("health", "healthz"):
        res = checker.health_check()
    elif args.command in ("ready", "readiness"):
        res = checker.readiness_check()
    else:
        res = checker.liveness_check()

    if args.format == "json":
        print(json.dumps(res, indent=2))
    else:
        # Human-friendly output used in tests
        print(f"Status: {res.get('status')}")
        ts = res.get("timestamp")
        if ts:
            print(f"Timestamp: {ts}")
        if "uptime" in res:
            print(f"Uptime: {res['uptime']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
