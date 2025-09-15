from __future__ import annotations

"""Simple execution profiler for pipeline stages (time + memory)."""

import contextlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PipelineProfiler:
    timers: Dict[str, float] = field(default_factory=dict)
    memory_mb: Dict[str, float] = field(default_factory=dict)

    @contextlib.contextmanager
    def timer(self, name: str):
        t0 = time.time()
        try:
            yield
        finally:
            self.timers[name] = self.timers.get(name, 0.0) + (time.time() - t0)
            # capture memory rss if available
            try:
                import psutil  # type: ignore
                rss = psutil.Process().memory_info().rss / 1024 / 1024
                self.memory_mb[name] = rss
            except Exception:
                pass

    def profile_execution(self, name: str, func, *args, **kwargs):
        with self.timer(name):
            return func(*args, **kwargs)

    def generate_report(self) -> Dict[str, Any]:
        return {"timers_sec": self.timers, "memory_mb": self.memory_mb}

