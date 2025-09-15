from __future__ import annotations

"""Resilience helpers: retry and simple checkpointing."""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Awaitable


@dataclass
class ResilienceConfig:
    enabled: bool = True
    max_retries: int = 2
    checkpoint_enabled: bool = True


class ResilientPipeline:
    def __init__(self, output_dir: Path, cfg: ResilienceConfig | None = None) -> None:
        self.output_dir = output_dir
        self.cfg = cfg or ResilienceConfig()
        self.ckpt_dir = self.output_dir / "_checkpoints"
        if self.cfg.checkpoint_enabled:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, stage: str, data: Dict[str, Any]) -> Path | None:
        if not self.cfg.checkpoint_enabled:
            return None
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self.ckpt_dir / f"{stage}_{ts}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            return path
        except Exception:
            return None

    def execute_with_retry(self, stage: str, operation: Callable[[], Any]) -> Any:
        attempts = self.cfg.max_retries + 1 if self.cfg.enabled else 1
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                return operation()
            except Exception as e:
                last_exc = e
                # checkpoint error state
                self.save_checkpoint(stage, {"error": str(e), "attempt": i + 1})
                time.sleep(min(2 ** i, 30))
        if last_exc:
            raise last_exc
        return None

    async def execute_with_retry_async(self, stage: str, operation: Callable[[], Awaitable[Any]]) -> Any:
        """Async version of execute_with_retry for coroutine operations."""
        attempts = self.cfg.max_retries + 1 if self.cfg.enabled else 1
        last_exc: Exception | None = None
        for i in range(attempts):
            try:
                return await operation()
            except Exception as e:
                last_exc = e
                # checkpoint error state
                self.save_checkpoint(stage, {"error": str(e), "attempt": i + 1})
                await asyncio.sleep(min(2 ** i, 30))
        if last_exc:
            raise last_exc
        return None

