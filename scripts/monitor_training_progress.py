#!/usr/bin/env python3
"""Lightweight CLI to display current training progress.

Reads heartbeat.json / latest_metrics.json under runs/last/ and prints
human-readable status, including epoch, batches, elapsed hours, and the
latest validation metrics if present.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parents[1] / "runs" / "last"
HEARTBEAT = RUN_DIR / "heartbeat.json"
METRICS = RUN_DIR / "latest_metrics.json"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # File may be mid-write; retry once after short delay
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None


def _fmt_ts(ts_str: str | None) -> str:
    if not ts_str:
        return "unknown"
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        local = ts.astimezone()
        delta = datetime.now(tz=local.tzinfo) - local
        mins = delta.total_seconds() / 60.0
        recency = f" ({mins:.1f} min ago)" if mins >= 0 else ""
        return f"{local:%Y-%m-%d %H:%M:%S %Z}{recency}"
    except Exception:
        return ts_str


def main() -> None:
    hb = _load_json(HEARTBEAT)
    metrics = _load_json(METRICS)

    if hb is None and metrics is None:
        print("No active training run found (runs/last/* missing).")
        return

    print("=== Training Heartbeat ===")
    if hb:
        status = hb.get("status", "unknown")
        epoch = hb.get("epoch")
        batch = hb.get("batch")
        step = hb.get("global_step")
        elapsed = hb.get("elapsed_hours")
        print(f"Status      : {status}")
        if epoch is not None:
            print(f"Epoch       : {epoch}")
        if batch is not None:
            print(f"Batch       : {batch}")
        if step is not None:
            print(f"Global step : {step}")
        if isinstance(elapsed, (int, float)) and not math.isnan(elapsed):
            print(f"Elapsed     : {elapsed:.2f} h")
        print(f"Heartbeat   : {_fmt_ts(hb.get('timestamp'))}")
    else:
        print("Heartbeat file not found.")

    print("\n=== Latest Metrics ===")
    if metrics:
        epoch = metrics.get("epoch")
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        sharpe = metrics.get("avg_sharpe") or metrics.get("val_sharpe_ratio")
        if epoch is not None:
            print(f"Epoch        : {epoch}")
        if train_loss is not None:
            print(f"Train loss   : {train_loss:.6f}")
        if val_loss is not None:
            print(f"Val loss     : {val_loss:.6f}")
        if sharpe is not None:
            print(f"Sharpe       : {sharpe:.4f}")
        print(f"Metrics time : {_fmt_ts(metrics.get('timestamp'))}")
    else:
        print("Metrics file not found.")


if __name__ == "__main__":
    main()
