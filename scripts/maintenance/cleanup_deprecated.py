#!/usr/bin/env python3
"""
Remove deprecated shim scripts after a grace period.

Rules:
  - Scans the 'scripts/' tree for files containing:
      DEPRECATED_SHIM: true
      DEPRECATED_ON: YYYY-MM-DD
      DEPRECATION_WINDOW_DAYS: N
  - If today >= DEPRECATED_ON + N days, the file is deleted.

Usage:
  - Dry run (default):
      python scripts/maintenance/cleanup_deprecated.py
  - Apply deletions:
      python scripts/maintenance/cleanup_deprecated.py --apply

You can schedule via cron to run daily. Example (runs at 03:10):
  10 3 * * * cd /home/ubuntu/gogooku3-standalone && \
      /usr/bin/python3 scripts/maintenance/cleanup_deprecated.py --apply >> _logs/cleanup.log 2>&1
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import re


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"

PATTERN_SHIM = re.compile(r"^DEPRECATED_SHIM:\s*true\s*$", re.IGNORECASE)
PATTERN_ON = re.compile(r"^DEPRECATED_ON:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$", re.IGNORECASE)
PATTERN_WINDOW = re.compile(r"^DEPRECATION_WINDOW_DAYS:\s*([0-9]+)\s*$", re.IGNORECASE)


def parse_deprecation_header(text: str) -> tuple[bool, datetime | None, int | None]:
    shim = False
    on: datetime | None = None
    window: int | None = None
    for line in text.splitlines():
        line = line.strip()
        if PATTERN_SHIM.match(line):
            shim = True
        m = PATTERN_ON.match(line)
        if m:
            on = datetime.strptime(m.group(1), "%Y-%m-%d")
        m = PATTERN_WINDOW.match(line)
        if m:
            window = int(m.group(1))
    return shim, on, window


def find_deprecated_files() -> list[Path]:
    candidates: list[Path] = []
    for path in SCRIPTS_DIR.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        shim, on, window = parse_deprecation_header(text)
        if not shim or on is None or window is None:
            continue
        due = on + timedelta(days=window)
        if datetime.now() >= due:
            candidates.append(path)
    return candidates


def find_deprecated_dirs() -> list[Path]:
    """Find directories marked deprecated via README metadata.

    Looks for a README.md containing:
      DEPRECATED_DIR: true
      DEPRECATED_ON: YYYY-MM-DD
      DEPRECATION_WINDOW_DAYS: N
    """
    victims: list[Path] = []
    for dirpath in SCRIPTS_DIR.iterdir():
        if not dirpath.is_dir():
            continue
        readme = dirpath / "README.md"
        if not readme.exists():
            continue
        text = readme.read_text(encoding="utf-8", errors="ignore")
        if "DEPRECATED_DIR:" not in text:
            continue
        # Reuse file header parser on README text
        # Interpret SHIM as DIR for simplicity
        shim, on, window = parse_deprecation_header(text.replace("DEPRECATED_DIR", "DEPRECATED_SHIM"))
        if not shim or on is None or window is None:
            continue
        due = on + timedelta(days=window)
        if datetime.now() >= due:
            victims.append(dirpath)
    return victims


def main() -> int:
    ap = argparse.ArgumentParser(description="Cleanup deprecated shim scripts")
    ap.add_argument("--apply", action="store_true", help="Actually delete files (default: dry run)")
    args = ap.parse_args()

    victims = find_deprecated_files()
    dir_victims = find_deprecated_dirs()
    if not victims and not dir_victims:
        print("No deprecated scripts or directories due for deletion.")
        return 0

    if victims:
        print("Deprecated scripts due for deletion (>= window days):")
        for v in victims:
            print(f"  - {v}")
    if dir_victims:
        print("Deprecated directories due for deletion (>= window days):")
        for d in dir_victims:
            print(f"  - {d}")

    if args.apply:
        for v in victims:
            try:
                v.unlink()
                print(f"Deleted: {v}")
            except Exception as e:
                print(f"Failed to delete {v}: {e}")
        for d in dir_victims:
            try:
                shutil.rmtree(d)
                print(f"Deleted directory: {d}")
            except Exception as e:
                print(f"Failed to delete directory {d}: {e}")
    else:
        print("(dry-run) Use --apply to remove them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
