#!/usr/bin/env python3
"""
Analyse git working tree changes against documented allowlist groups.

Returns a JSON report indicating whether all changes are covered by known
in-progress initiatives. Exit code:
  0 -> all paths covered (or no changes)
  1 -> unexpected paths detected
  2 -> execution error (git/config issues)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path


@dataclass
class WorktreeEntry:
    status: str
    path: str


def _load_config(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise SystemExit(f"Config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def _normalise_pattern(pattern: str) -> str:
    pattern = pattern.strip()
    if not pattern:
        return pattern
    # Treat trailing slash as "match directory and descendants"
    if pattern.endswith("/"):
        return pattern + "*"
    return pattern


def _matches_any(path: str, patterns: Iterable[str]) -> bool:
    normalised_path = path.replace("\\", "/")
    for pattern in patterns:
        if not pattern:
            continue
        if fnmatch(normalised_path, _normalise_pattern(pattern)):
            return True
    return False


def _parse_git_status() -> list[WorktreeEntry]:
    try:
        proc = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Failed to interrogate git status: {exc}") from exc

    entries: list[WorktreeEntry] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        status = line[:2].strip()
        raw_path = line[3:]
        # Handle rename with format "R  old -> new"
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        entries.append(WorktreeEntry(status=status or line[:2], path=raw_path.strip()))
    return entries


def audit_worktree(config_path: Path) -> dict:
    config = _load_config(config_path)
    groups = config.get("groups", [])
    entries = _parse_git_status()

    report = {
        "total": len(entries),
        "matched": [],
        "unexpected": [],
    }

    if not entries:
        return report

    group_matches: dict[str, dict] = {}

    for entry in entries:
        matched_group = None
        for group in groups:
            patterns = group.get("patterns", [])
            if _matches_any(entry.path, patterns):
                matched_group = group["name"]
                break
        if matched_group is None:
            report["unexpected"].append({"status": entry.status, "path": entry.path})
        else:
            bucket = group_matches.setdefault(
                matched_group,
                {
                    "name": matched_group,
                    "description": next(
                        (
                            g.get("description", "")
                            for g in groups
                            if g.get("name") == matched_group
                        ),
                        "",
                    ),
                    "files": [],
                },
            )
            bucket["files"].append({"status": entry.status, "path": entry.path})

    report["matched"] = [
        {
            "name": data["name"],
            "description": data["description"],
            "count": len(data["files"]),
            "files": data["files"],
        }
        for data in group_matches.values()
    ]
    return report


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to worktree allowlist JSON",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output for CLI use",
    )
    args = parser.parse_args(argv)

    report = audit_worktree(args.config)
    unexpected = report.get("unexpected", [])

    if args.pretty:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
    else:
        json.dump(report, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")

    if unexpected:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
