#!/usr/bin/env python3
"""
Fail if any .env* files are tracked, except approved examples.

Allowed:
- .env.example
- .env.*.example

Disallowed examples:
- .env
- .env.local
- .env.production
- .env.optuna
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def git_ls_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, check=True, capture_output=True
    ).stdout
    parts = out.split(b"\x00")
    return [p.decode("utf-8") for p in parts if p]


def is_disallowed_env(path: str) -> bool:
    name = Path(path).name
    if not name.startswith(".env"):
        return False
    if name == ".env.example":
        return False
    # .env.*.example allowed
    if name.startswith(".env.") and name.endswith(".example"):
        return False
    return True


def main() -> int:
    tracked = git_ls_files()
    offenders = [p for p in tracked if is_disallowed_env(p)]
    if offenders:
        print("Disallowed .env* files tracked:\n")
        for p in offenders:
            print(f"- {p}")
        print("\nRemove from index: git rm --cached <path> (keep locally)")
        return 1
    print("No tracked disallowed .env* files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

