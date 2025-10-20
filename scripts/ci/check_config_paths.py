#!/usr/bin/env python3
"""
Fail CI if legacy config paths are referenced in tracked, non-archive files.

Checked patterns (forbid):
- configs/training/production.yaml
- configs/model/atft/train.yaml
- safe_production.yaml (outside archive/)

Scope:
- docs/**, README.md, CLAUDE.md, docs/architecture/migration.md, docs/development/agents.md, scripts/**, src/**
- Excludes: archive/** and .git/**
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

FORBIDDEN = [
    r"configs/training/production\.ya?ml",
    r"configs/model/atft/train\.ya?ml",
]

# Allow 'safe_production.yaml' only under archive/
FORBID_SAFE_PROD = r"safe_production\.ya?ml"

SEARCH_ROOTS = [
    REPO / "docs",
    REPO / "scripts",
    REPO / "src",
    REPO / "README.md",
    REPO / "CLAUDE.md",
    REPO / "docs" / "architecture" / "migration.md",
    REPO / "docs" / "development" / "agents.md",
]

EXCLUDES_DIRS = {".git", "archive", "_logs", "output", "runs", "mlruns", "wandb"}

TEXT_EXTS = {
    ".md",
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".txt",
}


def iter_text_files(roots: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            yield root
            continue
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_dir():
                if p.name in EXCLUDES_DIRS:
                    continue
                if any(part in EXCLUDES_DIRS for part in p.relative_to(REPO).parts):
                    continue
                continue
            if p.suffix.lower() in TEXT_EXTS:
                if any(part in EXCLUDES_DIRS for part in p.relative_to(REPO).parts):
                    continue
                if p in seen:
                    continue
                seen.add(p)
                yield p


def main() -> int:
    forb_patterns = [re.compile(p) for p in FORBIDDEN]
    safe_prod = re.compile(FORBID_SAFE_PROD)

    offenders: list[tuple[str, int, str]] = []

    self_path = Path(__file__).resolve()
    for f in iter_text_files(SEARCH_ROOTS):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = f.relative_to(REPO)
        # Skip this checker script itself
        if f.resolve() == self_path:
            continue
        # Skip anything in archive/ explicitly (already excluded above, but be safe)
        if str(rel).startswith("archive/"):
            continue
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            # Standard forbidden patterns
            if any(p.search(line) for p in forb_patterns):
                offenders.append((str(rel), idx, line.strip()))
            # 'safe_production.yaml' anywhere outside archive
            if safe_prod.search(line):
                offenders.append((str(rel), idx, line.strip()))

    if offenders:
        print("Forbidden legacy config path references found:\n")
        for path, lineno, snippet in offenders[:200]:
            print(f"- {path}:{lineno}: {snippet}")
        print("\nPlease replace with: configs/atft/train/production.yaml")
        return 1
    print("No legacy config path references found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
