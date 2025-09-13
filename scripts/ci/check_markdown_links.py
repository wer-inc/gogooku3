#!/usr/bin/env python3
"""
Check Markdown files for broken local links/images (no network requests).

Rules:
- Validate relative file links and images exist.
- Ignore external links (http/https/mailto/ftp) and anchor-only (#...).
- Exclude archive/** and typical runtime dirs.

Usage: python scripts/ci/check_markdown_links.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]

MD_SOURCES: list[Path] = [
    REPO / "docs",
    REPO / "README.md",
    REPO / "AGENTS.md",
    REPO / "MIGRATION.md",
    REPO / "CLAUDE.md",
]

EXCLUDES_DIRS = {".git", "archive", "_logs", "output", "runs", "mlruns", "wandb"}

LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<link>[^)]+)\)")
IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")


def unescape_markdown(path: str) -> str:
    # Unescape common markdown escapes in link URLs
    return re.sub(r"\\([()_#\-\s])", r"\1", path)


def iter_md_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".md":
            yield root
        elif root.is_dir():
            for p in root.rglob("*.md"):
                if any(part in EXCLUDES_DIRS for part in p.relative_to(REPO).parts):
                    continue
                yield p


def is_external(url: str) -> bool:
    return url.startswith(("http://", "https://", "mailto:", "ftp://", "tel:"))


def normalize_target(md_file: Path, target: str) -> Path | None:
    # Strip anchors and query
    path_only = target.split("#", 1)[0].split("?", 1)[0]
    if not path_only or path_only.startswith("#"):
        return None
    if is_external(path_only):
        return None
    # Absolute from repo root
    if path_only.startswith("/"):
        return (REPO / path_only.lstrip("/")).resolve()
    # Relative to md file
    return (md_file.parent / path_only).resolve()


def main() -> int:
    offenders: list[str] = []
    for md in iter_md_files(MD_SOURCES):
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pattern, group in ((LINK_RE, "link"), (IMG_RE, "src")):
            for m in pattern.finditer(text):
                raw = m.group(group).strip()
                raw = unescape_markdown(raw)
                target = normalize_target(md, raw)
                if target is None:
                    continue
                # Skip excluded directories
                try:
                    rel_parts = target.relative_to(REPO).parts
                except Exception:
                    # Target outside repo; ignore
                    continue
                if any(part in EXCLUDES_DIRS for part in rel_parts):
                    continue
                if not target.exists():
                    offenders.append(f"{md.relative_to(REPO)} -> {raw}")
    if offenders:
        print("Broken local links/images found (no network checks):\n")
        for o in offenders[:200]:
            print(f"- {o}")
        return 1
    print("Markdown local links/images OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
