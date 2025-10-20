#!/usr/bin/env python3
"""
Validate external links in Markdown files by performing HTTP HEAD/GET requests.

Behavior:
- Only checks external links (http/https). Skips local anchors and relative paths.
- HEAD first; fallback to GET on 405/403/>=400. Timeout per request ~5s.
- Allows 2xx and 3xx. Fails otherwise.
- Skips localhost/127.0.0.1/0.0.0.0 links.

Note: This is intended for CI (GitHub Actions) where network is available.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import re
from collections.abc import Iterable
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[2]
MD_SOURCES = [
    REPO / "docs",
    REPO / "README.md",
    REPO / "CLAUDE.md",
    REPO / "docs" / "development" / "agents.md",
    REPO / "docs" / "architecture" / "migration.md",
]

LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<link>[^)]+)\)")
IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")

SKIP_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}
# Common domains that frequently throttle or block bots; skip to reduce flakiness.
ALLOWLIST_DOMAINS = {
    "img.shields.io",
    "github.com",
    "python.org",
    "pytorch.org",
    "developer.nvidia.com",
    "pypi.org",
    "readthedocs.io",
}
TIMEOUT = 5
MAX_WORKERS = 12


def iter_md_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".md":
            yield root
        elif root.is_dir():
            for p in root.rglob("*.md"):
                # Skip archive and VCS directories
                parts = p.relative_to(REPO).parts
                if parts and parts[0] in {"archive", ".git"}:
                    continue
                yield p


def external_links(md_file: Path) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    text = md_file.read_text(encoding="utf-8", errors="ignore")
    for pattern, group in ((LINK_RE, "link"), (IMG_RE, "src")):
        for m in pattern.finditer(text):
            url = m.group(group).strip()
            if url.startswith(("http://", "https://")):
                out.append((md_file, url))
    return out


def host_of(url: str) -> str | None:
    try:
        return re.split(r"/", url.split("://", 1)[1], 1)[0]
    except Exception:
        return None


def check_url(url: str) -> tuple[bool, int | None, str | None]:
    try:
        h = host_of(url)
        # Extend allowlist via env var (comma-separated)
        extra = os.environ.get("GOGOOKU_LINKCHECK_ALLOW", "").strip()
        if extra:
            for dom in extra.split(","):
                dom = dom.strip()
                if dom:
                    ALLOWLIST_DOMAINS.add(dom)

        if h and (
            h in SKIP_HOSTS
            or h in ALLOWLIST_DOMAINS
            or (not any(ch.isalpha() for ch in h))
        ):
            return True, None, None
        # HEAD first
        r = requests.head(url, allow_redirects=True, timeout=TIMEOUT)
        if 200 <= r.status_code < 400:
            return True, r.status_code, None
        # Fallback to GET for sites that don't support HEAD
        r = requests.get(url, allow_redirects=True, timeout=TIMEOUT, stream=True)
        if 200 <= r.status_code < 400:
            return True, r.status_code, None
        return False, r.status_code, r.reason
    except requests.RequestException as e:
        return False, None, str(e)


def main() -> int:
    links: list[tuple[Path, str]] = []
    for md in iter_md_files(MD_SOURCES):
        links.extend(external_links(md))
    # Deduplicate
    seen = set()
    uniq: list[tuple[Path, str]] = []
    for md, url in links:
        key = (str(md), url)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((md, url))

    failures: list[tuple[Path, str, int | None, str | None]] = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(check_url, url): (md, url) for md, url in uniq}
        for fut in cf.as_completed(futs):
            md, url = futs[fut]
            ok, status, reason = fut.result()
            if not ok:
                failures.append((md, url, status, reason))

    if failures:
        print("External link check failed:\n")
        for md, url, status, reason in failures[:200]:
            loc = md.relative_to(REPO)
            print(f"- {loc} -> {url} [{status}] {reason or ''}")
        return 1
    print("External links OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
