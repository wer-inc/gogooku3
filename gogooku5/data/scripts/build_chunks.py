"""CLI for building dataset chunks."""
from __future__ import annotations

import argparse
import json
import os
import sys
import site
import sysconfig
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1] / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _extend_system_site_packages() -> None:
    """Allow venv python to reuse base interpreter packages (pydantic, etc.)."""

    base_prefix = sys.base_prefix
    if sys.prefix == base_prefix:
        return

    seen = set(sys.path)

    def _add_path(path: str | None) -> None:
        if not path:
            return
        if path in seen:
            return
        if not os.path.isdir(path):
            return
        site.addsitedir(path)
        seen.add(path)

    for key in ("purelib", "platlib"):
        path = sysconfig.get_path(key, vars={"base": base_prefix, "platbase": base_prefix})
        _add_path(path)

    major, minor = sys.version_info[:2]
    candidate_dirs = [
        f"/usr/local/lib/python{major}.{minor}/dist-packages",
        f"/usr/local/lib/python{major}/dist-packages",
        f"/usr/lib/python{major}.{minor}/dist-packages",
        f"/usr/lib/python{major}/dist-packages",
        f"/usr/lib/python{major}.{minor}/site-packages",
        f"/usr/lib/python{major}/site-packages",
    ]
    for candidate in candidate_dirs:
        _add_path(candidate)


_extend_system_site_packages()

from builder.chunks import ChunkPlanner, ChunkSpec
from builder.pipelines.dataset_builder import DatasetBuilder
from builder.utils import ensure_env_loaded
from builder.utils.logger import get_logger

LOGGER = get_logger("scripts.build_chunks")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset in quarterly chunks")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--refresh-listed",
        action="store_true",
        help="Refresh listed securities metadata before building the first chunk",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip chunks whose status.json reports state=completed",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if status.json indicates completion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned chunks without executing builds",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Only build the latest chunk covering the end date",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(os.getenv("CHUNK_JOBS", "1")),
        help="Number of parallel workers (currently limited to 1)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    if args.jobs != 1:
        print("⚠️  Parallel chunk builds (jobs>1) are not supported yet. Use --jobs 1.", file=sys.stderr)
        return 1

    ensure_env_loaded()

    planner = ChunkPlanner()
    try:
        chunk_specs = planner.plan(start=args.start, end=args.end)
    except RuntimeError as exc:
        LOGGER.warning(
            "Chunk planner failed to compute warmup (holiday calendar missing?). "
            "Falling back to zero warmup: %s",
            exc,
        )
        planner = ChunkPlanner(
            settings=planner.settings,
            warmup_days=0,
            output_root=planner.output_root,
        )
        chunk_specs = planner.plan(start=args.start, end=args.end)
    if args.latest_only and chunk_specs:
        chunk_specs = [chunk_specs[-1]]

    if args.dry_run:
        _print_plan(chunk_specs)
        return 0

    specs_to_run = [spec for spec in chunk_specs if not _should_skip(spec, args)]
    if not specs_to_run:
        print("✅ No chunks to build (all completed or skipped).")
        return 0

    refresh_flag = args.refresh_listed
    for spec in specs_to_run:
        print(f"🚧 Building chunk {spec.chunk_id}: {spec.output_start} → {spec.output_end}")
        builder = DatasetBuilder()
        builder.build_chunk(spec, refresh_listed=refresh_flag)
        refresh_flag = False  # refresh listed at most once
        print(f"✅ Completed chunk {spec.chunk_id}")

    return 0


def _print_plan(specs: Iterable[ChunkSpec]) -> None:
    print("Chunk build plan:")
    for spec in specs:
        status = _read_status(spec)
        state = status.get("state", "pending") if status else "pending"
        rows = status.get("rows") if status else None
        extra = f"rows={rows}" if rows is not None else ""
        print(f"  - {spec.chunk_id}: {spec.output_start} → {spec.output_end} (state={state} {extra})")


def _should_skip(spec: ChunkSpec, args: argparse.Namespace) -> bool:
    if args.force:
        return False
    if not args.resume:
        return False
    status = _read_status(spec)
    if not status:
        return False
    return status.get("state") == "completed"


def _read_status(spec: ChunkSpec) -> dict | None:
    try:
        return json.loads(spec.status_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return {"state": "invalid"}


if __name__ == "__main__":
    raise SystemExit(main())
