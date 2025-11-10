"""CLI for building dataset chunks."""
from __future__ import annotations

import argparse
import json
import os
import site
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any, Iterable, List

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

from builder.chunks import ChunkPlanner, ChunkSpec  # noqa: E402
from builder.pipelines.dataset_builder import DatasetBuilder  # noqa: E402
from builder.utils import ensure_env_loaded  # noqa: E402
from builder.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("scripts.build_chunks")


SCRIPT_PATH = Path(__file__).resolve()


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dataset in chunked batches")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
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
    parser.add_argument(
        "--chunk-months",
        type=int,
        default=int(os.getenv("CHUNK_MONTHS", "3")),
        help="Calendar months per chunk (default: 3; set to 1 for monthly safe mode)",
    )
    parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Disable per-chunk process isolation (testing/debug only)",
    )
    parser.add_argument(
        "--chunk-spec",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if not args.chunk_spec and (args.start is None or args.end is None):
        parser.error("--start and --end are required unless --chunk-spec is provided")
    return args


def _serialize_chunk_spec(spec: ChunkSpec) -> dict[str, Any]:
    return {
        "chunk_id": spec.chunk_id,
        "input_start": spec.input_start,
        "input_end": spec.input_end,
        "output_start": spec.output_start,
        "output_end": spec.output_end,
        "output_dir": str(spec.output_dir),
    }


def _load_chunk_spec(path: Path) -> ChunkSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ChunkSpec(
        chunk_id=payload["chunk_id"],
        input_start=payload["input_start"],
        input_end=payload["input_end"],
        output_start=payload["output_start"],
        output_end=payload["output_end"],
        output_dir=Path(payload["output_dir"]),
    )


def _run_single_chunk(spec: ChunkSpec, refresh_listed: bool) -> None:
    ensure_env_loaded()
    builder = DatasetBuilder()
    builder.build_chunk(spec, refresh_listed=refresh_listed)


def _run_chunk_subprocess(spec: ChunkSpec, refresh_listed: bool) -> None:
    """Execute chunk build via a fresh Python interpreter."""

    payload = _serialize_chunk_spec(spec)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
        json.dump(payload, tmp)
        tmp.flush()
        spec_path = Path(tmp.name)

    cmd = [sys.executable, str(SCRIPT_PATH), "--chunk-spec", str(spec_path)]
    if refresh_listed:
        cmd.append("--refresh-listed")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Chunk {spec.chunk_id} failed (exit_code={result.returncode})")
    finally:
        try:
            spec_path.unlink(missing_ok=True)
        except OSError:
            pass


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    if args.chunk_spec:
        spec = _load_chunk_spec(Path(args.chunk_spec))
        try:
            _run_single_chunk(spec, refresh_listed=args.refresh_listed)
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            LOGGER.error("Chunk %s failed: %s", spec.chunk_id, exc)
            return 1
        return 0

    if args.jobs != 1:
        print("⚠️  Parallel chunk builds (jobs>1) are not supported yet. Use --jobs 1.", file=sys.stderr)
        return 1

    ensure_env_loaded()
    use_isolation = not args.no_isolation

    planner = ChunkPlanner(months_per_chunk=args.chunk_months)
    try:
        chunk_specs = planner.plan(start=args.start, end=args.end)
    except RuntimeError as exc:
        LOGGER.warning(
            "Chunk planner failed to compute warmup (holiday calendar missing?). " "Falling back to zero warmup: %s",
            exc,
        )
        planner = ChunkPlanner(
            settings=planner.settings,
            warmup_days=0,
            output_root=planner.output_root,
            months_per_chunk=planner.months_per_chunk,
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
        try:
            if use_isolation:
                _run_chunk_subprocess(spec, refresh_flag)
            else:
                builder = DatasetBuilder()
                builder.build_chunk(spec, refresh_listed=refresh_flag)
        except Exception as exc:  # pragma: no cover - surfaced to CLI
            print(f"❌ Chunk {spec.chunk_id} failed: {exc}", file=sys.stderr)
            return 1
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
