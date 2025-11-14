"""CLI utilities for raw data chunking + manifest generation."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from gogooku5.data.tools.raw_source_specs import RAW_SOURCE_SPECS, RawSourceSpec

CHUNK_ID_COL = "__chunk_id__"
CHUNK_DATE_COL = "__chunk_date__"


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def _resolve_sources(selected: Sequence[str] | None) -> list[RawSourceSpec]:
    if not selected:
        return list(RAW_SOURCE_SPECS.values())
    missing = [name for name in selected if name not in RAW_SOURCE_SPECS]
    if missing:
        raise KeyError(f"Unknown sources: {', '.join(sorted(missing))}")
    return [RAW_SOURCE_SPECS[name] for name in selected]


def _resolve_input_path(spec: RawSourceSpec) -> Path:
    paths = sorted(Path().glob(spec.input_glob))
    files = [p for p in paths if p.is_file()]
    if not files:
        raise FileNotFoundError(f"No base parquet files match {spec.input_glob}")
    if spec.prefer_latest:
        return max(files, key=lambda p: p.stat().st_mtime)
    return files[0]


def _chunk_prefix(spec: RawSourceSpec, base_path: Path) -> str:
    if spec.chunk_name_override:
        return spec.chunk_name_override
    return base_path.stem


def _build_chunk_dataframe(
    base_path: Path,
    spec: RawSourceSpec,
    *,
    start: date | None,
    end: date | None,
) -> pl.DataFrame:
    lf = pl.scan_parquet(base_path)
    if spec.date_kind == "string":
        date_expr = pl.col(spec.date_column).str.strptime(
            pl.Date, format=spec.date_format, strict=False
        )
    elif spec.date_kind == "datetime":
        date_expr = pl.col(spec.date_column).dt.date()
    else:
        date_expr = pl.col(spec.date_column).cast(pl.Date)

    lf = lf.with_columns(date_expr.alias(CHUNK_DATE_COL)).filter(
        pl.col(CHUNK_DATE_COL).is_not_null()
    )

    if start:
        lf = lf.filter(pl.col(CHUNK_DATE_COL) >= pl.lit(start))
    if end:
        lf = lf.filter(pl.col(CHUNK_DATE_COL) <= pl.lit(end))

    lf = lf.with_columns(
        pl.format(
            "{}Q{}",
            pl.col(CHUNK_DATE_COL).dt.year(),
            pl.col(CHUNK_DATE_COL).dt.quarter(),
        ).alias(CHUNK_ID_COL)
    )

    df = lf.collect(streaming=True)

    if df.is_empty():
        return df

    sort_cols = list(spec.sort_by) or [spec.date_column]
    df = df.sort(sort_cols)
    if CHUNK_DATE_COL in df.columns:
        df = df.drop(CHUNK_DATE_COL)
    return df


def _chunk_source(
    spec: RawSourceSpec,
    *,
    start: date | None,
    end: date | None,
    overwrite: bool,
) -> None:
    try:
        base_path = _resolve_input_path(spec)
    except FileNotFoundError as exc:
        print(f"[WARN] {spec.name}: {exc}")
        return

    chunk_prefix = _chunk_prefix(spec, base_path)
    output_dir = spec.chunk_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _build_chunk_dataframe(base_path, spec, start=start, end=end)
    if df.is_empty():
        print(f"[WARN] {spec.name}: no rows after filtering; skipping")
        return

    partitions = df.partition_by(
        CHUNK_ID_COL, maintain_order=True, include_key=False, as_dict=True
    )
    if not partitions:
        print(f"[WARN] {spec.name}: no chunkable partitions found")
        return

    print(
        f"[INFO] {spec.name}: {len(partitions)} quarter partitions "
        f"from {base_path.name}"
    )
    written = 0
    for key in sorted(partitions.keys(), key=lambda k: k[0]):
        chunk_id = key[0]
        chunk_df = partitions[key]
        out_path = output_dir / f"{chunk_prefix}_{chunk_id}.parquet"
        if out_path.exists() and not overwrite:
            print(f"  [SKIP] {chunk_id} -> {out_path} (exists)")
            continue
        chunk_df.write_parquet(out_path, compression="zstd")
        print(f"  [WRITE] {chunk_id} -> {out_path} ({chunk_df.height:,} rows)")
        written += 1

    if written == 0:
        print(f"[INFO] {spec.name}: no files written (all present)")


def chunk_command(args: argparse.Namespace) -> None:
    start = _parse_date(args.start)
    end = _parse_date(args.end)
    sources = _resolve_sources(args.sources)
    for spec in sources:
        _chunk_source(spec, start=start, end=end, overwrite=args.overwrite)


def _relative_to_raw(path: Path, raw_root: Path) -> Path:
    path = path.resolve()
    raw_root = raw_root.resolve()
    return path.relative_to(raw_root)


def _extract_chunk_id(path: Path) -> str:
    parts = path.stem.split("_")
    if not parts:
        raise ValueError(f"Unable to infer chunk id from {path}")
    candidate = parts[-1]
    if len(candidate) == 6 and candidate.startswith("20") and candidate[-2] == "Q":
        return candidate
    if candidate.startswith("20") and "Q" in candidate:
        return candidate
    raise ValueError(f"Invalid chunk id inferred from {path}")


def _compute_chunk_stats(path: Path, spec: RawSourceSpec) -> tuple[str | None, str | None, int]:
    if spec.date_kind == "string":
        date_expr = pl.col(spec.date_column).str.strptime(
            pl.Date, format=spec.date_format, strict=False
        )
    elif spec.date_kind == "datetime":
        date_expr = pl.col(spec.date_column).dt.date()
    else:
        date_expr = pl.col(spec.date_column).cast(pl.Date)

    stats = (
        pl.scan_parquet(path)
        .select(
            date_expr.min().alias("start"),
            date_expr.max().alias("end"),
            pl.len().alias("rows"),
        )
        .collect()
    )
    start_val = stats["start"][0]
    end_val = stats["end"][0]
    rows = int(stats["rows"][0])

    def _fmt(value: date | datetime | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.date().isoformat()
        return value.isoformat()

    return _fmt(start_val), _fmt(end_val), rows


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sorted_chunk_files(spec: RawSourceSpec, chunk_prefix: str) -> list[Path]:
    if not spec.chunk_dir.exists():
        return []
    pattern = f"{chunk_prefix}_20??Q[1-4].parquet"
    return sorted(spec.chunk_dir.glob(pattern))


def manifest_command(args: argparse.Namespace) -> None:
    raw_root = Path(args.raw_root)
    manifest_path = Path(args.manifest)
    sources = _resolve_sources(args.sources)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_root": str(raw_root),
        "sources": {},
    }

    for spec in sources:
        try:
            base_path = _resolve_input_path(spec)
        except FileNotFoundError as exc:
            print(f"[WARN] {spec.name}: {exc}")
            continue
        chunk_prefix = _chunk_prefix(spec, base_path)
        chunk_files = _sorted_chunk_files(spec, chunk_prefix)
        if not chunk_files:
            print(f"[WARN] {spec.name}: no chunk files found under {spec.chunk_dir}")
            continue

        entries = []
        for path in chunk_files:
            chunk_id = _extract_chunk_id(path)
            start, end, rows = _compute_chunk_stats(path, spec)
            rel_path = str(_relative_to_raw(path, raw_root))
            entry = {
                "source": spec.name,
                "file": rel_path.replace(os.sep, "/"),
                "start": start,
                "end": end,
                "chunk_id": chunk_id,
                "rows": rows,
                "hash": _file_hash(path),
                "format": "parquet",
            }
            entries.append(entry)

        entries.sort(key=lambda item: (item["start"] or "", item["chunk_id"]))
        payload["sources"][spec.name] = entries
        print(f"[INFO] {spec.name}: recorded {len(entries)} chunks ({chunk_prefix}_*)")

    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[OK] Wrote manifest -> {manifest_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Raw data chunk/manifest utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chunk_parser = subparsers.add_parser("chunk", help="Chunk raw parquet sources by quarter.")
    chunk_parser.add_argument(
        "--sources",
        nargs="+",
        help="Subset of sources to process (default: all).",
    )
    chunk_parser.add_argument("--start", help="Optional ISO start date filter (YYYY-MM-DD).")
    chunk_parser.add_argument("--end", help="Optional ISO end date filter (YYYY-MM-DD).")
    chunk_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite chunk files even if they already exist.",
    )
    chunk_parser.set_defaults(func=chunk_command)

    manifest_parser = subparsers.add_parser(
        "manifest", help="Regenerate output/raw_manifest.json from chunk directories."
    )
    manifest_parser.add_argument(
        "--manifest",
        default="output/raw_manifest.json",
        help="Destination manifest path (default: %(default)s).",
    )
    manifest_parser.add_argument(
        "--raw-root",
        default="output/raw",
        help="Base directory used by RawDataStore (default: %(default)s).",
    )
    manifest_parser.add_argument(
        "--sources",
        nargs="+",
        help="Subset of sources to include (default: all).",
    )
    manifest_parser.set_defaults(func=manifest_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
