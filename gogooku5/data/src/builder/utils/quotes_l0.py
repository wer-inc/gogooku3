"""L0 (monthly) quote shard utilities."""
from __future__ import annotations

import contextlib
import io
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import polars as pl

from ..features.utils.lazy_io import lazy_load
from .logger import get_logger

LOGGER = get_logger("quotes_l0")


def month_key(date_str: str) -> str:
    """Convert YYYY-MM-DD to YYYYMM."""

    if len(date_str) < 7:
        raise ValueError(f"Invalid ISO date string: {date_str!r}")
    return f"{date_str[:4]}{date_str[5:7]}"


def month_range(start: str, end: str) -> list[str]:
    """Return inclusive list of YYYYMM between start and end date strings."""

    if start > end:
        raise ValueError(f"month_range requires start <= end, got {start} > {end}")

    start_year = int(start[:4])
    start_month = int(start[4:6]) if len(start) == 6 else int(start[5:7])
    end_year = int(end[:4])
    end_month = int(end[4:6]) if len(end) == 6 else int(end[5:7])

    items: list[str] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        items.append(f"{year:04d}{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return items


def schema_fingerprint(df: pl.DataFrame) -> str:
    """Return deterministic schema fingerprint (column name + dtype)."""

    import hashlib

    parts = "|".join(f"{name}:{dtype}" for name, dtype in df.schema.items())
    return hashlib.md5(parts.encode("utf-8")).hexdigest()  # nosec - cache metadata only


def compute_checksum(df: pl.DataFrame) -> str:
    """Return xxhash digest for dataframe content."""

    try:
        import xxhash
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("xxhash is required for cache checksum computation.") from exc

    algo = xxhash.xxh64()
    buffer = io.BytesIO()
    df.write_parquet(buffer, compression="zstd", statistics=False, use_pyarrow=True)
    algo.update(buffer.getvalue())
    return algo.hexdigest()


def atomic_write_parquet(df: pl.DataFrame, path: Path, create_ipc: bool = True) -> None:
    """Write parquet to temporary file then atomically replace target.

    Also creates IPC cache for 3-5x faster subsequent reads.

    Args:
        df: DataFrame to save
        path: Target parquet file path
        create_ipc: Also create .arrow IPC cache. Defaults to True.
    """

    tmp_path = path.with_suffix(".tmp.parquet")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(tmp_path, compression="zstd", statistics=True, use_pyarrow=True)
    os.replace(tmp_path, path)

    # Create IPC cache for faster reads (3-5x speedup)
    if create_ipc:
        try:
            ipc_path = path.with_suffix(".arrow")
            tmp_ipc_path = ipc_path.with_suffix(".tmp.arrow")
            df.write_ipc(tmp_ipc_path, compression="lz4")
            os.replace(tmp_ipc_path, ipc_path)
            LOGGER.debug("Created IPC cache: %s (3-5x faster reads)", ipc_path)
        except Exception as exc:
            LOGGER.warning("Failed to create IPC cache for %s (non-blocking): %s", path, exc)


@dataclass
class QuoteShard:
    """Metadata describing a single month shard."""

    yyyymm: str
    min_date: str
    max_date: str
    n_rows: int
    checksum: str
    schema_fp: str
    created_at: int


class QuoteShardIndex:
    """SQLite-backed metadata store for quote shards and request windows."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._ensure_initialized()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30.0, isolation_level=None, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    def _ensure_initialized(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            con.execute("PRAGMA temp_store=MEMORY;")
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS shards (
                    yyyymm TEXT PRIMARY KEY,
                    min_date TEXT NOT NULL,
                    max_date TEXT NOT NULL,
                    n_rows INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    schema_fp TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS windows (
                    id TEXT PRIMARY KEY,
                    start TEXT NOT NULL,
                    end TEXT NOT NULL,
                    codes_fp TEXT NOT NULL,
                    coverage REAL NOT NULL,
                    months TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
                """
            )

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            with self._connect() as con:
                con.execute("BEGIN IMMEDIATE;")
                try:
                    yield con
                    con.execute("COMMIT;")
                except Exception:
                    con.execute("ROLLBACK;")
                    raise

    def upsert_shard(self, shard: QuoteShard) -> None:
        """Insert or update shard metadata."""

        with self._transaction() as con:
            con.execute(
                """
                INSERT INTO shards (yyyymm, min_date, max_date, n_rows, checksum, schema_fp, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(yyyymm) DO UPDATE SET
                    min_date=excluded.min_date,
                    max_date=excluded.max_date,
                    n_rows=excluded.n_rows,
                    checksum=excluded.checksum,
                    schema_fp=excluded.schema_fp,
                    created_at=excluded.created_at;
                """,
                (
                    shard.yyyymm,
                    shard.min_date,
                    shard.max_date,
                    shard.n_rows,
                    shard.checksum,
                    shard.schema_fp,
                    shard.created_at,
                ),
            )

    def list_available_months(self) -> set[str]:
        """Return set of months currently cached."""

        with self._connect() as con:
            rows = con.execute("SELECT yyyymm FROM shards;").fetchall()
        return {row["yyyymm"] for row in rows}

    def all_shards(self) -> list[QuoteShard]:
        """Return all shards ordered by creation time."""

        with self._connect() as con:
            rows = con.execute("SELECT * FROM shards ORDER BY created_at ASC;").fetchall()
        return [
            QuoteShard(
                yyyymm=row["yyyymm"],
                min_date=row["min_date"],
                max_date=row["max_date"],
                n_rows=row["n_rows"],
                checksum=row["checksum"],
                schema_fp=row["schema_fp"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_shards(self, months: Iterable[str]) -> list[QuoteShard]:
        """Return metadata for given month keys (missing months ignored)."""

        placeholders = ",".join("?" for _ in months)
        items: list[QuoteShard] = []
        if not placeholders:
            return items
        with self._connect() as con:
            rows = con.execute(
                f"SELECT * FROM shards WHERE yyyymm IN ({placeholders}) ORDER BY yyyymm;",
                list(months),
            ).fetchall()
        for row in rows:
            items.append(
                QuoteShard(
                    yyyymm=row["yyyymm"],
                    min_date=row["min_date"],
                    max_date=row["max_date"],
                    n_rows=row["n_rows"],
                    checksum=row["checksum"],
                    schema_fp=row["schema_fp"],
                    created_at=row["created_at"],
                )
            )
        return items

    def register_window(
        self, window_id: str, *, start: str, end: str, codes_fp: str, coverage: float, months: list[str]
    ) -> None:
        """Record logical request → months mapping."""

        with self._transaction() as con:
            con.execute(
                """
                INSERT INTO windows (id, start, end, codes_fp, coverage, months, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    start=excluded.start,
                    end=excluded.end,
                    codes_fp=excluded.codes_fp,
                    coverage=excluded.coverage,
                    months=excluded.months,
                    created_at=excluded.created_at;
                """,
                (
                    window_id,
                    start,
                    end,
                    codes_fp,
                    coverage,
                    ",".join(months),
                    int(time.time()),
                ),
            )

    def months_for_window(self, window_id: str) -> Optional[list[str]]:
        """Return cached month list for window id if present."""

        with self._connect() as con:
            row = con.execute("SELECT months FROM windows WHERE id = ?;", (window_id,)).fetchone()
        if row is None:
            return None
        return [month for month in row["months"].split(",") if month]

    def delete_shard(self, month: str) -> None:
        """Remove shard metadata row."""

        with self._transaction() as con:
            con.execute("DELETE FROM shards WHERE yyyymm = ?;", (month,))

    def vacuum(self) -> None:
        """Reclaim sqlite file space."""

        with self._connect() as con:
            con.execute("VACUUM;")


class QuoteShardStore:
    """Manage reading/writing normalized month shards."""

    def __init__(self, root: Path, index: QuoteShardIndex) -> None:
        self.root = root
        self.index = index
        self.lock_dir = root / "locks"
        self.lock_dir.mkdir(parents=True, exist_ok=True)

    def _shard_path(self, month: str) -> Path:
        return self.root / "raw" / "quotes" / f"{month}.parquet"

    def shard_path(self, month: str) -> Path:
        """Public accessor for shard path."""

        return self._shard_path(month)

    @contextlib.contextmanager
    def _month_lock(self, month: str) -> Iterator[None]:
        lock_path = self.lock_dir / f"{month}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w", encoding="utf-8") as handle:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def append_month(self, month: str, df: pl.DataFrame) -> QuoteShard:
        """Append data into the given month shard (merge on code/date)."""

        if df.is_empty():
            raise ValueError(f"Cannot append empty dataframe to month {month}")

        target_path = self._shard_path(month)
        with self._month_lock(month):
            if target_path.exists():
                existing = lazy_load(target_path, prefer_ipc=True)
                # Check if schemas match (same columns and types)
                if existing.schema != df.schema:
                    LOGGER.warning(
                        "[CACHE] Month %s schema mismatch: existing has %d columns (%s), "
                        "new has %d columns (%s). Overwriting existing shard.",
                        month,
                        len(existing.columns),
                        ",".join(existing.columns[:5]) + ("..." if len(existing.columns) > 5 else ""),
                        len(df.columns),
                        ",".join(df.columns[:5]) + ("..." if len(df.columns) > 5 else ""),
                    )
                    # Schema mismatch: overwrite with new data
                    df = df.sort(["Date", "Code"])
                else:
                    # Schemas match: merge and deduplicate
                    df = pl.concat([existing, df], how="vertical", rechunk=True)
                    df = df.unique(subset=["Code", "Date"], keep="last").sort(["Date", "Code"])
            else:
                df = df.sort(["Date", "Code"])

            atomic_write_parquet(df, target_path)

        shard_meta = QuoteShard(
            yyyymm=month,
            min_date=df["Date"].min(),
            max_date=df["Date"].max(),
            n_rows=df.height,
            checksum=compute_checksum(df),
            schema_fp=schema_fingerprint(df),
            created_at=int(time.time()),
        )
        self.index.upsert_shard(shard_meta)
        return shard_meta

    def remove_month(self, month: str) -> None:
        """Delete month shard parquet and metadata."""

        path = self._shard_path(month)
        with self._month_lock(month):
            if path.exists():
                path.unlink()
        self.index.delete_shard(month)

    def collect_months(
        self,
        months: Iterable[str],
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        codes: Optional[set[str]] = None,
    ) -> pl.DataFrame:
        """Read requested months with optional predicate pushdown and streaming execution.

        Performance optimizations:
        - Lazy scanning with predicate pushdown (date/code filters)
        - Streaming collection for low memory usage (30-50% reduction)
        - Column pruning ready (extend with columns parameter if needed)
        """

        scans: list[pl.LazyFrame] = []
        code_filter = None
        if codes:
            code_filter = list({str(code) for code in codes if str(code).strip()})

        for month in months:
            path = self._shard_path(month)
            if not path.exists():
                LOGGER.debug("Skipping missing shard %s at %s", month, path)
                continue
            # Prefer IPC cache for 3-5x faster reads
            ipc_path = path.with_suffix(".arrow")
            if ipc_path.exists():
                try:
                    lf = pl.scan_ipc(str(ipc_path))
                    LOGGER.debug("Using IPC cache for shard %s: %s", month, ipc_path)
                except Exception as exc:
                    LOGGER.warning("Failed to scan IPC cache for %s, falling back to Parquet: %s", month, exc)
                    lf = pl.scan_parquet(path.as_posix())
            else:
                lf = pl.scan_parquet(path.as_posix())
            if start:
                lf = lf.filter(pl.col("Date") >= start)
            if end:
                lf = lf.filter(pl.col("Date") <= end)
            if code_filter:
                lf = lf.filter(pl.col("Code").is_in(code_filter))
            scans.append(lf)

        if not scans:
            return pl.DataFrame()

        return (
            pl.concat(scans, how="vertical")
            .unique(subset=["Code", "Date"], keep="last")
            .sort(["Date", "Code"])
            .collect(streaming=True)  # Streaming collection for lower memory usage
            .rechunk()
        )
