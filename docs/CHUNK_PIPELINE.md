# Chunked Dataset Pipeline

This document summarises the quarterly chunk workflow for the gogooku5 dataset
builder.

## Overview

The chunk pipeline splits a long dataset build into quarterly chunks with a
fixed warmup of **85 trading days**. Each chunk is stored under
`output/chunks/<chunk_id>/` alongside metadata and status files. A dedicated
merge tool concatenates completed chunks into the canonical
`ml_dataset_latest.parquet`.

```
└── output/
    ├── chunks/
    │   ├── 2020Q1/
    │   │   ├── ml_dataset.parquet
    │   │   ├── metadata.json
    │   │   └── status.json
    │   └── 2020Q2/
    │       └── ...
    └── ml_dataset_latest.parquet         # produced by merge_chunks.py
```

## Building Chunks

Use the planner-aware CLI to enumerate and build chunks. Warmup handling,
status tracking, and resume semantics are managed automatically.

```bash
# Dry-run planner output
python gogooku5/data/scripts/build_chunks.py --start 2020-01-01 --end 2020-12-31 --dry-run

# Build chunks sequentially (skip completed by default when --resume is set)
python gogooku5/data/scripts/build_chunks.py --start 2020-01-01 --end 2020-12-31 --resume

# Latest-chunk-only or force rebuild
python gogooku5/data/scripts/build_chunks.py --start 2024-01-01 --end 2024-12-31 --latest-only
python gogooku5/data/scripts/build_chunks.py --start 2024-01-01 --end 2024-12-31 --force
```

Shortcuts are provided via `Makefile.dataset`:

```bash
make build-chunks START=2020-01-01 END=2020-12-31 RESUME=1
make build-chunks START=2024-01-01 END=2024-12-31 DRY_RUN=1
```

Chunks write:

- `ml_dataset.parquet` – finalized rows restricted to the output window
- `metadata.json` – schema summary, date range, warmup context, builder meta
- `status.json` – state (`running`, `completed`, `failed`) for resuming

## Merging Chunks

Once the desired chunks report `state=completed`, run the merge tool:

```bash
python gogooku5/data/tools/merge_chunks.py --chunks-dir output/chunks
# or
make merge-chunks
```

The merger validates schema consistency, orders chunks by date, concatenates the
data, updates `ml_dataset_latest.parquet`, and refreshes the standard metadata
and symbolic links. Extra metadata includes per-chunk summaries for auditing. If
any chunk directories are still running or failed, the command exits with an
error; pass `--allow-partial` explicitly if you want to merge completed chunks
while skipping the incomplete ones (a warning is still emitted).

## Maintenance Tips

- `status.json` is authoritative for resume logic; delete or set `state=failed`
  to force reruns without `--force`.
- Run `tools/clean_empty_cache.py` before long batch runs to clear zero-row
  cache artefacts.
- Integration tests in `data/tests` cover planner, builder persistence, CLI
  resume logic, and chunk merging. Run `pytest gogooku5/data/tests` after
  making changes.

For existing single-run workflows, `DatasetBuilder.build()` continues to work
unchanged; chunks simply provide a failure-resilient alternative for large
rebuilds or incremental updates.
