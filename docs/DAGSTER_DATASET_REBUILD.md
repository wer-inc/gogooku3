# Dagster-Based gogooku5 Dataset Rebuild

This guide explains how to rebuild the gogooku5 dataset (2020-01-01 → latest) via Dagster.  
It replaces ad-hoc shell commands with a reproducible Dagster job that builds chunks, runs the schema gate, and merges the final dataset.

## 1. Environment prerequisites

Ensure the following environment variables are set before launching Dagster:

```bash
export DIM_SECURITY_PATH=/workspace/gogooku3/output_g5/dim_security.parquet
export RAW_MANIFEST_PATH=/workspace/gogooku3/output_g5/raw_manifest.json
export CATEGORICAL_COLUMNS="Code,SecId,SectorCode"
```

The `.env` file used by Dagster should also contain the usual J-Quants credentials and cache settings.

## 2. Command reference

From the repo root:

```bash
cd /workspace/gogooku3
PYTHONPATH=gogooku5/data/src \
  dagster job execute \
    -f gogooku5/data/src/dagster_gogooku5/defs.py \
    -j g5_dataset_rebuild_job \
    -c run_configs/dagster_phase3_full.yaml
```

- `g5_dataset_rebuild_job` runs the `g5_dataset_chunks`, `g5_schema_gate`, and `g5_dataset_full` assets in order.
- The supplied config (`run_configs/dagster_phase3_full.yaml`) builds every chunk from 2020-01-01 to 2025-11-07 into `/workspace/gogooku3/output_g5`.

## 3. Monitoring

- Dagster CLI will stream logs while the job runs.  
- To inspect progress outside of Dagster, use:
  ```bash
  python3 scripts/check_chunk_status.py
  ```
- The chunk status script verifies completion state and schema hash (`89f27cdf7eb9c285` for manifest v1.3.0).

## 4. Post-build steps

After the job finishes successfully:

1. **Validate**  
   `python3 scripts/check_chunk_status.py`
2. **Merge** (automatically done by the job, but re-run if needed)  
   ```bash
   PYTHONPATH=gogooku5/data/src \
     python gogooku5/data/tools/merge_chunks.py \
       --chunks-dir /workspace/gogooku3/output_g5/chunks \
       --output /workspace/gogooku3/output_g5/ml_dataset_full.parquet \
       --validate
   ```
3. **GCS sync**  
   ```bash
   python scripts/sync_multi_dirs_to_gcs.py --dirs output_g5
   ```

## 5. Troubleshooting tips

- **Schema hash mismatch**: regenerate the manifest from a reference chunk or delete stale chunks before rerunning.
- **Dim security not found**: verify the symlink `output/cache/dim_security.parquet` and `DIM_SECURITY_PATH`.
- **Missing targets**: run `add_future_returns.py` after the job to materialise `target_*` columns for Apex Ranker.

By running the Dagster job, the entire dataset rebuild is now reproducible, logged, and ready for scheduling or sensor-based automation.
