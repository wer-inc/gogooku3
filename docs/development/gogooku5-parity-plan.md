# gogooku5 vs gogooku3 Parity Validation Plan

## Objective
Verify that the modular `gogooku5` dataset builder reproduces the key feature columns from the legacy `gogooku3` pipeline over overlapping date ranges before switching training/inference workloads.

## Scope
- Datasets: equity panel covering 2024-01-01 → 2024-01-31 (expand after initial pass)
- Columns: core price fields, margin features, sector/peer/graph/technical/advanced indicators, volatility metrics
- Exclusions (pending migration): earnings, margin weekly, GPU-only graph features, futures/option sentiment

## Approach
1. **Build Baseline**
   - Use `make dataset-bg` (gogooku3) to generate parquet for target window → `output/ml_dataset_YYYYMMDD.parquet`
   - Extract subset columns using helper script `scripts/data/export_subset.py`
2. **Build Modular Output**
   - Run `make -C gogooku5 build START=2024-01-01 END=2024-01-31`
   - Copy resulting `gogooku5/data/output/ml_dataset_latest.parquet`
3. **Column Alignment**
   - Normalise column names (lowercase, consistent prefixes)
   - Drop columns not yet migrated; log omissions in `docs/development/gogooku5-parity-plan.md`
4. **Diff Engine**
   - Use Polars to compute:
     - `abs_diff = |v_g5 - v_g3|`
     - `rel_diff = abs_diff / (|v_g3| + 1e-9)`
   - Summaries: max/mean/std per column, histogram of relative diff buckets
5. **Acceptability Criteria**
   - Absolute diff < 1e-6 for deterministic transforms (price-based)
   - Relative diff < 1e-4 for rolling stats; investigate >1e-3 outliers
6. **Reporting**
   - Store diff summary in `reports/gogooku5/parity_2024-01_parquet.json`
- Document unresolved discrepancies

## Sample Comparison (2024-01-01 one-day synthetic)
```
python tools/compare_datasets.py \
  --g3 tmp_g3.parquet --g5 tmp_g5.parquet \
  --columns close volume --output tmp_diff.json
```
Summary (`tmp_diff.json`):
- `close`: abs_max ≈ 1.0e-5 (below rel threshold), caused by intentional 1e-7 scaling difference
- `volume`: abs_max = 1.0 (rel_max ≈ 1e-3) due to +1 offset in gogooku5 sample

Next run should use real dataset outputs to replace synthetic baseline.

## Next Steps
- Implement comparison script (`tools/compare_datasets.py`) feeding from both parquet paths ✅
- **Run real builds**:
  - gogooku3: `make dataset START=2024-01-01 END=2024-01-31`
  - gogooku5: `make -C gogooku5 build START=2024-01-01 END=2024-01-31`
- Execute diff: `python tools/compare_datasets.py --g3 output/ml_dataset_20240131.parquet --g5 gogooku5/data/output/ml_dataset_latest.parquet --output reports/gogooku5/parity_202401.json`
- Review JSON summary, log columns above threshold, and adjust pipeline or backlog gaps (earnings, margin-weekly, GPU graph, etc.)
- After parity confirmation, update pipeline docs and training configs to consume gogooku5 outputs by default
