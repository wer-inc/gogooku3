# Evaluation Protocol

## Split Strategy
- Purged walk‑forward with 20‑day embargo to prevent look‑ahead via overlap.
- Use chronological folds of the equity trading calendar; no shuffling.

## Tools
- Split preview: `python scripts/tools/split_purged_wf.py --dataset output/ml_dataset_latest_full.parquet --n-splits 5 --embargo-days 20`
- Data checks: `python scripts/tools/data_checks.py output/ml_dataset_latest_full.parquet`
- Snapshot: `python scripts/tools/baseline_snapshot.py output/ml_dataset_latest_full.parquet`
- Research bundle (snapshot + checks + splits + metrics): `make research-plus DATASET=output/ml_dataset_latest_full.parquet`
- Research report (with optional fold-level section): `make research-report DATASET=output/ml_dataset_latest_full.parquet REPORT=output/reports/research_report.md`
  - Uses `output/eval_splits_$(NSPLITS)fold_$(EMBARGO)d.json` if present to append fold-level top factors (RankIC mean ± CI95) tables.

## Metrics
- RankIC (Spearman) by horizon, ICIR, hit rate, Sharpe (simple daily portfolio, 0 RF).
- Report 95% CIs via bootstrap; note p‑values for changes (Diebold‑Mariano if needed).

## Label Policy
- Forward returns (1/5/10/20d) defined explicitly; last N rows per Code are masked.
- Only past‑available features; EOD data effective T+1 (as‑of/backward join).

## Acceptance Gates
- No leakage flags; uniqueness OK; high‑null columns reviewed or masked.
- Baseline metrics frozen; improvements must be reproduced under this protocol.
