# Research Plan

## Objectives
- Data integrity: eliminate leak paths by clarifying labels, lags, and fills.
- Evaluation: standardize purged walk‑forward with 20‑day embargo and report IC/Sharpe with CIs.
- Modeling: rebalance horizon weights and reduce redundant features.

## Key Questions
- Labels: forward return horizons and exact definitions (1/5/10/20d; log vs simple).
- Lags: weekly/daily disclosures (credit/short/sector) — publish lag tables, verify T+1.
- Fills: forward‑fill scope and masks for long breaks/halts.
- Dividends: confirm adjustment policy for returns.
- Features: prune unstable or collinear indicators (RSI/MACD variants, etc.).

## Method
- Data audit: run `scripts/tools/data_checks.py` on latest parquet; compile lag & fill maps.
- Eval protocol: use `scripts/tools/split_purged_wf.py` to declare folds; 20‑day embargo.
- Baseline snapshot: `scripts/tools/baseline_snapshot.py` for coverage and sanity.
- Ablations: grid horizon weights, add regularization (Huber, dropout, weight decay) and prune features.

## Deliverables
- Audit report (lag table, fill policy, leakage risks) and acceptance gates.
- Eval spec (splits, metrics, significance tests) and baseline metrics tables.
- Ablation matrix and recommendations; rollout/rollback checklist.

## Timeline
- Week 1: scope + audits + label/eval specs
- Week 2: baselines + first ablations (horizon weights/regularization)
- Week 3: feature pruning + confirm gains + roadmap sign‑off

