Cross‑Sectional Day Ranking (Subproject)

Goal
- Maximize RankIC and improve Sharpe by training on per‑day cross‑sectional ordering instead of pure MSE regression.
- Minimal changes: reuse existing dataset and pipeline with ranking‑friendly settings.

Why this first
- The dataset is a large Date×Code panel with multi‑horizon targets and daily Z‑score normalization. Cross‑sectional ranking losses align directly with evaluation (RankIC), and are robust to non‑stationary scale changes.

What this runs
- Existing integrated pipeline (`scripts/train.py` → `scripts/integrated_ml_training_pipeline.py`) with:
  - Day batching enabled
  - RankIC loss emphasis (Pairwise rank + Rank preserving)
  - Multi‑worker DataLoader for throughput (A100 80GB)
  - bf16 mixed precision, gradient clip, EMA

Quick start
- Precondition: build or link the dataset symlink `output/ml_dataset_latest_full.parquet` (use the main dataset pipeline if needed).

Option A —— foreground quick check (3 epochs):
  make -C ../../ train-quick DATA_PATH=output/ml_dataset_latest_full.parquet BATCH_SIZE=2048

Option B —— full run (background, 120 epochs):
  ./run_ranking.sh --data output/ml_dataset_latest_full.parquet --epochs 120 --batch-size 2048 --lr 2e-4

What the runner sets
- USE_DAY_BATCH=1 (group by date)
- ALLOW_UNSAFE_DATALOADER=1, NUM_WORKERS=12, PERSISTENT_WORKERS=1, PREFETCH_FACTOR=4, PIN_MEMORY=1
- USE_RANKIC=1, RANKIC_WEIGHT=0.5, CS_IC_WEIGHT=0.3, SHARPE_WEIGHT=0.1
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, bf16 mixed precision

Notes
- This subproject does not fork configs; it wraps the existing integrated pipeline with environment knobs that are already supported by the repo’s training scripts.
- Next steps (optional):
  - Add exposure‑neutral penalties to the loss (market/sector neutrality)
  - Introduce confidence‑weighted ranking using distribution heads (Student‑t / quantile)
  - Add a pretraining stage (TS2Vec/Masked) and fine‑tune with day‑ranking
