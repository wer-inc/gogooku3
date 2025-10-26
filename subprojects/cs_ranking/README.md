Cross‑Sectional Day Ranking (Subproject)

目的（Purpose）
- 同一営業日内の銘柄の「順位」を直接最適化し、RankIC を最短距離で引き上げつつ Sharpe の底上げを狙う。
- 既存のクリーンなデータセット（T‑1ラグ、日次Zスコア）をそのまま活用し、最小改修で価値を出す。

方針（Policy / Strategy）
- 学習単位を Day（USE_DAY_BATCH）に統一し、Pairwise ランク損失（RankNet/LambdaRank系）＋RankIC 正則化を主軸にする。
- A100 80GB を前提にマルチワーカ＋大バッチ（bf16）でスループット最適化。early‑stopping/EMA/勾配クリップで安定化。
- 評価は固定検証窓の purged walk‑forward＋20日 embargo。再現2回で採択（実験間の公平性を担保）。
- フェーズ計画: 1) Dayランキング（本サブプロ） → 2) 分布ヘッド併設（Student‑t/分位；confidence‑weighted ranking） → 3) 自己教師ありpretrain → 4) 余力で資産トークン注意/E2Eポートフォリオ。

なぜこの順序か（Rationale）
- 本データは Date×Code のクロスセクション依存が強く、順位の安定性が価値。ランキング損失は評価指標（RankIC）と目的が一致し、スケール非定常にも頑健。
- 分布ヘッドは Sharpe の分母（リスク）を締める補完軸として相性がよく、段階的拡張が容易。

What this runs
- Existing integrated pipeline (`scripts/train.py` → `scripts/integrated_ml_training_pipeline.py`) with:
  - Day batching enabled
  - RankIC loss emphasis (Pairwise rank + Rank preserving)
  - Multi‑worker DataLoader for throughput (A100 80GB)
  - bf16 mixed precision, gradient clip, EMA

成果指標（Metrics & Targets）
- Primary: Val RankIC@5d（最後10エポックの平均）
- Secondary: Val Sharpe（コスト含む、年率化）、Q5–Q1 スプレッド、Turnover/コスト感度
- 初期目標（ベースライン比）: RankIC +0.01〜0.03、Sharpe +0.03〜0.08（120ep）

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

安全策 / ガードレール（Safety & Guardrails）
- データリーク回避: T‑1ラグを厳守。purged/embargo split を固定（再学習で不変）。
- 安定性: bf16、勾配クリップ、EMA、OOM時の自動リトライ（パイプライン既定）。
- 再現性: 乱数固定、同一検証窓、2回以上の再実行で採択判断。
