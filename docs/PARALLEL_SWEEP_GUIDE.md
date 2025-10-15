# Parallel Sweep & Auto-Optimization Guide

## Overview

自動化されたSharpe最適化パイプライン。Variance Collapse回避とSharpe最大化を並列探索・自動選抜で実現。

**Total ETA**: 6-8時間（並列スイープ45-60分 + 本学習5-7時間）

---

## Quick Start

### Option 1: 完全自動実行（推奨）

```bash
# 全自動: Sweep → Eval → Training → Backtest
bash scripts/auto_sharpe_optimization.sh --yes

# 確認プロンプトあり（デフォルト）
bash scripts/auto_sharpe_optimization.sh
```

### Option 2: 段階的実行

```bash
# Step 1: 並列スイープ（45-60分）
bash scripts/parallel_sweep.sh

# Step 2: 結果評価・自動選抜（2-3分）
python scripts/evaluate_sweep_results.py --sweep-dir output/sweep_results

# Step 3: ベスト設定で80-epoch本学習（5-7時間）
bash scripts/run_best_config.sh output/sweep_results

# Step 4: バックテスト（5-10分）
python scripts/backtest_sharpe_model.py \
  --checkpoint output/checkpoints/best_model.pth \
  --data-path output/ml_dataset_latest_full.parquet \
  --output-dir output/backtest_production
```

### Option 3: 既存スイープ結果を使う

```bash
# スイープをスキップして本学習のみ
bash scripts/auto_sharpe_optimization.sh --skip-sweep --yes
```

---

## 各スクリプトの詳細

### 1. `parallel_sweep.sh` - 並列スタビリティスイープ

**目的**: Variance Collapse回避設定を素早く特定

**Grid Search空間**:
```bash
TURNOVER_WEIGHT:   [0.0, 0.025, 0.05]       # 3値
PRED_VAR_WEIGHT:   [0.5, 0.8, 1.0]          # 3値
OUTPUT_NOISE_STD:  [0.02, 0.03]             # 2値
RANKIC_WEIGHT:     [0.2, 0.3]               # 2値
---------------------------------------------------------
Total: 3 × 3 × 2 × 2 = 36 configurations
```

**共通設定（Collapse防止）**:
```bash
FEATURE_CLIP_VALUE=10
DEGENERACY_GUARD=1
HEAD_NOISE_STD=0.02
PRED_VAR_MIN=0.012
CS_IC_WEIGHT=0.25
```

**実行パラメータ**:
- Epochs: 5（速度優先）
- Batch size: 2048
- Parallel jobs: 4（同時実行数、GPU memoryに応じて調整）
- Workers: 4（DataLoaderマルチワーカー有効）

**出力**:
```
output/sweep_results/
├── logs/
│   ├── tw0p0_pvw0p5_onu0p02_rw0p2.log
│   ├── tw0p0_pvw0p5_onu0p02_rw0p2.meta
│   └── ... (36 pairs)
└── checkpoints/ (optional)
```

**カスタマイズ**:
```bash
# 並列数を増やす（GPUメモリ十分な場合）
MAX_PARALLEL_JOBS=8

# Grid空間を拡大
TURNOVER_WEIGHTS=(0.0 0.01 0.025 0.05 0.1)
PRED_VAR_WEIGHTS=(0.3 0.5 0.8 1.0)
```

---

### 2. `evaluate_sweep_results.py` - 結果評価・自動選抜

**Gate Criteria（必須条件）**:
1. `pred_std > 0.010` - Variance Collapse回避
2. `Val Sharpe > -0.01` - 極端に悪くない
3. `Val RankIC > 0.02` - 予測力あり

**Composite Score（上位ランク決定）**:
```python
score = 0.3 × pred_std_norm      # Collapse回避 (30%)
      + 0.4 × sharpe_norm        # 収益性 (40%)
      + 0.3 × rankic_norm        # 予測力 (30%)
```

**出力**:
```
output/sweep_results/
├── all_results.csv           # 全36設定の結果
├── top_configs.csv           # 上位4設定（スコア順）
└── top_config_ids.txt        # ベスト設定ID（1行目がTop 1）
```

**使用例**:
```bash
# Top 4を選抜（デフォルト）
python scripts/evaluate_sweep_results.py --sweep-dir output/sweep_results

# Top 10を選抜
python scripts/evaluate_sweep_results.py --sweep-dir output/sweep_results --top-k 10

# 結果確認
head -1 output/sweep_results/top_config_ids.txt
# => tw0p025_pvw0p8_onu0p03_rw0p3
```

**カスタマイズ**:
```python
# Gate criteriaを緩和（evaluate_sweep_results.py内で編集）
gate_pass = (
    metrics["pred_std"] > 0.008          # 0.010 → 0.008
    and metrics["final_sharpe"] > -0.05  # -0.01 → -0.05
    and metrics["final_rankic"] > 0.01   # 0.02 → 0.01
)
```

---

### 3. `run_best_config.sh` - 2段階本学習

**Stage 1: Variance Bootstrap（5 epochs）**

目的: 安定したpred_std > 0.01を確立

```bash
TURNOVER_WEIGHT=0.0           # ターンオーバーペナルティなし
PRED_VAR_WEIGHT=1.0           # 分散保存を最大化
OUTPUT_NOISE_STD=0.03         # ノイズ注入（退化防止）
HEAD_NOISE_STD=0.02
```

**Stage 2: Sharpe Optimization（75 epochs）**

目的: Sharpe最大化（分散を維持しながら）

```bash
TURNOVER_WEIGHT=0.05-0.10     # Stage 1の2倍（スイープ値の2倍、最低0.05）
PRED_VAR_WEIGHT=0.4           # 緩和（1.0 → 0.4）
USE_SWA=1                     # Stochastic Weight Averaging
SWA_START_EPOCH=60            # 60 epoch目から開始
SNAPSHOT_ENS=1                # Snapshot Ensemble有効
SNAPSHOT_NUM=4                # 4スナップショット保存
```

**出力**:
```
output/production_training/
├── logs/
│   ├── stage1_bootstrap.log
│   └── stage2_sharpe_optimization.log
└── checkpoints/
    └── stage1_final.pth

output/checkpoints/
├── best_model.pth            # Best val Sharpe
├── epoch_080_*.pth           # Final epoch
└── snapshot_*.pth            # ENSスナップショット
```

**実行時間**:
- Stage 1: 15-20分
- Stage 2: 5-7時間
- Total: ~5.5-7.5時間

---

### 4. `backtest_sharpe_model.py` - トランザクションコスト込みバックテスト

**実装済みコストモデル**:
```python
TransactionCostModel:
  - Base cost: 10 bps (0.1% per trade)
  - Market impact: k × √(participation_rate)
  - Slippage: 2 bps (bid-ask spread)
```

**Portfolio Strategy**:
- Top 20%: Equal-weight long
- Bottom 20%: Equal-weight short（`--long-only`でショート無効化）
- Middle 60%: Zero weight

**Performance Metrics**:
- Sharpe ratio (annualized)
- Sortino ratio (downside risk only)
- Calmar ratio (return / max drawdown)
- IC / RankIC validation
- Daily turnover
- Transaction cost drag

**使用例**:
```bash
# 学習済みモデルでバックテスト
python scripts/backtest_sharpe_model.py \
  --checkpoint output/checkpoints/best_model.pth \
  --data-path output/ml_dataset_latest_full.parquet \
  --output-dir output/backtest_production

# 最近250日（約1年）だけテスト
python scripts/backtest_sharpe_model.py \
  --checkpoint output/checkpoints/best_model.pth \
  --data-path output/ml_dataset_latest_full.parquet \
  --days 250 \
  --output-dir output/backtest_1year

# Fast mode（モデルなし、実際のリターンを"予測"として使用）
python scripts/backtest_sharpe_model.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --days 100 \
  --fast-mode \
  --output-dir output/backtest_test
```

**出力**:
```
output/backtest_production/
├── daily_results.csv          # 日次パフォーマンス
└── summary_metrics.csv        # サマリメトリクス
```

---

## Troubleshooting

### Issue 1: 並列スイープでGPU OOM

**症状**:
```
RuntimeError: CUDA out of memory
```

**解決策**:
```bash
# parallel_sweep.sh内で調整
MAX_PARALLEL_JOBS=2            # 4 → 2に削減
export BATCH_SIZE=1024         # 2048 → 1024に削減
```

### Issue 2: DataLoader worker crashes

**症状**:
```
RuntimeError: DataLoader worker (pid XXXXX) is killed by signal: Aborted.
```

**解決策**:
```bash
# parallel_sweep.sh内で調整
export FORCE_SINGLE_PROCESS=1
export NUM_WORKERS=0           # マルチワーカー無効化
```

### Issue 3: 全設定がGate Criteriaで落ちる

**症状**:
```
❌ No configurations passed gate criteria!
```

**解決策1: Gate基準を緩和**
```python
# evaluate_sweep_results.py内で編集
gate_pass = (
    metrics["pred_std"] > 0.008          # より緩く
    and metrics["final_sharpe"] > -0.05
    and metrics["final_rankic"] > 0.01
)
```

**解決策2: Grid空間を拡大**
```bash
# parallel_sweep.sh内で追加
PRED_VAR_WEIGHTS=(0.3 0.5 0.8 1.0 1.5)  # より高い値も試す
OUTPUT_NOISE_STDS=(0.01 0.02 0.03 0.04) # ノイズ範囲拡大
```

### Issue 4: Stage 2でVariance Collapse再発

**症状**:
Stage 2後半でpred_std < 0.01に低下

**解決策**:
```bash
# run_best_config.sh内で調整
export PRED_VAR_WEIGHT=0.6     # 0.4 → 0.6に維持
export TURNOVER_WEIGHT=0.03    # 0.05 → 0.03に削減
```

---

## Performance Expectations

### 並列スイープ結果（5 epochs）

期待される通過率:
```
Total: 36 configs
Passed gate criteria: 15-25 configs (40-70%)
pred_std > 0.010: 20-30 configs
Val Sharpe > 0: 10-18 configs
```

Top 4の典型的なスコア分布:
```
Rank 1: score=0.72, pred_std=0.014, sharpe=0.018, rankic=0.051
Rank 2: score=0.68, pred_std=0.012, sharpe=0.015, rankic=0.048
Rank 3: score=0.65, pred_std=0.013, sharpe=0.012, rankic=0.052
Rank 4: score=0.61, pred_std=0.011, sharpe=0.014, rankic=0.046
```

### 本学習結果（80 epochs）

Target metrics:
```
Val Sharpe:  0.30 - 0.40  (Baseline: -0.0184)
Val RankIC:  0.04 - 0.06  (Baseline: 0.0578)
pred_std:    > 0.010      (Critical)
Turnover:    20-30%/day   (Baseline: ~50%/day)
```

Backtest metrics (with costs):
```
Sharpe ratio:       0.25 - 0.35
Annual return:      15% - 25%
Max drawdown:       -15% - -25%
Transaction costs:  8% - 15% annually
```

---

## Advanced Customization

### Grid Searchのカスタマイズ

`parallel_sweep.sh`でgrid空間を変更:

```bash
# 細かいgrid（実行時間2-3倍）
TURNOVER_WEIGHTS=(0.0 0.01 0.02 0.03 0.05 0.075 0.1)
PRED_VAR_WEIGHTS=(0.3 0.5 0.7 0.9 1.0 1.2)
OUTPUT_NOISE_STDS=(0.01 0.02 0.03 0.04)
RANKIC_WEIGHTS=(0.15 0.2 0.25 0.3)
# Total: 7 × 6 × 4 × 4 = 672 configs

# 粗いgrid（実行時間1/2）
TURNOVER_WEIGHTS=(0.0 0.05)
PRED_VAR_WEIGHTS=(0.5 1.0)
OUTPUT_NOISE_STDS=(0.03)
RANKIC_WEIGHTS=(0.3)
# Total: 2 × 2 × 1 × 1 = 4 configs
```

### Composite Scoreの重み調整

`evaluate_sweep_results.py`のrank_configs():

```python
# Sharpe重視（収益性優先）
passed["score"] = (
    0.2 * passed["pred_std_norm"]   # 20%
    + 0.6 * passed["sharpe_norm"]   # 60% ← 増
    + 0.2 * passed["rankic_norm"]   # 20%
)

# Variance安定性重視（Collapse回避優先）
passed["score"] = (
    0.5 * passed["pred_std_norm"]   # 50% ← 増
    + 0.3 * passed["sharpe_norm"]   # 30%
    + 0.2 * passed["rankic_norm"]   # 20%
)
```

### 3段階トレーニング（実験的）

`run_best_config.sh`を編集して3 stages実装:

```bash
# Stage 1: Variance bootstrap (5 epochs)
TURNOVER_WEIGHT=0.0, PRED_VAR_WEIGHT=1.0

# Stage 2: IC optimization (25 epochs)
TURNOVER_WEIGHT=0.025, PRED_VAR_WEIGHT=0.6

# Stage 3: Sharpe focus (50 epochs)
TURNOVER_WEIGHT=0.1, PRED_VAR_WEIGHT=0.3, SWA=ON
```

---

## Files Created

```
scripts/
├── parallel_sweep.sh                  # 並列スイープ実行
├── evaluate_sweep_results.py          # 結果評価・選抜
├── run_best_config.sh                 # 2段階本学習
├── auto_sharpe_optimization.sh        # 全体統合マスター
└── backtest_sharpe_model.py           # バックテストフレームワーク

docs/
└── PARALLEL_SWEEP_GUIDE.md            # このファイル
```

---

## Next Steps After Pipeline Completion

### 1. 結果の検証

```bash
# Training metricsの確認
python scripts/evaluate_trained_model.py \
  --log-file output/production_training/logs/stage2_sharpe_optimization.log

# Backtestの詳細確認
cat output/backtest_production/summary_metrics.csv
```

### 2. A/B比較（複数設定を並行実行した場合）

```bash
# Top 2設定を比較
python scripts/compare_backtest_results.py \
  --backtest-a output/backtest_config1/ \
  --backtest-b output/backtest_config2/
```

### 3. 本番デプロイ準備

```bash
# Best modelをexport（ONNX形式など）
python scripts/export_model.py \
  --checkpoint output/checkpoints/best_model.pth \
  --output-path models/production_model.onnx

# Inference速度テスト
python scripts/benchmark_inference.py \
  --model models/production_model.onnx \
  --batch-size 1024
```

---

## References

- Original diagnosis: `docs/PREDICTION_ANALYSIS.md`
- Sharpe optimization progress: `docs/SHARPE_OPTIMIZATION_PROGRESS.md`
- Backtest framework: `scripts/backtest_sharpe_model.py`
- Loss implementations: `src/losses/{sharpe_loss,transaction_cost}.py`

---

**Last Updated**: 2025-10-15
**Status**: Ready for production use
**Estimated Success Rate**: 70-80% (based on gate criteria passing rate)
