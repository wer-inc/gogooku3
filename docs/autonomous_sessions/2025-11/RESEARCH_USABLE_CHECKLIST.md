# Research-Usable 到達チェックリスト

**目的**: Quick Run → 短縮WF の2段階で「実際に使える状態」に到達

**作成**: 2025-11-02
**ステータス**: 実行待ち Ready ✅

---

## ✅ 合格ライン（Research-Usable）

### Stage 1: Quick Run (3 epochs, 15分)

**必須条件**:
- [ ] `RFI56 |` の行が **3本**出力（各epoch 1本）
- [ ] `gat_gate_mean ∈ [0.2, 0.7]`
- [ ] `gat_gate_std ∈ [0.05, 0.30]`
- [ ] `deg_avg ∈ [10, 40]`
- [ ] `isolates < 0.02`
- [ ] `RankIC > 0`（初期 0.02-0.10 でOK）
- [ ] `grad_ratio ∈ [0.5, 2.0]`
- [ ] OOM/segfault なし（exit code 0）

**判定**: `python scripts/accept_quick_p03.py rfi_56_metrics.txt` → `PASS`

### Stage 2: 短縮WF (3 splits, 30分)

**必須条件**:
- [ ] 全 split 完走（3/3）
- [ ] `RankIC 平均 > 0.05`
- [ ] `Sharpe 平均 > 0.30`
- [ ] `qx_rate < 0.05`

**判定**: ログ確認 + 手動検証

---

## 🚀 実行手順（コピペ実行）

### Pre-flight Check（実行前確認）

```bash
# 1. Dataset存在確認
ls -lh output/ml_dataset_latest_full.parquet
# 期待: 1-5GB

# 2. train_atft.py パッチ確認
grep "log_rfi_56_metrics" scripts/train_atft.py | wc -l
# 期待: 2 (import + 呼び出し)

# 3. ログディレクトリ作成
mkdir -p _logs

# 4. FAN/SAN有効確認
echo "BYPASS_ADAPTIVE_NORM=${BYPASS_ADAPTIVE_NORM:-0}"
# 期待: 0 または空

# 5. GAT有効確認
echo "BYPASS_GAT_COMPLETELY=${BYPASS_GAT_COMPLETELY:-0}"
# 期待: 0 または空
```

### Stage 1: Quick Run（RFI-5/6回収）

```bash
# 実行
USE_GAT_SHIM=1 BATCH_SIZE=1024 \
make train-quick EPOCHS=3 2>&1 | tee _logs/train_p03_quick.log

# RFI-5/6抽出
grep "RFI56 |" _logs/train_p03_quick.log > rfi_56_metrics.txt

# 自動判定
python scripts/accept_quick_p03.py rfi_56_metrics.txt
```

**期待出力**:
```
================================================================================
✅ PASS: P0-3 Quick Acceptance

Next steps:
1. Enable P0-4/6/7 coefficients
2. Run short WF validation (3 splits)
3. Monitor full training (120 epochs)
================================================================================
```

**Exit code**: 0 (PASS)

### Stage 2: P0-4/6/7有効化 + 短縮WF

```bash
# 係数確定（rfi_56_metrics.txt から qx_rate 確認後）
# qx_rate < 0.05 → LAMBDA_QC=2e-3 (下記のまま)
# qx_rate > 0.05 → LAMBDA_QC=5e-3 (コメント外す)

export QUANTILE_WEIGHT=1.0
export SHARPE_WEIGHT=0.30
export RANKIC_WEIGHT=0.20
export CS_IC_WEIGHT=0.15
export LAMBDA_QC=2e-3        # qx_rate < 0.05
# export LAMBDA_QC=5e-3      # qx_rate > 0.05 (必要時)
export SHARPE_EMA_DECAY=0.95

# 短縮WF実行
USE_GAT_SHIM=1 BATCH_SIZE=1024 \
python scripts/train_atft.py --max-epochs 30 \
  --data-path output/ml_dataset_latest_full.parquet \
  2>&1 | tee _logs/train_p0467_wf3.log
```

**期待結果**:
- All splits完走（3/3）
- RankIC avg > 0.05
- Sharpe avg > 0.30
- qx_rate < 0.05

---

## 🧯 Borderline/Fail 時の即応

### トリアージマトリクス（順に試す）

| 症状 | 1st Aid | 2nd Aid | 検証コマンド |
|------|---------|---------|-------------|
| **Gate飽和** (0/1付近) | `tau=1.5-2.0` | `edge_dropout=0.10-0.15` | `grep "gat_gate_mean" rfi_56_metrics.txt` |
| **Graph疎** (deg_avg<10) | k-NN増加 | threshold下げ | `grep "deg_avg" rfi_56_metrics.txt` |
| **孤立多** (isolates>2%) | 接続性確認 | GraphBuilder調整 | `grep "isolates" rfi_56_metrics.txt` |
| **RankIC負** | 重み維持(0.20/0.15) | 学習率 0.7× | `grep "RankIC" rfi_56_metrics.txt` |
| **交差多** (qx_rate>0.05) | `LAMBDA_QC=5e-3` | isotonic後処理 | `grep "qx_rate" rfi_56_metrics.txt` |
| **勾配不均衡** (<0.5 or >2.0) | `tau`+`edge_dropout`同時調整 | GAT lr 0.8× | `grep "grad_ratio" rfi_56_metrics.txt` |
| **OOM** | `BATCH_SIZE=512` | `BATCH_SIZE=256` | `dmesg \| grep -i oom` |
| **Segfault** | B-1案（PyTorch 2.8.0降格） | ソースビルド | `python scripts/diagnose_pyg_environment.py` |

### 再実行テンプレート

```bash
# パラメータ調整例（Gate飽和対策）
export GAT_TAU=1.5
export GAT_EDGE_DROPOUT=0.10

# Quick Run再実行
USE_GAT_SHIM=1 BATCH_SIZE=1024 \
make train-quick EPOCHS=3 2>&1 | tee _logs/train_p03_quick_retry.log

# 再判定
grep "RFI56 |" _logs/train_p03_quick_retry.log > rfi_56_metrics_retry.txt
python scripts/accept_quick_p03.py rfi_56_metrics_retry.txt
```

---

## 📦 受け渡し成果物（Research-Usable達成時）

### 必須ファイル

```bash
# 1. モデル重み
models/p0_research_usable_YYYYMMDD.tar

# 2. 設定ファイル
configs/p0_production_final.yaml

# 3. Feature ABI指紋
echo "5cc86ec5...bbc5" > feature_abi.txt

# 4. Git commit
git rev-parse HEAD > git_commit.txt

# 5. RFI-5/6実測値
rfi_56_metrics.txt

# 6. 短縮WF結果
_logs/train_p0467_wf3.log

# 7. 予測成果物サンプル（日次）
outputs/predictions_daily_sample.csv
# 列: code, date, horizon, y_point, y_q_0.1, y_q_0.25, y_q_0.5, y_q_0.75, y_q_0.9
```

### 成果物テンプレート

```markdown
## Research-Usable Achievement Report

### Environment
- PyTorch: 2.9.0+cu128
- CUDA: 12.8
- GPU: NVIDIA A100-SXM4-80GB
- GAT mode: Shim (GraphConvShim)
- Feature ABI: 5cc86ec5...bbc5
- Git commit: <commit_hash>
- Timestamp: 2025-11-02 HH:MM:SS

### Stage 1: Quick Run (3 epochs)
**Status**: ✅ PASS

**Metrics** (median of 3 epochs):
- gat_gate_mean: 0.4612
- gat_gate_std: 0.1198
- deg_avg: 25.92
- isolates: 0.0110
- RankIC: 0.0278
- qx_rate: 0.0236
- grad_ratio: 0.913

**Acceptance Test**: `python scripts/accept_quick_p03.py rfi_56_metrics.txt`
```
✅ PASS: P0-3 Quick Acceptance
```

### Stage 2: Short WF (3 splits, 30 epochs)
**Status**: ✅ PASS

**P0-4/6/7 Coefficients**:
- QUANTILE_WEIGHT: 1.0
- SHARPE_WEIGHT: 0.30
- RANKIC_WEIGHT: 0.20
- CS_IC_WEIGHT: 0.15
- LAMBDA_QC: 2e-3
- SHARPE_EMA_DECAY: 0.95

**Results**:
- All splits: 3/3 completed
- RankIC avg: 0.067
- Sharpe avg: 0.412
- qx_rate: 0.023

### Deliverables
- Model: `models/p0_research_usable_20251102.tar`
- Config: `configs/p0_production_final.yaml`
- Feature ABI: `5cc86ec5...bbc5`
- Git commit: `<commit_hash>`
- Predictions sample: `outputs/predictions_daily_sample.csv`

### Reproduce Command
```bash
make reproduce --run-id p0_research_usable_20251102
```

### Status
**Research-Usable**: ✅ ACHIEVED

### Next Steps
1. Long WF validation (5 splits, 120 epochs)
2. Production deployment preparation
3. SLO monitoring setup
```

---

## 🧭 その先（プロダクションへの階段）

### 1. PyG本実装へ切替（任意タイミング）

```bash
# PyTorch 2.8.0+cu128 降格
pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0+cu128
pip install torch_geometric
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.8.0+cu128.html

# 確認
python -c "from torch_geometric.nn import GATv2Conv; print('✅ PyG OK')"

# Shim OFF で再学習
make train EPOCHS=120
```

**効果**: 性能 60-80% → 100%

### 2. 長尺WF（Purge/Embargo）

```bash
# 5 splits, 120 epochs
python scripts/train_atft.py \
  --max-epochs 120 \
  --data-path output/ml_dataset_latest_full.parquet \
  --run-safe-pipeline \
  --adv-graph-train \
  2>&1 | tee _logs/train_p0_full_wf5.log
```

**期待結果**:
- Sharpe ratio > 0.849
- RankIC > 0.18
- qx_rate < 0.03

### 3. SLO定義（プロダクション基準）

**7日間移動平均での基準**:
- ✅ Sharpe ratio > 0.849
- ✅ RankIC > 0.18
- ✅ qx_rate < 0.03
- ✅ gat_gate_mean ∈ [0.3, 0.6]（安定）
- ✅ deg_avg ∈ [15, 35]（安定）

**アラート条件**:
- ⚠️ Sharpe ratio < 0.70（3日連続）
- ⚠️ RankIC < 0.10（3日連続）
- ⚠️ qx_rate > 0.05（1日）
- ⚠️ gat_gate_mean < 0.1 or > 0.9（Gate飽和、1日）
- ⚠️ isolates > 0.03（グラフ劣化、1日）

### 4. 監視・ロールバック

**監視対象**:
```bash
# 日次メトリクス抽出
python scripts/extract_daily_metrics.py \
  --log-dir _logs/training/ \
  --output metrics_daily.csv

# ダッシュボード生成
python scripts/generate_dashboard.py \
  --metrics metrics_daily.csv \
  --output dashboard.html
```

**ロールバック手順**:
```bash
# 昨日版へロールバック
cp models/p0_backup_yesterday.tar models/p0_current.tar

# GAT無効ルートへ切替
export BYPASS_GAT_COMPLETELY=1
make train EPOCHS=10
```

---

## 📞 次のアクション

### 即座実行

```bash
# Quick Run実行
USE_GAT_SHIM=1 BATCH_SIZE=1024 make train-quick EPOCHS=3 2>&1 | tee _logs/train_p03_quick.log

# RFI-5/6抽出
grep "RFI56 |" _logs/train_p03_quick.log > rfi_56_metrics.txt
```

### 結果報告

**rfi_56_metrics.txt を貼り付けてください**

実測値に基づいて:
1. P0-4/6/7の係数をピン留め（tau/edge_dropout/λ含む）
2. 短縮WFの具体コマンド生成
3. 想定される問題と対策を提示

---

## ⏱ 想定タイムライン

```
T+0:   Quick Run開始
T+15:  Quick Run完了 → rfi_56_metrics.txt取得
T+15:  受け入れ判定（30秒）
T+16:  係数確定・環境変数設定（1分）
T+17:  短縮WF開始
T+47:  短縮WF完了 → Research-Usable達成 ✅
T+48:  長尺WF/本番学習への移行判断
```

**合計所要時間**: 20-40分（問題なければ）

---

**作成**: 2025-11-02
**最終更新**: 2025-11-02
**バージョン**: 1.0.0
**ステータス**: 実行待ち Ready ✅
