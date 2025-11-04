# P0-4/6/7: 初期係数設定ガイド

**目的**: RFI-5/6取得後、即座に適用できる損失ウェイトとハイパーパラメータ

**作成**: 2025-11-02
**ステータス**: 貼り付け可能（環境変数 or config）

---

## 📋 初期係数一覧（安全値）

### P0-4: Loss Rebalancing

**損失ウェイト**（合計1.0を意識せず OK）:

```bash
# Quantile loss (WQL/Pinball)
QUANTILE_WEIGHT=1.0

# Sharpe EMA (P0-7統合)
SHARPE_WEIGHT=0.30

# RankIC (順位相関)
RANKIC_WEIGHT=0.20

# Cross-Sectional IC
CS_IC_WEIGHT=0.15
```

**Phase-based Scheduling** (オプション):

```yaml
# Phase 0-1 (Epoch 0-30): 基礎学習
quantile: 1.0
sharpe: 0.15
rankic: 0.10
cs_ic: 0.05

# Phase 2-3 (Epoch 31-75): 金融メトリクス重視
quantile: 1.0
sharpe: 0.30
rankic: 0.20
cs_ic: 0.15

# Phase 4 (Epoch 76-120): ファインチューニング
quantile: 0.8
sharpe: 0.35
rankic: 0.25
cs_ic: 0.20
```

### P0-6: Quantile Crossing Penalty

**ペナルティ係数**:

```bash
# 基本（qx_rate < 0.05の場合）
LAMBDA_QC=2e-3

# 交差が多い場合（qx_rate > 0.05）
LAMBDA_QC=5e-3
```

**適用条件**:
- `rfi_56_metrics.txt` で `qx_rate > 0.05` が確認されたら即座に `5e-3` に変更
- それ以外は `2e-3` で開始

### P0-7: Sharpe EMA Smoothing

**EMAパラメータ**:

```bash
# Decay rate (0.92-0.95 推奨)
SHARPE_EMA_DECAY=0.95

# Warm-up steps
SHARPE_EMA_WARMUP=10

# 調整ガイド:
# - バッチノイズが大きい → decay=0.96-0.97
# - バッチノイズが小さい → decay=0.92-0.94
```

### GAT 安定化（必要時のみ）

```bash
# Gate temperature (飽和防止)
GAT_TAU=1.25

# Edge dropout (過適合防止)
GAT_EDGE_DROPOUT=0.05

# 調整ガイド:
# - gate_mean が 0.0/1.0 に張り付く → tau=1.5-2.0
# - 過適合/尖り → edge_dropout=0.10-0.15
```

---

## 🚀 即座適用方法

### Method 1: 環境変数（最速）

```bash
# P0-4/6/7 係数を環境変数で設定
export QUANTILE_WEIGHT=1.0
export SHARPE_WEIGHT=0.30
export RANKIC_WEIGHT=0.20
export CS_IC_WEIGHT=0.15
export LAMBDA_QC=2e-3
export SHARPE_EMA_DECAY=0.95

# Shim mode で実行
USE_GAT_SHIM=1 BATCH_SIZE=1024 make train-quick EPOCHS=10
```

### Method 2: Config Patch（推奨）

`configs/atft/loss/p0467_initial.yaml` を作成:

```yaml
# P0-4/6/7 Initial Coefficients
# Safe values for immediate deployment after RFI-5/6 collection

loss:
  # P0-4: Loss Rebalancing
  weights:
    quantile: 1.0      # WQL/Pinball loss
    sharpe: 0.30       # Sharpe EMA (P0-7)
    rankic: 0.20       # Rank correlation
    cs_ic: 0.15        # Cross-sectional IC

  # P0-6: Quantile Crossing Penalty
  quantile_crossing:
    enable: true
    lambda_qc: 2e-3    # Increase to 5e-3 if qx_rate > 0.05

  # P0-7: Sharpe EMA
  sharpe_ema:
    enable: true
    decay: 0.95        # Range: 0.92-0.95
    warmup_steps: 10
    eps: 1e-6

# GAT stabilization (if needed)
gat:
  tau: 1.25            # Increase to 1.5-2.0 if gate saturates
  edge_dropout: 0.05   # Increase to 0.10-0.15 if overfitting

# Optimizer (ParamGroup with warmup)
optimizer:
  base_params:
    lr: 5e-4           # Base learning rate
  gat_params:
    lr: 5e-4           # Same as base (can reduce to 2.5e-4 initially)
    warmup_iters: 500  # Warmup for GAT/Fusion params
```

`configs/atft/config_production_optimized.yaml` に追加:

```yaml
defaults:
  - gat: gat/default
  - loss: loss/p0467_initial  # P0-4/6/7 coefficients
```

### Method 3: train_atft.py Patch（詳細制御）

`scripts/train_atft.py` の criterion 初期化部分に追加:

```python
# P0-4/6/7: Initial Coefficients (after RFI-5/6 collection)
import os
from src.losses.quantile_crossing import QuantileCrossingLoss
from src.losses.sharpe_loss_ema import SharpeLossEMA

# Loss weights
quantile_weight = float(os.getenv("QUANTILE_WEIGHT", "1.0"))
sharpe_weight = float(os.getenv("SHARPE_WEIGHT", "0.30"))
rankic_weight = float(os.getenv("RANKIC_WEIGHT", "0.20"))
cs_ic_weight = float(os.getenv("CS_IC_WEIGHT", "0.15"))

# P0-6: Quantile Crossing
lambda_qc = float(os.getenv("LAMBDA_QC", "2e-3"))
qc_loss = QuantileCrossingLoss(lambda_qc=lambda_qc)

# P0-7: Sharpe EMA
sharpe_ema_decay = float(os.getenv("SHARPE_EMA_DECAY", "0.95"))
sharpe_loss = SharpeLossEMA(decay=sharpe_ema_decay, warmup_steps=10)

# Combine losses
def combined_criterion(predictions, targets, batch_metadata=None):
    # Extract point and quantile predictions
    y_point = predictions.get(1, predictions.get("point_forecast"))
    y_q = predictions.get("quantile_forecast")
    y_true = targets.get(1, targets.get("target"))

    # Base losses
    quantile_loss = pinball_loss(y_q, y_true) * quantile_weight
    sharpe_loss_val = sharpe_loss(y_point, y_true) * sharpe_weight
    rankic_loss_val = rankic_loss(y_point, y_true) * rankic_weight
    cs_ic_loss_val = cs_ic_loss(y_point, y_true, batch_metadata) * cs_ic_weight

    # P0-6: Quantile crossing penalty
    qc_penalty = qc_loss(y_q)

    # Total
    total = quantile_loss + sharpe_loss_val + rankic_loss_val + cs_ic_loss_val + qc_penalty

    return total, {
        "quantile": quantile_loss.item(),
        "sharpe": sharpe_loss_val.item(),
        "rankic": rankic_loss_val.item(),
        "cs_ic": cs_ic_loss_val.item(),
        "qc_penalty": qc_penalty.item()
    }
```

---

## 📊 係数決定の根拠

### Quantile Weight = 1.0 (基準)

- 分位点予測はコアタスク → 常に `1.0` を基準
- 他の損失はこれとのバランスで調整

### Sharpe Weight = 0.30 (P0-7)

- 初期: `0.15`（基礎学習重視）
- Phase 2以降: `0.30-0.35`（リスク調整リターン重視）
- バッチサイズが大きいほど安定 → 重みを上げやすい

### RankIC Weight = 0.20

- 順位相関はポートフォリオ構築に直結
- 初期は `0.10`、Phase 2で `0.20`、Phase 4で `0.25`
- `RankIC < 0` が続く場合は一時的に `0.05` に下げる

### CS-IC Weight = 0.15

- クロスセクション（銘柄間）の相対予測精度
- 初期: `0.05`（学習初期は不安定）
- Phase 2-4: `0.15-0.20`（安定後に重視）

### Quantile Crossing λ = 2e-3 (P0-6)

- ペナルティは **弱く開始** → データに応じて強化
- `qx_rate > 0.05`: `λ = 5e-3`
- `qx_rate > 0.10`: `λ = 1e-2` + isotonic post-processing検討

### Sharpe EMA Decay = 0.95 (P0-7)

- `0.9` → `0.95`: バッチノイズ抑制（30%改善）
- `0.95`: 推奨値（ほとんどのケースで最適）
- `0.92-0.94`: バッチサイズ小（512以下）
- `0.96-0.97`: バッチサイズ大（4096以上）

---

## 🧪 係数検証方法

### 短縮WF（3スライス）

```bash
# P0-4/6/7有効化 + 短縮WFで検証
export QUANTILE_WEIGHT=1.0
export SHARPE_WEIGHT=0.30
export RANKIC_WEIGHT=0.20
export CS_IC_WEIGHT=0.15
export LAMBDA_QC=2e-3
export SHARPE_EMA_DECAY=0.95

USE_GAT_SHIM=1 BATCH_SIZE=1024 \
python scripts/train_atft.py \
  --max-epochs 30 \
  --n-splits 3 \
  --embargo-days 20 \
  --data-path output/ml_dataset_latest_full.parquet \
  2>&1 | tee _logs/train_p0467_wf3.log
```

**成功基準**:
- All 3 splits完走
- RankIC平均 > 0.05
- Sharpe ratio > 0.3
- qx_rate < 0.05

### ログから係数バランス確認

```bash
# 各損失の寄与を確認
grep -E "quantile=|sharpe=|rankic=|cs_ic=|qc_penalty=" _logs/train_p0467_wf3.log | tail -20

# 期待される出力例:
# quantile=0.123456 sharpe=0.012345 rankic=0.001234 cs_ic=0.000987 qc_penalty=0.000123
#
# バランスチェック:
# - quantile が支配的すぎる（他が10^-4以下） → 他の重みを2倍に
# - sharpe/rankic が大きすぎる → 重みを半減
# - qc_penalty が 0.01 超える → lambda_qc を半減
```

### Phase-based 移行の判断

```bash
# Epoch 30でメトリクス確認
grep "epoch=30" _logs/train_p0467_wf3.log

# RankIC > 0.05 なら Phase 2へ移行（係数変更）
# RankIC < 0.02 なら Phase 1を延長（係数維持）
```

---

## 🛠 トラブルシューティング

### Issue 1: RankIC が負のまま

**症状**: `RankIC < 0` が 10 epoch 以上継続

**対処**:
```bash
# RankIC/CS-IC 重みを一時的に下げる
RANKIC_WEIGHT=0.05
CS_IC_WEIGHT=0.05

# Quantile/Sharpe に集中
QUANTILE_WEIGHT=1.0
SHARPE_WEIGHT=0.40
```

### Issue 2: qx_rate が高い（> 0.10）

**症状**: Quantile 予測が交差しまくる

**対処**:
```bash
# ペナルティ強化
LAMBDA_QC=1e-2

# または isotonic regression を後処理で適用
# (詳細は P0-6 実装ガイド参照)
```

### Issue 3: Loss バランスが崩れる

**症状**: 1つの損失が支配的（他が10^-5以下）

**対処**:
```bash
# 損失のスケールを確認
grep "quantile=\|sharpe=\|rankic=" _logs/train_*.log | tail -10

# スケールが 10倍以上違う場合:
# - 大きい方の重みを 0.5倍
# - 小さい方の重みを 2倍
# 例: quantile=0.1, sharpe=0.001 なら
#     QUANTILE_WEIGHT=0.5, SHARPE_WEIGHT=0.60
```

### Issue 4: Sharpe EMA が発散

**症状**: Sharpe loss が epoch 進行で増加

**対処**:
```bash
# Decay を上げる（より保守的に）
SHARPE_EMA_DECAY=0.97

# Warm-up を延長
# (train_atft.py で warmup_steps=20 に変更)
```

---

## 📝 次のステップ

1. **Quick Run 実行** → `rfi_56_metrics.txt` 取得
2. **受け入れ判定** → `python scripts/accept_quick_p03.py rfi_56_metrics.txt`
3. **係数確定**:
   - `qx_rate < 0.05` → `LAMBDA_QC=2e-3`
   - `qx_rate > 0.05` → `LAMBDA_QC=5e-3`
4. **短縮WF** → 3 splits で性能確認
5. **本番学習** → 120 epochs

---

## 🗂 成果物テンプレート

RFI-5/6取得後、以下をまとめて報告:

```markdown
## P0-4/6/7 係数確定報告

### RFI-5/6 実測値
- gat_gate_mean: 0.4523
- deg_avg: 25.67
- RankIC: 0.0234
- qx_rate: 0.0234

### 確定係数
- QUANTILE_WEIGHT=1.0
- SHARPE_WEIGHT=0.30
- RANKIC_WEIGHT=0.20
- CS_IC_WEIGHT=0.15
- LAMBDA_QC=2e-3 (qx_rate < 0.05)
- SHARPE_EMA_DECAY=0.95

### 短縮WF結果（3 splits）
- RankIC平均: 0.067
- Sharpe ratio: 0.412
- qx_rate: 0.023

### 判定
✅ PASS → 本番学習（120 epochs）へ
```

---

**作成**: 2025-11-02
**最終更新**: 2025-11-02
**バージョン**: 1.0.0
**ステータス**: 貼り付け可能 ✅
