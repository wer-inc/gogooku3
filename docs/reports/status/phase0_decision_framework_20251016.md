# Phase 0 完了後の意思決定フレームワーク

**作成日時**: 2025-10-16 14:17 UTC
**Phase 0 完了予定**: ~14:23 UTC (約6分後)

---

## 📊 Phase 0 最終評価基準

### ✅ 継続条件（Trial 0をPhase 1へ進める）

**RankIC回復の兆候**:
```
Epoch 5の結果で以下のいずれかを確認:
  ✓ Val RankIC ≥ 0.015 (baseline 0.028の50%以上)
  ✓ RankIC上昇トレンド (Epoch 2→3→4→5で改善)
  ✓ IC ≥ 0 (負でない)
```

**理由**:
- Phase 1でGATが有効化されるとgraph edgesが効いてRank metricsが改善する可能性
- Spearman regularizerがmulti-horizonで効いてくる可能性
- 損失関数の重みバランスが後半で効果を発揮する可能性

**アクション**:
→ **Trial 0をそのまま継続** (Phase 1へ進める)
→ Phase 1完了後（Epoch 8-10頃）に再評価

---

### ❌ 停止条件（Trial 0を中止して設定変更）

**RankIC低迷の継続**:
```
Epoch 5の結果で以下の全てに該当:
  ✗ Val RankIC < 0.010 (baseline 0.028の35%未満)
  ✗ RankIC下降または横ばいトレンド
  ✗ IC < 0 (依然として負)
```

**理由**:
- このまま20 trials実行しても同じ傾向が続く可能性が高い
- 損失関数の重み調整が先に必要
- 23時間の機会損失を避ける

**アクション**:
→ **Trial 0を停止**
→ **損失関数の重み調整**
→ **短期HPO sweep再実行**（3-5 trials × 3-6時間）

---

## 🎯 Phase 0 メトリクス追跡

### Epoch別トレンド（現在）

| Epoch | Val Loss | Val Sharpe | Val IC | Val RankIC | 評価 |
|-------|----------|------------|--------|------------|------|
| 1 | 0.3661 | ? | ? | ? | 初期 |
| 2 | 0.3616 | 0.027 | -0.0048 | 0.0058 | ⚠️ RankIC低 |
| 3 | 0.3621 | ? | ? | ? | Loss微増 |
| 4 | 0.3594 | ? | ? | ? | Loss改善 |
| 5 | **待機中** | **?** | **?** | **?** | **判断材料** |

### 🔍 Epoch 5で確認すべき指標

```bash
# Phase 0完了後に実行
tail -100 logs/ml_training.log | grep -A3 "Epoch 5/5"
```

**重要指標**:
1. **Val RankIC**: 0.010以上か？ トレンドは上昇か？
2. **Val IC**: 0以上か？
3. **Val Sharpe**: 0.030以上か？
4. **Loss**: 0.355以下か？（過学習していないか）

---

## 🚀 継続シナリオ（Phase 1へ）

### Phase 1の特徴
```
Phase 1: GAT有効化
- Epochs: 6-8 (推定)
- GAT layers: 3層
- Graph edges: 相関ベース（~5000 edges）
- 損失重み: Multi-horizon weighted
```

### Phase 1での期待
- **RankIC**: GATの順序学習でさらに改善
- **IC**: Graph構造から特徴抽出で改善
- **Sharpe**: 維持または改善

### Phase 1完了後の再評価タイミング
```
⏱️  ETA: ~14:45 UTC (Phase 1完了予定)

判断基準:
  ✓ RankIC ≥ 0.020 → Trial 0完了まで継続
  ✗ RankIC < 0.015 → 停止して損失調整
```

---

## 🛑 停止シナリオ（損失関数調整）

### 問題の診断

**現象**:
- Sharpe高 (0.027) だがRankIC低 (0.0058)
- IC負 (-0.0048)

**原因仮説**:
1. **損失関数の重みバランス不良**
   - MSE重視すぎる → ボラティリティ予測に偏る
   - RankIC/IC項の重みが不足

2. **Spearman regularizer未使用**
   - 実装済みだが環境変数で無効化されている可能性

3. **ハイパーパラメータの問題**
   - Learning rate高すぎ → 細かい順序を学習できない
   - Batch size大きすぎ → ランク情報が薄まる

---

## 🔧 停止後の改善策

### Option A: 損失関数の重み調整（推奨）

```bash
# 新しいHPO設定（短期：5 trials × 70分 = 6時間）
python scripts/hpo/run_optuna_atft.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --n-trials 5 \
  --max-epochs 10 \
  --study-name atft_loss_weight_tuning \
  --output-dir output/hpo_loss_tuning

# 環境変数で損失重み調整
export USE_RANKIC=1
export RANKIC_WEIGHT=0.5        # 現在より大幅増（0.2 → 0.5）
export CS_IC_WEIGHT=0.3         # IC重視（0.15 → 0.3）
export SHARPE_WEIGHT=0.1        # Sharpe重視を減らす（0.3 → 0.1）
```

**期待効果**:
- RankICの学習を強化
- Sharpeとのバランス改善
- より金融的に意味のある予測

---

### Option B: Spearman regularizer有効化

```bash
# Spearman rank-preserving loss を明示的に使用
# 実装済み: src/gogooku3/training/losses/rank_preserving_loss.py

# train_atft.py で環境変数チェック
export USE_SPEARMAN_REGULARIZER=1
export SPEARMAN_WEIGHT=0.1

# または train_atft.py を修正して直接統合
```

**期待効果**:
- Spearman相関を直接最適化
- RankICの改善
- ランク順序の学習強化

---

### Option C: ハイパーパラメータ範囲調整

```python
# run_optuna_atft.py の範囲を変更

# 現在:
lr: [1e-5, 1e-3]
batch_size: [2048, 4096, 8192]

# 変更後:
lr: [1e-6, 1e-4]           # より小さく（細かい学習）
batch_size: [512, 1024, 2048]  # より小さく（ランク情報保持）
```

**期待効果**:
- より繊細な学習
- クロスセクショナルなランク情報の保持

---

## 📋 Phase 0完了後の具体的手順

### Step 1: メトリクス確認（14:23 UTC）

```bash
# Epoch 5完了ログを確認
tail -100 logs/ml_training.log | grep -A5 "Epoch 5/5"

# 重要指標を抽出
tail -100 logs/ml_training.log | grep "Val Metrics" | tail -1
```

### Step 2: 判断実行

```bash
# RankIC値を確認
VAL_RANKIC=$(tail -100 logs/ml_training.log | grep "Val Metrics" | tail -1 | grep -oP 'RankIC: \K[0-9.]+')

echo "Val RankIC: $VAL_RANKIC"

# 判断基準
if (( $(echo "$VAL_RANKIC >= 0.015" | bc -l) )); then
    echo "✅ 継続: Phase 1へ進める"
else
    echo "❌ 停止: 損失関数調整が必要"
fi
```

### Step 3A: 継続の場合

```bash
# そのまま待機
echo "Phase 1の完了を待つ（14:45 UTC頃）"
```

### Step 3B: 停止の場合

```bash
# HPOプロセスを停止
kill $(ps aux | grep "run_optuna_atft.py" | grep -v grep | awk '{print $2}')

# トレーニングプロセスも停止
kill $(ps aux | grep "train_atft.py" | grep -v grep | awk '{print $2}')

# 損失重み調整版を起動
export USE_RANKIC=1
export RANKIC_WEIGHT=0.5
export CS_IC_WEIGHT=0.3
export SHARPE_WEIGHT=0.1

nohup python scripts/hpo/run_optuna_atft.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --n-trials 5 \
  --max-epochs 10 \
  --study-name atft_loss_weight_tuning \
  --output-dir output/hpo_loss_tuning \
  > /tmp/hpo_loss_tuning.log 2>&1 &
```

---

## 🎯 最終的な意思決定ツリー

```
Phase 0 完了 (Epoch 5)
    |
    ├─ RankIC ≥ 0.015 & IC ≥ 0
    |   └─ ✅ 継続 → Phase 1へ
    |       |
    |       └─ Phase 1完了後 (Epoch 8-10)
    |           |
    |           ├─ RankIC ≥ 0.020
    |           |   └─ ✅ Trial 0完了まで継続 → 残り19 trials
    |           |
    |           └─ RankIC < 0.020
    |               └─ ❌ 停止 → 損失調整
    |
    └─ RankIC < 0.015 OR IC < 0
        └─ ❌ 即座に停止 → 損失調整
            |
            └─ 損失重み調整 HPO (5 trials, 6時間)
                |
                └─ 改善確認後 → 本格HPO再開
```

---

## 📊 記録すべきデータ

### Phase 0完了時に保存

```bash
# Epoch 1-5の全メトリクス
grep "Epoch [0-9]/5" logs/ml_training.log | grep "Val Metrics" > phase0_metrics.txt

# 判断材料として保存
cat phase0_metrics.txt
```

### 分析用データ

```python
# 後で分析するため
import json

phase0_summary = {
    "trial": 0,
    "phase": 0,
    "epochs": 5,
    "config": {
        "lr": 5.61e-05,
        "batch_size": 2048,
        "hidden_size": 256,
        "gat_layers": 3
    },
    "metrics": {
        # Epoch 5の値をここに記録
        "final_val_sharpe": None,
        "final_val_rankic": None,
        "final_val_ic": None,
        "final_val_loss": None
    },
    "decision": None,  # "continue" or "stop"
    "reason": None
}

with open("output/phase0_decision.json", "w") as f:
    json.dump(phase0_summary, f, indent=2)
```

---

## 💡 推奨アクション（まとめ）

1. **今（14:17）**: 待機
2. **14:23頃**: Epoch 5完了を確認
3. **14:25**: メトリクス分析と判断実行
4. **継続の場合**: Phase 1完了（14:45）を待つ
5. **停止の場合**: 損失調整HPO起動（5 trials, 6時間）

**判断基準**: Val RankIC ≥ 0.015 かつ IC ≥ 0

---

**ステータス**: ⏳ Epoch 5完了待ち（あと6分）
**次回確認**: 2025-10-16 14:23 UTC
