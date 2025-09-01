# ATFT-GAT-FAN Improvements ロールバックガイド

## 概要
改善実装で問題が発生した場合の迅速なロールバック手順を説明します。

## 緊急ロールバック手順

### 即時ロールバック（5分以内）
```bash
# 1. 設定ファイルを既存設定に戻す
cp configs/atft/config.yaml.backup configs/atft/config.yaml

# 2. 改善機能全無効化
cat > configs/atft/config.yaml << EOF
improvements:
  output_head_small_init: false
  gat_temperature: 1.0
  freq_dropout_p: 0.0
  use_ema: false
  huber_loss: false
  online_normalization: false
  enable_wandb: false
  auto_recover_oom: false
EOF

# 3. トレーニング設定も戻す
cat > configs/atft/train/production.yaml << EOF
loss:
  type: mse
ema:
  enabled: false
grad_scaler:
  init_scale: 1024.0
EOF

# 4. サービス再起動
pkill -f "integrated_trainer"
python src/training/integrated_trainer.py &
```

### 段階的ロールバック（30分以内）

#### 問題別ロールバック設定

##### 学習不安定の場合
```yaml
# config.yaml
improvements:
  output_head_small_init: false  # 初期化改善無効
  huber_loss: false             # Huber損失無効
  freq_dropout_p: 0.0           # FreqDropout無効
  use_ema: false                # EMA無効
```

##### OOMが発生する場合
```yaml
# config.yaml
improvements:
  use_ema: false                # EMA無効（メモリ削減）
  online_normalization: false   # オンライン正規化無効
  auto_recover_oom: true        # 自動回復有効
```

##### 性能劣化の場合
```yaml
# config.yaml
improvements:
  gat_temperature: 1.0          # GAT温度既存値
  freq_dropout_p: 0.0           # FreqDropout無効
  output_head_small_init: false # small-init無効
```

## 完全ロールバック手順

### Gitリバート（推奨）
```bash
# 1. 現在の変更を確認
git status
git diff --name-only

# 2. 改善関連コミットを特定
git log --oneline -10 | grep -i "improve\|feat\|fix"

# 3. リバート実行
git revert <commit-hash> --no-edit

# 4. 変更反映
git push origin main
```

### 手動ロールバック
```bash
# 1. 新規ファイル削除
rm -f src/utils/settings.py
rm -f src/data/loaders/streaming_dataset.py
rm -f src/losses/multi_horizon_loss.py
rm -f src/training/robust_trainer.py
rm -f src/models/components/freq_dropout.py
rm -f src/utils/monitoring.py
rm -f src/utils/robust_executor.py
rm -f src/training/integrated_trainer.py
rm -f scripts/smoke_test.py
rm -f scripts/validate_improvements.py

# 2. 変更ファイル戻す
git checkout HEAD -- src/atft_gat_fan/models/architectures/atft_gat_fan.py
git checkout HEAD -- src/models/components/gat_layer.py
git checkout HEAD -- configs/atft/config.yaml
git checkout HEAD -- configs/atft/train/production.yaml

# 3. 依存関係更新
pip uninstall pydantic  # 不要になった場合
```

## 部分ロールバックオプション

### 機能別無効化
```python
# settings.pyで個別無効化
config = get_settings()
config.output_head_small_init = False
config.use_ema = False
config.freq_dropout_p = 0.0
```

### ランタイム設定変更
```python
# トレーニングスクリプト内で動的変更
if problem_detected:
    config['improvements']['use_ema'] = False
    config['improvements']['freq_dropout_p'] = 0.0
    # モデル再初期化
    model = ATFT_GAT_FAN(config)
```

## ロールバック後の検証

### 必須チェック項目
```bash
# 1. 基本機能テスト
python scripts/smoke_test.py

# 2. 既存性能確認
python -c "
import torch
from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN
config = {'model': {'hidden_size': 128}, 'data': {...}}
model = ATFT_GAT_FAN(config)
print('✓ Model loads successfully')
print(f'✓ Parameters: {sum(p.numel() for p in model.parameters()):,}')
"

# 3. データ処理確認
python -c "
from src.data.loaders.production_loader_v2_optimized import ProductionDatasetOptimized
dataset = ProductionDatasetOptimized('path/to/data')
print(f'✓ Dataset loaded: {len(dataset)} samples')
"
```

### 性能回帰テスト
```bash
# 改善前後の比較
python scripts/validate_improvements.py --baseline-only

# 期待値確認:
# - RankIC@1d: 改善前水準に戻っている
# - メモリ使用量: 改善前水準に戻っている
# - 学習時間: 改善前水準に戻っている
```

## 問題別対応表

| 問題症状 | 原因 | ロールバック対象 | 復旧時間 |
|----------|------|------------------|----------|
| OOMエラー | EMA/オンライン正規化 | `use_ema: false`, `online_normalization: false` | 5分 |
| 学習発散 | small-init/Huber | `output_head_small_init: false`, `huber_loss: false` | 5分 |
| 性能劣化 | GAT温度/FreqDropout | `gat_temperature: 1.0`, `freq_dropout_p: 0.0` | 10分 |
| 収束遅延 | Warmup不足 | `warmup_steps` 増加 or small-init無効 | 15分 |
| メモリリーク | ストリーミング処理 | `memory_map: false`, `online_normalization: false` | 10分 |

## 予防措置

### 事前バックアップ
```bash
# 設定ファイルバックアップ
cp configs/atft/config.yaml configs/atft/config.yaml.backup.$(date +%Y%m%d_%H%M%S)
cp configs/atft/train/production.yaml configs/atft/train/production.yaml.backup.$(date +%Y%m%d_%H%M%S)

# モデルバックアップ
cp models/best_model.pth models/best_model.pth.backup.$(date +%Y%m%d_%H%M%S)
```

### 段階的デプロイ
```bash
# Phase 1: 基本機能のみ
improvements:
  output_head_small_init: true
  gat_temperature: 1.0

# Phase 2: 学習安定化追加
improvements:
  use_ema: true
  huber_loss: true

# Phase 3: 高度機能追加
improvements:
  freq_dropout_p: 0.1
  online_normalization: true
```

### モニタリング強化
```python
# 異常検知
if loss > threshold or torch.isnan(loss):
    logger.error(f"Anomaly detected: loss={loss}")
    # 自動ロールバック
    rollback_to_baseline()
```

## 復旧後の対応

### ログ分析
```bash
# エラーログ確認
tail -f logs/training.log | grep -i "error\|fail\|rollback"

# 性能ログ確認
python -c "
import pandas as pd
logs = pd.read_csv('logs/metrics.csv')
print(logs.tail())
print('Performance after rollback:')
print(f'RankIC: {logs[\"rankic_h1\"].mean():.4f}')
"
```

### 再有効化計画
```python
# 問題分析後の再有効化
def gradual_reenable():
    # Phase 1: 低リスク機能から
    config['improvements']['output_head_small_init'] = True

    # Phase 2: 中リスク機能
    if no_issues_detected:
        config['improvements']['use_ema'] = True

    # Phase 3: 高リスク機能
    if performance_improved:
        config['improvements']['freq_dropout_p'] = 0.05
```

## 連絡先
- **緊急時**: Slack #ml-alerts または電話
- **技術的問題**: GitHub Issues またはメール
- **性能問題**: MLチームLeadに直接連絡

## 最終確認リスト

ロールバック実行後の確認:
- [ ] モデル初期化正常
- [ ] データローダー正常
- [ ] トレーニング開始可能
- [ ] 既存性能維持
- [ ] エラーログクリア
- [ ] メモリ使用量正常
- [ ] サービス安定稼働

---

*最終更新: 2024年*
*対応責任者: ML Platform Team*
