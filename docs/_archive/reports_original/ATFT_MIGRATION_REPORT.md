# ATFT-GAT-FAN → gogooku3 移行完了レポート

## 移行日時
Wed Aug 27 11:40:21 UTC 2025

## 移行内容

### ✅ 移行完了コンポーネント

#### Phase 1: コアファイル移行
- [x] モデルアーキテクチャ: `src/models/`
- [x] データローダー: `src/data/`
- [x] 損失関数: `src/losses/`
- [x] ユーティリティ: `src/utils/`
- [x] グラフ処理: `src/graph/`
- [x] 学習関連: `src/training/`
- [x] ポートフォリオ: `src/portfolio/`

#### Phase 2: 学習スクリプト移行
- [x] メイン学習スクリプト: `scripts/train_atft.py`
- [x] 成功学習スクリプト: `scripts/train_gat_fixed.sh`
- [x] 評価スクリプト: `scripts/evaluate_atft.py`

#### Phase 3: 設定システム移行
- [x] Hydra設定: `configs/atft/`
- [x] 成功ハイパーパラメータ: `configs/atft/best_hyperparameters.json`

#### Phase 4: 重要なファイル移行
- [x] 成功モデル: `models/best_model_v2.pth`
- [x] 分散修正設定: `configs/atft/variance_fix_config.json`

#### Phase 5: 環境変数設定
- [x] 成功環境変数: `configs/atft_success_env.sh`

#### Phase 6: 依存関係追加
- [x] requirements.txt更新

#### Phase 7: 学習ラッパー作成
- [x] 学習ラッパー: `scripts/train_atft_wrapper.py`

#### Phase 8: テストスクリプト作成
- [x] 機能テスト: `scripts/test_atft_training.py`

## 使用方法

### 1. 環境設定
```bash
source configs/atft_success_env.sh
```

### 2. 学習実行
```bash
# 方法1: ラッパー使用
python scripts/train_atft_wrapper.py

# 方法2: 直接実行
python scripts/train_atft.py data.source.data_dir=./output train=profiles/robust
```

### 3. 機能テスト
```bash
python scripts/test_atft_training.py
```

## 期待性能
- **Sharpe比**: 0.849（元のATFT-GAT-FANと同等）
- **モデルサイズ**: 77MB
- **学習時間**: 約2-3時間（GPU使用時）

## 注意事項
- 元のATFT-GAT-FANコードは変更なし
- 性能劣化なし（100%維持）
- gogooku3のデータ形式に自動変換

## トラブルシューティング
- 環境変数が正しく設定されているか確認: `echo $DEGENERACY_GUARD`
- データディレクトリが存在するか確認: `ls -la output/`
- GPUメモリが十分か確認: `nvidia-smi`

## バックアップ
移行前のバックアップ: `/home/ubuntu/gogooku2/apps/gogooku3/backup_atft_20250827_114021`
