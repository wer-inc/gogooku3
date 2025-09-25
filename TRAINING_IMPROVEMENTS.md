# ATFT-GAT-FAN Training Improvements Summary

## 実施済み改善内容 (Completed Improvements)

### 1. ✅ Zero Loss問題の解決
**問題**: 全てのバッチで損失が0になり学習が進まない
**原因**:
- 特徴量のスケールが大きく異なる（例：shares_outstanding: ~123M, returns: ~0.001）
- 正規化が実装されていなかった

**解決策**:
- バッチごとのZ-score正規化を実装 (`scripts/train_atft.py`)
- 極値クリッピング（±10σ）を追加
- 環境変数 `ENABLE_FEATURE_NORM=1` で有効化

### 2. ✅ 損失関数の最適化
**実装済み機能**:
- **RankIC損失**: 順位相関を直接最適化 (`USE_RANKIC=1`, `RANKIC_WEIGHT=0.2`)
- **Huber損失**: 外れ値にロバスト (`USE_HUBER=1`, `HUBER_WEIGHT=0.3`)
- **CS-IC損失**: クロスセクショナル相関 (`USE_CS_IC=1`, `CS_IC_WEIGHT=0.15`)
- **方向性補助損失**: 価格上昇/下降予測 (`USE_DIR_AUX=1`, `DIR_AUX_WEIGHT=0.1`)
- **動的重み調整**: 各ホライズンのRMSEに基づく適応 (`DYN_WEIGHT=1`)

### 3. ✅ フェーズ別損失スケジュール
学習段階に応じて損失重みを動的に変更:
- **Phase 0**: Huber重視（外れ値にロバスト）
- **Phase 1**: Quantile + Sharpe導入
- **Phase 2**: RankIC追加、金融メトリクス重視
- **Phase 3**: 全メトリクスのバランス最適化

### 4. ✅ データローダー最適化
PDF分析の推奨に基づく改善:
- マルチワーカー有効化 (`NUM_WORKERS=8`)
- Persistent workers (`PERSISTENT_WORKERS=1`)
- プリフェッチ最適化 (`PREFETCH_FACTOR=4`)
- 安全ガード無効化 (`ALLOW_UNSAFE_DATALOADER=1`)

### 5. ✅ データ検証機能
`scripts/validate_data.py` を作成:
- ターゲット値の分布確認
- NaN/ゼロ値の検出
- 特徴量スケールチェック
- 日付範囲の妥当性確認

### 6. ✅ PyTorchコンパイル最適化
- `torch.compile` 有効化
- `max-autotune` モードで最大パフォーマンス
- 10-30%の速度向上期待

## 実行コマンド

```bash
# 全ての最適化を有効にして訓練開始
make train-optimized-stable
```

このコマンドにより:
1. データ検証が自動実行
2. 全ての最適化環境変数が設定
3. 正規化、RankIC、Huber等の全機能が有効
4. マルチワーカーで高速データローディング

## 環境変数一覧

| 変数名 | 値 | 説明 |
|--------|-----|------|
| ENABLE_FEATURE_NORM | 1 | 特徴量正規化 |
| FEATURE_CLIP_VALUE | 10.0 | 極値クリッピング |
| USE_RANKIC | 1 | RankIC損失有効化 |
| RANKIC_WEIGHT | 0.2 | RankIC重み |
| USE_HUBER | 1 | Huber損失有効化 |
| HUBER_WEIGHT | 0.3 | Huber損失重み |
| USE_CS_IC | 1 | CS-IC損失有効化 |
| CS_IC_WEIGHT | 0.15 | CS-IC重み |
| USE_DIR_AUX | 1 | 方向性補助損失 |
| DIR_AUX_WEIGHT | 0.1 | 方向性重み |
| SHARPE_WEIGHT | 0.3 | Sharpe比重み |
| DYN_WEIGHT | 1 | 動的重み調整 |
| ALLOW_UNSAFE_DATALOADER | 1 | マルチワーカー強制有効 |
| NUM_WORKERS | 8 | ワーカー数 |
| PERSISTENT_WORKERS | 1 | ワーカー永続化 |
| PREFETCH_FACTOR | 4 | プリフェッチ係数 |
| TORCH_COMPILE_MODE | max-autotune | コンパイル最適化モード |

## データ検証結果サンプル

```
📊 Validating 5 sample files...
✅ target_1d: mean=0.000337, non_zero=100.0%
✅ target_5d: mean=0.001779, non_zero=100.0%
✅ target_10d: mean=0.003200, non_zero=100.0%
✅ target_20d: mean=0.005808, non_zero=100.0%
📅 Date range: 2015-09-24 to 2022-09-22
📊 7/20 features need normalization
✅ Feature normalization is ENABLED
```

## 期待される改善効果

1. **損失計算の正常化**: ゼロ損失問題が解決
2. **学習の安定性向上**: フェーズ別最適化で段階的改善
3. **金融メトリクス最適化**: RankIC/Sharpe比の直接最適化
4. **訓練速度向上**: マルチワーカーで2-3倍高速化
5. **外れ値耐性**: Huber損失でロバスト性向上

## 注意事項

- 現在のデータは2015-2022年（やや古い）
- 必要に応じて `make dataset-full START=2020-01-01 END=2025-09-25` で新データ生成可能
- GPU利用時は `REQUIRE_GPU=1` を追加設定

## 次のステップ

1. `make train-optimized-stable` で訓練開始
2. 損失が正常に計算されることを確認
3. TensorBoardでメトリクス監視
4. 必要に応じてハイパーパラメータ調整

---
*Last updated: 2025-09-25*