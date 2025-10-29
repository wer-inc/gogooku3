# ISSUE.md - 課題管理

**最終更新**: 2025-10-21
**プロジェクト**: ATFT-GAT-FAN (Japanese Stock Market Prediction)

---

## 📋 現在の課題一覧

### 🔴 緊急課題（Critical）

#### 1. ❌ 学習結果の検証完了：Sharpe 1.367は誤報告（実際は失敗）

**発生日**: 2025-10-21
**重要度**: Critical
**ステータス**: ✅ 検証完了（学習失敗を確認）

**概要**:
報告された異常値Sharpe Ratio 1.367を検証した結果、**バッチ0の値が誤って最終結果として報告されていた**ことが判明しました。実際のエポックレベルのSharpe Ratioは極めて低く（最良0.025）、**学習は失敗しています**。

**検証結果の詳細**:

**🔴 Critical Finding: Sharpe 1.367は誤報告**
- **報告値**: 1.367202（バッチ0の単一値）
- **実際の最良エポックレベルSharpe**: 0.024531（Phase 1 Epoch 10）
- **全エポック平均Sharpe**: -0.006006（ほぼゼロ学習）
- **ターゲット**: 0.849（達成度わずか2.9%）

**📊 フェーズ別エポックレベルSharpe分析**:
```
Phase 0 (5 epochs):   Range [-0.027, 0.017]   Mean: 0.002   Best: 0.017
Phase 1 (10 epochs):  Range [-0.037, 0.025]   Mean: -0.004  Best: 0.025
Phase 2 (8 epochs):   Range [-0.025, 0.013]   Mean: -0.004  Best: 0.013
Phase 3 (6 epochs):   Range [-0.037, 0.000]   Mean: -0.018  Best: 0.000

総エポック数: 29エポック（期待30エポック）
全体統計: Mean: -0.006, Max: 0.025, Min: -0.037
```

**❌ Phase 3で性能悪化** - 最終フェーズで最悪の結果（mean -0.018）

**🔍 確認された問題点**:

1. **モデルパラメータ数の不一致**:
   - 期待値: 5,611,803 params
   - 初期値: 1,550,779 params
   - 最終値: 1,705,399 params（再ビルド後）
   - **原因**: "Dynamic feature dimension mismatch detected (expected 306, got 99)"
   - 変数選択ネットワークが自動再構築され、小型化

2. **DataLoaderワーカークラッシュ**:
   ```
   Fatal Python error: Aborted
   RuntimeError: DataLoader worker (pid 1084361) is killed by signal: Aborted
   ```
   - マルチワーカーDataLoader（workers=8）で発生
   - トレーニングは完了したが不安定

3. **バッチレベルの極端なボラティリティ**:
   - Batch 0: Sharpe 1.367（この値が誤って報告された）
   - Batch 1: Sharpe -0.027
   - Batch 2: Sharpe -0.468
   - Batch 8: Sharpe -1.006
   - **範囲**: [-1.01, +1.37]（極めて不安定）

**🔴 根本原因**:
- 特徴量次元の不一致（期待306 vs 実際99）
- モデルアーキテクチャの自動再構築による容量不足
- 99特徴量に対して最適化されていない学習設定

**✅ 検証済み項目**:
- [x] 全エポックのSharpe Ratioを確認（29エポック分を分析）
- [x] モデルパラメータ数の不一致原因を調査（特徴量不一致が原因）
- [x] Phase早期終了を確認（Phase 0-3すべて実行、合計29エポック）
- [x] DataLoaderクラッシュの発生を確認
- [x] バッチレベルのメトリクス分布を分析

**⏳ 未検証項目**:
- [ ] テストセットでの評価結果を確認
- [ ] 新特徴量のデータ品質を検証（NaN, Inf, 外れ値）
- [x] 特徴量次元不一致の原因を特定 → **解明済み（下記参照）**

**🔍 特徴量次元不一致の詳細解析**:

**モデルのハードコード期待値**:
```
✅ Using simplified feature dimensions (category split DISABLED):
   total=306, current=46, historical=260
```

**実際のデータ**:
```
✅ Auto-detected 99 feature columns
[input_dim] detected from data: F=99 (was: 13)
```

**不一致の原因**:
1. モデルアーキテクチャが306特徴量を想定してハードコード
2. 実際のparquetファイルには99特徴量のみ
3. 自動的に変数選択ネットワークを再構築（パラメータ数が減少）
4. 99特徴量に最適化されていない設定で学習開始
5. **結果**: モデル容量不足 + 学習失敗

**99特徴量の内訳**:
- feature_categories.yaml期待値: 198特徴量
- 実際の使用特徴量: 99特徴量
- **差分**: 99特徴量が欠落または使用されていない

**🔬 データセット詳細分析（完全解明済み）**:

実際のデータセット構造:
```
総カラム数: 395
├─ メタデータ列: 6 (Date, Code, target_*, is_valid, split)
├─ Raw OHLCV列: 10 (Close, Open, High, Low, Volume等)
└─ 派生特徴量: 379
   ├─ 自動検出使用: 99特徴量 ✅
   └─ フィルタ除外: 280特徴量 ❌ (74%が除外！)
```

**根本原因の全体像**:

1. **データモジュールの過剰なフィルタリング**:
   - 379個の派生特徴量が存在
   - データモジュールが99個のみ選択（280個を除外）
   - おそらくNaN/欠損値の多い特徴量を自動除外
   - **新特徴量9個は含まれている** ✅

2. **特徴量の欠損パターン**:
   - `dmi_*`（日次マージン）: 100%欠損（データ未整備）
   - `stmt_*`（財務諸表）: 100%欠損（データ未整備）
   - `x_*`（拡張特徴量）: 100%欠損（実装未完了）
   - 合計280特徴量が高率欠損により除外

3. **モデル期待値とのミスマッチ**:
   - モデルハードコード: 306特徴量期待
   - データモジュール自動検出: 99特徴量
   - 変数選択ネットワーク自動再構築 → パラメータ数減少（5.6M → 1.7M）

4. **学習失敗の連鎖**:
   ```
   99特徴量のみ使用
   → モデル容量不足（1.7M params）
   → 学習能力不足
   → Sharpe 0.025（ターゲット0.849の3%）
   → 学習失敗
   ```

**✅ 新特徴量9個の状態**:
- yz_vol_20, yz_vol_60 ✅ 含まれている
- pk_vol_20, pk_vol_60 ✅ 含まれている
- rs_vol_20, rs_vol_60 ✅ 含まれている
- vov_20, vov_60 ✅ 含まれている
- amihud_20 ✅ 含まれている

**すべて99特徴量に含まれており、学習に使用されている**

**関連ファイル**:
- 学習ログ: `/tmp/tier1_reproduction_new_features.log`
- 結果JSON: `output/results/complete_training_result_20251021_161827.json`
- チェックポイント: `models/checkpoints/atft_gat_fan_final.pt` (6.5MB)

**結論と推奨対応**:
🚫 **この学習結果は使用不可** - 報告されたSharpe 1.367は誤報告であり、実際の性能は極めて低い。

**次のアクション（優先度順）**:
1. ✅ **Safe Modeで再学習** - FORCE_SINGLE_PROCESS=1で安定実行
2. 🔧 **特徴量次元の整合性を確認** - なぜ306期待なのに99なのかを調査
3. 📊 **新特徴量の統計検証** - 9個の新特徴量が正しく生成されているか確認
4. 🎯 **前回成功設定を再現** - Tier 1 (Sharpe 0.779) の設定を厳密に再現

---

### 🟡 中優先度課題（Medium）

#### 2. 新特徴量の統合完了（✅ 一部完了）

**発生日**: 2025-10-21
**重要度**: Medium
**ステータス**: ✅ 実装完了（検証待ち）

**完了した作業**:
- ✅ 高度なボラティリティ推定量を実装（8特徴量）
  - Yang-Zhang volatility (yz_vol_20, yz_vol_60)
  - Parkinson volatility (pk_vol_20, pk_vol_60)
  - Rogers-Satchell volatility (rs_vol_20, rs_vol_60)
  - Volatility-of-Volatility (vov_20, vov_60)
- ✅ 流動性指標を実装（1特徴量）
  - Amihud illiquidity (amihud_20)
- ✅ スキーマ更新（`full_dataset.py`のallowed_prefixes）
- ✅ データセット再生成（395列、4,643,494行）
- ✅ feature_categories.yaml更新（学習設定反映）
- ✅ Git commit & push (401b846)

**データカバレッジ**:
| 特徴量 | カバレッジ |
|--------|-----------|
| yz_vol_20 | 91.8% |
| yz_vol_60 | 85.6% |
| pk_vol_20 | 92.0% |
| pk_vol_60 | 85.8% |
| rs_vol_20 | 92.0% |
| rs_vol_60 | 85.8% |
| vov_20 | 88.5% |
| vov_60 | 78.8% |
| amihud_20 | 99.2% |

**残タスク**:
- [ ] 新特徴量の統計的検証（分布、外れ値チェック）
- [ ] 学習への効果測定（Ablation study）
- [ ] 特徴量の重要度分析

---

### 🟢 低優先度課題（Low）

#### 3. ベータ列のクリーンアップ

**発生日**: 2025-10-21
**重要度**: Low
**ステータス**: ⏸️ Pending

**概要**:
バリデーションスクリプトで中間計算用のベータ列が検出されました。

**詳細**:
```
Helper beta columns should not be present: ['beta_60d_raw', 'beta_20d_raw']
```

**影響**:
- 軽微（学習には影響なし）
- データセットサイズへの小さな影響（2列分）

**対応方針**:
- データセット再生成時に中間列を除去
- 優先度低（機能的な問題なし）

---

## 📊 学習実行履歴

### 実行1: Tier 1再現 + 新特徴量（2025-10-21）

**設定**:
- Loss Weight: SHARPE=1.0, RANKIC=0.0, IC=0.0
- Epochs: 30 (Phase-based: 0→1→2→3)
- Batch Size: 1024
- Learning Rate: 2.0e-4
- GPU: A100 80GB (bf16-mixed)
- Workers: 8 (multi-worker DataLoader)
- Dataset: 4.6M rows × 395 cols (99 features used)

**結果**:
- **Reported Sharpe**: 1.367 ⚠️ （要検証）
- **Best Val Loss**: -0.0284
- **Training Time**: 88.0 min (5,281s)
- **Model Params**: 1.7M (期待値: 5.6M) ⚠️

**問題**:
- DataLoaderワーカークラッシュ
- パラメータ数の不一致
- Phase 0で早期終了の可能性
- 異常に高いSharpe値（要検証）

**ログファイル**:
- `/tmp/tier1_reproduction_new_features.log`
- `output/results/complete_training_result_20251021_161827.json`

---

## 🔧 技術的な問題

### DataLoaderクラッシュ問題

**エラーメッセージ**:
```
Fatal Python error: Aborted
terminate called without an active exception
RuntimeError: DataLoader worker (pid 1084361) is killed by signal: Aborted
```

**発生タイミング**:
- 学習実行中（Phase 0）
- マルチワーカーDataLoader使用時（workers=8）

**可能性のある原因**:
1. メモリ不足（OOM）
2. Polars/PyArrowとPyTorchのマルチプロセス競合
3. 共有メモリの枯渇
4. データローダーのバグ

**対策案**:
- [ ] Safe mode（workers=0）で再実行
- [ ] メモリ使用量のモニタリング
- [ ] Shared memory設定の確認
- [ ] データローダーのデバッグログ有効化

---

## 🎯 次のステップ

### 短期（今週中）

1. **Sharpe 1.367の検証** ⭐⭐⭐
   - [ ] 全エポックのメトリクスを抽出
   - [ ] テストセット評価を実行
   - [ ] バッチレベルの分布を分析
   - [ ] モデルパラメータ数の不一致を調査

2. **Safe Modeで再学習** ⭐⭐
   - [ ] `FORCE_SINGLE_PROCESS=1`で再実行
   - [ ] DataLoaderクラッシュを回避
   - [ ] 同じ設定で完全な30エポック実行

3. **新特徴量の検証** ⭐⭐
   - [ ] 統計的検証（分布、外れ値）
   - [ ] 相関分析
   - [ ] Ablation study（新特徴量なしとの比較）

### 中期（今月中）

4. **モデルパラメータ数の調査**
   - [ ] アーキテクチャ定義を確認
   - [ ] 期待値5.6M vs 実際1.7Mの原因特定
   - [ ] 必要に応じて修正

5. **Tier 2実験の準備**
   - [ ] hidden_size=256の設定
   - [ ] 検証済みのベストプラクティスで実行

### 長期（次回以降）

6. **本番デプロイの準備**
   - [ ] Walk-forward validation
   - [ ] リスク分析
   - [ ] アンサンブル戦略

---

## 📝 備考

### 環境情報

- **GPU**: NVIDIA A100-SXM4-80GB (CUDA 12.4)
- **PyTorch**: 2.8.0+cu128
- **Python**: 3.12.3
- **Dataset**: 2020-10-21 ～ 2025-10-20 (5年間)
- **Features**: 99 used / 395 total

### 参考資料

- Tier 1レポート: `output/reports/tier1_sharpe_weight_optimization_20251021.md`（生成後参照）
- Model Preservation: `docs/playbooks/model_preservation_playbook.md`
- Loss Curriculum: `scripts/utils/loss_curriculum.py`

---

**最終更新**: 2025-10-21 16:30 JST
