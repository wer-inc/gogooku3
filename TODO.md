# TODO.md - gogooku3-standalone

**最終更新**: 2025-10-06 11:32
**バックアップ**: `TODO.md.backup-*` (旧9,829行版を自動バックアップ済み)

---

## 📌 現在の状況 (2025-10-06)

### 🔄 進行中のタスク

#### GAT勾配ゼロ問題の検証トレーニング
- **状態**: Batch 8/20完了 (11:32時点)
- **速度**: ~123s/batch
- **ログ**: `/tmp/dimension_fix_verification.log`
- **予定**:
  - Batch 10到達: ~11:36 (残り2バッチ)
  - Batch 20完了: ~11:57 (最終検証)

**修正内容**:
- GAT未実行時もゼロパディングで次元512に統一
- `_ensure_backbone_projection()`呼び出し削除
- 初期化パラメータのみ使用（Optimizer登録済み）

**検証結果（重要）**:
- ✅ モデル初期化成功（46M params）
- ✅ **"Adjusting backbone projection"メッセージなし** → 修正成功の証拠
- ⏳ Batch 0-8: GAT勾配ゼロ（Warmup期間中、正常範囲）
- **🎯 Batch 10-20で勾配非ゼロ確認が必要**

### ⏳ 次のタスク（優先順）

1. **検証完了待ち**:
   - [ ] Batch 10-20でGAT勾配が非ゼロになることを確認
   - [ ] 期待結果: `grad_norm(gat) > 0.00e+00`

2. **検証成功時**:
   - [ ] `PHASE_MAX_BATCHES`制限解除
   - [ ] 本番トレーニング開始（全エポック実行）
   - [ ] TODO.mdに最終成功レポート追加

3. **検証失敗時**:
   - [ ] 第5次Deep Reasoning実施
   - [ ] 別の根本原因を調査
   - [ ] Parameter IDとOptimizer param_groupsの対応チェック

---

## ✅ 完了した主要タスク

### 2025-10-06: GAT勾配ゼロ問題 - 調査・修正 🎯

#### 問題の背景
- ATFT-GAT-FANモデルのトレーニング中、GATレイヤーの勾配が常にゼロになる現象が発生
- GAT自体は実行されている（256→512次元変換確認済み）が、勾配がGATパラメータまで伝播しない
- 複数の修正試行を実施するも解決せず

#### 調査プロセス（4回のDeep Reasoning）

**第1次調査**: `edge_index`渡し忘れ仮説
- 仮説: `_forward_with_optional_graph()`がedge_indexをbatch dictに含めていない
- 修正: train_atft.py内の関数を2箇所修正
- 結果: ❌ GAT実行されるようになったが、勾配ゼロは継続

**第2次調査**: `.detach()`仮説
- 仮説: edge cacheの`.detach()`が勾配を切断
- 修正: train_atft.py:6785-6786の`.detach()`削除
- 検証: PHASE_MAX_BATCHES=20で実行、Batch 10（新規edge構築時）確認
- 結果: ❌ Batch 10でもGAT勾配ゼロ - 仮説は不正解

**第3次調査**: `torch.compile dynamic=False`仮説
- 仮説: torch.compileのdynamic=False設定が動的グラフと非互換
- 修正: model/atft_gat_fan.yamlでcompile.enabled=false設定
- 検証: PHASE_MAX_BATCHES=10で実行
- 結果: ❌ torch.compile無効でもGAT勾配ゼロ - torch.compileは原因ではない

**第4次調査**: `backbone_projection`動的再作成 ✅ **真の原因**
- 発見: `_ensure_backbone_projection()`がforward pass中に新しいLinear層を作成
- メカニズム:
  1. `__init__`: GAT有効時 → `Linear(512, 256)`作成
  2. Optimizer初期化: この時点のパラメータのみ登録
  3. Forward pass: 次元変化検出 → 新しい`Linear(256, 256)`作成
  4. 新層のパラメータはOptimizer未登録 → 勾配計算されるが更新されない
- 証拠: `/tmp/torch_compile_disabled_verification.log`に「Adjusting backbone projection from 512 to 256」「from 256 to 512」メッセージ

#### 実装した修正

**修正ファイル**: `src/atft_gat_fan/models/architectures/atft_gat_fan.py` (Line 600-624)

**修正内容**:
```python
# 🔧 FIX (2025-10-06): Always use consistent dimensions
if gat_features is not None:
    combined_features = torch.cat([tft_output, gat_features], dim=-1)
else:
    # GAT disabled or no edges: pad with zeros to match expected dimension
    if self.gat is not None:
        zero_pad = torch.zeros(
            tft_output.size(0), tft_output.size(1), self.gat_output_dim,
            device=tft_output.device, dtype=tft_output.dtype
        )
        combined_features = torch.cat([tft_output, zero_pad], dim=-1)
    else:
        combined_features = tft_output

# 🔧 FIX: No longer need dynamic dimension check
combined_features = self.backbone_projection(combined_features)
```

**修正のポイント**:
1. GAT未実行時も**ゼロパディング**で次元を512に統一
2. `_ensure_backbone_projection()`呼び出しを完全削除（Line 611）
3. `backbone_projection`の動的再作成を防止
4. 初期化時のパラメータのみ使用 → Optimizer登録済みパラメータで学習

---

### 2025-10-01: 395列データセット実装 ✅

**実装概要**:
ドキュメント仕様（`docs/ml/dataset_new.md`）の395列を完全実装

**完了項目**:
- Phase 1: Makefileフラグ有効化 (+56列)
  - `--enable-sector-cs`: セクター内クロスセクショナル特徴（15列）
  - `--enable-daily-margin`: 日次マージン特徴（41列）
- Phase 2: セクター集約機能 (+30列)
  - 新規ファイル: `src/gogooku3/features/sector_aggregation.py`
  - セクター等加重リターン、時系列、ボラティリティ、個別-セクター相対
- Phase 3: セクターOne-Hot (+17列)
- Phase 4: ウィンドウ成熟フラグ (+8列)
- Phase 5: インタラクション特徴 (+18列)
- Phase 6: 拡張ローリング統計 (+20列)
- Phase 7: カレンダー・レジーム特徴 (+30列)

**列数進捗**:
- 目標: 395列
- 実装完了: 392列 (99.2%)
- 残り3列: GPU-ETL特有の高度な特徴（オプション）

---

### 2025-09-16: 統合MLトレーニングパイプライン ✅

**主要機能**:
- `integrated_ml_training_pipeline.py`: 統合トレーニングスクリプト
- `SafeTrainingPipeline`: 7ステップ検証パイプライン
  1. データローディング (ProductionDatasetV3)
  2. 品質特徴生成 (QualityFinancialFeaturesGenerator)
  3. クロスセクショナル正規化 (CrossSectionalNormalizerV2)
  4. Walk-Forward分割 (WalkForwardSplitterV2、20日embargo)
  5. GBMベースライン (LightGBMFinancialBaseline)
  6. グラフ構築 (FinancialGraphBuilder)
  7. パフォーマンスレポート

**検証結果**:
- データ: 606,127サンプル、644銘柄、2021-2025 (4年+)
- 速度: 1.9s 全パイプライン実行
- メモリ: 7.0GB ピーク使用量 (目標<8GB達成)
- グラフ: 50ノード、266相関エッジ

---

### 2025-09-07: セクターエンリッチメント (一部完了)

**実装済み**:
- セクター集約モジュール (`sector_aggregation.py`)
- セクター等加重リターン計算
- セクター相対特徴（beta, alpha, z-score）

**現在の状態**: 基本機能実装完了、本番データでの大規模検証は未実施

---

## 🐛 既知の課題

### 🔍 進行中の課題

#### 1. GAT勾配ゼロ問題 (2025-10-06)
- **状態**: 修正実装完了、検証トレーニング実行中（Batch 8/20）
- **背景**: Optimizer初期化後に`backbone_projection`が動的に再作成されていた
- **根本原因**: Forward pass中の層再作成がOptimizer未登録パラメータを生成
- **修正内容**: ゼロパディングで次元統一、動的層作成を防止
- **次のステップ**: Batch 10-20でGAT勾配非ゼロ確認

### ⚠️ 未解決の課題

- (検証完了後に更新予定)

---

## 📚 重要な技術的学習

### PyTorchの重要な制約

#### 1. Optimizer初期化後の動的層追加は危険
- **問題**: `model.parameters()`は初期化時点のスナップショット
- **結果**: Forward pass中の層再作成はOptimizer未登録になる
- **対策**: 動的な層作成は`__init__`のみで行う

#### 2. 動的グラフの正しい実装
- **原則**: 次元変化が予想される場合は**ゼロパディング**で統一
- **理由**: 条件分岐でも出力次元を一定に保つ必要がある
- **実装例**:
  ```python
  if feature is None:
      # ゼロパディングで次元統一
      feature = torch.zeros(size, dim, device=device)
  combined = torch.cat([base, feature], dim=-1)
  ```

#### 3. 勾配デバッグのベストプラクティス
- **重要**: 勾配ゼロ ≠ 必ずしも計算グラフ切断
- **確認項目**:
  1. 計算グラフの連続性 (`requires_grad=True`)
  2. Optimizer登録状況 (`optimizer.param_groups`)
  3. Parameter IDとOptimizerの対応
- **ツール**:
  - `torch.autograd.grad()`: 勾配の手動確認
  - `register_hook()`: 中間層の勾配監視

---

## 📋 参考情報

### よく使うコマンド

```bash
# トレーニング（本番）
cd /home/ubuntu/gogooku3-standalone
make train-optimized                    # 最適化設定でトレーニング
make train-integrated                   # 統合パイプライン使用

# トレーニング（検証用）
PHASE_MAX_BATCHES=20 python scripts/train_atft.py \
  --config-path ../configs/atft \
  --config-name config_production_optimized

# データセット生成（GPU-ETL推奨）
make dataset-full-gpu START=2015-09-27 END=2025-09-26
make dataset-full START=2020-09-06 END=2025-09-06  # CPU版

# 検証・監視
tail -f /tmp/dimension_fix_verification.log | grep -E 'GAT grad|GUARD|Batch'
grep -E "GAT grad|GUARD|step=" /tmp/dimension_fix_verification.log

# プロセス管理
ps aux | grep "train_atft.py"           # 実行中プロセス確認
pkill -f "PHASE_MAX_BATCHES"            # 検証トレーニング停止
```

### 環境情報

```
GPU: NVIDIA A100 80GB PCIe
CPU: 24-core AMD EPYC 7V13
Memory: 216GB RAM
Storage: 291GB SSD (167GB free)
```

### 重要なファイル

```
# モデル実装
src/atft_gat_fan/models/architectures/atft_gat_fan.py  # ATFT-GAT-FANモデル

# 設定
configs/atft/config_production_optimized.yaml    # 本番最適化設定
configs/atft/model/atft_gat_fan.yaml             # モデル設定
configs/atft/train/production_improved.yaml      # トレーニング設定

# トレーニングスクリプト
scripts/train_atft.py                             # メインス クリプト
scripts/integrated_ml_training_pipeline.py        # 統合パイプライン

# データパイプライン
scripts/pipelines/run_full_dataset.py            # 完全データセット生成
src/gogooku3/features/sector_aggregation.py      # セクター機能
```

---

## 📝 履歴・変更ログ

### 2025-10-06
- GAT勾配ゼロ問題の4回目のDeep Reasoning完了
- backbone_projection動的再作成が根本原因と特定
- ゼロパディングによる次元固定修正を実装
- 検証トレーニング開始（PHASE_MAX_BATCHES=20）
- TODO.md大幅整理（9,829行→約350行、92%削減）

### 2025-10-01
- 395列データセット実装完了（392列達成、99.2%）
- 全7フェーズの特徴エンジニアリング完了

### 2025-09-16
- 統合MLトレーニングパイプライン実装
- SafeTrainingPipeline（7ステップ）実装
- 実データ検証完了（606K samples、644 stocks）

### 2025-09-07
- セクターエンリッチメント基本機能実装

---

**注**: 旧TODO.mdは`TODO.md.backup-20251006-*`として保存済み。詳細な実装履歴はバックアップファイルを参照してください。
