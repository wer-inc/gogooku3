# TODO.md - gogooku3-standalone

**最終更新**: 2025-10-06 15:10
**バックアップ**: `TODO.md.backup-*` (旧9,829行版を自動バックアップ済み)

---

## 📌 現在の状況 (2025-10-06)

### 🔄 進行中のタスク

#### GAT勾配ゼロ問題 - 第5次Deep Reasoning実施中
- **最終更新**: 2025-10-06 15:10
- **状態**: 新たな根本原因候補を特定、診断ログ追加完了
- **前回修正**: backbone_projection動的再作成を防止（第4次）
- **結果**: ✅ 動的層作成は解決したが、GAT勾配はゼロのまま
- **新発見**: GAT entropy/edge weight が 0 の可能性（第5次）

### ⏳ 次のタスク（優先順）

1. **第5次調査 - 診断ログ分析**（最優先）:
   - [x] 診断ログ追加完了
   - [ ] 診断トレーニング実行完了待ち
   - [ ] ログから以下を確認:
     - [ ] `[GAT-INIT]` で entropy_weight/edge_weight の値
     - [ ] `[CONFIG-DEBUG]` でモデルの設定値
     - [ ] `[GAT-DEBUG]` で gat_features の状態
     - [ ] `[GAT-LOSS]` で Loss 追加の有無

2. **根本原因特定後の修正**:
   - [ ] Case A: 設定読み込み修正
   - [ ] Case B: 分岐ロジック修正
   - [ ] Case C: Loss 計算修正

3. **修正後の検証**:
   - [ ] PHASE_MAX_BATCHES=2 で短期検証
   - [ ] GAT勾配が非ゼロになることを確認
   - [ ] 成功後、本番トレーニング開始

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

**第4次調査**: `backbone_projection`動的再作成 ✅ **部分的に解決**
- 発見: `_ensure_backbone_projection()`がforward pass中に新しいLinear層を作成
- メカニズム:
  1. `__init__`: GAT有効時 → `Linear(512, 256)`作成
  2. Optimizer初期化: この時点のパラメータのみ登録
  3. Forward pass: 次元変化検出 → 新しい`Linear(256, 256)`作成
  4. 新層のパラメータはOptimizer未登録 → 勾配計算されるが更新されない
- 証拠: `/tmp/torch_compile_disabled_verification.log`に「Adjusting backbone projection from 512 to 256」「from 256 to 512」メッセージ
- 修正実装: ゼロパディングで次元統一、動的層作成を防止
- **結果**: ✅ 動的層作成は解決、❌ GAT勾配はゼロのまま

**第5次調査**: GAT Loss Weight ゼロ仮説 🔍 **進行中**
- **経緯**: 第4次修正後もGAT勾配ゼロが継続
- **検証状況**:
  - ✅ GAT層は実行されている（ログ確認済み: `[GAT-EXEC]`）
  - ✅ edge_index は正しく渡されている（shape: [2, 2786]）
  - ✅ GAT出力は生成されている（(256, 256) → (256, 20, 256)）
  - ✅ backbone_projection 動的再作成は解決済み
  - ❌ **GAT勾配は依然としてゼロ**

- **新たな発見**（atft_gat_fan.py:577）:
  ```python
  return_attention = self.training and self.gat is not None and self.gat_entropy_weight > 0
  ```
  - この条件により、`gat_entropy_weight == 0` なら attention weights が返されない
  - attention weights がないと、GAT entropy loss が計算されない
  - 結果として GAT 出力が total loss に貢献しない可能性

- **設定ファイル確認結果**:
  - ✅ `configs/atft/model/atft_gat_fan.yaml` には設定あり:
    ```yaml
    regularization:
      edge_weight_penalty: 0.01
      attention_entropy_penalty: 0.001
    ```
  - ❓ **実際に読み込まれているかは未確認**

- **現在の仮説**:
  1. `gat_entropy_weight` が 0 に初期化されたまま
  2. または設定ファイルの値が読み込まれていない
  3. そのため `return_attention = False` となる
  4. GAT entropy loss が計算されない
  5. GAT edge regularization も weight が 0 の可能性
  6. **結果**: GAT 出力が loss に貢献せず、勾配が発生しない

- **実装した診断ログ** (2025-10-06 14:50):
  1. **GAT初期化時** (atft_gat_fan.py:337):
     ```python
     logger.info(f"[GAT-INIT] gat_entropy_weight={self.gat_entropy_weight}, gat_edge_weight={self.gat_edge_weight}")
     ```

  2. **モデル初期化後** (train_atft.py:5798-5801):
     ```python
     logger.info(f"[CONFIG-DEBUG] model.gat_entropy_weight={model.gat_entropy_weight}")
     logger.info(f"[CONFIG-DEBUG] model.gat_edge_weight={model.gat_edge_weight}")
     logger.info(f"[CONFIG-DEBUG] model.gat is None: {model.gat is None}")
     ```

  3. **Forward pass時** (atft_gat_fan.py:606-608, 616-618):
     ```python
     logger.info(f"[GAT-DEBUG] gat_features is None: {gat_features is None}")
     logger.info(f"[GAT-DEBUG] Checking concatenation: gat_features is not None = ...")
     logger.info(f"[GAT-DEBUG] Using GAT features branch" / "Using zero-padding branch")
     logger.info(f"[GAT-DEBUG] combined_features.requires_grad=...")
     ```

  4. **Loss計算時** (atft_gat_fan.py:821, 828):
     ```python
     logger.info(f"[GAT-LOSS] Adding edge_reg={edge_reg.item():.6f}")
     logger.info(f"[GAT-LOSS] Adding entropy_reg={entropy_reg.item():.6f}")
     ```

- **次のステップ**:
  1. ⏳ 診断ログ付き検証トレーニング実行中
  2. 📊 ログ分析で以下を確認:
     - `gat_entropy_weight` の実際の値
     - `gat_features` が None かどうか
     - GAT loss が total_loss に追加されているか
  3. 🔧 原因特定後、以下のいずれかを実施:
     - **Case A**: weight が 0 → 設定読み込みロジック修正
     - **Case B**: gat_features が None → 分岐ロジック修正
     - **Case C**: Loss 計算漏れ → Loss 集計ロジック修正

- **状態**: 🟡 診断ログ実装完了、検証トレーニング実行中

#### 実装した修正（第4次）

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

#### 1. GAT勾配ゼロ問題 (2025-10-06) - 第5次調査中
- **状態**: 第5次Deep Reasoning実施中、新たな仮説を検証
- **進捗**:
  - ✅ 第1次: edge_index 渡し修正
  - ✅ 第2次: .detach() 削除
  - ✅ 第3次: torch.compile 無効化
  - ✅ 第4次: backbone_projection 動的再作成防止
  - 🔍 第5次: GAT Loss Weight ゼロ仮説（進行中）
- **最新仮説**: `gat_entropy_weight=0` により GAT loss が計算されていない
- **診断**: 詳細ログ追加完了、検証トレーニング実行中
- **次のステップ**: ログ分析 → 根本原因特定 → 修正実装

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

**午前（~12:00）**:
- GAT勾配ゼロ問題の4回目のDeep Reasoning完了
- backbone_projection動的再作成が根本原因と特定
- ゼロパディングによる次元固定修正を実装
- 検証トレーニング開始（PHASE_MAX_BATCHES=20）
- TODO.md大幅整理（9,829行→約350行、92%削減）

**午後（14:50~15:10）**:
- 第4次修正後もGAT勾配ゼロが継続することを確認
- 第5次Deep Reasoning開始：GAT Loss Weight ゼロ仮説
- 新たな発見：`return_attention` 条件で `gat_entropy_weight > 0` が必要
- 4種類の診断ログ追加実装:
  - `[GAT-INIT]`: 初期化時の weight 値
  - `[CONFIG-DEBUG]`: モデル設定確認
  - `[GAT-DEBUG]`: Forward pass 状態
  - `[GAT-LOSS]`: Loss 計算追跡
- TODO.md更新：第5次調査の詳細を記録

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
