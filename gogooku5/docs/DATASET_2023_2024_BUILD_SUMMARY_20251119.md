# 2023-2024統合データセット構築サマリー

**作成日**: 2025-11-19
**作成者**: Claude Code (Autonomous Session)

## 📊 構築結果

### 最終データセット

**ファイル**: `output_g5/datasets/ml_dataset_2023_2025_final_pruned.parquet`
**シンボリックリンク**: `output_g5/datasets/ml_dataset_2023_2025_final.parquet`

**統計情報**:
- **行数**: 1,880,466
- **カラム数**: 3,542
- **期間**: 2023-01-04 → 2024-12-30 (484営業日)
- **ファイルサイズ**: 14GB

### 処理フロー

```
Phase A: チャンク生成完了確認 ✅
  └─ 2023Q1-Q4, 2024Q1-Q4 (8チャンク完了)

Phase B: チャンクマージ ✅
  └─ 1,880,466行 × 4,174列

Phase C: Post-Processing ✅
  ├─ Step 1: Beta/Alpha特徴量追加 (+12列)
  ├─ Step 2: Basis Gate特徴量追加 (+9列)
  ├─ Step 3: Graph特徴量追加 (+33列)
  └─ Step 4: 全NULL列削除 (-667列)

最終結果: 1,880,466行 × 3,542列
```

## 🔧 使用されたツール

### Post-Processing Tools

1. **add_beta_alpha_bd_features_full.py**
   - Beta/Alpha (60日窓) + bd_net_adv60特徴量
   - 追加カラム: 12列
   - 実行時間: ~1分

2. **add_basis_gate_full.py**
   - Basis gate特徴量 + derivatives
   - 追加カラム: 9列
   - 実行時間: ~1分

3. **add_graph_features_full.py**
   - Graph network特徴量（相関ベース）
   - window_days=60, correlation_threshold=0.3
   - 追加カラム: 33列
   - 実行時間: ~5-10分

4. **drop_all_null_columns.py**
   - 全行NULL列の削除
   - 削除カラム: 667列
   - 実行時間: ~30秒

## ⚠️ 既知の問題

### 2025データのスキーマミスマッチ

**症状**:
- 2024Q1: 4298列
- 2023Q1, 2025Q1: 4174列
- **差分**: 124列

**影響**:
- 2025チャンク（2025Q1-Q4）を2023-2024と統合できない
- `merge_chunks.py`が「スキーマミスマッチ」エラーでスキップ

**対応方針**:
1. **短期**: 2023-2024データのみで学習開始（本ドキュメント対応）
2. **中期**: 2025チャンクの124列差分を調査・統一
3. **長期**: 2023-2025統合データセット再構築

### Code列のCategorical型問題

**症状**:
- マージ時にCode列がCategorical型になる
- Graph特徴量追加時にString型との型ミスマッチエラー

**解決策**:
- 中間ファイルでCode列をString型に変換
- `pl.col("Code").cast(pl.String)`

## 📁 ファイル構成

```
output_g5/
├── chunks/
│   ├── 2023Q1/ml_dataset.parquet (1.5GB)
│   ├── 2023Q2/ml_dataset.parquet (1.6GB)
│   ├── 2023Q3/ml_dataset.parquet (1.6GB)
│   ├── 2023Q4/ml_dataset.parquet (1.6GB)
│   ├── 2024Q1/ml_dataset.parquet (1.5GB)
│   ├── 2024Q2/ml_dataset.parquet (1.6GB)
│   ├── 2024Q3/ml_dataset.parquet (1.6GB)
│   └── 2024Q4/ml_dataset.parquet (1.7GB)
│
└── datasets/
    ├── ml_dataset_2023_2025_final_pruned.parquet (14GB) ← 最終データ
    └── ml_dataset_2023_2025_final.parquet → symlink
```

## 🎯 APEX-Ranker設定更新

以下のconfigファイルを更新済み：

**apex-ranker/configs/v0_base.yaml**:
```yaml
data:
  parquet_path: output_g5/datasets/ml_dataset_2023_2025_final_pruned.parquet
  # NOTE: 2025 data excluded due to schema mismatch (124-column差分)
```

**apex-ranker/configs/v0_short_term.yaml**:
```yaml
data:
  parquet_path: output_g5/datasets/ml_dataset_2023_2025_final_pruned.parquet
  # Dataset: 2023-2024 ONLY (484 trading days, 1.88M samples)
```

## 📝 次回セッションでの作業項目

### Priority 1: 2025スキーマ問題解決

1. **124列差分の調査**
   ```bash
   # 2024Q1 vs 2023Q1のカラム差分確認
   python3 -c "
   import polars as pl
   df1 = pl.scan_parquet('output_g5/chunks/2023Q1/ml_dataset.parquet')
   df2 = pl.scan_parquet('output_g5/chunks/2024Q1/ml_dataset.parquet')
   cols1 = set(df1.collect_schema().names())
   cols2 = set(df2.collect_schema().names())
   print('2024Q1のみ:', sorted(cols2 - cols1))
   print('2023Q1のみ:', sorted(cols1 - cols2))
   "
   ```

2. **スキーマ統一方針決定**
   - オプションA: 2024Q1の+124列を他チャンクに追加（NULL埋め）
   - オプションB: 2024Q1の+124列を削除して4174列に統一
   - オプションC: 全チャンク再ビルド（最新スキーマで統一）

3. **2023-2025統合データセット再構築**

### Priority 2: NULL率検証

```bash
# TTM特徴量のNULL率確認（100% → 35%改善検証）
python scripts/check_null_columns.py \
  --dataset output_g5/datasets/ml_dataset_2023_2025_final_pruned.parquet \
  --output docs/NULL_RATE_REPORT_2023_2024_FIXED_20251119.md
```

### Priority 3: 学習実行

```bash
# APEX-Ranker v0_base学習
cd /workspace/gogooku3
python apex-ranker/scripts/train_v0.py \
  --config apex-ranker/configs/v0_base.yaml \
  --output models/apex_ranker_v0_2023_2024.pt \
  --max-epochs 50

# Short-term特化学習
python apex-ranker/scripts/train_v0.py \
  --config apex-ranker/configs/v0_short_term.yaml \
  --output models/apex_ranker_v0_2023_2024_short.pt \
  --max-epochs 50
```

## 🔍 トラブルシューティング

### ディスククォータエラー

**症状**: `OSError: [Errno 122] Disk quota exceeded`

**解決策**:
1. 中間ファイル削除（beta_bd, basis, with_graph33など）
2. Arrow IPCファイル生成スキップ（Parquetのみ保存）
3. 古いデータセット削除（2025単年データなど）

### Makefile変数展開バグ

**症状**: `build-range-dataset`で`$$yQ$$qi`が展開されない

**修正**: Makefile:241行目
```makefile
# Before
q="$$yQ$$qi"

# After
q="$${y}Q$${qi}"
```

### drop_all_null_columns.pyの型エラー

**症状**: `TypeError: int() argument must be a string... not 'Series'`

**修正**: `drop_all_null_columns.py:40-48`
```python
# Before
null_counts = df.null_count()
for name, null_count in zip(df.columns, null_counts):
    if int(null_count) == height:
        ...

# After
null_counts_df = df.null_count()
null_counts_dict = null_counts_df.to_dict(as_series=False)
for name, null_list in null_counts_dict.items():
    null_count = null_list[0]
    if null_count == height:
        ...
```

## 📚 関連ドキュメント

- `gogooku5/CLAUDE.md`: プロジェクト全体ガイド
- `gogooku5/Makefile`: Dataset/Training targets
- `apex-ranker/EXPERIMENT_STATUS.md`: APEX-Ranker実験状況
- `apex-ranker/INFERENCE_GUIDE.md`: 推論実行ガイド

## ✅ チェックリスト

- [x] Phase A: チャンク生成完了確認
- [x] Phase B: 2023-2024マージ
- [x] Phase C: Post-processing (beta/alpha, basis_gate, graph, NULL削除)
- [x] APEX-Ranker config更新
- [x] ドキュメント作成
- [ ] 2025スキーマ問題解決
- [ ] 2023-2025統合データセット構築
- [ ] NULL率検証レポート
- [ ] APEX-Ranker学習実行
