# output/ と outputs/ クリーンアップ完了レポート

**実行日時**: 2025-11-04 08:30 UTC
**実行者**: Claude Code (Autonomous Mode)

---

## 📊 削除結果サマリー

### output/ ディレクトリ
- **削除前**: 44GB (17,742ファイル)
- **削除後**: 20GB (12,420ファイル)
- **削減量**: **24GB (54%削減)**
- **削除ファイル数**: 5,322ファイル

### outputs/ ディレクトリ
- **サイズ**: 8.9MB (変更なし)
- **理由**: 最近使用されたHydra設定とログのみ

---

## 🗑️ 削除されたファイル

### 1. 古いデータセットファイル (約23GB)

削除されたファイル:
```
✓ ml_dataset_latest_full_filled_89feat.parquet (3.9GB) - Nov 3
✓ ml_dataset_latest_clean_with_adv.parquet (3.9GB) - Nov 2
✓ ml_dataset_latest_clean_final.parquet (3.8GB) - Nov 2
✓ ml_dataset_latest_clean_v2.parquet (3.9GB) - Nov 2
✓ ml_dataset_latest_clean.parquet (4.0GB) - Nov 2
✓ ml_dataset_latest_full_filled.parquet (3.9GB) - Oct 29
```

**理由**: 最新データセット (`ml_dataset_with_csz.parquet` 5.2GB) が存在するため、古いバージョンは不要

### 2. 古いサンプルデータセット (約656MB)

削除されたディレクトリ:
```
✓ output/atft_data_sample_300000/ (595MB)
✓ output/atft_data_sample_20000/ (50MB)
✓ output/atft_data_sample_5000/ (11MB)
✓ output/atft_data_sample_1000/ (2KB)
```

**理由**: テスト用サンプルで、本番データ (`output/atft_data/` 7.3GB) が存在

### 3. 古いグラフキャッシュ (約690MB)

削除されたディレクトリ:
```
✓ output/graph_cache/202509/ (471MB) - 19日前
✓ output/graph_cache/202510/ (217MB) - 19日前
```

**理由**: 19日以上前のキャッシュで再生成可能

### 4. テストディレクトリ (数KB)

削除されたディレクトリ:
```
✓ output/test/
✓ output/overfitting_test/
✓ output/hpo_test/
```

**理由**: 古いテストディレクトリで現在未使用

---

## ✅ 保持されたファイル

### 最新データセット
```
✓ output/ml_dataset_with_csz.parquet (5.2GB) - Nov 3 最新
✓ output/ml_dataset_latest.parquet -> (symlink)
✓ output/datasets/ (4.9GB) - Oct 27
```

### 使用中のデータ
```
✓ output/atft_data/ (7.3GB) - Nov 3 使用中
  ├── train/ (4.4GB, 12,086ファイル)
  ├── val/ (1.5GB)
  └── test/ (1.5GB)

✓ output/panel_cache_test_csz/ (1.6GB) - Nov 3
✓ output/checkpoints/ (160MB) - 最近7日以内
✓ output/raw/ (397MB) - キャッシュデータ
```

### その他
```
✓ output/results/ (78MB)
✓ output/baselines/ (1006KB)
✓ output/hpo_production/ (993KB)
✓ output/macro/ (1.1MB)
✓ output/reports/ (3.6MB)
✓ outputs/inference/ (8.9MB) - Hydra設定
```

---

## 📈 ディスク使用量改善

| 項目 | Before | After | 削減量 |
|------|--------|-------|--------|
| **output/** | 44GB | 20GB | **-24GB (54%)** |
| **outputs/** | 8.9MB | 8.9MB | 0 |
| **合計** | 44GB | 20GB | **-24GB** |
| **ファイル数** | 17,742 | 12,420 | **-5,322 (30%)** |

---

## 🔍 削除されたファイルの内訳

### カテゴリ別
```
古いデータセット:       23GB (6ファイル)
サンプルデータセット:   656MB (4ディレクトリ)
グラフキャッシュ:       690MB (2ディレクトリ)
テストディレクトリ:     数KB (3ディレクトリ)
```

### 日付別
```
2025-10-29以前:   3.9GB (1ファイル)
2025-11-02:       約16GB (4ファイル)
2025-11-03:       3.9GB (1ファイル)
2025-10-16:       690MB (グラフキャッシュ)
```

---

## 💾 ディスク容量状況

### 削除前
```bash
$ df -h /workspace
Filesystem                Size  Used Avail Use% Mounted on
mfs#euro.runpod.net:9421  2.3P  1.8P  549T  77% /workspace
```

### 削除後
```bash
24GB の容量を解放
総使用量: 77% → 約77% (0.001%改善)
```

---

## 🎯 推奨事項

### 1. 定期的なクリーンアップ

#### 週次クリーンアップ (推奨)
```bash
# crontab -e で以下を追加
# 毎週日曜日 午前2時に実行
0 2 * * 0 find /workspace/gogooku3/output -name "ml_dataset_*.parquet" -mtime +14 -delete
0 2 * * 0 find /workspace/gogooku3/output/graph_cache -type d -mtime +30 -exec rm -rf {} +
```

#### 月次クリーンアップ (オプション)
```bash
# 毎月1日 午前3時に実行
0 3 1 * * find /workspace/gogooku3/output/checkpoints -type f -mtime +60 -delete
```

### 2. データセット命名規則

最新データセットを明確にするため:
```bash
# 推奨: タイムスタンプ付き命名
ml_dataset_YYYYMMDD_HHMMSS.parquet

# 最新へのシンボリックリンク
ln -sf ml_dataset_20251104_083000.parquet ml_dataset_latest.parquet
```

### 3. キャッシュ管理

```bash
# 古いキャッシュを定期削除
find output/graph_cache -type d -mtime +30 -exec rm -rf {} +
find output/panel_cache_* -type f -mtime +30 -delete
```

---

## ✅ 完了確認

- [x] 古いデータセット削除 (23GB)
- [x] サンプルデータセット削除 (656MB)
- [x] 古いグラフキャッシュ削除 (690MB)
- [x] テストディレクトリ削除
- [x] 最新データ保持確認
- [x] ディスク容量削減確認 (24GB)

**ステータス**: 🎉 **完了**

---

## 📚 参考情報

### 削除コマンド (実行済み)
```bash
# 1. 古いデータセットファイル
rm -f output/ml_dataset_latest_full_filled_89feat.parquet
rm -f output/ml_dataset_latest_clean*.parquet
rm -f output/ml_dataset_latest_full_filled.parquet

# 2. サンプルデータセット
rm -rf output/atft_data_sample_*

# 3. 古いグラフキャッシュ
rm -rf output/graph_cache/202509/
rm -rf output/graph_cache/202510/

# 4. テストディレクトリ
rm -rf output/test/
rm -rf output/overfitting_test/
rm -rf output/hpo_test/
```

### 検証コマンド
```bash
# ディスク使用量確認
du -sh output/ outputs/

# ファイル数確認
find output/ outputs/ -type f | wc -l

# 最新データセット確認
ls -lh output/ml_dataset_*.parquet
```

---

🤖 **Generated with [Claude Code](https://claude.com/claude-code)**
Co-Authored-By: Claude <noreply@anthropic.com>
