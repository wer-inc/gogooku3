# Dagster運用ガイド

このガイドは、gogooku3プロジェクトでDagsterを使用したデータセット構築の運用方法を説明します。

## 目次

1. [クイックスタート](#クイックスタート)
2. [設定ファイル](#設定ファイル)
3. [よくある使用ケース](#よくある使用ケース)
4. [トラブルシューティング](#トラブルシューティング)
5. [パフォーマンス最適化](#パフォーマンス最適化)

---

## クイックスタート

### 基本コマンド

```bash
# ヘルパースクリプトを使用（推奨）
./scripts/dagster_run.sh production           # 本番環境で全体構築
./scripts/dagster_run.sh incremental          # 増分更新のみ
./scripts/dagster_run.sh merge                # チャンクマージのみ

# バックグラウンド実行
./scripts/dagster_run.sh production --background
./scripts/dagster_run.sh incremental --background

# ドライラン（設定確認）
./scripts/dagster_run.sh production --dry-run
```

### 手動実行

```bash
# 環境変数設定
export DAGSTER_HOME=/workspace/gogooku3/gogooku5
export PYTHONPATH=/workspace/gogooku3/gogooku5/data/src

# チャンク構築
dagster asset materialize -m dagster_gogooku5.defs \
  -c run_configs/dagster_production.yaml \
  --select g5_dataset_chunks

# チャンクマージ
dagster asset materialize -m dagster_gogooku5.defs \
  -c run_configs/dagster_production.yaml \
  --select g5_dataset_full

# チャンク構築 + マージ (両方実行)
dagster asset materialize -m dagster_gogooku5.defs \
  -c run_configs/dagster_production.yaml \
  --select '*'
```

### Dagsterアセット構成

- `g5_dataset_chunks`: DatasetBuilder を呼び出してチャンクを構築。`status.json` を `completed` / `failed_schema_mismatch` などで更新。
- `g5_schema_gate`: 直近のチャンクをすべて検証し、マニフェストとハッシュ・dtype が一致しない場合は `Failure` を発生させてマージ処理を止める。
- `g5_dataset_full`: `g5_schema_gate` が成功した場合のみ実行され、`data/tools/merge_chunks.py` を呼び出して最新データセットを生成。

---

## 設定ファイル

### 本番環境設定 (`dagster_production.yaml`)

**用途**: 全期間データセット構築 (2020-2024)

**特徴**:
- `resume: true` - 完了済みチャンクをスキップ
- `index_option_parallel_concurrency: 75` - 75並列で高速化
- `source_cache_mode: read_write` - キャッシュを有効活用

**推奨ユースケース**:
- 初回データセット構築
- 週次/月次の全体更新
- データ品質検証

### 増分更新設定 (`dagster_incremental.yaml`)

**用途**: 最新データのみ更新

**特徴**:
- `latest_only: true` - 最新チャンクのみ処理
- `force: true` - 完了済みでも再構築
- `source_cache_force_refresh: true` - キャッシュ強制更新
- `index_option_cache_ttl_days: 1` - 1日で期限切れ

**推奨ユースケース**:
- 日次データ更新
- 最新四半期の再計算
- デバッグ/検証

---

## よくある使用ケース

### ケース1: 初回データセット構築 (2020-2024)

```bash
# 1. 本番設定で全チャンク構築（バックグラウンド）
./scripts/dagster_run.sh production --background

# 2. 進捗確認
tail -f logs/dagster_production_*.log

# 3. 完了後、チャンクをマージ
./scripts/dagster_run.sh merge
```

**所要時間**: 約6-12時間 (75並列の場合)

### ケース2: 日次データ更新

```bash
# 毎日実行（cronジョブ推奨）
./scripts/dagster_run.sh incremental --background
```

**所要時間**: 約15-30分

**cronジョブ例**:
```cron
# 毎日朝8時に実行
0 8 * * * cd /workspace/gogooku3 && ./scripts/dagster_run.sh incremental --background >> logs/cron_incremental.log 2>&1
```

### ケース3: 特定期間の再構築

**カスタム設定ファイル作成**:
```yaml
# run_configs/dagster_custom_2023.yaml
ops:
  g5_dataset_chunks:
    config:
      start: "2023-01-01"
      end: "2023-12-31"
      chunk_months: 3
      resume: false
      force: true
      refresh_listed: true

resources:
  dataset_builder:
    config:
      data_output_dir: "/workspace/gogooku3/output"
      source_cache_force_refresh: true  # 2023年データを強制再取得
      index_option_parallel_concurrency: 75
```

**実行**:
```bash
./scripts/dagster_run.sh custom --config run_configs/dagster_custom_2023.yaml
```

### ケース4: 並列度のオンデマンド調整

**方法1: 環境変数で上書き**
```bash
export INDEX_OPTION_PARALLEL_CONCURRENCY=50
./scripts/dagster_run.sh production
```

**方法2: カスタム設定ファイル**
```yaml
# run_configs/dagster_slow.yaml (負荷軽減版)
resources:
  dataset_builder:
    config:
      index_option_parallel_concurrency: 20  # 並列度を下げる
```

---

## トラブルシューティング

### 問題1: チャンク構築が途中で止まる

**症状**: 特定のチャンクでハング

**原因**: Index Options APIのタイムアウト

**解決策**:
```bash
# 1. プロセスを停止
kill $(cat /tmp/dagster_production.pid)

# 2. 並列度を下げて再実行
export INDEX_OPTION_PARALLEL_CONCURRENCY=30
./scripts/dagster_run.sh production --background
```

### 問題2: キャッシュが古い

**症状**: 最新データが反映されていない

**解決策**:
```bash
# 強制リフレッシュ
./scripts/dagster_run.sh incremental  # source_cache_force_refresh=true
```

### 問題3: "completed" チャンクが再実行されない

**原因**: `resume: true` が有効

**解決策**:
```yaml
# 設定ファイルで force: true に変更
ops:
  g5_dataset_chunks:
    config:
      force: true  # 完了済みでも再構築
```

### 問題4: Dagster asset not found

**症状**: `Error: Unknown asset g5_dataset_chunks`

**解決策**:
```bash
# 1. 環境変数確認
export DAGSTER_HOME=/workspace/gogooku3/gogooku5
export PYTHONPATH=/workspace/gogooku3/gogooku5/data/src

# 2. アセット一覧確認
dagster asset list -m dagster_gogooku5.defs

# 期待される出力:
# g5_dataset_chunks
# g5_dataset_full
```

---

## パフォーマンス最適化

### Index Options並列度のチューニング

| 並列度 | 所要時間 (590日) | 推奨環境 |
|-------|----------------|---------|
| **4** (default) | ~22分 | 開発環境 |
| **8** (推奨) | ~11分 | 標準環境 |
| **30** | ~4分 | 本番環境（安定） |
| **75** (最速) | ~18秒 | 本番環境（高速） |

**推奨設定**:
- 開発/デバッグ: `8-20`
- 本番環境: `30-50`
- 高速構築: `75` (ただしAPI負荷に注意)

### キャッシュ戦略

**キャッシュモード**:
| モード | 説明 | ユースケース |
|--------|------|-------------|
| `read_write` | キャッシュ読み書き両方 | 通常運用（推奨） |
| `read` | キャッシュ読み取りのみ | デバッグ |
| `off` | キャッシュ無効 | データ検証 |

**TTL設定**:
```yaml
resources:
  dataset_builder:
    config:
      # Index Options: 2週間保持
      index_option_cache_ttl_days: 14

      # 日次更新時は短く
      index_option_cache_ttl_days: 1
```

### ASOF/Tagによるスナップショット管理

**本番環境でのバージョン管理**:
```yaml
resources:
  dataset_builder:
    config:
      source_cache_asof: "2024-11-11"  # 特定日付のスナップショット
      source_cache_tag: "v1.0.0"       # バージョンタグ
```

これにより、同じ日付範囲でも異なるキャッシュキーを生成:
- `index_option_raw_2020-01-01_2024-12-31_asof-2024-11-11_v1.0.0`

---

## スキーマ検証とデータ品質管理

### 概要

データセットの品質を保証するため、すべてのチャンクは標準化されたスキーマに従う必要があります。スキーマ検証システムは、チャンク作成時とマージ前にスキーマの一貫性を自動的にチェックします。

- チャンクビルド: `SchemaValidator` に失敗した場合は `status.json.state = "failed_schema_mismatch"` となり、Dagster の `resume` 対象から除外されます。
- マージ: `g5_schema_gate` が manifest と異なるチャンクや `failed_schema_mismatch` を検出した場合は Dagster run を失敗させ、`merge_chunks.py` も複数の schema hash を拒否します。

### スキーママニフェスト

**マニフェストの場所**: `/workspace/gogooku3/gogooku5/data/schema/feature_schema_manifest.json`

**マニフェストの内容**:
- **version**: スキーマバージョン (例: "1.0.0")
- **schema_hash**: スキーマの16文字SHA256ハッシュ
- **total_columns**: カラム総数 (例: 2788)
- **columns**: 全カラムの定義 (name, dtype, nullable)

**スキーマハッシュの計算方法**:
```
SHA256("col1:dtype1;col2:dtype2;...")[:16]
```

### チャンク検証コマンド

**基本的な検証**:
```bash
# すべてのチャンクの状態確認
cd /workspace/gogooku3/gogooku5/data
python tools/check_chunks.py

# スキーマ検証を有効化
python tools/check_chunks.py --validate-schema

# スキーマミスマッチで失敗（CI/CD用）
python tools/check_chunks.py --validate-schema --fail-on-schema

# スキーマミスマッチを警告のみ（開発用）
python tools/check_chunks.py --validate-schema --no-fail-on-schema
```

**期待される出力**:
```
[INFO] Using schema manifest: /workspace/gogooku3/gogooku5/data/schema/feature_schema_manifest.json
[INFO] Expected schema hash: 2875957eecefb206
Chunk                 Rows State              Schema       Issues
-------------------------------------------------------------------------------------
2020Q1              213155 completed    ✓ 2875957e
2020Q2              224538 completed    ✓ 2875957e
2020Q3              226374 completed    ✓ 2875957e
...
[OK] All 14 chunks validated successfully
```

### スキーマミスマッチの対処方法

**症状**: チャンクがスキーママニフェストと一致しない

**原因**:
1. 古いバージョンのコードでチャンクを作成
2. フィーチャー生成ロジックの変更
3. データソースAPIの変更
4. マニュアルでのデータ編集

**対処手順**:

**Step 1: ミスマッチの詳細確認**
```bash
python tools/check_chunks.py --validate-schema --no-fail-on-schema

# 詳細なミスマッチ情報を表示
# - Missing columns: マニフェストにあるがチャンクにない
# - Extra columns: チャンクにあるがマニフェストにない
# - Type mismatches: 型が一致しない (例: Int16 vs Int64)
```

**Step 2: 影響範囲の評価**

**Option A: チャンク再構築（推奨）**
```bash
# ミスマッチのあるチャンクを特定
python tools/check_chunks.py --validate-schema | grep "✗"

# 該当チャンクのステータスをクリア
rm /workspace/gogooku3/gogooku5/data/output/chunks/2020Q1/status.json

# 再構築
./scripts/dagster_run.sh production
```

**Option B: マニフェスト更新（非推奨）**
```bash
# 現在のチャンクを新しい標準として採用する場合のみ
# WARNING: 既存の検証が無効化されます
python gogooku5/data/tools/regenerate_schema_manifest.py \
  --reference-chunk /workspace/gogooku3/output/chunks/2023Q1/ml_dataset.parquet \
  --output /workspace/gogooku3/gogooku5/data/schema/feature_schema_manifest.json
```

**Option C: 年次部分マージ（一時的回避策）**
```bash
# スキーマが一致する期間のみマージ
python tools/merge_chunks.py \
  --chunks-dir output/chunks \
  --output-dir output \
  --filter-by-year 2020,2021,2022 \
  --validate-schema \
  --fail-on-schema-mismatch
```

### マニフェスト再生成手順

**状況**: 全チャンクのスキーマを統一的に更新する場合

**前提条件**:
- 信頼できる参照データセット（最新かつ正しいスキーマ）
- すべてのチャンクを再構築する準備

**手順**:

**1. 参照データセットの準備**
```bash
# 最新のコードで1チャンクを構築
./scripts/dagster_run.sh custom --config run_configs/dagster_single_chunk.yaml

# スキーマ確認
python -c "
import polars as pl
df = pl.read_parquet('output/chunks/2024Q4/ml_dataset.parquet', n_rows=0)
print(f'Columns: {len(df.columns)}')
print(f'Schema: {df.schema}')
"
```

**2. マニフェスト生成**
```bash
# マニフェスト生成スクリプト実行
python gogooku5/data/tools/regenerate_schema_manifest.py \
  --reference-chunk /workspace/gogooku3/output/chunks/2024Q4/ml_dataset.parquet \
  --output schema/feature_schema_manifest.json \
  --version 1.1.0

# 確認
cat schema/feature_schema_manifest.json | jq '.schema_hash, .total_columns'
```

**3. 全チャンク再構築**
```bash
# 既存チャンクのステータスをクリア
find output/chunks -name "status.json" -delete

# 全チャンクを再構築（スキーマ検証有効）
./scripts/dagster_run.sh production --validate-schema
```

**4. 検証**
```bash
# すべてのチャンクが新しいマニフェストに一致することを確認
python tools/check_chunks.py --validate-schema --fail-on-schema

# 期待される出力: すべて ✓ マーク
```

### 自動検証の統合

**チャンク作成時の検証** (実装予定):
```python
# src/builder/utils/artifacts.py
from builder.utils.schema_validator import SchemaValidator

class DatasetArtifactWriter:
    def __init__(self, ...):
        self.schema_validator = SchemaValidator()

    def save_chunk(self, df: pl.DataFrame, chunk_dir: Path) -> None:
        # スキーマ検証
        result = self.schema_validator.validate_dataframe(df)
        if not result.is_valid:
            raise ValueError(f"Schema mismatch: {result}")

        # 保存
        df.write_parquet(chunk_dir / "ml_dataset.parquet")

        # メタデータにスキーマ情報を追加
        metadata["feature_schema_hash"] = result.schema_hash
        metadata["feature_schema_version"] = self.schema_validator.manifest["version"]
```

**マージ前の検証** (実装予定):
```python
# tools/merge_chunks.py
from builder.utils.schema_validator import validate_chunks_directory

def main():
    # マージ前にすべてのチャンクを検証
    results = validate_chunks_directory(
        chunks_dir=Path(args.chunks_dir),
        fail_fast=True  # 最初のミスマッチで停止
    )

    # 検証失敗時はマージを中止
    failed = [cid for cid, res in results.items() if not res.is_valid]
    if failed:
        print(f"❌ Schema validation failed for: {failed}")
        print("Please rebuild these chunks or update manifest")
        sys.exit(1)
```

### ヘルスチェックとの統合

**定期的なスキーマ検証** (5分ごと):
```bash
# tools/dagster-health-check.sh にスキーマチェックを追加済み
# 自動的に以下を実行:
# - check_chunks.py --validate-schema --fail-on-schema
# - 失敗時はステータスを "failed_schema_mismatch" に設定

# 手動実行
./tools/dagster-health-check.sh

# cron設定例
*/5 * * * * /workspace/gogooku3/tools/dagster-health-check.sh >> logs/health-check.log 2>&1
```

### ベストプラクティス

**1. チャンク作成前にマニフェストを確認**
```bash
cat /workspace/gogooku3/gogooku5/data/schema/feature_schema_manifest.json | \
  jq '{version, schema_hash, total_columns}'
```

**2. 新機能追加時はマニフェストを更新**
```bash
# 新しいフィーチャーを追加した場合
# 1. テストチャンクを作成
# 2. スキーマ確認
# 3. マニフェスト再生成
# 4. 全チャンク再構築
```

**3. マージ前に必ずスキーマ検証**
```bash
python tools/check_chunks.py --validate-schema --fail-on-schema
# ✅ すべて合格してからマージ実行
./scripts/dagster_run.sh merge
```

**4. CI/CDパイプラインに組み込む**
```yaml
# .github/workflows/dataset-build.yml
- name: Validate chunk schemas
  run: |
    python tools/check_chunks.py --validate-schema --fail-on-schema
```

---

## ステップ3: CLIプロセス完了後の移行計画

### 現在の状況 (2025-11-11 22:46 JST時点)

- CLIプロセス 2つ実行中: 2021Q2, 2021Q4を構築中
- 完了予定: 明日朝6:00-8:00 JST頃
- 完了後のチャンク数: 23/23 (100%)

### 完了後のワークフロー

**1. チャンクマージ (Dagster推奨)**

```bash
# CLIプロセス完了確認
ps aux | grep gogooku5-dataset | grep -v grep
# → 何も表示されなければ完了

# チャンク数確認
ls -1 /workspace/gogooku3/output/chunks/*/ml_dataset.parquet | wc -l
# → 23 であればOK

# Dagsterでマージ実行
./scripts/dagster_run.sh merge
```

**2. Future returns追加**

```bash
# マージ後のデータセットにFuture returnsを追加
# (この手順は別途実装必要)
```

**3. 以降は Dagster で運用**

```bash
# 日次更新（cronジョブ化推奨）
./scripts/dagster_run.sh incremental --background

# 月次全体更新（月末）
./scripts/dagster_run.sh production --background
```

---

## まとめ

**今後はDagsterをメインツールとして使うための準備が完了しました**:

✅ **完了した対応**:
1. 環境変数の互換性修正 (`INDEX_OPTION_PARALLEL_CONCURRENCY`)
2. 本番環境/増分更新用の設定ファイル作成
3. ヘルパースクリプト (`dagster_run.sh`) 作成
4. 運用ガイド作成

✅ **利用可能な機能**:
- YAMLファイルでの簡単な設定管理
- バックグラウンド実行対応
- 完了済みチャンクの自動スキップ
- キャッシュ制御 (ASOF/tag対応)
- Index Options 75並列対応

📋 **次のステップ**:
1. CLIプロセス完了まで待機 (明日朝)
2. `./scripts/dagster_run.sh merge` でチャンクマージ
3. Future returns追加（実装必要）
4. 日次更新をcronジョブ化

何か質問や追加の調整が必要な箇所はありますか？
