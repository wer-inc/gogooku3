# Gogooku3-standalone クイックスタートガイド

## 🚀 10分で始める（セキュリティ強化版）

### 前提条件
- Docker & Docker Compose インストール済み
- Python 3.10+
- 最低16GB RAM、50GB ディスク空き容量
- セキュリティ意識（環境変数管理）

### Step 1: クローン & セキュアセットアップ
```bash
cd /home/ubuntu/gogooku3-standalone

# .env.example をコピーして編集
cp .env.example .env
nano .env  # 必須の環境変数を設定

# 必須環境変数（最低限）:
# MINIO_ROOT_USER=your_secure_username
# MINIO_ROOT_PASSWORD=your_secure_password
# CLICKHOUSE_USER=default
# CLICKHOUSE_PASSWORD=your_secure_ch_password
# REDIS_PASSWORD=your_secure_redis_password
# JQUANTS_AUTH_EMAIL=your_email@example.com
# JQUANTS_AUTH_PASSWORD=your_secure_api_password

# ディレクトリ初期化
mkdir -p logs data/processed output results
```

### Step 2: セキュアDocker起動
```bash
# セキュア設定で全サービス起動（推奨）
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

# または従来通り起動（開発用のみ）
docker compose up -d

# 起動確認（数分かかる場合があります）
docker compose ps
```

### Step 3: セキュリティ検証 & アクセス確認
```bash
# セキュリティ検証（必須）
python ops/health_check.py health

# サービス確認
curl http://localhost:9001    # MinIO Console
curl http://localhost:8123    # ClickHouse HTTP
curl http://localhost:3000    # Grafana (監視)
curl http://localhost:5000    # MLflow (実験管理)

# 新機能確認
curl http://localhost:8000/healthz   # ヘルスチェック
curl http://localhost:8000/metrics   # Prometheusメトリクス
```

## 📊 UIアクセス

### MinIO（オブジェクトストレージ）
- URL: http://localhost:9001
- ユーザー: `${MINIO_ROOT_USER}` (.envで設定)
- パスワード: `${MINIO_ROOT_PASSWORD}` (.envで設定)
- バケット: gogooku, feast, mlflow, dagster

### Grafana（監視ダッシュボード）
- URL: http://localhost:3000
- ユーザー: admin
- パスワード: admin (初回ログイン後に変更推奨)
- ダッシュボード: gogooku3-overview (自動作成)

### MLflow（ML実験管理）
- URL: http://localhost:5000
- Experiments タブで実験確認
- Models タブでモデル管理
- ATFT-GAT-FANモデルが自動追跡

### Prometheusメトリクス（監視）
- メトリクスURL: http://localhost:8000/metrics
- REDメトリクス: Rate, Error, Duration
- SLAメトリクス: コンプライアンス監視
- カスタムメトリクス: トレーニング/データ品質

## 🎯 基本的な使い方

### 1. システム検証（必須）
```bash
# 全体ヘルスチェック
python ops/health_check.py health

# データ品質チェック有効化
export DATA_QUALITY_ENABLED=1

# パフォーマンス最適化有効化（オプション）
export PERF_POLARS_STREAM=1
export PERF_MEMORY_OPTIMIZATION=1
```

### 2. データパイプライン実行
```bash
# CLIから直接実行（推奨）
python main.py ml-dataset

# データ品質検証
python data_quality/great_expectations_suite.py validate --input data/processed/dataset.parquet

# 特徴量エンジニアリング
python main.py direct-api-dataset
```

### 3. MLモデル学習
```bash
# 高速学習モード
python main.py safe-training --mode quick

# 本番学習モード（最適化適用）
export PERF_POLARS_STREAM=1
export PERF_CACHING_ENABLED=1
python main.py safe-training --mode full

# ATFT完全学習
python main.py complete-atft
```

### 4. 監視・品質確認
```bash
# リアルタイムメトリクス
python ops/metrics_exporter.py --once

# パフォーマンスベンチマーク
pytest tests/ -k "performance" --benchmark-only

# ログ監視
tail -f logs/main.log

# バックアップ検証
ls -la backups/
```
```

### 3. 特徴量ストア利用
```python
from feast import FeatureStore

# Feature Store接続
store = FeatureStore(repo_path="scripts/feature_store")

# 特徴量取得
features = store.get_online_features(
    features=["price_features:close", "price_features:returns_1d"],
    entity_rows=[{"ticker": "7203"}]
).to_dict()
```

## 🛑 停止 & クリーンアップ

### サービス停止
```bash
docker-compose down
```

### 完全クリーンアップ
```bash
docker-compose down -v
rm -rf dagster_home/storage/* output/*
```

## 🆘 トラブルシューティング

### サービスが起動しない
```bash
# ログ確認
docker-compose logs [service-name]

# 再起動
docker-compose restart [service-name]
```

### メモリ不足エラー
```bash
# Docker Desktop設定でメモリ増加
# Settings → Resources → Memory → 16GB以上
```

### ポート競合
```bash
# 使用中のポート確認
lsof -i :3001  # Dagster
lsof -i :5000  # MLflow
lsof -i :9001  # MinIO

# 別プロセスを停止するか、docker-compose.ymlでポート変更
```

## 📚 詳細ドキュメント

- [実装状況レポート](IMPLEMENTATION_STATUS.md)
- [設計仕様書](docs/archive/gogooku3-spec.md)
- [MLデータセット仕様](docs/ML_DATASET_COLUMNS.md)

## 💡 便利なコマンド

```bash
# サービス状態確認
docker-compose ps

# ログ表示（リアルタイム）
docker-compose logs -f --tail=100

# 個別サービスログ
docker-compose logs dagster-webserver
docker-compose logs mlflow
docker-compose logs feast-server

# リソース使用状況
docker stats

# コンテナ内部アクセス
docker exec -it gogooku3-clickhouse clickhouse-client
docker exec -it gogooku3-redis redis-cli -a gogooku123
```

---
*サポートが必要な場合は、[docs/brain.md](docs/brain.md)を参照*
