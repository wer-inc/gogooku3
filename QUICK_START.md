# Gogooku3 クイックスタートガイド

## 🚀 5分で始める

### 前提条件
- Docker & Docker Compose インストール済み
- Python 3.10+
- 最低16GB RAM、50GB ディスク空き容量

### Step 1: クローン & セットアップ
```bash
cd /home/ubuntu/gogooku2/apps/gogooku3

# 環境変数設定
cat > .env << EOF
JQUANTS_AUTH_EMAIL=your_email@example.com
JQUANTS_AUTH_PASSWORD=your_password
EOF

# ディレクトリ初期化
mkdir -p dagster_home/storage output/dagster logs data/raw
```

### Step 2: Docker起動
```bash
# 全サービス起動
docker-compose up -d

# 起動確認（数分かかる場合があります）
docker-compose ps
```

### Step 3: アクセス確認
```bash
# サービス確認
curl http://localhost:9001    # MinIO
curl http://localhost:3001    # Dagster
curl http://localhost:5000    # MLflow
curl http://localhost:3000    # Grafana
```

## 📊 UIアクセス

### Dagster（パイプライン管理）
- URL: http://localhost:3001
- 初回アクセス時は数分待つ
- "Materialize all" でパイプライン実行

### MLflow（ML実験管理）
- URL: http://localhost:5000
- Experiments タブで実験確認
- Models タブでモデル管理

### Grafana（監視）
- URL: http://localhost:3000
- ユーザー: admin
- パスワード: gogooku123

### MinIO（ストレージ）
- URL: http://localhost:9001
- ユーザー: minioadmin
- パスワード: minioadmin123

## 🎯 基本的な使い方

### 1. データパイプライン実行
```bash
# Dagster UI から実行（推奨）
# http://localhost:3001 → Assets → Materialize all

# または CLI から
cd scripts
python pipelines/run_pipeline.py
```

### 2. MLモデル学習
```bash
# サンプルデータで学習
cd scripts
python mlflow/trainer.py

# MLflow UI で結果確認
# http://localhost:5000
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
