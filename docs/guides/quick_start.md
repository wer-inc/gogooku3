# Gogooku3-standalone クイックスタート（コンテナレス版）

> ℹ️ 2025-10 以降、Docker Compose スタックは廃止されました。本ガイドは GPU サーバ上で直接実行する前提の最短手順です。詳細は [`docs/getting-started.md`](../getting-started.md) を参照してください。

## ✅ 前提条件
- Python 3.11 以上（推奨: 仮想環境）
- CUDA 12.4 対応 GPU（A100 80GB 推奨）と最新ドライバ
- 16GB 以上のシステムメモリ / 50GB 以上の空きストレージ
- MinIO / ClickHouse / Redis など外部サービスへの接続権限
- J-Quants API 資格情報

## 🚀 5 ステップでセットアップ

### 1. リポジトリ準備
```bash
git clone git@github.com:your-org/gogooku3-standalone.git
cd gogooku3-standalone
make setup              # venv + 依存関係
```

### 2. 環境変数設定
```bash
cp .env.example .env
editor .env             # 認証情報・ホスト名を編集
```

主要変数:
- `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`
- `CLICKHOUSE_HOST`, `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`
- `REDIS_HOST`, `REDIS_PASSWORD`
- `MLFLOW_BASE_URL`, `DAGSTER_BASE_URL`
- `JQUANTS_AUTH_EMAIL`, `JQUANTS_AUTH_PASSWORD`

### 3. 周辺サービスの接続確認
```bash
# MinIO (S3 API互換)
aws --endpoint-url "$MLFLOW_S3_ENDPOINT_URL" s3 ls

# ClickHouse
clickhouse-client --host "$CLICKHOUSE_HOST" --query "SELECT 1"

# Redis
redis-cli -h "$REDIS_HOST" -a "$REDIS_PASSWORD" PING
```

### 4. 動作確認
```bash
make smoke                 # 1 epoch スモークテスト
python ops/health_check.py ready
```

### 5. 本番ワークフロー起動
```bash
python scripts/pipelines/run_full_dataset.py --jquants
python scripts/integrated_ml_training_pipeline.py
```

## 🔍 運用に役立つコマンド
```bash
make test                  # pytest (unit + integration)
make lint                  # ruff + mypy
python ops/metrics_exporter.py --once   # メトリクス確認
journalctl -u gogooku3.service --since "10 minutes ago"  # ログ確認（systemd運用例）
```

## 🧹 クリーンアップ
```bash
systemctl stop gogooku3.service      # 運用プロセス停止（例）
make clean                            # 仮想環境・キャッシュ削除
rm -rf output/experiments/*          # 生成物を手動削除
```

## 📚 追加リソース
- [docs/getting-started.md](../getting-started.md): 詳細なセットアップとトラブルシューティング
- [docs/operations/runbooks.md](../operations/runbooks.md): 本番運用手順
- [docs/ml/model-training.md](../ml/model-training.md): 学習パイプライン全体像

