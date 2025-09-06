# 開発: ローカル環境構築

Gogooku3 をローカルで動かすための手順です。

## 要件
- Python 3.10+
- Docker/Docker Compose（任意、フル環境向け）

## セットアップ
```bash
make setup            # 依存インストール（pip install -e .[dev]）
cp .env.example .env  # 環境変数テンプレートをコピーして値を設定
```

## 実行
```bash
make dev              # 開発モード起動（必要に応じて）
make test             # テスト実行
make lint             # 品質チェック（ruff/mypy/bandit）
```

## Docker 環境
```bash
docker-compose up -d   # 各種サービス起動（ClickHouse/MinIO/Redis等）
make docker-logs       # ログ確認
make docker-down       # 停止
```

## トラブルシューティング
- `.env` の必須キー設定を確認（`JQUANTS_*`, `WANDB_API_KEY` 等）
- 権限問題: `chown -R $USER:$USER output/` で修正
- 依存問題: `pip install -U pip setuptools wheel` 実行

