# 開発: ローカル環境構築

Gogooku3 をローカルで動かすための手順です。

## 要件
- Python 3.11+
- CUDA 12.x 対応 GPU（任意）

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

# Docker ベースのローカル環境は 2025-10 時点で廃止しました。MinIO や ClickHouse が必要な場合は既存のクラウド/社内インフラへ接続してください。

## トラブルシューティング
- `.env` の必須キー設定を確認（`JQUANTS_*`, `WANDB_API_KEY` 等）
- 権限問題: `chown -R $USER:$USER output/` で修正
- 依存問題: `pip install -U pip setuptools wheel` 実行
