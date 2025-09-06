# 開発: テストガイド

`pytest` を用いたテスト運用ガイドです。ユニット/結合/スモーク/遅延（slow）を使い分けます。

## 原則
- 決定論: 乱数固定・時刻依存の隔離
- 独立性: テスト間の依存禁止
- 速度: 単体テストは秒単位、遅い処理は `-m slow` に分離
- ネットワーク: ユニットテストは外部ネットワーク禁止（モック）

## マーカー
- `unit`, `integration`, `smoke`, `slow`, `requires_api`

## 実行コマンド
```bash
pytest -m "not slow"
pytest --cov=src/gogooku3 --cov-report=term-missing
```

## 品質ゲート
```bash
pre-commit run --all-files
ruff check src/ --fix
mypy src/gogooku3
bandit -r src/
```

## テストデータ
- `tests/fixtures/` に固定化データを配置
- 機微情報や大容量データは保持しない（サンプル化）

