# 開発: コーディング規約・命名

## フォーマット/静的解析
- Black: 88列（`black .`）
- isort: black プロファイル（`ruff check --fix` に統合）
- Ruff: Lint 一式（`ruff check src/ --fix`）
- mypy: 厳しめ設定（`mypy src/gogooku3`）

## 命名規約
- モジュール/関数/変数: `snake_case`
- クラス: `CamelCase`
- ファイル: `snake_case.py`、ディレクトリは意味的に

## コミット規約（Conventional Commits）
- `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
- 例: `feat(training): add EMA teacher`

## PR 方針
- 変更は最小限・関連範囲に限定
- 何を/なぜを明確化、関連Issue/メトリクスを添付
- CI グリーンを必須（テスト/品質ゲート合格）

