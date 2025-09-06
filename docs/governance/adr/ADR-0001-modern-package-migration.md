# ADR-0001: モダンPythonパッケージ移行（v2.0.0）

日付: 2025-08-28  
ステータス: Accepted

## 背景 / Context
- 旧構成（scripts直下・相対import・単一CLI）がスケールに限界
- 型/品質/配布の一貫性確保、後方互換性維持が課題

## 決定 / Decision
- `src/` 配下のモダンパッケージ構成へ移行（`src/gogooku3`）
- 統一CLI: `gogooku3` を導入（legacyはcompat経由で段階的廃止）
- 設定管理: `.env` + Pydantic settings
- 品質ゲート: pre-commit + ruff + mypy + bandit

## 根拠 / Rationale
- importの健全化とテスト/型/ビルドの標準化
- 配布/再利用性の向上（ライブラリ/アプリ双方で整合）

## 影響 / Consequences
- メリット: 可読性・保守性・CI一貫性向上、互換レイヤーで移行容易
- デメリット: 初期移行コスト、フォルダ再配置に伴うリンク更新
- マイグレーション: `MIGRATION.md` の手順に従い段階的移行

## 代替案 / Alternatives
- 旧構成維持: 技術的負債の固定化、スケールに非対応のため不採用

## 参照 / References
- MIGRATION: `/MIGRATION.md`
- 変更履歴: `docs/releases/changelog.md`

