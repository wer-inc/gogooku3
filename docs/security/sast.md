# セキュリティ: SAST と品質ゲート

静的アプリケーションセキュリティテスト（SAST）と品質チェックの実施方針です。

## 推奨ツール
- Ruff: Python Lint（セキュリティルール含む）
- Bandit: 代表的な脆弱性ルール（Bxxx）検出
- mypy: 型安全性の担保（不正な型操作の抑止）
- pre-commit: 変更の入口での統合ゲート

## コマンド
```bash
# まとめ実行（推奨）
pre-commit run --all-files

# 個別
ruff check src/ --fix
bandit -r src/
mypy src/gogooku3
```

## ポリシー
- 重大度: High/Medium は原則リリース前に解消
- 例外申請: 根拠（CVSS/影響評価）と期限を記載して一時除外
- 継続改善: フレームワーク/依存の定期アップデート

## CI 連携
- PR で SAST を自動実行し、失敗時はマージブロック
- 依存脆弱性スキャン（pip-audit など）を併用可

---

参考: `pyproject.toml` の Ruff/mypy 設定、`.pre-commit-config.yaml` のフック

