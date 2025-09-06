# セキュリティ: 認証情報と機密管理

このドキュメントは、認証情報（Secrets）と機密設定の安全な取り扱い方法をまとめます。リポジトリに秘密情報をコミットしないでください。

## 基本原則
- .env: 秘密は `.env` に保存。`.env` はGit追跡対象外。
- 参照用テンプレート: `.env.example` にキーのみ記載し、値は空にする。
- 最小権限: APIキーやDBユーザーは必要最小の権限に限定。
- ローテーション: キー流出時や権限変更時は速やかにローテーション。

## 必須環境変数（一例）
- JQuants: `JQUANTS_AUTH_EMAIL`, `JQUANTS_AUTH_PASSWORD`
- Weights & Biases: `WANDB_API_KEY`
- ストレージ/DB: `MINIO_*`, `CLICKHOUSE_*`, `POSTGRES_*`, `REDIS_*`

`.env.example` を参照の上、実運用では値を設定してください。

## 運用上の注意
- ログ出力禁止: Secrets をログ・例外メッセージに出さない。
- 権限分離: 開発/検証/本番でキーを分離し、交差使用しない。
- CI/CD: リポジトリSecrets（GitHub Actions等）で安全に注入。

## 失効とローテーション
- 失効条件: 退職・権限変更・漏洩兆候を検知した場合。
- 即時対応: 影響範囲を把握し、鍵ローテーションと監査を実施。

---

参考: `docs/ml/atft/ATFT_CRITICAL_ENV_VARS.md`（ATFT 特有の環境変数）

