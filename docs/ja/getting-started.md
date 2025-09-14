# はじめに（日本語・暫定版）

このページはプレースホルダーです。セットアップ手順の日本語版は準備中です。

- 英語版: `docs/getting-started.md`
- 用語集: `docs/glossary.md`
- FAQ: `docs/faq.md`

## データセット構築（ワンショット）

### 日次マージン（dmi_）を含める
```bash
python scripts/pipelines/run_full_dataset.py --jquants \
  --start-date 2020-09-06 --end-date 2025-09-06 \
  --enable-daily-margin \
  --daily-margin-parquet output/daily_margin_interest_*.parquet
```
注意:
- dmi_* は `PublishedDate` の **翌営業日(T+1)** から有効です（as‑of backward 結合）。
- 週次マージン（margin_*）と併存します。日次を使う場合は学習で dmi_* 列を選択してください。
