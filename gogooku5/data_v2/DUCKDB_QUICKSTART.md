DuckDB をメイン DB にして J-Quants データを回す最小パターンです。
パスは `data_v2/output/jquants.duckdb` をデフォルトにしています。

## 1) テーブル初期化

```bash
PYTHONPATH=.. python data_v2/scripts/duckdb_loader.py init
# Make 経由なら:
#   make duckdb-init DUCKDB=data_v2/output/jquants.duckdb
```

## 2) カレンダーを API から直接 upsert（Parquet を書かない）

```bash
# CLI
PYTHONPATH=.. python data_v2/scripts/duckdb_loader.py \
  --db data_v2/output/jquants.duckdb \
  fetch-calendar-direct \
  --from 2015-01-01 \
  --to 2025-12-31

# Make
make -C data_v2 duckdb-load-calendar FROM_DATE=2015-01-01 TO_DATE=2025-12-31
```

## 3) /listed/info を API から直接 upsert（Parquet を書かない）

```bash
# CLI
PYTHONPATH=.. python data_v2/scripts/duckdb_loader.py \
  --db data_v2/output/jquants.duckdb \
  fetch-listed-direct \
  --start 2015-01-01 \
  --end 2025-12-31 \
  --holiday-division 1,2 \
  --auto-fetch-calendar

# Make（calendar が無ければ自動取得）
make -C data_v2 duckdb-fetch-listed START=2015-01-01 END=2025-12-31
```

## 4) 既存 Parquet を DuckDB に取り込む（必要なときだけ）

例: 2015–2020 の listed_info スナップショットを取り込む場合

```bash
PYTHONPATH=.. python data_v2/scripts/duckdb_loader.py \
  --db data_v2/output/jquants.duckdb \
  load-parquet \
  --table listed_info \
  --path /path/to/listed_2015_2020.parquet
```

trading_calendar も同様に `--table trading_calendar` で投入できます。

## 5) DuckDB から Parquet に書き出す

```bash
PYTHONPATH=.. python data_v2/scripts/duckdb_loader.py \
  --db data_v2/output/jquants.duckdb \
  export-parquet \
  --table listed_info \
  --out data_v2/output/listed_2015_2025.parquet
```

年別に分割したい場合は `--split-yearly` を付けます。

## 拡張のヒント

- 他のエンドポイント（prices, statements など）も、同じ「normalize → upsert → COPY TO」で流用できます。
- `--threads` オプションで DuckDB の並列度を明示できます（デフォルトは DuckDB 任せ）。
- trading_calendar も `load-parquet` / `export-parquet` サブコマンドで同様に扱えます。
- listed_info には `available_ts`（JST 09:00 基準の as-of）を付与し、最新スナップショットビュー `listed_info_latest` を自動作成します。
- yfinance からの価格ヒストリも `fetch-yf-history --start 2020-01-01 --end 2020-12-31` で DuckDB に直接 upsert できます（Parquet不要）。ティッカーは data と同じ固定セット（SPY, QQQ, ^VIX, DX-Y.NYB ほか）を使用し、期間は `.env` の `START`/`END` を参照します。
- マクロ用のデフォルトティッカーセット（SPY, QQQ, ^VIX, DX-Y.NYB ほか）は `--use-macro-defaults` で一括取得できます（必要なら `SLEEP_SEC` でレートを調整）。
