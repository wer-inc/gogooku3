上場銘柄一覧(/listed/info)
株価四本値(/prices/daily_quotes)
財務情報(/fins/statements)
決算発表予定日(/fins/announcement)
取引カレンダー(/markets/trading_calendar)
投資部門別情報(/markets/trades_spec)
TOPIX指数四本値(/indices/topix)
指数四本値(/indices)
日経225オプション四本値(/option/index_option)
信用取引週末残高(/markets/weekly_margin_interest)
業種別空売り比率(/markets/short_selling)
空売り残高報告(/markets/short_selling_positions)
日々公表信用取引残高(/markets/daily_margin_interest)

## 実行時の注意
- `JQUANTS_MIN_AVAILABLE_DATE`（例: `2015-09-23`）を環境変数に設定しておくと、パイプラインが自動的にサブスクリプションの下限日付を尊重し、API の `from`/`date` パラメータが範囲外になって 400 を返す問題を防げます。
- 上記が未設定の場合は `ML_PIPELINE_START_DATE` を下限として利用します。

python scripts/pipelines/run_full_dataset.py --jquants --start-date 2020-09-16 --end-date 2025-09-16 2>&1 | tee /tmp/full_dataset_run.log
