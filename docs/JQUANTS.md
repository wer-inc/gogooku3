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

## 営業日カレンダー統合（2025-09 以降）
- 標準フロー（run_full_dataset.py）に Trading Calendar API を導入しました。
- `--use-calendar-api`（既定: 有効）で、日次API呼び出し（daily_margin_interest / short_selling / short_selling_positions）を営業日配列に限定し、無駄なAPIコールを削減します。
- T+1 の付与（dmi_*, ss_*）には、営業日配列から生成した「次営業日」関数を注入します（祝日跨ぎも正しく処理）。

補助ユーティリティ（src/features/calendar_utils.py）
- `build_next_bday_expr_from_dates(dates)`
- `build_next_bday_expr_from_quotes(df)`
- `build_next_bday_expr_from_calendar_df(calendar_df, include_divisions=[1,2])`

注: 株式用途の既定は HolidayDivision in [1,2] です（祝日取引 3 は既定では含めません）。
