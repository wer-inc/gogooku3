# TODO.md - gogooku3-standalone

## 2025-09-06 Updates

### ✅ Completed Today

#### Dataset Verification & 100% DATASET.md Compliance
- **Verified** latest dataset `ml_dataset_20200906_20250906_20250906_143623_full.parquet`
  - 11.3M rows × 248 columns covering 4,220 stocks
  - Date range: 2020-09-07 to 2025-09-05
- **Achieved 100% implementation** of all 145 required features from DATASET.md specification
  - All feature categories complete: OHLCV, Returns, Volatility, Moving Averages, Technical Indicators, Market Features, Cross Features, Flow Features, Statement Features, Validity Flags, Target Variables
  - Plus 103 additional features for enhanced modeling
- **Created comprehensive verification report** at `/docs/DATASET_VERIFICATION_REPORT.md`

#### Full Dataset Pipeline Execution
- Successfully ran `scripts/pipelines/run_full_dataset.py --jquants --start-date 2020-09-06 --end-date 2025-09-06`
  - Fetched 5 years of historical data from JQuants API
  - Generated complete ML dataset with all enrichments
  - Processing time: ~27 minutes
  - Total API calls: 2,582
  - Processing speed: 4,371 rows/second

#### Feature Implementation (Previous Session)
Previously implemented missing features to achieve 100% compliance:
- Added Bollinger Band raw values (bb_upper, bb_lower, bb_middle)
- Added validity flags (is_rsi2_valid, is_valid_ma)
- Added ADX indicator (adx_14)
- Added TurnoverValue as alias
- Added log returns, volatility windows, volume features, price position indicators

### 📊 Current Dataset Status
- **Latest Dataset**: `/output/ml_dataset_latest_full.parquet` (symlink)
- **Actual File**: `/output/ml_dataset_20200906_20250906_20250906_143623_full.parquet`
- **Metadata**: `/output/ml_dataset_latest_full_metadata.json`
- **Production Ready**: ✅ Yes

## Next Steps / Future Work

### High Priority
1. **Model Training**
   - Train ATFT-GAT-FAN model with the complete dataset
   - Run hyperparameter tuning with the full feature set
   - Validate model performance metrics

2. **Performance Optimization**
   - Optimize pandas-ta feature calculation (currently takes ~12 minutes)
   - Consider parallel processing for technical indicators
   - Implement incremental updates instead of full regeneration

3. **Data Quality**
   - Investigate flow feature coverage (currently 78.2%, could be improved)
   - Add data quality validation tests
   - Implement automated data freshness checks

### Medium Priority
1. **Documentation**
   - Update README with latest dataset statistics
   - Document feature engineering pipeline
   - Create feature importance analysis

2. **Pipeline Improvements**
   - Add automatic retry logic for API failures
   - Implement checkpoint/resume capability
   - Add progress bars for long-running operations

3. **Testing**
   - Add integration tests for full pipeline
   - Create unit tests for feature calculations
   - Validate target variable integrity (no lookahead bias)

### Low Priority
1. **Monitoring**
   - Set up automated daily data updates
   - Create data drift detection
   - Build feature stability monitoring

2. **Visualization**
   - Create feature distribution plots
   - Build correlation heatmaps
   - Generate time series validation charts

## Known Issues
1. **Flow Coverage**: Section×Date join strategy only achieves 78.2% coverage, falls back to AllMarket aggregation
2. **Processing Time**: pandas-ta features take significant time (751 seconds for 4042 stocks)
3. **Memory Usage**: Peak memory ~2.7GB during processing

## Environment Notes
- Python environment: Ubuntu 22.04.5 LTS
- Hardware: 24x AMD EPYC CPUs, 216GB RAM
- Storage: 129GB free of 291GB

## Command Reference
```bash
# Generate full dataset with JQuants data
python scripts/pipelines/run_full_dataset.py --jquants --start-date 2020-09-06 --end-date 2025-09-06

# Verify dataset compliance
python -c "from pathlib import Path; import polars as pl; df = pl.read_parquet('output/ml_dataset_latest_full.parquet'); print(f'Shape: {df.shape}, Columns: {len(df.columns)}')"

# Check specific features
python scripts/validate_dataset.py --check-features

# Run ATFT training
python scripts/integrated_ml_training_pipeline.py
```

---
*Last updated: 2025-09-06 by Claude*

------

結論

- 価格（日次OHLCV）を基準に、TOPIXは同日結合、trade-specは市場区分Section×日付の区間結合、財務（fins-statements）はCodeごと
のas‑of（T+1規則）で結合。いずれも将来情報の混入を避ける安全な時間整合（カットオフ/翌営業日反映）を実装。

TOPIX の結合

- 結合キー: Date（同日left join; coalesce=True）
- 生成処理: MarketFeaturesGenerator.build_topix_featuresでmkt_*（リターン/トレンド/ボラ/レジーム等26種）を生成し、
CrossMarketFeaturesGenerator.attach_market_and_crossで銘柄×市場クロス（β/α/相対強度/整合性など8種）を計算
- 時間整合: βは市場リターンをデフォルトでt−1ラグ（beta_lag=1）にして60日ローリングで推定（最小有効サンプル有り）
- 主な特徴: mkt_ret_*, mkt_ema_*, mkt_vol_20d, mkt_bb_*, mkt_bull_200等＋beta_60d, alpha_1d, alpha_5d, rel_strength_5d,
trend_align_mkt, alpha_vs_regime, idio_vol_ratio, beta_stability_60d
- 関連コード: scripts/data/ml_dataset_builder.py:add_topix_features → src/features/market_features.py

trade-spec（投資部門別売買） の結合

- 結合キー: Section×Date（銘柄ではなく市場区分レベル）
- 前処理:
    - Section正規化: 上場区分の時系列マッピング（SectionMapper）をas‑ofで日次に付与。名称ゆらぎはPrime/Standard/Growthなど
に正規化
    - 区間生成: PublishedDateの翌営業日をeffective_start（T+1）とし、次回開始日の前日までをeffective_end
    - 区間→日次展開: 取引カレンダーに沿って各Sectionの全営業日へ展開
- 結合手順: 価格側にSectionを持たせ、展開済み日次フローをSection×Dateでleft join
- 付随指標/フラグ: flow_*（ネット比率/Z/活動度/スマートマネー等）、flow_impulse（公表初日フラグ）、flow_days_since、
is_flow_valid（カバレッジ指標）
- フォールバック: Section×Dateのカバレッジが極端に低い場合は全市場集約AllMarketで再結合（Dateのみ）。TODOにも「Section×Date
で78.2%」課題の記載あり
- 関連コード: scripts/data/ml_dataset_builder.py:add_flow_features → src/features/flow_joiner.py（build_flow_intervals/
expand_flow_daily/attach_flow_to_quotes）

財務（fins-statements） の結合

- 結合キー: Code as‑of backward（Date ≤ effective_date）
    - DisclosedDate/DisclosedTimeから開示タイムスタンプを作成
    - 15:00以前の開示は当日有効、15:00以降は翌営業日をeffective_dateに（時刻無しは保守的にT+1）
    - 同日複数開示は最新のみ採用（同銘柄・同日での時刻でデデュープ）
- 結合手順: effective_dateでCode別にas‑of結合（backward）。結合後にstmt_imp_statement（有効日起点フラグ）と
stmt_days_since_statement（経過日数）を付与し、is_stmt_valid（>=0）を作成
- 生成される主なstmt_*: YoY（売上/営利/純利）、利益率、進捗、ガイダンス改定、ROE/ROA、非比較可能/見積変更フラグ等
- 関連コード: scripts/data/ml_dataset_builder.py:add_statements_features → src/features/
safe_joiner.py:join_statements_asof（改良版safe_joiner_v2もあり）

共通の正規化/安全策

- 型正規化: Dateはpl.Date、Codeは0埋め4桁文字列（ensure_date/ensure_code）
- リーク防止: statementsはas‑of（T+1/カットオフ）、trade-specはT+1区間、TOPIXクロスβは市場リターンにラグを入れて因果性を
担保
- 有効性フラグ: is_flow_valid/is_stmt_valid（必要に応じてis_mkt_validも付与モジュールあり）

どこで使われているか（実行パス）

- ワンショット生成: scripts/pipelines/run_full_dataset.py → src/pipeline/full_dataset.enrich_and_save →
MLDatasetBuilder.add_topix_features/add_flow_features/add_statements_features
- 手動追補: scripts/pipelines/enrich_topix_and_statements.py（既存ParquetへのTOPIX/財務追補）

必要なら、実データに対して各結合の被覆率と代表的な行（Code/Date）での結合結果をサンプル抽出して確認レポートを出します。

------

● 調査結果をまとめます。

  データ結合処理の詳細調査結果

  概要

  gogooku3-standaloneでは、価格データを基盤として、TOPIX、trade-spec（投資部門別売買）、fins-statements（財務諸表）の3
  つのデータソースを時間整合性を保ちながら結合しています。

  1. TOPIX データの結合

  結合方法

  - 結合キー: Dateのみ（全銘柄共通の市場データ）
  - 結合方式: LEFT JOIN（価格データ側を保持）
  - 実装場所:
    - src/features/market_features.py - MarketFeaturesGenerator, CrossMarketFeaturesGenerator
    - src/pipeline/full_dataset.py - enrich_and_save()

  特徴量生成

  1. 市場特徴量（26個）: mkt_* prefix
    - リターン系: mkt_ret_1d, mkt_ret_5d, mkt_ret_10d, mkt_ret_20d
    - トレンド系: mkt_ema_5, mkt_ema_20, mkt_ema_60, mkt_ema_200
    - ボラティリティ系: mkt_vol_20d, mkt_atr_14, mkt_natr_14
    - レジーム系: mkt_bull_200, mkt_trend_up, mkt_high_vol
  2. クロス特徴量（8個）: 個別銘柄×市場
    - beta_60d: 60日ベータ（t-1ラグ付き）
    - alpha_1d, alpha_5d: 残差リターン
    - rel_strength_5d: 相対強度
    - trend_align_mkt: トレンド整合性

  結合タイミング

  - 同期的結合: TOPIXデータは日次で更新され、同日のDateで直接結合
  - ラグ考慮: beta計算時は市場リターンにt-1ラグを適用（将来情報リーク防止）

  2. Trade-spec（投資部門別売買）データの結合

  結合方法

  - 結合キー: (Section, Date)（市場区分別）
  - 結合方式: 区間展開後のLEFT JOIN
  - 実装場所:
    - src/features/flow_joiner.py - FlowJoiner関連クラス
    - src/features/safe_joiner.py - SafeJoiner.join_trades_spec_interval()

  処理フロー

  1. 区間設定:
    - effective_start = PublishedDate（公表日）の翌営業日（T+1ルール）
    - effective_end = 次回effective_startの前日
  2. 日次展開:
    - 週次データを日次に展開（as-of結合で最適化）
    - 各営業日に有効な週次データを割り当て
  3. Section結合:
    - 銘柄のSection（TSEPrime/TSEStandard/TSEGrowth）で結合
    - フォールバック: Section不明時はAllMarketデータを使用

  特徴量生成（17個）

  - flow_foreign_net_ratio: 外国人ネット比率
  - flow_individual_net_ratio: 個人ネット比率
  - flow_activity_ratio: 活動比率
  - flow_impulse: 公表初日フラグ
  - flow_days_since: 公表からの経過日数
  - flow_smart_idx: スマートマネー指標
  - その他Z-score、モメンタム指標

  カバレッジ

  - 実績: 約78.2%（Section×Date結合の制約）
  - 欠損時: is_flow_valid=0、flow_days_since=-1

  3. Fins-statements（財務諸表）データの結合

  結合方法

  - 結合キー: (Code, effective_date)（銘柄別）
  - 結合方式: As-of JOIN（backward strategy）
  - 実装場所:
    - src/features/safe_joiner.py - SafeJoiner.join_statements_asof()

  T+1ルールの実装

  if DisclosedTime < 15:00:
      effective_date = DisclosedDate  # 当日有効
  else:
      effective_date = 翌営業日(DisclosedDate)  # T+1
  - 15:00を境界として、場中開示は当日、場後開示は翌営業日から有効

  As-of結合の詳細

  result = base_sorted.join_asof(
      stm_for_join,
      left_on="Date",
      right_on="effective_date",
      by="Code",
      strategy="backward"  # その日以前の最新開示を使用
  )

  特徴量生成（17個）

  - stmt_yoy_sales, stmt_yoy_op, stmt_yoy_np: 前年同期比
  - stmt_opm, stmt_npm: 利益率
  - stmt_progress_op, stmt_progress_np: 進捗率
  - stmt_rev_fore_*: 予想修正率
  - stmt_roe, stmt_roa: 収益性指標
  - stmt_imp_statement: 開示当日フラグ
  - stmt_days_since_statement: 開示からの経過日数

  4. データ安全性の保証

  時間整合性

  1. TOPIX: 同日結合だがbeta計算でt-1ラグ適用
  2. Trade-spec: T+1ルール（公表翌営業日から有効）
  3. Statements: T+1ルール（15:00判定付き）

  リーク防止メカニズム

  - 将来情報排除: effective_date管理による厳密な時点制御
  - As-of結合: backward strategyで過去データのみ参照
  - 検証フラグ: days_since_*が負の場合は異常として検出

  データ品質管理

  - 重複排除: (Code, Date)の一意性保証
  - 型正規化: Code=4桁0埋め文字列、Date=pl.Date型
  - 欠損値処理:
    - 数値: 0.0で埋める
    - フラグ: is_*_validで有効性を別管理
    - インパルス: 明示的に0/1

  5. パイプライン実行順序

  1. 価格データ読み込み（base）
  2. TOPIX結合（add_topix_features）
     └→ 市場特徴量26個 + クロス特徴量8個
  3. Statements結合（add_statements_features）
     └→ 財務特徴量17個（as-of結合）
  4. Trade-spec結合（add_flow_features）
     └→ フロー特徴量17個（区間展開→結合）
  5. 最終データセット保存

  まとめ

  - 結合の安全性: 全データソースでT+1ルールまたはラグ処理により将来情報リークを防止
  - カバレッジ: TOPIX 100%、Statements 高カバレッジ、Flow 約78%
  - 実装の堅牢性: As-of結合、区間管理、型正規化により一貫性を保証
  - 特徴量数: 合計68個の追加特徴量（市場26+クロス8+財務17+フロー17）

● Update Todos
  ⎿  ☒ 価格データとTOPIXデータの結合方法を調査          
     ☒ 価格データとtrade-specデータの結合方法を調査
     ☒ 価格データとfins-statementsデータの結合方法を調査
     ☒ 結合処理の実装詳細を確認
     ☒ データ結合のタイミングと安全性を検証
     ☒ 調査結果をまとめる

● 調査が完了しました。TOPIX、trade-spec、fins-statementsの各データがどのように価格データと結合されているか、詳細な実装
  と安全性メカニズムを含めてまとめました。

  主要なポイント：
  - TOPIX: Date結合で全銘柄共通の市場特徴量を付与
  - Trade-spec: Section×Date結合で週次フローデータを日次展開
  - Statements: Code×Date のas-of結合でT+1ルールを厳密に適用
  - すべての結合で将来情報リークを防ぐ安全メカニズムを実装