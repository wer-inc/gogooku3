# ML データセット完全仕様 v1.1（フル版）

## 0) 共通ルール（キー・結合・欠損の扱い）

* **主キー**：`(Code: str[5], Date: pl.Date)` の**日次格子**。
* **結合規則**

  * **/prices/daily\_quotes**：土台（全行の出自）。
  * **/listed/info**：`as‑of 区間結合`（`valid_from/valid_to` で期間一致）。
  * **/indices/topix**：**同日結合**（Z 系は 252 営業日 warmup を別途取得）。
  * **/markets/trades\_spec**：**T+1 区間→営業日ブロードキャスト→(Section,Date) 結合**。
    `effective_start = next_business_day(PublishedDate)` / `effective_end = next_effective_start − 1日`。
  * **/fins/statements**：**15:00 カットオフ** → `effective_date` → `join_asof(backward)` by `Code`。
    `DisclosedTime < 15:00 → 当日`, それ以外→**翌営業日**。同 `(Code,effective_date)` は最新1件。
* **欠損方針**：窓不足・非対象日は **NaNのまま**＋**有効フラグ列**（`is_*_valid`）を並走。安易な補完はしない（リーク/歪み防止）。

---

## 1) 識別子・メタ（12列）

| 列名                      | 型           | 計算/付与                 | なぜ必要か             |
| ----------------------- | ----------- | --------------------- | ----------------- |
| `Code`                  | str         | J-Quants `LocalCode`  | 銘柄キー              |
| `Date`                  | date        | 営業日                   | 時系列キー             |
| `Section`               | str         | `/listed` 区間 as-of    | trades\_spec 結合キー |
| `MarketCode`            | str         | `/listed` より          | 履歴参照用             |
| `sector17_code/name/id` | str/str/int | `/listed` as-of＋辞書ID化 | 粗い業種 prior        |
| `sector33_code/name/id` | str/str/int | 同上                    | 細かい業種 prior／集約に使用 |
| `row_idx`               | int         | 銘柄内累積カウント             | 有効窓判定             |
| `SharesOutstanding`     | float       | `/listed` or別ソース      | 回転率/時価総額          |

> **補足**：`sector*_id` は学習用 Embedding、`*_code/name` は監査/可視化・後工程用。

---

## 2) 価格・出来高（約 60–70 列）

### 2.1 リターン

* `returns_h = Close/Close.shift(h) - 1`（`h∈{1,5,10,20,60,120}`）
* `log_returns_h = log1p(returns_h)`（`h∈{1,5,10,20}`）
  **背景**：スケール不変・短期(1–5)は反転/ニュース解消、中期(10–20)は継続傾向が出やすい。

### 2.2 ボラティリティ

* `volatility_w = std(returns_1d, w)*sqrt(252)`（`w∈{5,10,20,60}`）
* `realized_volatility = sqrt( Σ (log(High/Low))^2 / (4*log 2) )`
  **背景**：レジーム識別（高ボラ＝反転優位、低ボラ＝トレンド優位）。

### 2.3 移動平均・乖離

* `sma_{5,10,20,60,120}`, `ema_{5,10,20,60,200}`
* `price_to_sma5 = Close/(sma5+ε)`、`price_to_sma20`、`price_to_sma60`
* `ma_gap_5_20 = (ema5-ema20)/(ema20+ε)`、`ma_gap_20_60`
* `ema5_slope = ema5.pct_change()`, `ema20_slope`, `ema60_slope`
* `dist_to_200ema = (Close-ema200)/(ema200+ε)`
  **背景**：過熱/乖離とトレンド成熟度。

### 2.4 レンジ・出来高

* `high_low_ratio = High/Low`
* `close_to_high = (High−Close)/(High−Low+ε)`、`close_to_low = (Close−Low)/(High−Low+ε)`
* `volume_ma_{5,20}`, `volume_ratio_{5,20} = Volume/MA(Volume,w)`
* `turnover_rate = Volume/(SharesOutstanding+ε)`、`dollar_volume = Close*Volume`
  **背景**：需給・注意度合いの代理。

---

## 3) テクニカル（pandas‑ta、約 15–20 列）

* `rsi_2`, `rsi_14`, `rsi_delta = diff(rsi_14)`
* `macd_signal (MACDs_12_26_9)`, `macd_histogram (MACDh_12_26_9)`
* `bb_pct_b = (Close−BBL)/(BBU−BBL+ε)`、`bb_bw = (BBU−BBL)/(BBM+ε)`
  * 互換カラム: 出力では `bb_position`/`bb_width` も同値エイリアスとして並走（旧名対応）
* `atr_14`, `adx_14`, `stoch_k`
  **背景**：反転/継続の二面性を少数の安定窓で表現。

---

## 4) 市場（TOPIX）特徴（26 列＋Z 系）

### 4.1 当日系列

* `mkt_ret_{1,5,10,20}d`
* `mkt_ema_{5,20,60,200}`, `mkt_dev_20 = (Close−ema20)/(ema20+ε)`、`mkt_gap_5_20`
* `mkt_vol_20d`, `mkt_atr_14`, `mkt_natr_14`
* `mkt_bb_pct_b`, `mkt_bb_bw`
* `mkt_dd_from_peak = (Close − cum_max(Close))/cum_max(Close)`
* フラグ：`mkt_bull_200 = (Close>ema200).int8()`、`mkt_trend_up = (mkt_gap_5_20>0).int8()`、`mkt_big_move_flag = (abs(mkt_ret_1d) >= 2*std_60).int8()`、`mkt_high_vol = (mkt_vol_20d_z>1.0).int8()`

### 4.2 長期 Z（要 warmup≥252）

* `mkt_ret_1d_z`, `mkt_vol_20d_z`, `mkt_bb_bw_z`, `mkt_dd_from_peak_z`
  **背景**：市場レジーム・ショック耐性を同日で供給。Z は長期基準差の把握。

---

## 5) 銘柄×市場クロス（7–8 列）

* `beta_60d = Cov(returns_1d, mkt_ret_1d;60)/Var(mkt_ret_1d;60)`（`min_periods=60`）
* `alpha_1d = returns_1d − beta_60d*mkt_ret_1d`、`alpha_5d`
* `rel_strength_5d = returns_5d − mkt_ret_5d`
* `idio_vol_ratio = volatility_20d/(mkt_vol_20d+ε)`
* `beta_stability_60d = 1/(STD(beta_60d,20)+ε)`
* `trend_align_mkt = (sign(ma_gap_5_20)==sign(mkt_gap_5_20)).int8()`
  **背景**：共通因子と固有成分の分離、地合い整合の成否。

---

## 6) セクター（/listed/info）特徴（33業種ベース推奨）

### 6.1 基礎（as‑of 区間）

* `sector17_code/name/id`, `sector33_code/name/id`
* One‑Hot（推奨：17のみ）`sec17_onehot_*`、日別頻度 `sec17_daily_freq`, `sec33_daily_freq`
  **背景**：構造 prior（業種）を軽負荷で投入。

### 6.2 セクター集約（33業種）

* 集合 `G_s(t)={i | sector33_id(i,t)=s}`
* **ロバスト代表**：`sec_ret_1d_eq(s,t) = median_{i∈G_s(t)} returns_1d(i,t)`
* `sec_ret_5d_eq`, `sec_ret_20d_eq`（日次合成 or 直接中央値）
* `sec_mom_20 = Σ_{u=t-19..t} sec_ret_1d_eq(s,u)`、`sec_ema_{5,20}`、`sec_gap_5_20`
* `sec_vol_20 = std(sec_ret_1d_eq,20)*√252`、`sec_vol_20_z`（時間Z）
* 監査補助：`sec_member_cnt`、`sec_small_flag=(cnt<5).int8()`
  **背景**：業種の地合い／トレンド／不安定性を補足。

### 6.3 個別の相対化

* `rel_to_sec_5d = returns_5d − sec_ret_5d_eq(sector_of(i,t))`
* `beta_to_sec_60 = Cov(returns_1d, sec_ret_1d_eq;60)/Var(sec_ret_1d_eq;60)`
* `alpha_vs_sec_1d = returns_1d − beta_to_sec_60*sec_ret_1d_eq`
* セクター内 demean / z：`ret_1d_demeaned`, `z_in_sec_{returns_5d,ma_gap_5_20,...}`
  **背景**：共通因子除去と相対強弱の抽出（短期予測の雑音低減）。

---

## 7) 週次フロー（/markets/trades\_spec）特徴（**プレフィックス `flow_` に統一**、\~37列）

### 7.1 区間展開

* `effective_start = next_business_day(PublishedDate)`
* `effective_end = next_effective_start − 1日`（同 `Section` 内）
* 営業日テーブルに**日次ブロードキャスト** → `(Section, Date)` で付与。
  **背景**：公表値が次回更新まで有効という自然解釈。**リークなし**で日次化。

### 7.2 指標（**すべて `flow_` 接頭辞**）

* **比率**

  * `flow_foreign_net_ratio = ForeignersBalance/(ForeignersTotal+ε)`
  * `flow_individual_net_ratio = IndividualsBalance/(IndividualsTotal+ε)`
  * `flow_activity_ratio = TotalTotal/(Σ Section 全体の TotalTotal + ε)` *（任意：市場全体比）*
  * `flow_foreign_share = ForeignersTotal/(TotalTotal+ε)`
* **ブレッドス**

  * `flow_breadth_pos = mean( [ForeignersBalance>0, IndividualsBalance>0, TrustBanksBalance>0, InvestmentTrustsBalance>0, ProprietaryBalance>0, BrokerageBalance>0] )`
* **標準化（週次52）**

  * `flow_foreign_net_z = (ForeignersBalance − MA_52)/STD_52`
  * `flow_individual_net_z`, `flow_activity_z`
* **スマートマネー**

  * `flow_smart_idx = flow_foreign_net_z − flow_individual_net_z`
  * `flow_smart_mom4 = flow_smart_idx − MA_4(flow_smart_idx)`
  * `flow_smart_pos = (flow_smart_idx > 0).int8()`
  * `flow_shock_flag = (abs(flow_smart_idx) ≥ 2).int8()`
* **タイミング**

  * `flow_impulse = (Date == effective_start).int8()`
  * `flow_days_since = (Date − effective_start).days()`
* **追加指標（実装済み）**

  * `flow_activity_high`, `flow_activity_low`: 活動水準の上下分解
  * `days_since_flow_right`: 右側結合用の日数カウント
  * 各指標の`_right`サフィックス版（結合処理用の重複列）
    **背景**：主体別需給の強弱と異常フロー検知（短期の方向・続伸/反落に寄与）。拡張された37列は詳細な需給分析を可能にする。

> **互換性エイリアス**：旧名（`foreigners_net_ratio`等）を使うコードがある場合は、`flow_*` を**正本**にし、旧名を **ビュー/別名** として残すのが安全。

---

## 8) マージン特徴（週次・日次両方、合計\~86列）

### 8a) 週次マージン（**接頭辞 `margin_`**、\~45列）

週次の信用取引データから生成される特徴量：

* **基本指標**
  * `margin_long_tot`, `margin_short_tot`: 買い建・売り建総額
  * `margin_net`, `margin_total_gross`: ネット・グロス
  * `margin_credit_ratio`: 信用倍率
  * `margin_imbalance`: インバランス指標

* **ADV正規化**
  * `margin_long_to_adv20`, `margin_short_to_adv20`: 20日平均売買代金比

* **変化率**
  * `margin_d_long_wow`, `margin_d_short_wow`: 週次変化率
  * `margin_d_net_wow`, `margin_d_ratio_wow`: ネット・比率変化

* **Z-score（52週）**
  * `margin_gross_z52`, `long_z52`, `short_z52`, `ratio_z52`: 52週標準化

* **モメンタム**
  * `margin_gross_mom4`: 4週モメンタム
  * `margin_impulse`: インパルス指標

* **シェア分解**
  * `margin_neg_share_long/short`: ネゴシエイティブマージンシェア
  * `margin_std_share_long/short`: 制度信用シェア

### 8b) 日次マージン（**接頭辞 `dmi_`**、\~41列）

#### 8b.1 キー・結合・リーク防止

* **補正処理**：同一 `(Code, ApplicationDate)` について **最新の `PublishedDate`** を採用（API訂正の吸収）。
* **有効日**：`effective_start = next_business_day(PublishedDate)`（T+1 ルール）。
* **結合**：銘柄内 `effective_start` に対して **as‑of backward** で日次格子 `(Code, Date)` に付与。
* **Null規約**：`effective_start` 前は `null`、有効日に `dmi_impulse=1`、`is_dmi_valid=1`。

#### 8b.2 指標群

* **水準・比率**：`dmi_long`, `dmi_short`, `dmi_net`, `dmi_total`, `dmi_credit_ratio`, `dmi_imbalance`, `dmi_short_long_ratio`
* **変化・Z**：`dmi_d_long_1d`, `dmi_d_short_1d`, `dmi_d_net_1d`, `dmi_d_ratio_1d`, `dmi_z26_long/short/total/d_short_1d`
* **ADV正規化**（有効時）：`dmi_long_to_adv20`, `dmi_short_to_adv20`, `dmi_total_to_adv20`, `dmi_d_long_to_adv1d`, `dmi_d_short_to_adv1d`, `dmi_d_net_to_adv1d`
* **規制・イベント**：`dmi_reason_restricted`, `dmi_reason_dailypublication`, `dmi_reason_monitoring`, `dmi_reason_restrictedbyjsf`, `dmi_reason_precautionbyjsf`, `dmi_reason_unclearorseconalert`, `dmi_reason_count`, `dmi_tse_reg_level`
* **タイミング**：`dmi_impulse`, `dmi_days_since_pub`, `dmi_days_since_app`, `is_dmi_valid`

#### 8b.3 パイプライン有効化

* `scripts/pipelines/run_full_dataset.py` で：
  * `--enable-daily-margin` で有効化。
  * `--daily-margin-parquet <path>` 指定可。未指定時は `output/daily_margin_interest_*.parquet` を自動探索。

> 週次マージン（`margin_*`）と日次（`dmi_*`）は併存します。学習で日次を優先する場合は `dmi_*` を選択してください。

---

## 9) 財務（/fins/statements）特徴（\~20列＋タイミング）

### 9.1 有効日

* `effective_date = (DisclosedTime < 15:00 ? DisclosedDate : next_business_day(DisclosedDate))`
* as‑of backward で `(Code, Date)` に**直近の有効開示**を付与。
  **背景**：PEADや開示吸収のタイミングをリークなしで取り込む。

### 9.2 指標

* **YoY 成長**

  * `stmt_yoy_sales = Δ4Q(NetSales)/|NetSales.shift(4)|`
  * `stmt_yoy_op`, `stmt_yoy_np`（同様）
* **マージン**

  * `stmt_opm = OperatingProfit/(NetSales+ε)`、`stmt_npm = Profit/(NetSales+ε)`
* **進捗率**

  * `stmt_progress_op = OperatingProfit/(ForecastOperatingProfit+ε)`、`stmt_progress_np`
* **ガイダンス改定率（前回比）**

  * `stmt_rev_fore_op = (ForecastOperatingProfit − prev_ForecastOperatingProfit)/(|prev|+ε)`
  * `stmt_rev_fore_np`, `stmt_rev_fore_eps`, `stmt_rev_div_fore`
* **財務比率**

  * `stmt_roe = Profit/(Equity+ε)`、`stmt_roa = Profit/(TotalAssets+ε)`
* **品質/タイミング**

  * `stmt_change_in_est = (ChangesInAccountingEstimates ∈ {"true","1"}).int8()`
  * `stmt_nc_flag = ((ChangesBasedOnRevisionsOfAccountingStandard=="true") | (RetrospectiveRestatement=="true")).int8()`
  * `stmt_imp_statement = (Date==effective_date).int8()`
  * `stmt_days_since_statement = (Date − effective_date).days()`
    **背景**：サプライズ方向/大きさ、進捗、品質で短期の再価格付けを説明。

---

## 10) 有効フラグ（マスク）（14 列）

* **窓成熟**：`is_rsi2_valid (row_idx≥5)`, `is_ema5_valid (≥15)`, `is_ema10_valid (≥30)`, `is_ema20_valid (≥60)`, `is_ema200_valid (≥200)`
* **市場Z**：`is_mkt_z_valid (warmup≥252)`
* **β/α**：`is_beta_valid (≥60)`
* **セクター集約**：`is_sec_valid (member_cnt≥3)`
* **フロー**：`is_flow_valid (Section一致 & 日次展開成功)`
* **財務**：`is_stmt_valid (開示取得済)`
* **総合**：`is_valid_ma (row_idx≥60)` 等
  **背景**：欠損を埋めず、**使えるときだけ使う**意思をモデルに伝える。

---

## 11) 目的変数（ターゲット）

* **回帰**：`target_{1,2,3,5,10} = Close.shift(-h)/Close − 1`
* **分類**：`target_{1,5,10}_binary = (target_h>0).int8()`
  **背景**：業務要件（短期 1–3 日重視）に合わせてマルチホライズン。

---

## 12) 正規化（任意・学習時に fold 内で）

* **クロスセクション Z**（その日ごと）：`x_cs_z = (x − mean_by(Date))/(std_by(Date)+ε)`
* **グループ内 Z**（セクター内）：`z_in_sec_*`（上記 6.3）
  **背景**：地合い除去・相対化で汎化を高める（**学習データ内で fit**、推論は transform のみ）。

---

## 13) 列総数（実装済み）

* 識別子・メタ：**12**
* 価格/出来高：**\~70**（returns, log_returns, volatility, moving averages, volume ratios等）
* テクニカル：**\~20**（RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic等）
* 市場（TOPIX）：**\~30**（market returns, EMA, volatility, Z-scores等）
* クロス：**\~8**（beta, alpha, relative strength, idiosyncratic vol等）
* セクター：**\~40**（基礎、集約、相対、One-Hot、頻度含む）
* 週次フロー（flow\_\*）：**\~37**（拡張版：foreign, individual, smart money, timing等）
* 週次マージン（margin\_\*）：**\~45**（買建、売建、Z-score、シェア分解等）
* 日次マージン（dmi\_\*）：**\~41**（水準、変化、ADV正規化、規制フラグ等）
* 財務（stmt\_\*）：**\~20**（YoY成長、マージン、進捗、ガイダンス改定等）
* 有効フラグ：**\~14**
* ターゲット：**\~12**（multi-horizon targets and binary versions）
* その他の特徴：**\~146**（earnings events, short selling, options, cross-features等）

**実装済み合計：** **395 列**

※ 仕様の225列を大幅に超え、実際の実装では拡張された特徴量が追加されています。主な拡張：
- フロー特徴の詳細化（18→37列）
- マージン特徴の分離（週次45列＋日次41列）
- その他の追加特徴（earnings, short selling, options, cross-features等）

---

## 14) 互換性・命名の最終確認（漏れ対策）

* **flow 列は `flow_` 接頭辞で統一**（旧名は alias として残す）：
  `foreigners_net_ratio → flow_foreign_net_ratio` 等。
* **TOPIX Z 系**は warmup が無い期間は**NaN で正常**（`is_mkt_z_valid` で管理）。
* **セクター集約**は `sector33` をデフォルト、`member_cnt<5` は `sec_small_flag=1`。
* **β/α** は `min_periods=60`、不足は NaN＋`is_beta_valid=0`。
* **財務**は開示日以外は NaN（補完しない）。
* **全ローリング**は `over('Code')`（銘柄内ソート）・`min_periods` 厳守。

---

## 15) 実装チェック（落ちやすいポイント）

1. **Flow カバレッジ 0%** → `Section` が `/listed` as‑of でなく **コード範囲擬似**になっていないか／区間展開が**営業日**か／`(Section,Date)` の **型一致**。
2. **TOPIX Z が 100% NaN** → warmup 252 営業日の取得不足（取れない期間は `*_z` は仕様上 NaN）。
3. **MACD/β/集約の NaN 多数** → `min_periods` の想定通り（**有効フラグ**で弾くのが正）。
4. **`activity_ratio` NaN** → 定義が「Section比」か「市場全体比」かの選択ズレに注意。推奨は `flow_activity_ratio = TotalTotal / Σ(全セクション TotalTotal)`（同日に計算）。

---

### 付録：代表的な Polars 断片（再掲・命名確定版）

```python
# --- returns / vol ---
df = df.sort(['Code','Date']).with_columns([
    (pl.col('Close') / pl.col('Close').shift(1) - 1).over('Code').alias('returns_1d'),
    pl.col('returns_1d').over('Code')
      .rolling_std(window_size=20, min_periods=21) * (252.0 ** 0.5)
      .alias('volatility_20d'),
])

# --- TOPIX join (same-day) ---
df = df.join(mkt_df, on='Date', how='left')  # mkt_df で mkt_* を計算済

# --- beta/alpha (実装は明示ローリング共分散) ---
# beta_60d = cov(ret, mkt, 60) / var(mkt, 60)
# alpha_1d = ret - beta_60d * mkt

# --- trades_spec expand & join ---
# flow_daily(Section, Date, flow_*) を作成済みとして:
df = df.join(flow_daily, left_on=['Section','Date'], right_on=['section','Date'], how='left')

# --- statements as-of (backward) ---
df = df.sort(['Code','Date']).join_asof(
    stm.sort(['Code','effective_date']),
    left_on='Date', right_on='effective_date', by='Code', strategy='backward'
)
```

---

これが**フル版の列カタログ**です。
この通りに揃えれば、これまでログに出ていた **Flow 0%** や **TOPIX Z 100% NaN** は「仕様上あり得る/設定不足」を切り分けて扱えます（有効フラグで制御）。
もし「この列名が実ファイルにない/別名だ」という箇所があれば **alias マップ**を一枚作って吸収しましょう（特に `flow_*` と旧名）。

---

## 仕様追記（2025-09-19 更新）

### 2025-09-19 更新内容
- セクション12（列総数）を実装実態に合わせて更新（225列→395列）
- フロー特徴を37列に拡張（flow_* プレフィックス統一）
- マージン特徴を週次（margin_*）45列と日次（dmi_*）41列に分離
- 財務特徴を20列に拡張
- その他の特徴146列（earnings events, short selling, options, cross-features等）を明記

## 仕様追記（2025-09-07）

最新データセットで運用・検証して明確化されたポイントを追記します（既存仕様の明文化であり、モデル性能に影響しない変更）。

### 1) セクター・ターゲットエンコーディング（TE）
- 生成列: `te33_sec_target_5d`（エイリアス: `te_sec_target_5d`）
- ベース: 33 セクター（`sector33_id`）
- リーク防止: `t-1` 遅れでの統計（当日情報は不使用）
- 学習安定化: k-fold クロスフィット排他 + ベイズ平滑（`m=100`）
  - セクター平均（除外 fold）= μ_sec_excl(t)
  - グローバル平均（除外 fold）= μ_all_excl(t)
  - TE = (n_sec·μ_sec_excl + m·μ_all_excl) / (n_sec + m)
- 欠損規約: `sector33_id` 未付与行は TE を `null` とする（セクター不明のため）

注: きわめて初期の期間など、fold 排他の窓が成立しない日は `t-1` までの統計のみで μ を構成し、ベイズ平滑により安定化する。

### 2) セクター属性付与のルール（`/listed/info`）
- キー: `LocalCode`/`Code` を UTF-8 文字列として扱う（本データでは同一値）
- 期間結合: as-of backward（`valid_from ≤ Date ≤ valid_to`）。将来日スナップショットのみが存在する場合は、`valid_from` をデータセット最小日付まで引き下げ、全期間に適用可能とする（スナップショット型 `listed_info` を許容）
- 優先規則: 同名列が両辺に存在する場合、`listed_info`（右辺）を優先
  - 例: `Sector33Code` → `sector33_code`/`sector33`、`MarketCode` を右辺優先で採用
- `MarketCode`→`Section` 変換を標準化（JASDAQ 等は互換ルールで統合）
- 期待カバレッジ: `sector33_id ≥ 0` は概ね 95〜97% 程度（上場直後・市場移行直後などは `null` になり得る）

### 3) 命名の正規化（レガシー → 正式名）
- 有効フラグ: `is_ema_5_valid` → `is_ema5_valid`、`is_ema_10_valid` → `is_ema10_valid`、`is_rsi_2_valid` → `is_rsi2_valid`
- ボリンジャー: `bb_bandwidth` → `bb_width`、`bb_pct_b` → `bb_position`
- 旧名は原則出力しない。互換が必要な場合はビュー/別名で吸収する

### 4) フロー特徴の表記について
- 文書中の**プレフィクスの説明**は「`flow_*` グループ」として扱う（ワイルドカードは列集合の説明。個々の列名は代表列の定義を参照）
- 代表列: `flow_foreign_net_ratio`, `flow_individual_net_ratio`, `flow_activity_ratio`, `flow_smart_idx`, `flow_smart_mom4`, `flow_shock_flag`, `flow_impulse`, `flow_days_since` など（合計 ≥ 17）

### 5) 拡張（任意列）
- セクター日次頻度: `sec33_daily_freq`
  - 含意: 同一 `Date` における 33 セクター内の相対頻度（その日のスナップショット）
  - 算出: `count(Date×sector33_id) / count(Date)` を window 集計で算出（join 不要）
  - 位置づけ: 任意（必要に応じて生成）。本仕様の必須列ではない


---

### 追加

了解です。いただいた **v1.1 フル仕様** は土台が十分に厚いので、ここからは **「掛け合わせ＝相互作用・ゲーティング・残差化」** を狙って、少数精鋭で効く組み合わせを追加します。
以下は **(A) 高優先度（まず入れる） → (B) 中優先度 → (C) 実装パターン（Polars断片）** の順です。すべて **t 時点の情報のみ** を使い、`is_*_valid` を尊重して **Null伝播/マスク** を行います。

---

## A. 高優先度（まず入れる：汎化とSharpe底上げに効く）

> **設計原則**
>
> 1. 連続値どうしは **Z もしくはスケール整合**（÷volなど）してから積。
> 2. **レジーム/フラグ × 連続値** は「フラグを float 化して乗算」＝ゲーティング。
> 3. 相互作用は **“市場→セクター→個別” の三階層** を意識（ノイズ減らし）。

### 1) 市場×個別トレンド整合（Trend Alignment Intensity）

* **狙い**：市場トレンドと個別トレンドが同方向のときだけ、個別トレンドを強調。
* **式**：`trend_intensity = ma_gap_5_20 * mkt_gap_5_20`
* **ゲート版**：`trend_intensity_g = ma_gap_5_20 * (mkt_trend_up.cast(Float64) * 2 - 1)`
  （`mkt_trend_up∈{0,1}` を ±1 に射影して乗算）
* **効果想定**：継続相場（10–20日）で IC 上乗せ、弱地合いでは過信抑制。

### 2) セクター相対モメンタムの強気/弱気整合

* **式**：`rel_to_sec_5d * sec_mom_20`
  さらに `z_in_sec_ma_gap_5_20 * sec_mom_20` も有効。
* **効果**：**“強いセクターで相対的にも強い銘柄”** を抽出（5–10日リスキー逆風を回避）。

### 3) リスク調整モメンタム（局所 Sharpe）

* **式**：`mom_sh_5 = returns_5d / (volatility_20d + ε)`、`mom_sh_10`
* **mkt 中立**：`(returns_5d - beta_60d*mkt_ret_5d) / (volatility_20d + ε)`
* **効果**：**ボラに見合った上昇**を拾い、ノイズの高い小型株で特に有効。

### 4) 出来高ショック × 価格方向（フロー実体の確認）

* **式**：`rvol_5 = Volume/volume_ma_5`（既存）
  相互作用：`rvol_5 * returns_1d.sign()`、`rvol_5 * bb_pct_b`
* **効果**：**出来高伴う上昇/下落**だけを強調（フェイク・ブレイク排除）。

### 5) スクイーズ検知（ショート×上昇）

* **式**：`squeeze_pressure = dmi_short_to_adv20 * rel_strength_5d.clip_min(0)`
  代替：`dmi_short_to_adv20 * (returns_1d.clip_min(0))`
* **効果**：**ショート高水準×上昇転換**で踏み上げ捕捉（1–5日寄与）。

### 6) 信用フローの偏り × 反転/継続の切替

* **式**：`rev_bias = (dmi_credit_ratio - 1).zscore_26 * (-z_close_20).clip_min(0)`
  （**信用買い偏重**が高いほど **過熱×逆張り**を強く出す）
* **効果**：短期リバーサルの精度向上（特に高ボラ日）。

### 7) PEADゲーティング（決算ドリフトの時間減衰）

* **式**：`pead_score = (stmt_rev_fore_op + stmt_progress_op).fill_null(0)`（出力時は `stmt_progress_*`/`stmt_rev_fore_*` を ±100 にクリップして外れ値を抑制）
  `pead_effect = pead_score * exp(- stmt_days_since_statement / τ)`（τ≈5営業日）
* **効果**：**決算サプライズ→ドリフト**の典型を連続値で表現（5–10日）。

### 8) マーケット・レジームゲート × リバーサル/ブレイク

* **式**：`rev_gate = (mkt_high_vol.cast(Float64)) * (-z_close_20).clip_min(0)`
  `bo_gate = ((~mkt_high_vol).cast(Float64)) * donchian_break_20`
* **効果**：**高ボラ＝逆張り優位／低ボラ＝ブレイク優位** を自動切替。

### 9) αの平均回帰 × β安定度

* **式**：`alpha_meanrev = (-alpha_1d) * beta_stability_60d`
* **効果**：βが安定している銘柄ほど **αは翌日戻りやすい** 傾向を利用（1–3日）。

### 10) フロー（週次）× 個別相対強弱

* **式**：`flow_smart_idx * rel_strength_5d`、`flow_foreign_net_z * rel_to_sec_5d`
* **効果**：**主体別フローが支える相対強さ**を抽出（5–10日）。

---

## B. 中優先度（余力があれば：ノイズを削る・特殊局面で効く）

### 11) 三階層の合成（市場×セクター×個別）

* **式**：`tri_align = (mkt_gap_5_20>0).int8() * (sec_mom_20>0).int8() * (ma_gap_5_20)`
* **効果**：**追い風×追い風×追い風** のときのみ個別トレンド採用（偽陽性低減）。

### 12) 乖離の符号別相互作用（Hinge）

* **式**：`pos_b = (bb_pct_b - 1).clip_min(0)`、`neg_b = (0 - bb_pct_b).clip_min(0)`
  `pos_b * rvol_5`、`neg_b * rvol_5`
* **効果**：**上限側/下限側**で反応を分離（非線形性の直交化）。

### 13) 流動性ショック × 短期モメンタム

* **式**：`liquidity_shock = (turnover_rate / turnover_rate.shift(1) - 1).clip(−p, p)`
  `liquidity_shock * returns_5d`
* **効果**：出来高急変を伴う継続/反転の識別。

### 14) 相関変化感度（対TOPIX）

* **式**：`corr_shift = (beta_60d - beta_20d)`（なければβ差近似）
  `corr_break = corr_shift * returns_5d`
* **効果**：**共動崩れ**をとらえてリスクイベント検知。

### 15) 決算×市場地合いの符号整合

* **式**：`pead_effect * (mkt_trend_up.cast(Float64)*2-1)`
* **効果**：良決算でも弱地合いは伸びにくい／逆も然りを織り込む。

### 16) dmi\_impulse × 短期方向

* **式**：`dmi_impulse.cast(Float64) * returns_1d`、`dmi_days_since_pub` を指数減衰で重み付け
* **効果**：**発表直後だけ** 影響が強い日次信用のクセを抽出。

### 17) ブレッドス × 個別

* **式**：`flow_breadth_pos * rel_strength_5d`
* **効果**：**広範な買い越し日**に相対強い銘柄の追随期待（5–10日）。

---

## C. 実装パターン（Polars 最小断片）

> **方針**：
>
> * 相互作用は **Null を尊重**（どちらかが NaN なら NaN）。
> * ただし **“ゲート（0/1）×連続値”** は `fill_null(0.0)` で 0 に倒すと扱いやすい。
> * 過度な次元爆発を避けるため **上記 10〜15 本に限定**。

```python
import polars as pl

def hinge_pos(col): return pl.col(col).clip_min(0.0)
def hinge_neg(col): return (-pl.col(col)).clip_min(0.0)

df = df.with_columns([
    # 1) 市場×個別トレンド
    (pl.col("ma_gap_5_20") * pl.col("mkt_gap_5_20")).alias("x_trend_intensity"),
    (pl.col("ma_gap_5_20") * (pl.col("mkt_trend_up").cast(pl.Float64)*2-1)).alias("x_trend_intensity_g"),

    # 2) セクター相対×セクターモメンタム
    (pl.col("rel_to_sec_5d") * pl.col("sec_mom_20")).alias("x_rel_sec_mom"),

    # 3) 局所Sharpe
    (pl.col("returns_5d") / (pl.col("volatility_20d")+1e-12)).alias("x_mom_sh_5"),
    ((pl.col("returns_5d") - pl.col("beta_60d")*pl.col("mkt_ret_5d")) /
        (pl.col("volatility_20d")+1e-12)).alias("x_mom_sh_5_mktneu"),

    # 4) 出来高ショック×方向
    (pl.col("volume_ratio_5") * pl.col("returns_1d").sign()).alias("x_rvol5_dir"),
    (pl.col("volume_ratio_5") * pl.col("bb_pct_b")).alias("x_rvol5_bb"),

    # 5) スクイーズ
    (pl.col("dmi_short_to_adv20") * hinge_pos("rel_strength_5d")).alias("x_squeeze_pressure"),

    # 6) 信用過熱×逆張り
    ((pl.col("dmi_credit_ratio") - 1.0).rolling_mean(26).over("Code")
        .fill_null(0) * hinge_neg("z_close_20")).alias("x_credit_rev_bias"),

    # 7) PEAD 減衰
    ((pl.col("stmt_rev_fore_op").fill_null(0) + pl.col("stmt_progress_op").fill_null(0)) *
     ( (-pl.col("stmt_days_since_statement")/5.0).exp() )).alias("x_pead_effect"),

    # 8) レジームゲート
    (pl.col("mkt_high_vol").cast(pl.Float64) * hinge_neg("z_close_20")).alias("x_rev_gate"),
    ((1.0 - pl.col("mkt_high_vol").cast(pl.Float64)) *
     pl.col("ma_gap_5_20").gt(0).cast(pl.Float64)).alias("x_bo_gate"),

    # 9) α平均回帰×β安定
    ((-pl.col("alpha_1d")) * pl.col("beta_stability_60d")).alias("x_alpha_meanrev_stable"),

    # 10) 週次フロー×相対強弱（Nullは落ちる）
    (pl.col("flow_smart_idx") * pl.col("rel_strength_5d")).alias("x_flow_smart_rel"),
    (pl.col("flow_foreign_net_z") * pl.col("rel_to_sec_5d")).alias("x_foreign_relsec"),

    # 11) 三階層の合成（ブール×ブール×連続）
    (pl.col("mkt_gap_5_20").gt(0).cast(pl.Float64) *
     pl.col("sec_mom_20").gt(0).cast(pl.Float64) *
     pl.col("ma_gap_5_20")).alias("x_tri_align"),

    # 12) 乖離の符号別×出来高
    (hinge_pos("bb_pct_b") * pl.col("volume_ratio_5")).alias("x_bbpos_rvol5"),
    (hinge_neg("bb_pct_b") * pl.col("volume_ratio_5")).alias("x_bbneg_rvol5"),

    # 13) 流動性ショック×モメンタム
    (((pl.col("turnover_rate")/(pl.col("turnover_rate").shift(1)+1e-12)) - 1.0)
        .clip(-0.5, 0.5) * pl.col("returns_5d")).alias("x_liquidityshock_mom"),

    # 15) 決算×地合い
    (pl.col("x_pead_effect") * (pl.col("mkt_trend_up").cast(pl.Float64)*2-1)).alias("x_pead_times_mkt"),

    # 16) dmi インパルス×方向（インパルス Null は0に）
    (pl.col("dmi_impulse").cast(pl.Float64).fill_null(0.0) * pl.col("returns_1d")).alias("x_dmi_impulse_dir"),

    # 17) ブレッドス×個別
    (pl.col("flow_breadth_pos") * pl.col("rel_strength_5d")).alias("x_breadth_rel"),
])
```

> **メモ**
>
> * `volume_ratio_5` は `Volume/MA5(Volume)` 相当（仕様に合わせて置換）。
> * `donchian_break_20` が未定義なら `Close > rolling_max_20(Close)` を bool→float 化で代替。
> * 項目が未算出の列は、まず既存パイプで生成・正規名に合わせてください。

---

## どう効くか（期待・検証の着眼点）

* **短期（1–3日）**：`x_rvol5_dir`、`x_rev_gate`、`x_alpha_meanrev_stable`、`x_squeeze_pressure`。
* **中期（5–10日）**：`x_trend_intensity(_g)`、`x_rel_sec_mom`、`x_mom_sh_5(_mktneu)`、`x_pead_effect`。
* **イベント横断**：`x_pead_times_mkt`、`x_dmi_impulse_dir`。
* **ロバスト化**：`x_tri_align`（偽陽性を下げ Sharpe 底上げ）。

**Ablation 推奨**：Base → +A（高優先度10本）→ +B（中優先度）で **Purged KFold+Embargo** 評価。
**期待**：RankIC +0.01〜+0.03（短期側）、Sharpe +0.1〜+0.3 の底上げ（データ期間・ユニバース次第）。

---

## 追加の運用ヒント

* **学習時標準化**：相互作用列も **fold内 fit → transform**（`x_cs_z`等）を徹底。
* **過学習抑制**：似通う相互作用は **L2/Dropout** か **学習率減衰＋早停** で制御。
* **特徴選択**：Permutation / SHAP で寄与小を整理（ただしゲート類は残すと安定）。
* **GAT併用**：`x_rel_sec_mom` や `x_flow_smart_rel` を **エッジ重み候補**（同業内）にも使える。

---

このセットは、**価格×需給×イベント×レジーム**をそれぞれ単独で足すのではなく、**“条件が揃ったときだけ強調する”** ための掛け合わせです。まずは **高優先度10本** を入れて再学習→効果を見てから中優先度を段階追加、の順で進めるのが最短で効果的です。

