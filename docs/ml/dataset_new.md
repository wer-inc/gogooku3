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

## 7) 週次フロー（/markets/trades\_spec）特徴（**プレフィックス `flow_` に統一**、\~18列）

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
* **ブレッドス**

  * `flow_breadth_pos = mean( [ForeignersBalance>0, IndividualsBalance>0, TrustBanksBalance>0, InvestmentTrustsBalance>0, ProprietaryBalance>0, BrokerageBalance>0] )`
* **標準化（週次52）**

  * `flow_foreign_net_z = (ForeignersBalance − MA_52)/STD_52`
  * `flow_individual_net_z`, `flow_activity_z`
* **スマートマネー**

  * `flow_smart_idx = flow_foreign_net_z − flow_individual_net_z`
  * `flow_smart_mom4 = flow_smart_idx − MA_4(flow_smart_idx)`
  * `flow_shock_flag = (abs(flow_smart_idx) ≥ 2).int8()`
* **タイミング**

  * `flow_impulse = (Date == effective_start).int8()`
  * `flow_days_since = (Date − effective_start).days()`
    **背景**：主体別需給の強弱と異常フロー検知（短期の方向・続伸/反落に寄与）。

> **互換性エイリアス**：旧名（`foreigners_net_ratio`等）を使うコードがある場合は、`flow_*` を**正本**にし、旧名を **ビュー/別名** として残すのが安全。

---

## 8) 財務（/fins/statements）特徴（\~17列＋タイミング）

### 8.1 有効日

* `effective_date = (DisclosedTime < 15:00 ? DisclosedDate : next_business_day(DisclosedDate))`
* as‑of backward で `(Code, Date)` に**直近の有効開示**を付与。
  **背景**：PEADや開示吸収のタイミングをリークなしで取り込む。

### 8.2 指標

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

## 9) 有効フラグ（マスク）（10–14 列）

* **窓成熟**：`is_rsi2_valid (row_idx≥5)`, `is_ema5_valid (≥15)`, `is_ema10_valid (≥30)`, `is_ema20_valid (≥60)`, `is_ema200_valid (≥200)`
* **市場Z**：`is_mkt_z_valid (warmup≥252)`
* **β/α**：`is_beta_valid (≥60)`
* **セクター集約**：`is_sec_valid (member_cnt≥3)`
* **フロー**：`is_flow_valid (Section一致 & 日次展開成功)`
* **財務**：`is_stmt_valid (開示取得済)`
* **総合**：`is_valid_ma (row_idx≥60)` 等
  **背景**：欠損を埋めず、**使えるときだけ使う**意思をモデルに伝える。

---

## 10) 目的変数（ターゲット）

* **回帰**：`target_{1,2,3,5,10} = Close.shift(-h)/Close − 1`
* **分類**：`target_{1,5,10}_binary = (target_h>0).int8()`
  **背景**：業務要件（短期 1–3 日重視）に合わせてマルチホライズン。

---

## 11) 正規化（任意・学習時に fold 内で）

* **クロスセクション Z**（その日ごと）：`x_cs_z = (x − mean_by(Date))/(std_by(Date)+ε)`
* **グループ内 Z**（セクター内）：`z_in_sec_*`（上記 6.3）
  **背景**：地合い除去・相対化で汎化を高める（**学習データ内で fit**、推論は transform のみ）。

---

## 12) 列総数の目安（重複除去・一部任意含む）

* 識別子・メタ：**\~12**
* 価格/出来高：**\~65**
* テクニカル：**\~18**
* 市場（TOPIX）：**\~26 + Z系(\~4) = \~30**
* クロス：**\~7**
* セクター基礎/集約/相対/One‑Hot：**基礎6 + 集約~~8 + 相対~~5 + One‑Hot(17) + 頻度2 ≈ \~38**
* 週次フロー（flow\_\*）：**\~18**
* 財務（stmt\_\*）：**\~17**
* 有効フラグ：**\~12**
* ターゲット：**\~8**

**合計目安：** **\~225 列**（選択制の One‑Hot/頻度/Z を外すと \~170–185 列）。
※ 実際の総数は設定フラグ（One‑Hot有無、Z有無、freq有無）で前後します。

---

## 13) 互換性・命名の最終確認（漏れ対策）

* **flow 列は `flow_` 接頭辞で統一**（旧名は alias として残す）：
  `foreigners_net_ratio → flow_foreign_net_ratio` 等。
* **TOPIX Z 系**は warmup が無い期間は**NaN で正常**（`is_mkt_z_valid` で管理）。
* **セクター集約**は `sector33` をデフォルト、`member_cnt<5` は `sec_small_flag=1`。
* **β/α** は `min_periods=60`、不足は NaN＋`is_beta_valid=0`。
* **財務**は開示日以外は NaN（補完しない）。
* **全ローリング**は `over('Code')`（銘柄内ソート）・`min_periods` 厳守。

---

## 14) 実装チェック（落ちやすいポイント）

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
