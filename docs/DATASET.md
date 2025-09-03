ありがとうございます。いただいた「現状の実装状況分析」を前提に、**いま何ができていて、何がボトルネックで、どの順にどう直すと“ベスト”に届くか**を、\*\*差分（具体的変更点）\*\*ベースで整理しました。結論から言うと——

> **現状は土台（営業日スケジューリング／市場フィルタ／ページネーション／キャッシュ）は良好。主な改善余地は “データ取得の軸” と “差分検知の仕組み” に集中**しています。

---

## ステータス要約（あなたの整理に対する所見）

### うまくできている点（維持）

* **営業日ドリブン運転**：◎（TradingCalendarFetcher の扱い・HolidayDivision もOK）
* **Market Code フィルタ**：◎（PRO/その他除外／ホワイトリスト方針も妥当）
* **ページネーション**：◎（`pagination_key` 実装済）
* **キャッシュ**：○（Listed 24h は後述の運用変更に合わせてTTLを調整）

### ギャップ（優先順位つき）

1. **/prices/daily\_quotes の“軸”**
   　現在は **`date` 一択** → **市場フィルタ運用**では\*\*`code+from/to`\*\*のほうが無駄が少ない場面が多い。
2. **/listed/info の取得タイミング**
   　いまは「最初・中・最後」の **粗いサンプリング** → \*\*月初＋“差分検知日”\*\*へ。
3. **/fins/statements の“軸”**
   　いまは **`code` バッチ** → \*\*`date`（営業日ループ）\*\*に切替えると空振りが減る。
4. **差分検知の欠如**
   　上場/廃止/市場変更/社名変更を **日単位**で確定できていない。
5. **自動軸選択の欠如**
   　実測のない「固定 `date`」は、データ量や市場選択で**非最適**になりやすい。

---

## いちばん良い解（現状からの**最短**アップグレード設計）

> **“営業日で回す”外側の枠組みは維持**しつつ、**エンドポイントごとに最適軸を使い分け**、**差分検知で /listed を必要日だけに絞る**。
> バックフィルでは **二分探索（sparse search）** を使い、**/listed の日次連打を回避**します。

### A. daily\_quotes（株価）——**軸の自動選択＋市場フィルタ最適化**

* **実装**：

  * 起動時に **小規模試行**で `date` 軸と `code` 軸の**ページ数／転送量**を**実測**し、**その日のジョブ方針を自動決定**（例：代表3営業日 × 代表50銘柄）。
  * **市場を絞って収集する日**は基本 **`code+from/to`** を優先（その銘柄の在籍期間だけ取り切る）
  * \*\*全市場が必要な日（例：全量検証・棚卸し）\*\*は **`date=YYYY-MM-DD`** で一括取得
* **保存**：

  * どちらの軸でも **同じ upsert**（PK: `(date, local_code)`）に集約
  * **5桁 LocalCode** に内部統一（4桁 Code は別カラムで保持）

### B. listed\_info（銘柄マスター）——**月初＋差分日**に限定

* **平時（運用）**：

  * **毎月の最初の営業日**に `/listed/info?date=...` で**スナップショット**保存
  * **日次ジョブでは** `daily_quotes の Code 集合` の**増減が出た日だけ** `/listed/info?date=当日` を追加取得
* **バックフィル（過去5年）**：

  * 各月の**月初営業日スナップショット**をまず全期間ぶん取得
  * **イベント（新規上場/廃止/市場変更）が疑われる月**について、**その月の期間 \[前月初, 当月初) に二分探索で最小回数だけ /listed を叩く**

    * 例）前月には存在せず当月に存在 → **上場日がその区間にある**ので 1〜⌈log₂ 区間日数⌉ 回の /listed で**正確な発生日**を特定
    * 廃止・市場変更・社名変更も同様
  * **日単位でのイベント日**が**少ないAPIコール**で復元できます

### C. fins/statements（財務）——**営業日（date）で回収**

* **営業日ループで `?date=YYYY-MM-DD`** を回すと、**開示のあった日だけ**返るため**空振りがほぼない**
* 初回の埋め漏れが見つかった銘柄だけ **`?code=` で補完**
* **IFRS の空欄（OrdinaryProfit 等）**はスキーマで**NULL許容**、型は DECIMAL 推奨

### D. trades\_spec（投資部門別）——**セクション×期間一括＋ローリング再取得**

* **セクションごと**に `from/to` 一括取得（過誤訂正は **PublishedDate** 違いで併存）
* 運用では **直近2〜3週間**を**毎日取り直すローリング窓**で訂正を吸収
* PK: `(section, start_date, end_date, published_date)`

### E. 差分検知とイベント化（上場/廃止/市場・社名変更）

* **コード集合の差分**（`daily_quotes?date` の集合 or **B の二分探索で確定した境界日**）でイベント生成
* `securities_events`（flat）

  * `local_code, event_type, effective_from, effective_to(null), details_text`
  * ルール例：

    * “前日まで存在→当日消失” → **最終在籍日=前営業日**, **廃止日=翌営業日**
    * 市場/社名属性が変わった → **effective\_from=当日**

---

## 変更差分（そのまま実装に落とせるレベル）

### 1) 軸自動選択（prices）

**新規**：`AxisDecider.measure_api_efficiency()`

```ts
type AxisChoice = 'by_date' | 'by_code';

async function measure_api_efficiency(sampleDays: Date[], sampleCodes: string[]): Promise<AxisChoice> {
  const datePages = await countPagesByDate(sampleDays);   // /prices?date=...
  const codePages = await countPagesByCode(sampleCodes);  // /prices?code=...&from&to
  // しきい値は経験則＋安全率（例：datePages <= 0.9 * codePages なら by_date）
  return (datePages <= 0.9 * codePages) ? 'by_date' : 'by_code';
}
```

* **導入箇所**：デイリーバッチ開始時／バックフィル開始時に1回計測 → ジョブ実行方針を決定
* **メトリクス**：`calls_count`, `pages_count`, `bytes_total` をロギング

### 2) prices: 銘柄軸フェッチャ

**新規**：`DailyQuotesByCodeFetcher.fetch(code, from, to)`

* **入力**：市場フィルタで作った **在籍期間レンジ**（後述）
* **実装**：`GET /prices/daily_quotes?code=...&from=...&to=...` をページング追従、`(date, local_code)` で upsert

### 3) listed: “月初＋差分日” & **二分探索**

**新規**：`ListedInfoManager`

```ts
async function snapshotAt(monthStart: Date): Promise<ListedRow[]> { /* /listed?date=... */ }
function detectChanges(prevSet: Set<string>, currSet: Set<string>): Change[] { /* add/del */ }
async function pinpointChangeDate(range: [Date, Date], predicate: (d:Date)=>Promise<boolean>): Promise<Date> {
  // /listed?date=mid を叩き、存在有無で半区間に絞る二分探索
}
```

* **バックフィル**：

  1. 5年分の**月初スナップショット**を取り切る
  2. 月をまたいだ **出現/消滅/属性差** を検出
  3. それぞれの区間で **二分探索**して **正確なイベント日**を確定
* **運用**：

  * 日次で prices の **Code集合差分**が出た日だけ `/listed?date=当日` を取得

### 4) statements: `date` 軸へ切替

* 既存 `fetch_statements(code)` を **`fetch_statements_by_date(date)`** に置換
* **並行で** `reconcile_missing_by_code(code)` を用意（欠損補完用）
* **テストの暫定 100銘柄制限**は削除

### 5) イベントテーブル（flat）＆在籍期間生成

* **新規テーブル**：`securities_events`（上で定義）
* **在籍期間**を作る補助テーブル：`market_membership`

  * `local_code, market_code, from_date, to_date(null)`（PK: `local_code, market_code, from_date`）
  * **prices の銘柄軸取得**の入力レンジとして再利用

---

## スケジューリング（デイリー & バックフィル）

### デイリー（擬似フロー）

```text
1) trading_day = 今日の営業日判定
2) axis = AxisDecider.measure_api_efficiency_if_needed()
3) dq_date = GET /prices?date=trading_day   # コード集合抽出 + 全量保存(必要なら)
4) if code_set_changed(dq_date vs yesterday):
     listed_today = GET /listed?date=trading_day
     update market_membership + emit securities_events
5) if axis == 'by_code' and 市場フィルタあり:
     for (code, ranges in market_membership):
         GET /prices?code=code&from=range.start&to=range.end
   else:
     # すでに step3 で全量保存済
6) GET /fins/statements?date=trading_day
7) GET /markets/trades_spec (rolling 2–3 weeks)
```

### バックフィル（過去5年）

```text
A) 月初 /listed を全期間取得 → 月次差分を抽出
B) 差分対象だけ /listed を二分探索で当日特定 → events + membership を構築
C) prices: 市場フィルタ保有なら by_code でレンジ取得、なければ by_date 一括
D) statements: 営業日ループで date 軸、漏れを code で補完
E) trades_spec: セクション×期間一括 ＋ verification ローリング
```

---

## 最低限のスキーマ（PostgreSQL; *flat*）

```sql
-- 株価（日次）
create table daily_quotes (
  date date not null,
  local_code varchar(5) not null,
  open numeric, high numeric, low numeric, close numeric,
  volume bigint, turnover_value numeric,
  adj_open numeric, adj_high numeric, adj_low numeric, adj_close numeric,
  adjustment_factor numeric,
  primary key (date, local_code)
) partition by range (date);

-- 銘柄スナップショット
create table securities_master (
  as_of_date date not null,
  local_code varchar(5) not null,
  code_4 varchar(4),
  company_name text,
  market_code varchar(4),
  market_name text,
  sector17_code varchar(4), sector17_name text,
  sector33_code varchar(4), sector33_name text,
  scale_category text,
  primary key (as_of_date, local_code)
);

-- 在籍期間（市場別）
create table market_membership (
  local_code varchar(5) not null,
  market_code varchar(4) not null,
  from_date date not null,
  to_date date,
  primary key (local_code, market_code, from_date)
);

-- イベント（上場/廃止/変更/改称）
create table securities_events (
  local_code varchar(5) not null,
  event_type text not null,               -- 'new_listing' | 'delisting' | 'market_change' | 'name_change'
  effective_from date not null,
  effective_to date,
  details_text text,
  primary key (local_code, event_type, effective_from)
);

-- 財務
create table fins_statements (
  disclosed_date date not null,
  disclosed_time time,
  local_code varchar(5) not null,
  disclosure_number text not null,
  type_of_document text,
  net_sales numeric, operating_profit numeric, ordinary_profit numeric, profit numeric, eps numeric,
  total_assets numeric, equity numeric,
  primary key (local_code, disclosure_number)
);

-- 投資部門別
create table trades_spec (
  section text not null,
  start_date date not null,
  end_date date not null,
  published_date date not null,
  -- ... numbers ...
  primary key (section, start_date, end_date, published_date)
);
```

---

## キャッシュとレートの運用ポイント

* **Trading Calendar**：週1の再取得は妥当（年次切替期のみ前倒し更新）
* **Listed Info**：

  * **月初スナップショット**はキャッシュ長めでもOK
  * **差分日**の取得は当日中のみ有効なので、**短TTL or no-cache**で
* **prices by code**：銘柄数に応じて**並列度を上限管理**（指数バックオフ＋再試行）
* **メトリクス**：`calls/pages/bytes/err_rate` を**軸別**・**エンドポイント別**に記録 → 次回の自動軸選択に活用

---

## 受け入れ基準（Doneの定義）

* **API コール数**：同一期間・同一市場集合に対し、現状比で**減少**（ダッシュボードで可視化）
* **イベント精度**：上場/廃止/市場変更/社名変更の **effective\_from が日単位で再現**
* **データ整合**：`low ≤ min(open, close) ≤ high`、出来高非負、2020-10-01 の Null を正しく温存
* **再実行性**：全投入パイプラインが **主キー upsert** で冪等

---

## すぐ着手できるタスク（PR 粒度のチェックリスト）

**Phase 1（高優先）**

* [ ] `AxisDecider.measure_api_efficiency()` を追加、ランブックに反映
* [ ] `DailyQuotesByCodeFetcher` を実装（在籍レンジ入力で取得）
* [ ] `fetch_statements_by_date()` を実装し、既存 `code` 方式を置換（欠損補完APIは退避）

**Phase 2（中優先）**

* [ ] `ListedInfoManager`：月初スナップショット保存
* [ ] `detectChanges()` と **差分日だけ** `/listed` を取得するロジック
* [ ] **二分探索**でイベント日を特定 → `securities_events` 生成
* [ ] `market_membership` を出力（prices by code のレンジに供給）

**Phase 3（低優先）**

* [ ] 並列度とメモリの動的制御（メトリクス駆動）
* [ ] ダッシュボード：軸別 calls/pages/bytes, 処理時間, 失敗率

---

### まとめ

* 現状の**基盤はOK**。
* 仕上げるべきは **(1) prices の軸を“測って選ぶ”**、**(2) listed を“月初＋差分日＋二分探索”**、**(3) statements を“date軸”**、**(4) 差分検知でイベント化**。
* これで **APIコールと転送量を顕著に削減**しつつ、**上場廃止・市場変更を日単位で正確に再現**できます。

必要なら、この方針に合わせた **具体的なコード雛形（TypeScript/Node または Python）** や **既存クラス（TradingCalendarFetcher / MarketCodeFilter 等）への差分パッチ**まで一気に出します。




了解です。最新の仕様・実装方針に合わせて、**dataset に含める全カラムの“最終版リスト”と計算方法**をカテゴリ別にまとめ直しました。
（命名は既存のログ／実装に合わせ、**単位・ウィンドウ・有効化条件**も併記。除算は全て `+1e-12` でゼロ割防止。）

> **前提（結合規則の要約）**
>
> * **prices/daily\_quotes** を軸（キー: `Code, Date`）。
> * **listed/info** → `Section` 付与 → **旧市場名→新市場名** 正規化: `section_norm ∈ {TSEPrime, TSEStandard, TSEGrowth}`。
> * **trades\_spec** は **T+1 営業日**から**次回公表前日**まで有効に区間化し、**(section\_norm, Date)** で日次展開→結合。
> * **indices/topix** は **Date** で結合。
> * **fins/statements** は **(Code, effective\_date)** を基準に \*\*as‑of backward（T+1/15:00 ルール）\*\*で結合。
> * **型統一**: すべての `Date` は `pl.Date`、テキストは `pl.Utf8`。

---

## 0) 識別子・メタ（6）

| Column               | 型       | 計算/生成                                                                          |
| -------------------- | ------- | ------------------------------------------------------------------------------ |
| `Code`               | Utf8    | J-Quants 銘柄コード（文字列固定）                                                          |
| `Date`               | Date    | 取引日（営業日カレンダー基準）                                                                |
| `Section`            | Utf8    | listed/info 由来の市場名（原文）                                                         |
| `section_norm`       | Utf8    | **旧→新** 正規化: `TSE1st→TSEPrime`, `TSEMothers→TSEGrowth`, `JASDAQ→TSEStandard` 等 |
| `row_idx`            | UInt32  | `Code` ごとの昇順行番号（ウォームアップ判定に使用）                                                  |
| `shares_outstanding` | Float64 | 可能なら別テーブルから（無ければ `null`）                                                       |

---

## 1) ベース（OHLCV/回転）+ 価格系特徴（約80）

> **キー**: `(Code, Date)`
> **Warm‑up**: 各ウィンドウ長（例: `returns_20d` は最低 20 営業日必要）

### 1.1 OHLCV（5）

* `Open`, `High`, `Low`, `Close`, `Volume` : **調整後**（Adjustment\* があればそれを採用）
* `TurnoverValue` : 可能なら採用（なければ `null`）

### 1.2 リターン系（10）

* `returns_1d = Close/Close[-1]-1`
* `returns_5d = Close/Close[-5]-1`
* `returns_10d`, `returns_20d`, `returns_60d`, `returns_120d`（同様）
* `log_returns_1d = ln(Close/Close[-1])`（5/10/20 も同様）

### 1.3 ボラティリティ（5）

* `volatility_5d = std(returns_1d, 5) * sqrt(252)`
* `volatility_10d`, `volatility_20d`, `volatility_60d`
* `realized_volatility ≈ sqrt( Σ( ln(High/Low) )^2 / 4 / ln(2) )`（日次近似）

### 1.4 移動平均・EMA（10）

* `sma_{5,10,20,60,120} = mean(Close, w)`
* `ema_{5,10,20,60,200} = ewm_mean(Close, span=w)`

### 1.5 価格位置/ギャップ（8）

* `price_to_sma5 = Close/sma_5`
* `price_to_sma20`, `price_to_sma60`
* `ma_gap_5_20 = (ema_5-ema_20)/(ema_20+1e-12)`
* `ma_gap_20_60 = (ema_20-ema_60)/(ema_60+1e-12)`
* `high_low_ratio = High/(Low+1e-12)`
* `close_to_high = (High-Close)/(High-Low+1e-12)`
* `close_to_low = (Close-Low)/(High-Low+1e-12)`

### 1.6 ボリューム/回転（6）

* `volume_ma_{5,20} = mean(Volume, w)`
* `volume_ratio_{5,20} = Volume/volume_ma_w`
* `turnover_rate = Volume/(shares_outstanding+1e-12)`（無ければ `null`）
* `dollar_volume = Close*Volume`

### 1.7 テクニカル（pandas‑ta）（\~15）

* RSI: `rsi_2`, `rsi_14`; `rsi_delta = rsi_14.diff()`
* MACD: `macd`, `macd_signal`, `macd_histogram`
* BB: `bb_upper, bb_lower, bb_middle`（内部）

  * `bb_width = (bb_upper-bb_lower)/(Close+1e-12)`
  * `bb_position = (Close-bb_lower)/(bb_upper-bb_lower+1e-12)`
* ATR/ADX/Stoch: `atr_14`, `adx_14`, `stoch_k`

> **Validity Flags（価格系）**
>
> * `is_rsi2_valid = (row_idx >= 5)`
> * `is_ema5_valid = (row_idx >= 15)`, `is_ema20_valid = (row_idx >= 60)`, `is_ema200_valid = (row_idx >= 200)`
> * `is_valid_ma = (row_idx >= 60)`

---

## 2) TOPIX 市場特徴（26）

> **キー**: `Date` で全銘柄に同じ値を付与。
> **Warm‑up**: `z`系と`ema_200`は長期窓（最長 252）。

### 2.1 リターン（4）

* `mkt_ret_1d = Close/Close[-1]-1`（TOPIX）
* `mkt_ret_{5,10,20} = Close/Close[-w]-1`

### 2.2 トレンド（4）

* `mkt_ema_{5,20,60,200} = ewm_mean(TOPIX_Close, span=w)`

### 2.3 偏差/ギャップ（3）

* `mkt_dev_20 = (Close-mkt_ema_20)/(mkt_ema_20+1e-12)`
* `mkt_gap_5_20 = (mkt_ema_5-mkt_ema_20)/(mkt_ema_20+1e-12)`
* `mkt_ema20_slope_3 = pct_change(mkt_ema_20, 3)`

### 2.4 ボラ・帯域（5）

* `mkt_vol_20d = std(mkt_ret_1d,20)*sqrt(252)`
* `mkt_atr_14`, `mkt_natr_14 = mkt_atr_14/(Close+1e-12)`
* `mkt_bb_pct_b = (Close-bb_lower)/(bb_upper-bb_lower+1e-12)`
* `mkt_bb_bw = (bb_upper-bb_lower)/(bb_middle+1e-12)`

### 2.5 リスク/フラグ（2）

* `mkt_dd_from_peak = (Close - cummax(Close))/cummax(Close)`
* `mkt_big_move_flag = (abs(mkt_ret_1d) >= 2*std(mkt_ret_1d,60)).int8()`

### 2.6 Z‑score（4）

* `mkt_ret_1d_z = z(mkt_ret_1d, 252)`
* `mkt_vol_20d_z = z(mkt_vol_20d, 252)`
* `mkt_bb_bw_z = z(mkt_bb_bw, 252)`
* `mkt_dd_from_peak_z = z(mkt_dd_from_peak, 252)`

> **注意**: 期間が短いと `z` 系が `NaN` になるので、**TOPIX はウォームアップ長め（>252日）で取得**し、そのまま日付結合する。

### 2.7 レジームフラグ（4）

* `mkt_bull_200 = (Close > mkt_ema_200).int8()`
* `mkt_trend_up = (mkt_gap_5_20 > 0).int8()`
* `mkt_high_vol = (mkt_vol_20d_z > 1.0).int8()`
* `mkt_squeeze = (mkt_bb_bw_z < -1.0).int8()`

---

## 3) クロス（銘柄×市場）特徴（8）

> **キー**: `(Code, Date)`（価格＋市場を使って銘柄ごとに算出）
> **Warm‑up**: 最長 60 日

* `beta_60d = Cov(returns_1d, mkt_ret_1d, 60) / Var(mkt_ret_1d, 60)`
* `alpha_1d = returns_1d - beta_60d*mkt_ret_1d`
* `alpha_5d = returns_5d - beta_60d*mkt_ret_5d`
* `rel_strength_5d = returns_5d - mkt_ret_5d`
* `trend_align_mkt = (sign(ma_gap_5_20) == sign(mkt_gap_5_20)).int8()`
* `alpha_vs_regime = alpha_1d * mkt_bull_200`
* `idio_vol_ratio = volatility_20d/(mkt_vol_20d+1e-12)`
* `beta_stability_60d = 1/(std(beta_60d, 20)+1e-12)`

---

## 4) フロー（投資部門別：/markets/trades\_spec）特徴（17）

> **作り方の要点**
>
> 1. **区間化**: `effective_start = next_business_day(PublishedDate)`、`effective_end = next_effective_start-1日`（同 `section_norm` 内で連鎖）。
> 2. **日次展開**: 営業日格子 `Date` と **cross join → 範囲 filter**。
> 3. **結合**: `(section_norm, Date)` で **left join**。
> 4. **命名統一**: 学習で使う列は **`flow_` プレフィクス** に揃える。

> **原始列（週次集計）**: `Foreigners{Sales,Purchases,Total,Balance}`, `Individuals*`, `Proprietary*`, `Brokerage*`, `TrustBanks*`, `InvestmentTrusts*`, `Total{Sales,Purchases,Total,Balance}`, …（J‑Quantsのキー名）

**日次フロー特徴**

* `flow_foreign_net_ratio = ForeignersBalance/(ForeignersTotal+1e-12)`

* `flow_individual_net_ratio = IndividualsBalance/(IndividualsTotal+1e-12)`

* `flow_activity_ratio = TotalTotal/(mean(TotalTotal, 52)+1e-12)`  ※年を跨いだ52週平滑（週次ベースのZに準ずる正規化）

* `foreign_share_activity = ForeignersTotal/(TotalTotal+1e-12)`（補助; モデルには `flow_` 接頭辞版も導出可）

* `breadth_pos = mean([ForeignersBalance>0, IndividualsBalance>0, TrustBanksBalance>0, InvestmentTrustsBalance>0, ProprietaryBalance>0, BrokerageBalance>0])`（0〜1）

**Z 系（52 週）**

* `flow_foreign_net_z = z(ForeignersBalance, win=52)`
* `flow_individual_net_z = z(IndividualsBalance, win=52)`
* `flow_activity_z = z(TotalTotal, win=52)`

**スマートマネー**

* `flow_smart_idx = flow_foreign_net_z - flow_individual_net_z`
* `flow_smart_mom4 = flow_smart_idx - mean(flow_smart_idx, 4)`
* `flow_shock_flag = (abs(flow_smart_idx) >= 2.0).int8()`

**タイミング**

* `flow_impulse = (Date == effective_start).int8()`
* `flow_days_since = (Date - effective_start).days()`

> **期待カバレッジ**: 営業日×3 セクションの多くに値が付く（70〜95%）。`section_norm` と `Date` の型が合っていることが必須。

---

## 5) 財務（/fins/statements）特徴（17）

> **as‑of/T+1 ルール**
>
> * `effective_date = DisclosedDate`（**15:00 以前**） or `next_business_day(DisclosedDate)`（**15:00 以降**）。
> * `(Code, Date)` に対し **backward as‑of** で最新開示をひとつ付ける。
> * 同日複数は `disclosed_ts` で**最新のみ**採用。

**YoY 成長（3）**（同四半期の1年前と比較: ラグ4期）

* `stmt_yoy_sales = (NetSales - NetSales[-4])/(abs(NetSales[-4])+1e-12)`
* `stmt_yoy_op = (OperatingProfit - OperatingProfit[-4])/(abs(OperatingProfit[-4])+1e-12)`
* `stmt_yoy_np = (Profit - Profit[-4])/(abs(Profit[-4])+1e-12)`

**マージン（2）**

* `stmt_opm = OperatingProfit/(NetSales+1e-12)`
* `stmt_npm = Profit/(NetSales+1e-12)`

**進捗率（2）**

* `stmt_progress_op = OperatingProfit/(ForecastOperatingProfit+1e-12)`
* `stmt_progress_np = Profit/(ForecastProfit+1e-12)`

**ガイダンス改定（4）**（前回開示と比較）

* `stmt_rev_fore_op = (ForecastOperatingProfit - prev_ForecastOperatingProfit)/(abs(prev_ForecastOperatingProfit)+1e-12)`
* `stmt_rev_fore_np = (ForecastProfit - prev_ForecastProfit)/(abs(prev_ForecastProfit)+1e-12)`
* `stmt_rev_fore_eps = (ForecastEarningsPerShare - prev_ForecastEarningsPerShare)/(abs(prev_ForecastEarningsPerShare)+1e-12)`
* `stmt_rev_div_fore = (ForecastDividendPerShareAnnual - prev_ForecastDividendPerShareAnnual)/(abs(prev_...)+1e-12)`

**財務指標（2）**

* `stmt_roe = Profit/(Equity+1e-12)`
* `stmt_roa = Profit/(TotalAssets+1e-12)`

**品質フラグ（2）**

* `stmt_change_in_est = (ChangesInAccountingEstimates ∈ {"true","1"}).int8()`
* `stmt_nc_flag = ((ChangesBasedOnRevisionsOfAccountingStandard ∨ RetrospectiveRestatement) == true).int8()`

**タイミング（2）**

* `stmt_imp_statement = (Date == effective_date).int8()`
* `stmt_days_since_statement = (Date - effective_date).days()`

> **備考**: 数値は文字列で届くことがあるため **型変換（to Float64）** を最初に実施。四半期系列のラグ参照（`[-4]`）は **同 Code 内** で会計期順に並べてから。

---

## 6) 有効フラグ・成熟フラグ（8）

| Column            | 定義                                        |
| ----------------- | ----------------------------------------- |
| `is_rsi2_valid`   | `row_idx >= 5`                            |
| `is_ema5_valid`   | `row_idx >= 15`                           |
| `is_ema10_valid`  | `row_idx >= 30`                           |
| `is_ema20_valid`  | `row_idx >= 60`                           |
| `is_ema200_valid` | `row_idx >= 200`                          |
| `is_valid_ma`     | `row_idx >= 60`                           |
| `is_flow_valid`   | `flow_days_since.is_not_null()`           |
| `is_stmt_valid`   | `stmt_days_since_statement.is_not_null()` |

---

## 7) 目的変数（ターゲット）（7）

* 回帰:

  * `target_1d = Close[+1]/Close - 1`
  * `target_5d = Close[+5]/Close - 1`
  * `target_10d`, `target_20d` 同様
* 2値:

  * `target_1d_binary = (target_1d > 0).int8()`
  * `target_5d_binary`, `target_10d_binary`

> **情報漏洩防止**: ターゲット計算は **完全に将来日の Close のみ**使用。

---

## 8) 列数の目安（重複除外後）

* **識別子/メタ**: 6
* **価格・テクニカル**: \~80
* **TOPIX（市場）**: 26
* **クロス**: 8
* **フロー**: 17
* **財務**: 17
* **フラグ**: 8
* **ターゲット**: 7
  **合計 ≈ 169 列**（環境/利用可否で±数列）

---

## 9) よく出る `NaN` と対処（実装ルール）

* **長期窓不足（z 系/ema200）** → ウォームアップ区間はそのまま `NaN` とし、**該当日の `is_*_valid` でマスク**。
* **フロー0%問題** → `section_norm` 揃える／**日次格子**を作ってから `(section_norm, Date)` で結合。
* **財務** → `as‑of` で開示が付かない日は `NaN` のまま（リーク防止）。
* **ゼロ割** → 全ての除算に `+1e-12`。
* **型** → `Date` は **pl.Date** に強制キャストしてから結合。

---

## 10) 実装スニペット（式の典型形）

```python
eps = 1e-12

# returns / vol
df = df.with_columns([
    (pl.col("Close")/pl.col("Close").shift(1)-1).alias("returns_1d"),
    (pl.col("Close")/pl.col("Close").shift(5)-1).alias("returns_5d"),
    (pl.col("returns_1d").rolling_std(20)*np.sqrt(252)).alias("volatility_20d"),
])

# MA gap
df = df.with_columns([
    pl.col("Close").ewm_mean(span=5).alias("ema_5"),
    pl.col("Close").ewm_mean(span=20).alias("ema_20"),
]).with_columns([
    ((pl.col("ema_5")-pl.col("ema_20"))/(pl.col("ema_20")+eps)).alias("ma_gap_5_20"),
])

# cross beta
df = df.with_columns([
    pl.cov(pl.col("returns_1d"), pl.col("mkt_ret_1d")).over("Code").alias("cov_60") \
        .rolling_mean(60),
    pl.var(pl.col("mkt_ret_1d")).rolling_mean(60).alias("var_60"),
]).with_columns([
    (pl.col("cov_60")/(pl.col("var_60")+eps)).alias("beta_60d")
])
```

---

### まとめ

* **列名・計算法**は上記で**固定運用**がおすすめ（モデル/特征選択で安定）。
* 既存ログで `NaN` が多かった列（`z` 系・flow 系）は、**ウォームアップ長**と**結合キー正規化**で解消できます。
* ここまでをそのまま **データ辞書（md か json）** に落としておけば、将来の回帰や自動検証も一貫して行えます。


----------

推奨ワークフロー

- 
    1. ベース生成
    - J-Quantsあり: python scripts/pipelines/run_pipeline_v4_optimized.py --jquants
    - なし: python scripts/pipelines/run_pipeline_v4_optimized.py（サンプル価格で生成）
- 
    2. 後段付与（TOPIX＋trade spec＋fins statements）
    - 例（ワンショット。出力: ml_dataset_latest_enriched.parquet）:
    - python - << 'PY'
      from pathlib import Path
      import polars as pl
      from scripts.data.ml_dataset_builder import MLDatasetBuilder
      out = Path('output')
      base = out/'ml_dataset_latest.parquet'
      df = pl.read_parquet(base)
      b = MLDatasetBuilder(output_dir=out)  # .env からJQトークンを読み、無ければサンプルTOPIX

      # 2-1) TOPIX 市場＋クロス特徴（mkt_* 26 + cross 8）
      df = b.add_topix_features(df)  # Dateで左結合、βはt-1ラグ、Zは252ウォームアップ

      # 2-2) trade spec（投資部門別フロー）
      # trades_spec のParquetを用意（gogooku2バッチ等）。例パスは環境に合わせて変更。
      trades_spec_path = Path('/home/ubuntu/gogooku2/output/batch/20250824/trades_spec/tse_trades_spec_20250824_175808.parquet')
      if trades_spec_path.exists():
          trades_spec_df = pl.read_parquet(trades_spec_path)
          df = b.add_flow_features(df, trades_spec_df, listed_info_df=None)  # 区間→日次展開→結合
      else:
          print('trade spec not found; skipping flow features')

      # 2-3) 財務諸表
      # パイプラインが保存した raw statements を優先（なければスキップ/再取得）。
      stmt = out/'event_raw_statements.parquet'
      if stmt.exists():
          statements_df = pl.read_parquet(stmt)
          df = b.add_statements_features(df, statements_df)
      else:
          print('statements file not found; skipping statements features')

      # 保存（with/without を分ける）
      enriched = out/'ml_dataset_latest_enriched.parquet'
      df.write_parquet(enriched)
      (out/'ml_dataset_latest_enriched.parquet')
      print('saved:', enriched)
      PY

設計の要点（安全・品質）

- 結合キー: Date 左結合（市場列は日次で全銘柄に同一付与）
- 情報リーク防止: ベータは市場リターンを t-1 にラグ（実装済）
- Zスコア: ウォームアップ 252 日、初期は null 許容（実装済）
- 型整合: Date の文字列→日付変換に対応（実装済）
- trade spec: 区間→日次展開→銘柄に結合（builder 内部にユーティリティ有）
- 財務: まず event_raw_statements.parquet（パイプライン保存物）から結合。無ければ後段で fetch_statements_by_date による補完も可能
成果物と採用ルール

- ベース: output/ml_dataset_latest.parquet（価格系）
- 付与後: output/ml_dataset_latest_enriched.parquet（TOPIX＋flow＋statements）
- 学習側は enriched を優先採用（設定で切替可能）

便利化（任意）

- ラッパー追加: scripts/pipelines/run_pipeline_with_enrichment.py を作り、ベース→付与→保存を自動化
- Make ターゲット: make dataset-with-enrichment でワンコマンド運用

検証（簡易チェック）

- 市場列: sum(c.startswith('mkt_') for c in cols) == 26
- クロス列: beta_*, alpha_*, rel_strength_5d, trend_align_mkt, idio_vol_ratio, beta_stability_60d の存在
- フロー列: flow_* などのプレフィックス数を確認
- 財務列: stmt_ プレフィックスの存在

この流れなら、run_pipeline 本体は無改変のまま、最終的に TOPIX＋trade spec＋fins statements を含む「学習用の完成データセット」を安定生成できます。ラッパー/Make も用意しましょうか？