ありがとうございます。いただいた「現状の実装状況分析」を前提に、**いま何ができていて、何がボトルネックで、どの順にどう直すと“ベスト”に届くか**を、\*\*差分（具体的変更点）\*\*ベースで整理しました。結論から言うと——

> **現状は土台（営業日スケジューリング／市場フィルタ／ページネーション／キャッシュ）は良好。主な改善余地は “データ取得の軸” と “差分検知の仕組み” に集中**しています。

---

## 仕様追記（2025-09-07）

本節は、本データセットに既に含まれている機能・列の仕様を明文化し、将来の再現性を高めるための補足です。

### 1) セクター・ターゲットエンコーディング（TE）
- 生成列: `te33_sec_target_5d`（エイリアス: `te_sec_target_5d`）
- ベース: 33 セクター（`sector33_id`）
- リーク防止: `t-Δ` での計算（既定 Δ=1 営業日）。当日情報は使用しない。
- 学習安定化: k-fold クロスフィット排他 + ベイズ平滑（`m=100`）
  - セクター平均（除外 fold）= μ_sec_excl(t)
  - グローバル平均（除外 fold）= μ_all_excl(t)
  - TE = (n_sec·μ_sec_excl + m·μ_all_excl) / (n_sec + m)
- 欠損規約: `sector33_id` が未付与の行は TE を `null` とする。

注: 極端にデータが薄い日に対しては、fold 排他計算が成立しない場合があり、その際も `t-1` までの情報のみで μ を構成する。

### 2) セクター属性付与のルール
- データ源: `listed_info`（`Code`, `Date`, `MarketCode`, `Sector33Code` ほか）
- キー: 内部は `LocalCode`/`Code` を UTF-8 文字列として扱う（本データでは同一値）。
- 期間結合:
  - 原則は as-of backward（`valid_from ≤ Date ≤ valid_to`）。
  - 将来日スナップショットのみが存在する場合は、`valid_from` をデータセット最小日付まで引き下げ、全期間カバーできるよう補正する（スナップショット型の listed_info を許容）。
- 優先規則: 同名列が両辺にある場合、`listed_info` 側（右辺）を優先。
  - 例: `Sector33Code` → `sector33_code`/`sector33`、`MarketCode` を右辺優先で採用。
- Section 付与: `MarketCode`→`Section` へ標準化（JASDAQ 等は後方互換ルールで統合）。
- 期待カバレッジ: 上場銘柄の大半（約 95〜97% 程度）が `sector33_id ≥ 0` となる。上場直後・特殊市場移行直後などは `null` になり得る。

### 3) 命名の正規化（レガシー → 正式名）
- 有効フラグ: `is_ema_5_valid` → `is_ema5_valid`、`is_ema_10_valid` → `is_ema10_valid`、`is_rsi_2_valid` → `is_rsi2_valid` など。
- ボリンジャー: `bb_bandwidth` → `bb_width`、`bb_pct_b` → `bb_position`。
- その他: 旧名は極力出力しない。互換が必要な場合はビュー/エイリアスで吸収する。

### 4) フロー特徴の表記について
- 文書中の **プレフィクスの説明** は「flow_* グループ」と記述する（ワイルドカード `flow_*` は列集合の説明であり、個々の列名は明示列を参照）。
- 代表列: `flow_foreign_net_ratio`, `flow_individual_net_ratio`, `flow_activity_ratio`, `flow_smart_idx`, `flow_smart_mom4`, `flow_shock_flag`, `flow_impulse`, `flow_days_since` など（合計 ≥ 17）。

### 5) 拡張（任意列）
- セクター日次頻度: `sec33_daily_freq`
  - 含意: 同一 `Date` における 33 セクター内の相対頻度（集計窓無しのその日スナップショット）。
  - 算出: `count(Date×sector33_id) / count(Date)` を window 集計で算出（join 不要）。
  - 位置づけ: 任意（必要に応じて生成）。本仕様の必須列ではない。

### 6) 日次マージン（dmi_）ブロックの追加

- ソース: `/markets/daily_margin_interest`（J-Quants）。
- 補正: `(Code, ApplicationDate)` ごとに **最新の `PublishedDate`** を採用。
- 有効日: `effective_start = next_business_day(PublishedDate)`（T+1）。
- 結合: `(Code, Date)` に **as‑of backward** で付与。`dmi_impulse=1` は有効日当日。
- 主な列: `dmi_long/short/net/total/credit_ratio/imbalance/short_long_ratio`, `dmi_d_*_1d`, `dmi_z26_*`, `dmi_*_to_adv20`, `dmi_reason_*`, `dmi_reason_count`, `dmi_tse_reg_level`, `dmi_days_since_pub/app`, `is_dmi_valid`。
- 有効化: `scripts/pipelines/run_full_dataset.py --enable-daily-margin [--daily-margin-parquet PATH]`（未指定時は `output/daily_margin_interest_*.parquet` 自動探索）。

---

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

----------

承知しました。**“漏れなく”を最優先**に、現在の実装方針＋これまでの合意事項を統合した **フル版データセット仕様（列カタログ＋計算式＋背景）** をまとめ直します。
（※ `ε=1e-12` はゼロ割回避の微小量。`over('Code')` は銘柄内ローリング、`over('Section')` は市場区分内ローリングを意味します。）

---

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
