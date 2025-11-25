了解です。
ここまでの会話で出てきた **「/prices/daily_quotes から作る特徴量」** と
**「/markets/breakdown から作る特徴量」** を、**漏れなく**整理して出します。

ついでに、両者を掛け合わせる **複合特徴量** もまとめておきます。

---

# 1. `/prices/daily_quotes` 由来の特徴量リスト

## 1-0. 前処理・基本量

* `adj_open, adj_high, adj_low, adj_close, adj_volume`
  調整済み OHLCV。
* `dollar_volume = TurnoverValue`
  日次売買代金。
* `log_price = log(adj_close)`
  必要に応じて。
* `log_vol = log(adj_volume + 1)`
* `log_turnover = log(TurnoverValue + 1)`

---

## 1-1. リターン・モメンタム系

**単日・マルチホライズン**

* `ret_1d`
  `(P_t / P_{t-1} - 1)` または `log(P_t / P_{t-1})`。
* `logret_1d`
  `log(P_t / P_{t-1})`。
* `ret_5d, ret_20d, ret_60d`
  `log(P_t / P_{t-5/20/60})`。
* `ret_intraday`
  `log(adj_close_t / adj_open_t)`。
* `ret_overnight`
  `log(adj_open_t / adj_close_{t-1})`。

**モメンタム**

* `mom_5, mom_10, mom_20`
  過去 5/10/20 日リターン（`log(P_t / P_{t-N})`）。
* `mom_chg_5_20`
  `mom_5 - mom_20`（短期と中期の差）。
* （任意）`mom_accel`
  モメンタムの二階差分的な加速度。

**フォワードリターン（ターゲット）**

* `fwd_ret_1d, fwd_ret_5d, fwd_ret_10d, fwd_ret_20d`
  `log(P_{t+N} / P_t)`。

---

## 1-2. ボラティリティ系

**実現ボラ**

* `rv_5, rv_20`
  過去 5/20 日の `logret_1d` の標準偏差。
* `rv_ratio`
  `rv_5 / rv_20`（短期ボラ/中期ボラ）。
* （候補）`rv_z_20`
  `rv_5` の 20 日 Z スコア。

**レンジ系**

* `range`
  `(High - Low) / Close`。
* `range_z_20`
  `range` の 20 日 Z スコア。

**高値安値ベース**

* `parkinson_20`
  High/Low のみを使った 20 日実現ボラ。
* `gk_20`
  Garman–Klass ボラ（OHLC ベース）。

**テクニカル系ボラ（後述と重複）**

* `atr_14`
  True Range 由来の 14 日平均。

---

## 1-3. 出来高・売買代金・流動性系

**出来高レベルと異常度**

* `vol_ma_20`
  出来高の 20 日移動平均。
* `vol_ratio_5`
  当日出来高 / 過去 5 日平均出来高。
* `vol_ratio_20`
  当日出来高 / 過去 20 日平均出来高。
* `vol_ratio_5_20`
  `vol_ratio_5 / vol_ratio_20` 等。
* `vol_z_20`
  `log_vol` の 20 日 Z スコア。

**売買代金版**

* `val_ratio_5, val_ratio_20, val_z_20`
  上記の売買代金版。

**流動性・価格インパクト**

* `amihud_1d`
  `|ret_1d| / (TurnoverValue / スケール)`。
* `amihud_20`
  `amihud_1d` の 20 日平均。
* （任意）`vwap = TurnoverValue / Volume`
  日次 VWAP。

---

## 1-4. 価格位置・トレンド系

**レンジ内位置**

* `price_pos_20`
  `(P_t - min(P_{t-19..t})) / (max(P_{t-19..t}) - min(...) + ε)`。

**52週高値/安値との距離**

* `dist_from_high52`
  `(P_t - high_52) / high_52`。
* `dist_from_low52`
  `(P_t - low_52) / low_52`。

**移動平均 & 乖離率**

* （中間計算）`ma_5, ma_20, ma_60, ma_200`
* `price_dev_5`
  `(P_t - ma_5) / ma_5`。
* `price_dev_20`
  `(P_t - ma_20) / ma_20`。
* `dev_ma60`
  `(P_t - ma_60) / ma_60`。

---

## 1-5. ギャップ・イベント系

**ギャップ**

* `overnight_gap`
  `(adj_open_t - adj_close_{t-1}) / adj_close_{t-1}`（= ret_overnight の線形版）。
* `gap_abs`
  `|overnight_gap|`。
* `gap_dir`
  `sign(overnight_gap)`。

**ストップ高・安 / 分割イベント**

* `is_limit_up`
  `UpperLimit == 1`。
* `is_limit_down`
  `LowerLimit == 1`。
* `limit_up_count_20, limit_down_count_20`
  過去 20 日のストップ高/安回数。
* `days_since_limit_up, days_since_limit_down`
  最後にストップ高/安をつけてからの日数。
* `is_split`
  `AdjustmentFactor != 1` の日フラグ。

---

## 1-6. テクニカル指標（TA）

**移動平均系**

* `sma_5, sma_20, sma_60, sma_200`
* `ema_12, ema_26`
* 上記からの乖離率は `price_dev_*` でカバー。

**オシレーター・トレンド**

* `rsi_14`
  14 日 RSI。
* `macd_12_26`
* `macd_signal_9`
* `macd_hist`

**ボリンジャーバンド**

* `bb_mid_20`
* `bb_upper_20`
* `bb_lower_20`
* `bb_percent_b_20` (%B)
* `bb_width_20`

**ボラ・トレンド系**

* `atr_14`
* （任意）`adx_14`

---

## 1-7. ローソク足・連続パターン系

**単一日の形状**

* `cand_body`
  `|Close - Open| / Open`。
* `cand_body_ratio`
  `body / ((High - Low) / Close + ε)`。
* `cand_upper_shadow`
  `(High - max(Open, Close)) / (High - Low + ε)`。
* `cand_lower_shadow`
  `(min(Open, Close) - Low) / (High - Low + ε)`。
* `cand_is_bull`
  `Close > Open`（0/1）。

**連続パターン**

* `bull_run_len`
  連続陽線日数。
* `bear_run_len`
  連続陰線日数。
* `bull_ratio_5d, bull_ratio_20d`
  直近 N 日の陽線率。

---

## 1-8. セッション別特徴量（Premium: Morning/Afternoon）

**リターン関連**

* `ret_morning`
  `(MorningClose - MorningOpen) / MorningOpen`。
* `ret_afternoon`
  `(AfternoonClose - AfternoonOpen) / AfternoonOpen`。
* `session_divergence`
  `ret_afternoon - ret_morning`。

**出来高構造**

* `morning_vol_share`
  `MorningVolume / Volume`。
* `afternoon_vol_share`
  `AfternoonVolume / Volume`。

**ボラ・レンジ**

* `range_morning`
  `(MorningHigh - MorningLow) / MorningClose`。
* `range_afternoon`
  `(AfternoonHigh - AfternoonLow) / AfternoonClose`。
* `session_vol_ratio`
  `range_morning / (range_afternoon + ε)`。

**ギャップ**

* `lunch_gap`
  `(AfternoonOpen - MorningClose) / MorningClose`。

**統計系（時系列集計）**

* `morning_win_rate_20`
  直近 20 日で `ret_morning > 0` の割合。
* `afternoon_win_rate_20`
* `session_corr_20`
  直近 20 日の `ret_morning` vs `ret_afternoon` の相関。

---

## 1-9. クロスセクション（同日内の相対指標）

※「daily_quotes 由来」だけど、日付ごとに全銘柄まとめて計算する層。

* `cs_rank_ret_1d, cs_rank_ret_5d`
  その日の全銘柄中でのリターン順位（0〜1）。
* `cs_pct_dollar_volume`
  売買代金のパーセンタイル。
* `cs_z_ret_1d, cs_z_vol_z_20`
  クロスセクション Z スコア。

（このあたりは別テーブルにしてもOK）

---

# 2. `/markets/breakdown` 由来の特徴量リスト

まず値ベース・株数ベースの合計を作るところから。

```text
total_sell_value = LongSellValue
                 + ShortSellWithoutMarginValue
                 + MarginSellNewValue
                 + MarginSellCloseValue

total_buy_value  = LongBuyValue
                 + MarginBuyNewValue
                 + MarginBuyCloseValue

short_flow_value = ShortSellWithoutMarginValue + MarginSellNewValue
long_sell_value  = LongSellValue
margin_sell_new_value   = MarginSellNewValue
margin_sell_close_value = MarginSellCloseValue
long_buy_value          = LongBuyValue
margin_buy_new_value    = MarginBuyNewValue
margin_buy_close_value  = MarginBuyCloseValue

# Volume 版も同様
```

---

## 2-1. その日1日の「需給の顔」(Valueベース)

**売り側構成比**

* `sell_long_ratio`
  `LongSellValue / total_sell_value`。
* `sell_short_ratio`
  `(ShortSellWithoutMarginValue + MarginSellNewValue) / total_sell_value`。
* `sell_margin_close_ratio`
  `MarginSellCloseValue / total_sell_value`。

**買い側構成比**

* `buy_long_ratio`
  `LongBuyValue / total_buy_value`。
* `buy_margin_new_ratio`
  `MarginBuyNewValue / total_buy_value`。
* `buy_margin_close_ratio`
  `MarginBuyCloseValue / total_buy_value`。

**信用比率**

* `credit_buy_share`
  `(MarginBuyNewValue + MarginBuyCloseValue) / total_buy_value`。
* `credit_sell_share`
  `(MarginSellNewValue + MarginSellCloseValue) / total_sell_value`。
* `credit_turnover_share`
  `(MarginBuyNewValue + MarginBuyCloseValue + MarginSellNewValue + MarginSellCloseValue) / (total_buy_value + total_sell_value)`。

**総売買バランス**

* `net_flow_value`
  `total_buy_value - total_sell_value`。
* `flow_imbalance`
  `(total_buy_value - total_sell_value) / (total_buy_value + total_sell_value + ε)`。

**現物 vs 信用 新規**

* `net_long_value`
  `LongBuyValue - LongSellValue`。
* `net_long_ratio`
  `net_long_value / (LongBuyValue + LongSellValue + ε)`。
* `net_margin_new_value`
  `MarginBuyNewValue - MarginSellNewValue`。
* `margin_new_sentiment`
  `net_margin_new_value / (MarginBuyNewValue + MarginSellNewValue + ε)`。

**Bull vs Bear フロー**

* `bull_flow_value`
  `LongBuyValue + MarginBuyNewValue`。
* `bear_flow_value`
  `LongSellValue + ShortSellWithoutMarginValue + MarginSellNewValue`。
* `bull_bear_ratio`
  `bull_flow_value / (bull_flow_value + bear_flow_value + ε)`。

**既存カラムとの対応**

* `breakdown_buy_ratio`
  `total_buy_value / (total_buy_value + total_sell_value)`。
* `breakdown_sell_ratio`
  `total_sell_value / (total_buy_value + total_sell_value)`。
* `breakdown_volume_ratio`
  `総買い株数 / (総買い株数 + 総売り株数)` 等。

---

## 2-2. Volume ベースの構成比（任意）

* `sell_short_ratio_vol`
  `(ShortSellWithoutMarginVolume + MarginSellNewVolume) / 総売り株数`。
* `buy_margin_new_ratio_vol`
* その他、Value版と同様に Volume 版を数本（必要に応じて）。

低位株・高値株の影響を分離する補助指標として。

---

## 2-3. 時系列の変化・トレンド

銘柄ごとに `group_by(Code)` して計算。

**比率系のモメンタム**

* `short_ratio_ma_5, short_ratio_ma_20`

* `short_ratio_dev_5, short_ratio_dev_20`
  （当日値 − MA）

* `short_ratio_chg_1, short_ratio_chg_5`
  （前日比・5日前比）

* `long_buy_ratio_ma_5/20`、`long_buy_ratio_dev_*` 等も同様に作成可。

**Z スコア（異常度）**

* `short_ratio_z_60`
  60 日ローリング Z スコア。
* `long_buy_ratio_z_60`
* `flow_imbalance_z_60`

**ランレングス（需給トレンド持続）**

* `pos_flow_run_len`
  `flow_imbalance > 0` が何日連続か。
* `neg_flow_run_len`
  `flow_imbalance < 0` の連続日数。

---

## 2-4. 信用ポジション近似 & オーバーハング

Volume ベースで：

* `delta_margin_long_vol`
  `MarginBuyNewVolume - MarginSellCloseVolume`（信用買い残の「変化」）。
* `delta_margin_short_vol`
  `MarginSellNewVolume - MarginBuyCloseVolume`（信用売り残の「変化」）。

これを累積：

* `cum_margin_long_60`
  過去 60 日の `delta_margin_long_vol` 累積。
* `cum_margin_short_60`
  同じく売り側。

さらに、/prices 側から `AdjVolume` を持ってきて：

* `adv_vol_20`
  20 日平均出来高。
* `short_overhang_days`
  `cum_margin_short_60 / (adv_vol_20 + ε)`。
* `long_overhang_days`
  `cum_margin_long_60 / (adv_vol_20 + ε)`。

**リスク指標**

* `short_squeeze_risk`
  `short_overhang_days * max(0, ret_5d)`。
* `long_liquidation_risk`
  `long_overhang_days * max(0, -ret_5d)`。

---

## 2-5. クロスセクション（同日内での相対需給）

日付ごとに全銘柄で計算。

**ランク／パーセンタイル**

* `cs_rank_short_ratio, cs_pct_short_ratio`
* `cs_pct_flow_imbalance`
* `cs_pct_short_overhang_days`
* `cs_pct_credit_turnover_share`

**Z スコア**

* `cs_z_short_ratio`
* `cs_z_flow_imbalance`

---

# 3. `/prices` × `/markets/breakdown` の複合特徴量（オプション）

これは「どちらのテーブルにも依存する」ので別枠として。

**価格方向とフロー方向の一致度**

* `flow_price_alignment`
  `sign(ret_1d) * net_flow_value`。
* `short_price_alignment`
  `sign(ret_1d) * short_flow_value`。

**強さ（フロー/売買代金）**

* `flow_intensity`
  `net_flow_value / (dollar_volume + ε)`。
* `short_intensity`
  `short_flow_value / (dollar_volume + ε)`。

**ボラとの組み合わせ**

* `flow_vol_combo`
  `flow_intensity * rv_5`。
* `flow_vol_ratio`
  `flow_intensity / (rv_5 + ε)`。

---

## 4. まとめ

* 上のリストは、これまでの会話で出てきた
  **`daily_quotes` 由来の特徴量** と
  **`market_breakdown` 由来の特徴量**
  をカテゴリーごとに**全部まとめたもの**です。
* 実装・保存するテーブル（features_daily）では、この中から

  * コア（Tier1〜2）
  * 余力があれば追加（TA・ローソク足・Premium・クロスセクション）
    を選んでいく形になります。

もし、「この中から **実際に features_daily に入れる列名の最終セットを決めたい**」とか
「Polars の実装スケルトンがほしい」というフェーズに行くなら、
そこだけ切り出して一緒に設計しましょう。
