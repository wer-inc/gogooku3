# NULL率 ≥10% 特徴量リスト - 詳細サマリー

**作成日**: 2025-11-19
**データセット**: `output_g5/datasets/ml_dataset_2023_2025_final_pruned.parquet`
**対象期間**: 2023-01-04 → 2024-12-30 (1,880,466行)

---

## 📊 エグゼクティブサマリー

### 全体統計

| 指標 | 値 |
|------|-----|
| **全特徴量数** | 3,542列 |
| **NULL率 ≥10%の特徴量** | **732列 (20.67%)** |
| **NULL率 <10%の特徴量** | **2,810列 (79.33%)** |

### NULL率帯域別分布

| NULL率帯域 | 列数 | 割合 | 推奨対応 |
|------------|------|------|----------|
| **100%** | 45列 | 1.27% | 🔴 **即座に削除または修正** |
| **90-99%** | 237列 | 6.69% | 🟠 **削除または修正** |
| **50-90%** | 259列 | 7.31% | 🟡 **検討（モデルで使用可能か判断）** |
| **30-50%** | 10列 | 0.28% | ⚠️ **許容範囲（注意が必要）** |
| **20-30%** | 46列 | 1.30% | ⚠️ **許容範囲** |
| **10-20%** | 135列 | 3.81% | ℹ️ **許容範囲** |

---

## 📋 カテゴリ別分析

### TOP 10カテゴリ (NULL率≥10%)

| カテゴリ | 列数 | 平均NULL率 | 主要課題 |
|----------|------|------------|----------|
| **other (その他)** | 331列 | 69.35% | クロスセクション派生特徴量の連鎖NULL |
| **cs (クロスセクション)** | 180列 | 69.95% | 元特徴量のNULLが派生特徴量に伝播 |
| **ssp (空売り)** | 112列 | 64.39% | データ取得頻度の問題 |
| **div (配当)** | 24列 | 22.39% | 配当実施企業のみデータあり（正常） |
| **flow_signal (フロー信号)** | 22列 | 61.89% | 元特徴量の欠損による連鎖NULL |
| **preE (決算前フロー)** | 22列 | 99.15% | データソース未整備 |
| **topix (TOPIX)** | 11列 | 69.64% | 指数データの一部期間欠損 |
| **idxopt (指数オプション)** | 11列 | 60.16% | オプションデータの部分的欠損 |
| **fs (財務諸表)** | 9列 | 62.39% | YoY計算ロジック未実装 |
| **wm (週次信用)** | 8列 | 15.71% | 週次更新によるラグ（許容範囲） |

---

## 🔴 重大問題: NULL率100%の特徴量 (45列)

### 根本原因別分類

#### 1. YoY計算ロジック未実装 (3列)

| 特徴量 | 用途 | 対策 |
|--------|------|------|
| `fs_sales_yoy` | 売上高YoY成長率 | YoY計算ロジックの実装 |
| `fs_yoy_ttm_sales` | TTM売上高YoY成長率 | YoY計算ロジックの実装 |
| `fs_yoy_ttm_net_income` | TTM純利益YoY成長率 | YoY計算ロジックの実装 |

**実装例**:
```python
df = df.with_columns([
    ((pl.col("fs_ttm_sales") - pl.col("fs_ttm_sales").shift(252)) /
     pl.col("fs_ttm_sales").shift(252)).alias("fs_yoy_ttm_sales")
])
```

#### 2. 決算前フローシグナル (22列)

**パターン**: `preE_*` 系列の特徴量がすべて100% NULL

**影響特徴量**:
- `preE_margin_diff` + 派生特徴量 (zscore, outlier, roll_mean, roll_std等)
- `preE_margin_diff_z20` + 派生特徴量

**原因**: 決算イベント前後のフロー分析用データが未整備

**対策**: データソース整備または全列削除

#### 3. フロート調整信用取引特徴量 (20列)

**パターン**: `margin_long_pct_float*` 系列がすべて100% NULL

**原因**: フロート株数データ (`shares_free_float`) が94% NULL → 派生特徴量も100% NULL

**対策**:
1. フロート株数データソースの整備
2. または全列削除

#### 4. クラウディングスコア (6列)

**パターン**: `crowding_score*` 系列がすべて100% NULL

**原因**: 元特徴量の計算ロジック未実装

**対策**: 実装または削除

#### 5. 機関投資家フローシグナル (9列)

**パターン**: `institutional_accumulation*`, `foreign_sentiment*` 等の派生特徴量

**原因**: 元特徴量は78.61% NULLだが、派生特徴量は100% NULL

**対策**: 元特徴量の品質改善または削除

---

## 🟠 高NULL率: 90-99% (237列)

### 主要パターン

#### 1. クロスセクションランク特徴量の連鎖NULL

**例**: `rq_63_*` (Realized Quantile) 系列

- `rq_63_10`, `rq_63_50`, `rq_63_90`: **99.97% NULL**
- すべてのクロスセクション派生特徴量: `_cs_rank`, `_cs_pct`, `_cs_top20_flag` 等

**原因**: 元特徴量が99.97% NULL → すべての派生特徴量も同率NULL

#### 2. セクター変更関連 (7列)

- `days_since_sector33_change`: **99.92% NULL**
- すべての派生特徴量

**原因**: セクター変更イベントが稀

#### 3. フロート調整特徴量 (60列以上)

**パターン**:
- `float_turnover_pct*` 系列: 94-95% NULL
- `weekly_margin_long_pct_float*` 系列: 94-95% NULL
- `shares_total*`, `shares_free_float*` 系列: 94% NULL

**原因**: フロート株数データ (`shares_free_float`) が94% NULL

#### 4. SQ (特別清算指数) 関連 (14列)

- `days_to_sq`, `days_since_sq`: **95.32% NULL**
- すべての派生特徴量

**原因**: SQイベントが稀（年4回のみ）

#### 5. 時系列長期リターン

- `ret_prev_60d`: **97.40% NULL**

**原因**: 60日lookbackに必要なデータが不足している期間が多い

---

## 🟡 中NULL率: 50-90% (259列)

### 主要カテゴリ

#### 1. 信用取引（日次信用残高）関連 (30列)

**パターン**: `dmi_*` (Daily Margin Interest) 系列

- 平均NULL率: **89%**
- 全30列が80%以上のNULL率

**原因**: 日次信用残高データの取得率が低い

**対策**: データソースの改善、または特徴量削除を検討

#### 2. 空売り比率関連 (80列以上)

**パターン**: `ssp_*` (Short Selling Position) 系列

- 個別銘柄の空売りポジションデータ
- 平均NULL率: **64%**

**原因**: 報告義務がある大口空売りのみデータあり

**対策**: 許容範囲内だが、モデルでの有効性を検証

#### 3. 株主構成フロー (18列)

**パターン**: `institutional_accumulation*`, `foreign_sentiment*` 等

- 元特徴量: **78.61% NULL**
- すべての派生特徴量: **78-100% NULL**

**原因**: 週次更新データのラグ + 全銘柄カバレッジの不足

---

## ⚠️ 低NULL率: 30-50% (10列)

### 財務諸表TTM特徴量

| 特徴量 | NULL率 | 状態 |
|--------|--------|------|
| `fs_revenue_ttm` | 33.54% | ✅ 許容範囲 |
| `fs_net_income_ttm` | 33.43% | ✅ 許容範囲 |
| `fs_ttm_sales` | 33.54% | ✅ 許容範囲 |
| `fs_ttm_net_income` | 33.43% | ✅ 許容範囲 |

**原因**: 一部企業の財務データ報告遅延（正常な挙動）

**評価**: ✅ **許容範囲** - これらの特徴量は学習に使用可能

---

## 📝 推奨アクション

### Priority 1: 即座に削除すべき特徴量 (282列)

**対象**: NULL率 ≥90% (100% + 90-99%)

**アプローチ**:
```python
# NULL率90%以上の列を削除
high_null_cols = [col for col, rate in null_rates.items() if rate >= 90.0]
df_clean = df.drop(high_null_cols)
```

**期待効果**:
- 列数削減: 3,542列 → 3,260列
- 平均NULL率改善: 14.65% → 約8%

### Priority 2: YoY特徴量の実装 (3列)

**対象**:
- `fs_sales_yoy`
- `fs_yoy_ttm_sales`
- `fs_yoy_ttm_net_income`

**期待効果**: 財務諸表特徴量の品質向上

### Priority 3: フロート株数データの整備

**影響範囲**: 60列以上

**オプション**:
1. データソース改善（長期的）
2. 代替データソース検討
3. 特徴量削除（短期的）

### Priority 4: クロスセクション派生特徴量の見直し

**対象**: 180列

**アプローチ**:
- 元特徴量のNULL率を確認
- 元特徴量が高NULL率の場合、派生特徴量も削除

---

## 📊 詳細リスト

**CSV形式の完全リスト**: `gogooku5/docs/high_null_features_list_20251119.csv`

**内容**:
- 732行 (NULL率≥10%のすべての特徴量)
- カラム: `column`, `null_rate`, `null_count`, `category`, `band`
- ソート: NULL率降順

**使用例**:
```python
import pandas as pd
df_null = pd.read_csv('gogooku5/docs/high_null_features_list_20251119.csv')

# NULL率100%の列を抽出
null_100 = df_null[df_null['band'] == '100%']
print(null_100['column'].tolist())
```

---

## 🎯 結論

### ✅ 良好な点

1. **高品質特徴量の確保**: 2,810列 (79.33%) がNULL率<10%
2. **TTM特徴量の改善**: 平均12.18% NULL率（目標達成）
3. **許容範囲の特徴量**: 191列 (5.39%) が10-50% NULL率

### ⚠️ 改善が必要な点

1. **高NULL率特徴量**: 282列 (7.96%) がNULL率≥90%
2. **YoY特徴量未実装**: 3列が100% NULL
3. **フロート株数データ不足**: 60列以上に影響

### 📋 次回セッションでの作業

**Phase 1: クリーンアップ** (1-2時間)
1. NULL率100%の特徴量削除 (45列)
2. NULL率90-99%の特徴量削除 (237列)
3. 期待結果: 3,542列 → 3,260列

**Phase 2: YoY実装** (1-2時間)
1. `fs_sales_yoy` の実装
2. `fs_yoy_ttm_sales` の実装
3. `fs_yoy_ttm_net_income` の実装

**Phase 3: 検証** (30分)
1. 残存NULL率の確認
2. 特徴量品質レポート更新

---

**総合評価**: ✅ **現状でも学習可能だが、282列の削除で品質向上が見込める**

---

## ✅ 方針更新 (fs_E / free-float / イベント系)

### preE 系列の扱い

- `preE_*`（決算「予定」ベース）の特徴量は、/fins/announcement が過去の予定履歴を提供しない仕様のため、**過去時点の再現ができない**。
- 現世代ではこれらを **モデル入力から外し、feature group にも含めない** 方針とする。
- 代わりに、/fins/statements / fs_details 由来の実績決算イベント:
  - `fs_E_event_date`
  - `fs_days_since_E`
  - `fs_window_e_pm1`, `fs_window_e_pp3`, `fs_window_e_pp5`
  を短期モデルの主要な E イベント特徴量として採用する。

### free-float 系列の扱い

- 真のフリーフロート（`shares_free_float`, `market_cap_free_float`, `*_pct_float`）は **J-Quants単体では十分に再現できない**。
- 現在のパイプラインでは:
  - 発行済株数ベースの `fs_shares_outstanding` / `market_cap_total` を基盤とし、
  - /markets/daily_margin_interest の `*ListedShareRatio` を用いた `margin_long_listed_ratio` / `margin_short_listed_ratio`,
  - `float_turnover_pct_tradable`, `margin_long_pct_tradable`, `weekly_margin_long_pct_tradable`
    を主要な需給比率として採用する。
- `*_free_float*` プレフィックスを持つ列は **原則としてモデルでは使用しない（非推奨）**。

### セクター変更 / SQ 距離の扱い

- `days_since_sector33_change`, `sector33_changed_5d`, `days_to_sq`, `days_since_sq` は、
  - イベント自体がきわめて稀であるため **高NULL率は構造的に避けられない**。
- これらの列はデータセット内には保持するが、
  - 短期用 feature group（`g5_short_term`）からは除外し、
  - 将来のイベント専用モデルや分析用途での利用候補として扱う。
