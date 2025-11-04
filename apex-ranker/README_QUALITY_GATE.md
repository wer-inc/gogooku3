# APEX-Ranker 品質ゲート & バックテスト健全性チェック

データ品質管理とバックテスト検証のための3ステッププロセス。

## 📋 概要

### Step 1: データ品質ゲート
低品質データを除外し、クリーンなデータセットを生成

### Step 2: クリーンデータでバックテスト
autosupply機能で銘柄供給を自動調整

### Step 3: バックテスト結果の健全性チェック
JSON出力の検証と品質レポート生成

---

## 🚀 実行手順（コピペでOK）

### Step 1: クリーンデータ生成（品質ゲート付き）

```bash
# パス定数を使用（推奨）
python -c "from apex_ranker.scripts.path_constants import *; print(f'Input: {DATASET_RAW}\nOutput: {DATASET_CLEAN}\nReport: {QUALITY_REPORT}')"

# 実行（path_constantsで定義されたパスを使用）
python apex-ranker/scripts/filter_dataset_quality.py \
  --input output/ml_dataset_latest_full.parquet \
  --output output/ml_dataset_clean.parquet \
  --min-price 100 \
  --max-ret-1d 0.15 \
  --min-adv 50000000 \
  --report output/reports/quality_report.json
```

**チェック内容**:
- ✅ `share(|ret_1d|>10%) < 0.5%` を満たす
- ✅ `share(|ret_1d|>15%) ≈ 0%`（実装は ≤ 1e-6）
- ✅ `count(price < min_price) = 0`
- ✅ **price_freezes≥5日** の **post/pre ≤ 0.5** かつ **post ≤ 100日**

**ADV60計算**:
- 過去60営業日・**当日除外**（ルックアヘッド防止）
- `adv60_trailing = rolling_mean(turnover, 60).shift(1)`
- turnover列がない場合は `volume * price` から自動算出

---

### Step 2: クリーンデータで再バックテスト（週次×5d）

```bash
python apex-ranker/scripts/backtest_smoke_test.py \
  --model  models/apex_ranker_v0_pruned.pt \
  --config apex-ranker/configs/v0_pruned.yaml \
  --data   output/ml_dataset_clean.parquet \
  --start-date 2024-01-01 \
  --end-date   2024-12-31 \
  --horizon 5 \
  --top-k 50 \
  --output output/backtest/backtest_result.json
```

**autosupply統合（バックテストスクリプトに追加）**:

```python
# バックテストの選定直前にコピペ
from apex_ranker.scripts.autosupply_utils import calculate_dynamic_k

# リバランス日ごとに動的k調整
k_pick = calculate_dynamic_k(
    candidate_count=len(candidates),
    target_top_k=35,      # 目標保有銘柄数
    alpha=1.5,            # 供給倍率（35 × 1.5 = 53最低供給）
    floor_ratio=0.15,     # 最低供給率（15%）
    hard_floor=53,        # 絶対最低値
)

# 上位 k_pick 銘柄を選定（min で候補数上限を考慮）
selected = candidates[:min(k_pick, len(candidates))]
```

---

### Step 3: バックテスト結果の健全性チェック（JSON/ログ確認）

```bash
python apex-ranker/scripts/check_backtest_output.py \
  --input output/backtest/backtest_result.json \
  --k-min 53 \
  --fallback-threshold 0.20 \
  --abs-ret-day-max 0.15 \
  --report output/reports/backtest_health_report.json
```

**判定内容**（失敗なら即 `SystemExit`）:
- ✅ すべてのリバランス日で `selected_count >= 53`
- ✅ `fallback_rate < 20%`
- ✅ 日次 `|portfolio_return| > 15%` の発生 0

**JSON構造への寛容性**:
- `rebalances`/`timeline`/`days`/`events` 等を自動検出
- `selected_count` / `selected` / `positions` 等を柔軟に判定
- 構造が異なる場合でも集計結果を `/tmp/bt_health_report.json` に出力

---

## 📊 想定レンジ（クリーン後の目安）

| 指標 | 目標値 |
|------|--------|
| **Total Return (2024)** | 20–100% |
| **Sharpe Ratio（コスト込み）** | 0.5–2.0 |
| **MaxDD** | 10–30% |
| **Fallback Rate** | <20% |
| **供給銘柄数** | 毎回 ≥53銘柄 |

---

## 🔧 実装のポイント

### filter_dataset_quality.py

**機能**:
- 列名の自動検出: `code/ticker`・`timestamp/trading_date/date`・`adj_close/close`・`volume/turnover`
- 整列・重複排除: `code × timestamp` で安定ソート
- **ret_1d**: なければ `pct_change(price)` で生成
- **ADV60（当日除外）**: `rolling(60).mean().shift(1)`
- **フィルタ**: `price >= min_price`, `|ret_1d| <= max_ret_1d`, `adv60 >= min_adv`
- **Freeze検出**: 同値連続 ≥5 日のシーケンス数と日数を pre/post で算出

**出力**:
- `quality_report.json`: pre/post 指標（失敗時も必ず出力）
- クリーンデータ: `output/ml_dataset_clean.parquet`

### autosupply_utils.py

**機能**:
- `autosupply_k_ratio()`: 供給率を動的計算（15%～100%）
- `ensure_k_min()`: 絶対最低値（53銘柄）を保証
- `calculate_dynamic_k()`: ワンステップで動的k値を計算

**使用例**:
```python
from apex_ranker.scripts.autosupply_utils import calculate_dynamic_k

k_pick = calculate_dynamic_k(candidate_count=100, target_top_k=35)
# → 53銘柄（100 × 0.53）
```

### check_backtest_output.py

**機能**:
- JSONキーの自動探索: `rebalances`/`timeline`/`days` 等
- `selected_count` の推定: 数値または配列長から判定
- `fallback_used` のデフォルト処理: 無い場合は 0 とみなす
- `portfolio_return` の柔軟な抽出: 複数キー候補から検索

**出力**:
- `bt_health_report.json`: 統計と健全性判定
- SystemExit: 品質ゲート失敗時（CI/CD統合可能）

---

## 📂 ファイル構成

```
apex-ranker/
├── scripts/
│   ├── path_constants.py              # 🆕 パス定数（全スクリプトで共通使用）
│   ├── filter_dataset_quality.py      # Step 1: データ品質ゲート
│   ├── autosupply_utils.py            # Step 2: 銘柄供給自動調整
│   ├── check_backtest_output.py       # Step 3: BT結果検証
│   └── backtest_smoke_test.py         # 既存バックテストスクリプト
├── README_QUALITY_GATE.md             # 本ドキュメント
└── configs/
    ├── v0_pruned.yaml                 # バックテスト設定（pruned model）
    └── v0_base.yaml                   # バックテスト設定（enhanced model）

output/
├── ml_dataset_latest_full.parquet     # 元データ（DATASET_RAW）
├── ml_dataset_clean.parquet           # クリーンデータ（DATASET_CLEAN）
├── backtest/
│   ├── backtest_result.json           # BT結果（BACKTEST_JSON）
│   ├── backtest_daily.csv             # 日次データ（BACKTEST_DAILY_CSV）
│   └── backtest_trades.csv            # 取引履歴（BACKTEST_TRADES_CSV）
└── reports/
    ├── quality_report.json            # 品質レポート（QUALITY_REPORT）
    └── backtest_health_report.json    # 健全性レポート（BACKTEST_HEALTH_REPORT）

models/
├── apex_ranker_v0_pruned.pt           # Pruned model（MODEL_PRUNED）
└── apex_ranker_v0_enhanced.pt         # Enhanced model（MODEL_ENHANCED）
```

---

## 🔧 パス定数の使用方法

**問題**: 以前のバージョンではパス名が不一致（`ml_dataset_clean.parquet` vs `ml_dataset_latest_clean.parquet`）でCI/運用エラーが発生しやすかった。

**解決**: `scripts/path_constants.py` で全パスを一元管理。

### 基本的な使い方

```bash
# パス確認（存在チェック付き）
python apex-ranker/scripts/path_constants.py

# 出力例:
# ✅ DATASET_RAW          = /workspace/gogooku3/output/ml_dataset_latest_full.parquet
# ✅ DATASET_CLEAN        = /workspace/gogooku3/output/ml_dataset_clean.parquet
# ❌ BACKTEST_JSON        = /workspace/gogooku3/output/backtest/backtest_result.json
```

### Python スクリプトでの使用

```python
from scripts.path_constants import DATASET_RAW, DATASET_CLEAN, QUALITY_REPORT

# Step 1: Data quality gate
filter_dataset_quality(
    input_path=DATASET_RAW,
    output_path=DATASET_CLEAN,
    report_path=QUALITY_REPORT
)
```

### 環境変数でのオーバーライド

```bash
# カスタムパスを使用したい場合
export DATASET_RAW=/custom/path/to/dataset.parquet
export DATASET_CLEAN=/custom/path/to/clean.parquet

# 環境変数が優先される
python apex-ranker/scripts/filter_dataset_quality.py \
  --input $DATASET_RAW \
  --output $DATASET_CLEAN
```

---

## 🛠️ カスタマイズ

### データ列名が特殊な場合

`filter_dataset_quality.py` 冒頭の候補配列に追加:

```python
# 例: "stock_code" という列名を追加
CAND_CODE = ["code", "Code", "ticker", "symbol", "stock_code"]
```

### 閾値の調整

```bash
# より厳しい品質基準
python apex-ranker/scripts/filter_dataset_quality.py \
  --min-price 200 \              # 200円以上
  --max-ret-1d 0.10 \            # |ret_1d| <= 10%
  --min-adv 100000000 \          # 1億円以上
  --freeze-abs-max 50            # フリーズ50日以下
```

### autosupply パラメータ

```python
# より保守的な設定（70銘柄最低供給）
k_pick = calculate_dynamic_k(
    candidate_count=len(candidates),
    target_top_k=35,
    alpha=2.0,           # 35 × 2.0 = 70最低供給
    hard_floor=70,
)
```

---

## ⚠️ 重要な注意事項

### ADV60の当日除外（ルックアヘッド防止）

```python
# ✅ 正しい（当日除外）
adv60_trailing = rolling_mean(turnover, 60).shift(1)

# ❌ 間違い（当日含む = ルックアヘッド）
adv60_wrong = rolling_mean(turnover, 60)  # shift なし
```

### フリーズは"削除"ではなく"減少確認"

- 完全削除ではなく、**pre → post で削減されることを確認**
- 必要なら `--freeze-abs-max` 等で厳しく調整

---

## 🧪 テスト実行

```bash
# autosupply_utils の単体テスト
cd apex-ranker
python scripts/autosupply_utils.py

# 期待される出力:
# 十分な候補（100銘柄）:
#   - 候補数: 100
#   - 供給率: 53.0%
#   - 選定数: 53
```

---

## 📚 参考資料

- **ルックアヘッド防止**: `shift(1)` による当日除外が最重要
- **動的k調整**: 銘柄供給不足を自動補正（fallback_rate削減）
- **品質ゲート**: 異常データの混入を事前防止（BT精度向上）

---

**作成日**: 2025-11-02
**バージョン**: 1.0.0
**対応プロジェクト**: APEX-Ranker v0.1.0+
