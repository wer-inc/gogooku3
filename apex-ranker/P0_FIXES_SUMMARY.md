# APEX-Ranker P0修正完了サマリー

**実装日**: 2025-11-02
**ステータス**: ✅ P0-1, P0-2 完了
**次ステップ**: P1改善 → 統合テスト

---

## ✅ P0-1: effective_kmin 対応（完了）

### 問題

**供給判定の不一致**: `candidate_count < 53` のケースで、固定値 `k_min=53` による判定が破綻。

**具体例**:
```python
# 候補銘柄数が30の日
candidate_count = 30  # その日の全銘柄数
selected_count = 30   # 全銘柄を選定
k_min = 53            # 固定最低値

# 従来の判定（❌ 失敗）
if selected_count < k_min:  # 30 < 53 → 失敗!
    raise ValueError("Insufficient selection")
```

### 解決策

`effective_kmin = min(k_min, candidate_count)` による現実的な供給条件判定。

**実装箇所**: `apex-ranker/scripts/check_backtest_output.py`

#### 1. candidate_count 抽出関数（Line 74-91）

```python
def extract_candidate_count(entry: dict) -> int | None:
    """候補銘柄数を抽出（effective_kmin計算用）"""
    # 直接のキー検出
    candidates = ["candidate_count", "num_candidates", "universe_size", "n_candidates"]
    for key in candidates:
        if key in entry:
            val = entry[key]
            if isinstance(val, int):
                return val

    # selected + dropped から推定
    if "dropped" in entry and "selected_count" in entry:
        dropped = entry["dropped"]
        selected = entry["selected_count"]
        if isinstance(dropped, int) and isinstance(selected, int):
            return selected + dropped

    return None
```

#### 2. 統計収集の拡張（Line 139, 151-154）

```python
# candidate_count の収集
for idx, entry in enumerate(rebalances):
    # ... (selected_count 収集)

    # candidate_count 抽出
    cand_count = extract_candidate_count(entry)
    if cand_count is not None:
        candidate_counts.append(cand_count)
```

#### 3. 検証ロジックの書き換え（Line 174-216）

```python
# Check 1: selected_count >= effective_k_min
for i, sc in enumerate(selected_counts):
    # candidate_count がある場合は effective_kmin を計算
    if i < len(candidate_counts) and candidate_counts[i] is not None:
        effective_kmin = min(args.k_min, candidate_counts[i])
        if sc < effective_kmin:
            violations.append(sc)
            violation_details.append(
                f"#{i}: selected={sc}, effective_kmin={effective_kmin}, candidates={candidate_counts[i]}"
            )
    else:
        # candidate_count がない場合は従来通り k_min で判定
        if sc < args.k_min:
            violations.append(sc)
            violation_details.append(
                f"#{i}: selected={sc}, k_min={args.k_min} (candidate_count不明)"
            )
```

### 動作確認

**正常ケース**:
```
# 30銘柄しか候補がない日
candidate_count=30, selected_count=30
effective_kmin = min(53, 30) = 30
30 >= 30 → ✅ Pass
```

**異常ケース**:
```
# 30銘柄候補だが27銘柄しか選定できなかった
candidate_count=30, selected_count=27
effective_kmin = min(53, 30) = 30
27 < 30 → ❌ Fail (正しく検出)
```

---

## ✅ P0-2: パス名統一（環境変数化）（完了）

### 問題

**パス名の不一致によるCI/運用エラー**:

```bash
# README例
--output output/ml_dataset_clean.parquet

# filter_dataset_quality.py 例
--output output/ml_dataset_latest_clean.parquet  # ❌ 不一致!

# backtest_smoke_test.py 例
--data output/ml_dataset_latest_full_filled.parquet  # ❌ 不一致!
```

→ **Step 2 がこける**: Step 1の出力を見つけられない

### 解決策

**パス定数ファイル**による一元管理: `scripts/path_constants.py`

#### 1. パス定数ファイル作成（NEW）

**ファイル**: `apex-ranker/scripts/path_constants.py` (100行)

```python
#!/usr/bin/env python3
"""
Path constants for APEX-Ranker quality management pipeline

Centralized path definitions to prevent CI/CD failures.
"""
from pathlib import Path
import os

# Dataset paths (3-step quality pipeline)
DATASET_RAW = "output/ml_dataset_latest_full.parquet"
DATASET_CLEAN = "output/ml_dataset_clean.parquet"

# Backtest output paths
BACKTEST_JSON = "output/backtest/backtest_result.json"
BACKTEST_DAILY_CSV = "output/backtest/backtest_daily.csv"
BACKTEST_TRADES_CSV = "output/backtest/backtest_trades.csv"

# Quality report paths
QUALITY_REPORT = "output/reports/quality_report.json"
BACKTEST_HEALTH_REPORT = "output/reports/backtest_health_report.json"

# Model paths
MODEL_PRUNED = "models/apex_ranker_v0_pruned.pt"
MODEL_ENHANCED = "models/apex_ranker_v0_enhanced.pt"

# Environment variable overrides (optional)
DATASET_RAW = os.getenv("DATASET_RAW", DATASET_RAW)
DATASET_CLEAN = os.getenv("DATASET_CLEAN", DATASET_CLEAN)
```

**機能**:
- すべてのパスを一箇所で定義
- 環境変数オーバーライド対応
- 存在チェック機能（`python scripts/path_constants.py`）

#### 2. スクリプトへの統合

**filter_dataset_quality.py** (Line 34-41):
```python
# パス定数をインポート（デフォルト値として使用）
try:
    from path_constants import DATASET_CLEAN, DATASET_RAW, QUALITY_REPORT
except ImportError:
    # フォールバック: path_constants.py が見つからない場合
    DATASET_RAW = "output/ml_dataset_latest_full.parquet"
    DATASET_CLEAN = "output/ml_dataset_clean.parquet"
    QUALITY_REPORT = "output/reports/quality_report.json"
```

**argparse デフォルト値** (Line 171-206):
```python
parser.add_argument(
    "--input",
    default=DATASET_RAW,  # 定数を使用
    help=f"入力parquetファイル（デフォルト: {DATASET_RAW}）",
)
parser.add_argument(
    "--output",
    default=DATASET_CLEAN,  # 定数を使用
    help=f"出力parquetファイル（デフォルト: {DATASET_CLEAN}）",
)
```

**check_backtest_output.py** (Line 30-36, 107-123):
- 同様のパターンで `BACKTEST_JSON`, `BACKTEST_HEALTH_REPORT` を使用

#### 3. README更新

**README_QUALITY_GATE.md**:
- Step 1, 2, 3 の全コマンド例を統一パスに修正
- ファイル構成図に定数名を併記
- パス定数の使用方法セクションを追加（Line 197-239）

**新セクション**:
```markdown
## 🔧 パス定数の使用方法

### 基本的な使い方
python apex-ranker/scripts/path_constants.py

### Python スクリプトでの使用
from scripts.path_constants import DATASET_RAW, DATASET_CLEAN

### 環境変数でのオーバーライド
export DATASET_RAW=/custom/path/to/dataset.parquet
```

### 動作確認

```bash
# パス確認
$ python scripts/path_constants.py
======================================================================
📁 APEX-Ranker Path Constants
======================================================================
✅ DATASET_RAW               = /workspace/gogooku3/output/ml_dataset_latest_full.parquet
❌ DATASET_CLEAN             = /workspace/gogooku3/output/ml_dataset_clean.parquet
✅ MODEL_PRUNED              = /workspace/gogooku3/models/apex_ranker_v0_pruned.pt
...

# インポートテスト
$ python -c "import sys; sys.path.insert(0, 'scripts'); from path_constants import DATASET_RAW; print(DATASET_RAW)"
/workspace/gogooku3/output/ml_dataset_latest_full.parquet
```

---

## 📊 修正ファイル一覧

| ファイル | 変更内容 | 行数 |
|---------|---------|------|
| `scripts/path_constants.py` | 🆕 パス定数ファイル | +100 |
| `scripts/check_backtest_output.py` | effective_kmin 実装 + パス定数統合 | 修正 |
| `scripts/filter_dataset_quality.py` | パス定数統合 | 修正 |
| `README_QUALITY_GATE.md` | パス統一 + 使用方法追加 | 修正 |

---

## 🎯 CI/CD統合のベストプラクティス

### Step 1: データ品質ゲート

```bash
# 引数省略可能（デフォルトパス使用）
python apex-ranker/scripts/filter_dataset_quality.py \
  --min-price 100 \
  --max-ret-1d 0.15 \
  --min-adv 50000000

# または明示的に
python apex-ranker/scripts/filter_dataset_quality.py \
  --input $DATASET_RAW \
  --output $DATASET_CLEAN \
  --report $QUALITY_REPORT
```

### Step 2: バックテスト

```bash
python apex-ranker/scripts/backtest_smoke_test.py \
  --data $DATASET_CLEAN \
  --output $BACKTEST_JSON \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

### Step 3: 健全性チェック

```bash
python apex-ranker/scripts/check_backtest_output.py \
  --input $BACKTEST_JSON \
  --report $BACKTEST_HEALTH_REPORT \
  --k-min 53
```

---

## 🔜 次ステップ（P1改善）

P0修正完了により、CI/CDでの品質ゲートが安定運用可能になりました。次は以下のP1改善に進みます:

1. **AxisDecider ヒューリスティック改善**: 小規模候補数での性能向上
2. **ADV/Turnover 優先順位明確化**: J-Quants API の TradingValue 優先使用
3. **エラーハンドリング強化**: 429/5xx エラー時の自動フォールバック
4. **統合テスト**: 全ステップの連結動作確認

---

**作成日**: 2025-11-02
**バージョン**: 1.0.0
**対応プロジェクト**: APEX-Ranker v0.1.0+
