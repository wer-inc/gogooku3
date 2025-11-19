# J-Quants Statements API カラム損失調査レポート

**調査日**: 2025-11-18
**対象**: TTM (Trailing Twelve Months) 特徴量のNULL率問題
**結論**: API制約ではなく、実装バグによる96カラムの破棄が原因

---

## 📊 問題の概要

### 初期状態
TTM関連の18カラム中、12カラムが100% NULL:
- ✅ 改善済み (34.7% NULL): `fs_revenue_ttm`, `fs_net_income_ttm`, `fs_total_assets_ttm`, `fs_equity_ttm`
- ❌ 100% NULL: `fs_op_profit_ttm`, `fs_cfo_ttm`, `fs_roa_ttm`, `fs_roe_ttm` など12カラム

### 当初の仮説
J-Quants APIが必要なカラムを提供していない（API制約）

---

## 🔍 調査プロセス

### Phase 1: API仕様確認
**ドキュメント**: `/workspace/gogooku3/gogooku5/docs/external/jquants_api/j-quants-ja/api-reference/statements/index.md`

**発見**:
- ✅ `OperatingProfit` (営業利益): 定義あり
- ✅ `TotalAssets` (総資産): 定義あり
- ✅ `Equity` (純資産): 定義あり
- ✅ `CashFlowsFromOperatingActivities` (営業CF): 定義あり

→ **APIドキュメント上は全カラム提供されるはず**

### Phase 2: 実際のAPI接続テスト
**テストスクリプト**: `/tmp/test_jquants_statements_api.py`

**実行結果**:
```json
{
  "NetSales": "4434000000",
  "OperatingProfit": "1891000000",
  "OrdinaryProfit": "2316000000",
  "Profit": "1407000000",
  "TotalAssets": "279689000000",
  "Equity": "40525000000",
  "CashFlowsFromOperatingActivities": "",
  "CashFlowsFromInvestingActivities": "",
  "CashFlowsFromFinancingActivities": "",
  ...
  // 合計107カラム
}
```

**発見**: ✅ APIは107カラムを返す（全必要カラムを含む）

### Phase 3: Raw Data検証
**保存されたファイル**: `/workspace/gogooku3/output_g5/raw/earnings/earnings_2023-11-27_2024-06-30_20251118_145547.parquet`

**カラム数**: 11カラムのみ
```
- Code
- TypeOfDocument
- FiscalYear
- AccountingStandard
- DisclosedDate
- DisclosedTime
- NetSales
- Profit
- NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock
- NumberOfTreasuryStockAtTheEndOfFiscalYear
- AverageNumberOfShares
```

**損失**: 107 - 11 = **96カラムが破棄**

---

## 🎯 根本原因の特定

### 問題箇所
**ファイル**: `/workspace/gogooku3/gogooku5/data/src/builder/api/jquants_async_fetcher.py`

**Lines 1541-1575**: `target_labels` 辞書で9カラムのみ定義
```python
target_labels: dict[str, tuple[str, ...]] = {
    "NetSales": ("net sales", "netsales", "revenue", "sales", "operating revenue"),
    "OperatingProfit": ("operating profit", "operating income", "operating loss"),
    "Profit": ("profit", "profit (loss)", "net income", "net profit"),
    "Equity": ("equity attributable to owners of parent", "total equity", ...),
    "TotalAssets": ("total assets",),
    "CashAndCashEquivalents": ("cash and cash equivalents",),
    "InterestBearingDebt": ("interest-bearing debt", ...),
    "NetCashProvidedByOperatingActivities": ("net cash provided by ...", ...),
    "PurchaseOfPropertyPlantAndEquipment": ("purchase of property, ...", ...),
}
```

**Lines 1617-1628**: `_extract_financials` 関数
```python
def _extract_financials(fs_dict: dict[str, Any]) -> dict[str, Any]:
    lower_map = {k: set(v) for k, v in target_labels.items()}
    flat: dict[str, Any] = {}
    for key, value in _iter_items(fs_dict):
        norm_key = key.strip().lower()
        for target, aliases in lower_map.items():
            if norm_key in aliases and target not in flat:
                flat[target] = value  # ← target_labels に含まれるもののみ保存
        # ... share columns handling ...
    return flat  # ← 9カラム（+ share columns）のみ返却
```

**Lines 1675-1688**: Base情報 + 抽出データ = 最終出力
```python
base = {
    "Code": ...,
    "TypeOfDocument": ...,
    "FiscalYear": ...,
    "AccountingStandard": ...,
    "DisclosedDate": ...,
    "DisclosedTime": ...,
}
flat = _extract_financials(item)  # ← 9カラム（実際は5カラム程度がマッチング）
base.update(flat)
rows.append(base)
```

### 欠落している重要カラム

**TTM計算に必要だが定義されていないカラム**:
1. ❌ `OrdinaryProfit` (経常利益)
2. ❌ `CashFlowsFromInvestingActivities` (投資CF)
3. ❌ `CashFlowsFromFinancingActivities` (財務CF)
4. ⚠️ `CashFlowsFromOperatingActivities` (営業CF) - alias不一致

**Alias ミスマッチの詳細**:
- **コード定義** (Line 1566-1569):
  ```python
  "NetCashProvidedByOperatingActivities": (
      "net cash provided by (used in) operating activities",
      "cash flows from operating activities",
  )
  ```
- **API実際のフィールド名**: `"CashFlowsFromOperatingActivities"`
- **正規化後の比較**:
  - コード: `"cashflowsfromoperatingactivities"` (スペースなし)
  - Alias: `"cash flows from operating activities"` (スペースあり)
  - → 不一致のため取得失敗

**配当・会計変更フラグなど90+カラム**:
- `ResultDividendPerShare*` (配当実績)
- `ForecastDividendPerShare*` (配当予想)
- `ChangesBasedOnRevisionsOfAccountingStandard` (会計基準変更)
- `MaterialChangesInSubsidiaries` (子会社変更)
- など

---

## 💡 解決策の提案

### Option 1: 全カラム保存方式（推奨）

**変更内容**:
1. `_extract_financials` 関数を削除
2. APIレスポンスをそのまま保存（107カラム全て）
3. Feature engineeringの段階で必要カラムを選択

**メリット**:
- ✅ Raw dataは完全な状態で保存（再現性）
- ✅ 将来の機能拡張に対応しやすい（柔軟性）
- ✅ APIの仕様変更に強い（保守性）
- ✅ デバッグが容易（全データ参照可能）
- ✅ 107カラムすべてに有用な情報が含まれている

**デメリット**:
- ⚠️ Parquetファイルサイズが増加（11カラム → 107カラム、約10倍）
- ⚠️ 既存のfeature engineering codeの修正が必要

**実装の影響範囲**:
```python
# Before (Lines 1675-1688)
flat = _extract_financials(item)  # 9カラムのみ抽出
base.update(flat)

# After
# item 全体をそのまま使用（107カラム）
row = {
    "Code": item.get("LocalCode") or item.get("Code"),
    "TypeOfDocument": item.get("TypeOfDocument"),
    "FiscalYear": item.get("FiscalYear"),
    ...
}
# 全フィールドをそのまま追加
row.update({k: v for k, v in item.items() if k not in row})
```

### Option 2: target_labels拡張

**変更内容**:
1. `target_labels` に不足カラムを追加定義（90+カラム）
2. Aliasマッピングを修正

**メリット**:
- ✅ 既存コード変更が最小限
- ✅ Parquetファイルサイズは抑制可能

**デメリット**:
- ❌ 全107カラムのaliasマッピングを手動定義（保守コスト高）
- ❌ API仕様変更時に対応漏れのリスク
- ❌ 将来の拡張に弱い

---

## 📈 影響範囲の分析

### Raw Dataファイルサイズ
**現状** (11カラム):
```bash
$ ls -lh output_g5/raw/earnings/*.parquet
# 約50-200MB (3-5年分)
```

**Option 1実装後** (107カラム):
```
約500MB-2GB (推定10倍、圧縮効率により変動)
```

**ストレージへの影響**:
- ディスク使用量: 588T available / 2113T total (75% used)
- 増加分: +200MB~1.8GB → 影響は微小（0.0003%未満）

### Feature Engineering Code
**修正が必要なファイル**:
1. `/workspace/gogooku3/gogooku5/data/src/builder/features/fundamentals/breakdown_asof.py`
   - TTM計算ロジック（現在はNULL値を扱っている）
   - 新カラムを使用するように修正

2. `/workspace/gogooku3/gogooku5/data/src/builder/features/fundamentals/engineer.py`
   - 財務特徴量生成
   - 新カラムへのアクセス追加

---

## ✅ 推奨アクション

### 1. Option 1（全カラム保存）を採用
**理由**:
- Raw dataの完全性が最重要
- ストレージコストは無視できるレベル
- 将来の拡張性を確保

### 2. 段階的な実装
**Phase 1**: jquants_async_fetcher.py の修正
- `_extract_financials` を削除
- 全カラム保存に変更

**Phase 2**: Raw data再生成
- 2023-2025のデータを再取得
- 107カラム版parquetを生成

**Phase 3**: Feature engineering修正
- `breakdown_asof.py` のTTM計算を新カラムに対応
- 追加の財務特徴量を実装

**Phase 4**: 検証
- TTM NULL率の再検証
- 既存特徴量への影響確認

### 3. バックアップ戦略
- 現在の11カラム版Raw dataを保持（ロールバック用）
- 新107カラム版と並行運用して検証

---

## 📋 次のステップ

1. **Phase 2.4完了**: このレポートをドキュメント化（本ファイル）
2. **Phase 3.1開始**: jquants_async_fetcher.py の修正実装
3. **Phase 3.2**: テストスクリプトで動作確認
4. **Phase 3.3**: 2023-2025データの再生成
5. **Phase 3.4**: TTM NULL率の再検証
6. **Phase 3.5**: 結果レポート作成

---

## 📎 参考ファイル

- API Response サンプル: `/tmp/jquants_statements_api_response.json`
- テストスクリプト: `/tmp/test_jquants_statements_api.py`
- 問題のコード: `/workspace/gogooku3/gogooku5/data/src/builder/api/jquants_async_fetcher.py:1526-1733`
- Raw Data サンプル: `/workspace/gogooku3/output_g5/raw/earnings/*.parquet`

---

## 📝 結論

**TTM NULL問題はAPI制約ではなく、実装バグ（96カラムの意図的な破棄）が原因**

J-Quants APIは107カラムの完全なデータを提供しているが、`_extract_financials` 関数が9カラムのみを抽出する設計になっていたため、96カラムが失われていた。

**推奨対応**: 全カラム保存方式（Option 1）を採用し、Raw dataの完全性を確保する。
