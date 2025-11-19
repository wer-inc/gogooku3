# その他NULL特徴量調査レポート

**調査日**: 2025-11-18
**対象**: TTM以外のAPI由来特徴量のカラム破棄バグ
**方法**: jquants_async_fetcher.pyのコード解析

---

## 📋 調査対象エンドポイント

J-Quants APIの主要エンドポイント:
1. `/fins/statements` (財務諸表) - **✅ 修正済み** (TTM問題)
2. `/fins/dividend` (配当)
3. `/markets/trades_spec` (信用取引)
4. `/fins/announcement` (決算発表)
5. `/indices/topix` (TOPIX)
6. `/option/index_option` (オプション)
7. `/markets/short_selling` (空売り)
8. `/derivatives/futures` (先物)

---

## 🔍 検出結果サマリー

### 1. target_labels辞書（TTM問題パターン）
- **検出数**: 1個 (Line 1542)
- **状態**: ✅ **修正済み** (2025-11-18)
- **詳細**: `_extract_all_financials`関数に置き換え、107カラム全て保存

### 2. カラムフィルタリング関数
- **_extract_*関数**: 2個
  - `_extract_financials` (Line 1618) - ⚠️ **DEPRECATED** (96カラム破棄)
  - `_extract_all_financials` (Line 1638) - ✅ **現在使用中** (全107カラム保存)

- **_normalize_*関数**: 6個 (詳細調査が必要)
  - `_normalize_futures_data` (Line 2346)
  - `_normalize_index_option_data` (Line 2693)
  - `_normalize_earnings_announcement_data` (Line 3120)
  - `_normalize_sector_short_selling_data` (Line 3274)
  - `_normalize_short_selling_data` (Line 3333)
  - `_normalize_short_selling_positions_data` (Line 3424)

---

## 📊 エンドポイント別分析

### ✅ /fins/statements (財務諸表) - 修正済み

**状態**: ✅ **問題解決済み**

**問題**:
- Line 1542の`target_labels`辞書が9カラムのみ定義
- `_extract_financials`関数が96カラムを破棄
- TTM特徴量の12カラムが100% NULL

**修正内容**:
- `_extract_all_financials`関数を新規追加 (Line 1638)
- 全107カラムをそのまま保存
- 両パス（並列・逐次）で新関数を使用

**ドキュメント**: `gogooku5/docs/API_COLUMN_LOSS_INVESTIGATION_20251118.md`

---

### 🔍 /derivatives/futures (先物) - 要調査

**関数**: `_normalize_futures_data` (Line 2346)

**懸念点**:
- `_normalize_*`関数が`.select()`や`.drop()`を使用している可能性
- 先物APIは複数のカテゴリ（TOPIXF, NK225F, JN400F, REITF）を返す
- 20-68カラムが返されるはずだが、feature engineeringで使用しているのは一部のみ

**次のステップ**:
- Line 2346周辺のコードを読んで`.select()`/`.drop()`の有無を確認
- APIドキュメントと実装を照合

---

### 🔍 /option/index_option (オプション) - 要調査

**関数**: `_normalize_index_option_data` (Line 2693)

**懸念点**:
- オプションデータは複雑な構造（コール/プット、複数限月）
- カラムフィルタリングが行われている可能性

**次のステップ**:
- Line 2693周辺のコードを確認
- オプション関連特徴量のNULL率を調査

---

### 🔍 /fins/announcement (決算発表) - 要調査

**関数**: `_normalize_earnings_announcement_data` (Line 3120)

**懸念点**:
- 決算発表日時、予想値などの重要データを含む
- 一部カラムのみ抽出している可能性

**次のステップ**:
- Line 3120周辺のコードを確認
- `earnings_*`特徴量のNULL率を調査

---

### 🔍 /markets/short_selling (空売り) - 要調査

**関数**:
- `_normalize_short_selling_data` (Line 3333)
- `_normalize_short_selling_positions_data` (Line 3424)
- `_normalize_sector_short_selling_data` (Line 3274)

**懸念点**:
- 3つの空売り関連関数が存在
- 個別銘柄空売り、空売りポジション、セクター空売りのデータ
- データ正規化の過程でカラム損失の可能性

**次のステップ**:
- 各関数のコードを確認
- `short_*`および`sector_short_*`特徴量のNULL率を調査

---

## 🎯 優先度付き調査計画

### 優先度 HIGH（即時調査）

1. **`_normalize_futures_data`** (Line 2346)
   - 理由: 先物データは20-68カラムと規模が大きい
   - 影響: 先物特徴量が100% NULLの可能性

2. **`_normalize_index_option_data`** (Line 2693)
   - 理由: オプションデータも複雑で多数のカラムを含む
   - 影響: オプション特徴量のNULL率が高い可能性

### 優先度 MEDIUM（計画的調査）

3. **`_normalize_earnings_announcement_data`** (Line 3120)
   - 理由: 決算発表は重要なイベントデータ
   - 影響: `earnings_*`特徴量への影響

4. **空売り関連3関数** (Lines 3274, 3333, 3424)
   - 理由: 3つの関数が独立して処理
   - 影響: `short_*`および`sector_short_*`特徴量への影響

### 優先度 LOW（長期改善）

5. **その他のエンドポイント**
   - `/markets/trades_spec` (信用取引)
   - `/indices/topix` (TOPIX)
   - `/fins/dividend` (配当)

---

## 💡 推奨アクション

### Phase 4.2: _normalize_*関数の詳細調査

各関数について以下を確認:

1. **カラムフィルタリングの有無**
   ```python
   # 懸念パターン1: .select()による明示的な選択
   df = df.select(["col1", "col2", "col3"])  # 他のカラムが破棄される

   # 懸念パターン2: .drop()による削除
   df = df.drop(["unwanted_col"])  # 意図しない削除の可能性

   # 懸念パターン3: 辞書マッピングによるフィルタリング
   columns_map = {"api_col_name": "our_col_name"}  # target_labelsと同パターン
   ```

2. **APIドキュメントとの照合**
   - 各エンドポイントのAPIドキュメントで全カラムリストを確認
   - 実装で保存されているカラム数と比較
   - 差分があれば損失カラムをリストアップ

3. **NULL率の検証**
   - NULL_COLUMNS_REPORT.mdで該当特徴量のNULL率を確認
   - 95%以上NULLの特徴量は特に注意

### Phase 4.3: 修正方針の決定

**TTM修正と同じアプローチ**:
1. カラムフィルタリングが見つかった場合
2. `_normalize_all_*`関数を新規追加（全カラム保存）
3. 既存関数は`DEPRECATED`としてコメント化
4. 両パス（並列・逐次）で新関数を使用

### Phase 4.4: Raw Data再生成

修正が必要なエンドポイントについて:
1. 修正後のコードで2023-2025データを再生成
2. NULL率の改善を検証
3. Feature engineeringで新カラムを活用

---

## 📝 次のステップ

### 今日中（2025-11-18）

1. ✅ `_normalize_*`関数の検出完了
2. ⏳ **Phase 4.2開始**: 各関数のコードを読んでフィルタリングの有無を確認
3. ⏳ **NULL率との照合**: 対応する特徴量のNULL率を確認

### 明日以降

4. 問題が見つかった関数の修正実装
5. Raw data再生成
6. NULL率の改善検証
7. 最終レポート作成

---

## 🔗 関連ドキュメント

- TTM修正レポート: `gogooku5/docs/API_COLUMN_LOSS_INVESTIGATION_20251118.md`
- NULL率レポート: `gogooku5/docs/NULL_COLUMNS_REPORT.md`
- Feature Schema: `gogooku5/data/schema/feature_schema_manifest.json`
- API仕様: `gogooku5/docs/external/jquants_api/`

---

**調査ステータス**: 🔄 **Phase 4.1 完了 → Phase 4.2 開始**
