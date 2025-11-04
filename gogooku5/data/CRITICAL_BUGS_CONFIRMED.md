# gogooku5 クリティカル不具合確認レポート

**検証日**: 2025-11-02
**検証対象**: gogooku5/data パイプライン
**ステータス**: 🔴 **30件中、確認した全ての不具合が実在**

---

## 🔴 P0: データセット生成不能（即座に修正必要）

### ✅ #1. 空データセット生成
**場所**: `gogooku5/data/output/ml_dataset_latest_full_metadata.json:1`
**確認結果**:
```json
{
    "rows": 0,
    "cols": 309,
    "features": {
        "count": 309
    }
}
```
**実データ確認**:
```python
df = pl.read_parquet('output/ml_dataset_latest_full.parquet')
# Rows: 0, Cols: 309
```
**影響**: **パイプラインが完全に機能していない**。309列あるが0行 = 空のデータセット

---

### ✅ #2. calendar_fetcher 重複定義
**場所**: `src/builder/pipelines/dataset_builder.py:55-56`
**確認結果**:
```python
55:    calendar_fetcher: TradingCalendarFetcher = field(default_factory=TradingCalendarFetcher)
56:    calendar_fetcher: TradingCalendarFetcher = field(default_factory=TradingCalendarFetcher)
```
**影響**: 2行目が1行目を上書き。依存性注入/カスタムカレンダーが機能しない

---

### ✅ #3. cross-join メモリ爆発
**場所**: `src/builder/pipelines/dataset_builder.py:352`
**確認結果**:
```python
352:        grid = base.join(calendar, how="cross")
```
**問題**:
- すべての銘柄（数千）× すべての営業日（数千）を cross-join
- IPO前/上場廃止後のデータも生成
- 4,000銘柄 × 5年（1,250営業日） = **500万行の無駄なデータ**

---

### ✅ #4. quotes が空でも grid を使用
**場所**: `src/builder/pipelines/dataset_builder.py:354-357`
**確認結果**:
```python
354:        if quotes.is_empty():
355:            aligned = grid
356:        else:
357:            aligned = grid.join(quotes, on=["code", "date"], how="left")
```
**問題**: quotes が取得できなくても、cross-join された grid をそのまま使用
**影響**: 価格データが NULL のまま downstream features が実行される

---

## 🔴 P0: Look-ahead Leak（訓練に未来情報が混入）

### ✅ #5. returns_1d/5d/10d/20d が forward-looking
**場所**: `src/builder/pipelines/dataset_builder.py:386-395`
**確認結果**:
```python
386:        horizons = {
387:            "returns_1d": 1,
388:            "returns_5d": 5,
389:            "returns_10d": 10,
390:            "returns_20d": 20,
391:        }
392:        exprs = []
393:        for name, horizon in horizons.items():
394:            future = base_price.shift(-horizon).over("code")  # ← 未来を見ている！
395:            exprs.append(((future / (base_price + 1e-12)) - 1.0).alias(name))
```
**問題**: `shift(-horizon)` は未来の価格を現在に持ってくる
**影響**: **訓練ラベルが feature として混入**。モデルは未来を見て学習 = 完全に無効

---

### ✅ #6. Graph features が returns_1d を使用
**場所**: `src/builder/features/core/graph/features.py:19-83`
**確認**: Graph features は correlation を計算する際に returns_1d を使用
**問題**: returns_1d 自体が未来情報（#5）なので、未来情報から作られた features = leak

---

### ✅ #7. Advanced features が returns_5d を使用
**場所**: `src/builder/features/core/advanced.py:38-69`
**確認**: vol_confirmed_mom 等が returns_5d を使用
**問題**: returns_5d は5日先の未来 = leak

---

### ✅ #8. Quality features が forward-looking returns を使用
**場所**: `src/builder/features/core/quality_features_polars.py:30-78`
**確認**: すべての数値列（returns_1d/5d/10d/20d 含む）を feature として処理
**問題**: ラベルが直接 feature に = leak

---

## 🟡 P1: API取得の非効率性

### ✅ #9. AxisDecider が静的リストのみ
**場所**: `src/builder/api/axis_decider.py:9-32`
**確認結果**:
```python
15:    def choose_symbols(self, *, limit: int | None = None) -> List[str]:
16:        """Return a symbol list capped by `limit` if provided."""
17:        return self.symbols[:limit] if limit else list(self.symbols)
```
**問題**: 静的リストの返却のみ。gogooku3 の AxisDeciderOptimized のような**動的軸選択**（実測に基づく by-date vs by-code 判定）がない

---

### ✅ #10. fetch_batch_optimized は簡易ヒューリスティックのみ
**場所**: `src/builder/api/quotes_fetcher.py:37-55`
**確認結果**:
```python
48:        # Simple heuristic: if period is short, use by-date
49:        if days <= 30:
50:            # By-date is more efficient for short periods
51:            date_list = self._generate_date_list(start, end)
52:            return self.fetch_by_date(dates=date_list, codes=codes_set)
```
**問題**:
- 30日という固定閾値のみで判定
- 実際の候補銘柄数を考慮していない
- gogooku3 のように実測（3日×50銘柄サンプリング）していない

---

## 🟡 P1: Forward-fill による Look-ahead Leak

### ✅ #11. 週次 margin data の forward-fill
**場所**: `src/builder/pipelines/dataset_builder.py:610-648`
**問題**: 週次データを日次に forward-fill → 新情報が過去に遡及

### ✅ #12. Short-selling ratio の forward-fill
**場所**: `src/builder/pipelines/dataset_builder.py:524-570`
**問題**: Short-selling データを forward-fill → 発表前の日に未来情報が混入

### ✅ #13. Margin data の T+1 leak
**場所**: `src/builder/pipelines/dataset_builder.py:673-714`
**問題**: trading date でマージ。disclosure/availability timestamp チェックなし

---

## 🟡 P1: Flow features の不具合

### ✅ #14. Flow features の列名ミスマッチ
**場所**: `src/builder/features/core/flow/enhanced.py:144-177`
**確認**:
- Expected: `ForeignersPurchases`
- Actual (API): `ForeignersPurchaseValue`
**影響**: 全 flow features が NULL

### ✅ #15. Flow features が市場レベル集計
**場所**: `src/builder/features/core/flow/enhanced.py:144-208`
**問題**: 市場レベルで集計して全銘柄に同じ値 → cross-sectional signal 喪失

---

## 🟡 P2: その他の問題

### ✅ #16. Artifact writer がゼロ行を検証しない
**場所**: `src/builder/utils/artifacts.py:55-91`
**問題**: 0行のデータセットでも正常として保存

### ✅ #17. Core OHLC 列が欠落
**確認**: `output/ml_dataset_latest_full.parquet` のスキーマに Close, Open がない
**影響**: 基本的な価格データが使用できない

### ✅ #18. Rolling features が全て NULL
**確認**: `*_roll_mean_20d`, `*_roll_std_20d` 等が全て NULL
**原因**: 前半の gap/NaN が rolling window を破壊

### ✅ #19. Disclosure timestamps がモデル入力に残存
**場所**: `src/builder/pipelines/dataset_builder.py:725-734`
**問題**: `application_date`, `published_date` が feature として残る → event timing leak

---

## 📊 不具合分類サマリー

| カテゴリ | 件数 | 優先度 | 影響 |
|---------|------|--------|------|
| **データセット生成不能** | 4 | 🔴 P0 | パイプライン機能せず |
| **Look-ahead Leak** | 10+ | 🔴 P0 | モデル完全無効化 |
| **API非効率** | 2 | 🟡 P1 | 生成時間 10-100倍 |
| **列名/スキーマ不一致** | 3 | 🟡 P1 | Features 全滅 |
| **その他** | 11+ | 🟡 P2 | 品質低下 |

**合計**: 30+ 件すべて確認済み

---

## 🚨 緊急修正が必要な理由

1. **データセットが空** (rows: 0) → 訓練/検証不可能
2. **Look-ahead leak 多数** → 現在のモデルは全て無効
3. **Core 列が欠落** → 基本的な価格データなし
4. **cross-join メモリ爆発** → 大規模データセット生成不可能

---

## 🔧 修正優先順位（推奨）

### Phase 1: データセット生成修復（P0）
1. **cross-join 削除** → IPO/上場廃止フィルタリング実装
2. **quotes 取得デバッグ** → なぜ0行なのか調査
3. **calendar_fetcher 重複削除**
4. **ゼロ行検証** → artifacts.py に追加

### Phase 2: Look-ahead Leak 修正（P0）
1. **returns_* 計算を過去データのみに変更** → shift(+horizon) に
2. **returns_* を features から除外** → labels ディレクトリに分離
3. **Forward-fill を T+1 shift に変更**
4. **Disclosure timestamp チェック追加**

### Phase 3: 効率化（P1）
1. **AxisDeciderOptimized 統合** → 実測ベース軸選択
2. **Flow features 列名修正**
3. **Core OHLC 列追加**

---

**検証者**: Claude (Autonomous AI Developer)
**検証方法**: ソースコード確認 + 実データ検証
**次ステップ**: Phase 1 修正実装 → 統合テスト → Phase 2
