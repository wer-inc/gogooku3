# J-Quants Premium 非リーク特徴量ガイド

本ドキュメントは J-Quants Premium の生データから、未来リークが無い形で特徴量を作成するための実務ルールをまとめたものです。DatasetBuilder・モデル両方が同じ原則で動くよう、工程別に手順と Polars 実装ひな形を記載します。

---

## 1. 絶対ルール

| ルール | 内容 |
| --- | --- |
| **as-of join** | Date 時点で参照できた値のみを使う。`Date >= 公表日 (AnnDate/DisclosedDate/PublishedDate)` の範囲で最も近い過去を backward join。 |
| **forward fill (公表日→営業日)** | 公表日以降を営業日ベースで前方保持。公表日前には絶対に値を埋めない。 |
| **区間属性** | 業種・上場区分などは `valid_from <= Date < valid_to` 条件で join。 |
| **ターゲット独立生成** | 特徴量完成後に別パスで log return などを作り、特徴量とは混在させない。 |
| **PurgedGroupTimeSeriesSplit + 20d embargo** | Group=Code で同一銘柄の伝播を遮断。Fold 境界前後は purge、検証開始から 20 営業日は学習から除外。 |
| **標準化は Fold 単位** | 全期間をまとめて fit しない。各 Fold の train 統計で変換し、valid/test へ適用。 |

---

## 2. データ種別別の扱い

### 2.1 prices（株価）

- 営業日グリッドの基礎テーブル。
- すべての as-of join の左側に使い、`Date × Code` の完全格子を作る。

### 2.2 財務（`fins/*`）

| ファイル | 公表日カラム | 補足 |
| --- | --- | --- |
| statements | `DisclosedDate` | 決算期では join しない。 |
| forecasts | `DisclosedDate` | 修正履歴を時間順に扱う。 |
| dividends | `DisclosedDate` | 予想/実績で日付が異なる。 |

### 2.3 需給（`markets/*`）

- `PublishedDate` を as-of キーに使う。
- 計測期間（日次・週次）の開始日は参照せず、公開されたタイミングのみを見る。

### 2.4 企業属性（`security/*`）

- 業種・指数採用・上場区分などは `valid_from/valid_to` の区間条件で join。
- 変更日の境界でリークしやすいので必ず挟み込む。

---

## 3. Polars 実装テンプレ

```python
import polars as pl

# 1) 営業日格子（土台）
prices = (
    pl.scan_parquet("prices/daily_quotes.parquet")
    .select(["Date", "Code", "Open", "High", "Low", "Close", "Volume"])
)

# 2) 財務 as-of join
fins = (
    pl.scan_parquet("fins/statements.parquet")
    .select(["Code", "DisclosedDate", "FiscalYear", "NetSales", "EPS"])
    .with_columns(pl.col("DisclosedDate").alias("asof"))
)

features = prices.join_asof(
    fins.sort("asof"),
    left_on="Date",
    right_on="asof",
    by="Code",
    strategy="backward",
)

# 3) 公表日以降の forward fill
features = (
    features
    .groupby("Code", maintain_order=True)
    .agg(pl.all().forward_fill())
    .explode(pl.all().exclude("Code"))
)

# 4) 需給系 (PublishedDate ベース)
flows = (
    pl.scan_parquet("markets/margin_trading.parquet")
    .select(["Code", "PublishedDate", "MarginRatio", "ShortBalance"])
    .with_columns(pl.col("PublishedDate").alias("asof"))
)

features = features.join_asof(
    flows.sort("asof"),
    left_on="Date",
    right_on="asof",
    by="Code",
    strategy="backward",
)

# 5) 区間 join（業種など）
sector = pl.scan_parquet("security/sector.parquet")
features = sector_join(features, sector)  # 区間 join ユーティリティ（後述）

# 6) ターゲットは別パスで生成
k = 5
targets = (
    features.select(["Code", "Date", "Close"])
    .sort(["Code", "Date"])
    .with_columns(
        (pl.col("Close").shift(-k) / pl.col("Close")).log().alias(f"target_{k}d")
    )
)
```

---

## 4. クロスバリデーション

- `PurgedGroupTimeSeriesSplit`（López de Prado の PurgedKFold）を実装し、`groups=Code` で指定。
- 各 Fold で `train_end < val_start` を保証。`purge_days`（Fold 境界の両側を削除）と `embargo_days`（デフォルト 20 営業日）を必ず適用。
- Embargo 区間は train から除外し、valid/test にのみ残す。

---

## 5. チェックリスト

- [ ] すべての非日次データに as-of キー（公表日）を付与したか。
- [ ] `asof <= Date` の条件で join しているか。
- [ ] forward fill は公表日以降のみを埋めているか。
- [ ] 区間属性は `valid_from/valid_to` 条件で join しているか。
- [ ] 標準化/正規化は Fold の train 区間統計のみで fit しているか。
- [ ] `PurgedGroupTimeSeriesSplit + 20d embargo` を実装し、`groups=Code` を渡しているか。
- [ ] 目的変数を特徴量パイプラインと独立させ、as-of join に混在させていないか。

---

## 6. 今後の実装コンポーネント

1. **as-of / forward-fill ヘルパー**  
   `builder/utils/asof.py` などに共通関数を実装し、全特徴量モジュールから利用する。
2. **PurgedGroupTimeSeriesSplit**  
   `builder/validation/cv.py` として提供し、モデル学習コードはこの分割器のみを使う。
3. **feature_pipeline.py**  
   DatasetBuilder の特定ステージを抜き出した単独スクリプトを用意し、J-Quants Premium の raw → 特徴量→ターゲットを一気通貫で再現できるようにする。

このガイドを守ることで、新規特徴量を追加するときも非リーク性を保ちながら開発できます。
