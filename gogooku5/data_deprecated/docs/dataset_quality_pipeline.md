# 高品質データセット構築ガイド（リーク防止付き）

本書は、J-Quants Premium の生データを「学習に使える高品質マスターデータ」へ変換する工場ラインの実装ガイドです。ターゲット定義から品質テストまでの 10 ステップを、gogooku5 DatasetBuilder で実装する際の勘所と Polars ベースのサンプルコードで示します。

---

## 10 ステップの全体像

1. **ターゲット定義** – 例: `log(P_{t+5}/P_t)`、TOP/BOTTOM ラベルなど。ターゲット決定が期間・必要特徴量を規定する。
2. **スキーマ設計** – 主キー `(date, code)` と型/単位（円・%・bool）を全テーブルで統一。
3. **取引カレンダー × 日次グリッド** – 公式営業日カレンダーで `trade_dates` を作成し、全銘柄×営業日グリッドに展開。
4. **株価 + コーポレートアクション正規化** – 調整後株価を基準にし、異常値をフィルタ。リターン計算も調整後値で行う。
5. **財務/需給/属性の非リーク as-of 結合** – 公表日/PublishedDate/valid_from を基準に後方 as-of / 区間 join。前倒し禁止。
6. **欠損・異常値・重複クリーニング** – 欠損率集計、winsorize、`(date, code)` 重複ゼロを保証。
7. **投資ユニバース定義** – 例: ADV60 上位 1000 銘柄、東証プライムのみ、上場 60 日未満除外。
8. **特徴量生成** – Rolling/lag は過去データのみ。クロスセクション Z 値は同日情報のみで計算。
9. **ラベル生成** – 調整後価格を用いた連続ラベルや rank-based 分類ラベルを独立パスで生成。
10. **CV + 品質テスト** – PurgedGroupTimeSeriesSplit（code group + 20d embargo）と自動品質チェックを実施。

---

## 主要ステップの実装ポイント

### スキーマ & 主キー

```python
dup = (
    features
    .groupby(["date", "code"])
    .len()
    .filter(pl.col("len") > 1)
)
assert dup.collect().height == 0, "(date, code) duplicate detected"
```

### 日次グリッド

```python
calendar = pl.DataFrame({"date": trading_dates})
grid = (
    calendar.join(listed.select(["code"]), how="cross")
    .filter(pl.col("date").is_between(pl.col("listed_date"), pl.col("delisted_date").fill_null("2100-12-31")))
)
```

### as-of ユーティリティ

- `builder.utils.asof.prepare_snapshot_pl`
- `builder.utils.asof.interval_join_pl`
- `builder.utils.asof.forward_fill_after_publication`

DatasetBuilder からこれらを呼べば T+1 公表と forward fill が統一されます。

### 欠損/異常値

```python
null_ratio = features.select(pl.all().null_count() / pl.count())
winsorized = features.with_columns(
    pl.col("volume").clip(lower_quantile, upper_quantile)
)
```

### ユニバース

```python
adv = (
    prices
    .with_columns((pl.col("adj_close") * pl.col("volume")).alias("value"))
    .groupby("code")
    .tail(60)
    .groupby("code")
    .agg(pl.col("value").mean().alias("adv60"))
)
universe = adv.sort("adv60", descending=True).head(1000)["code"]
```

### ターゲット

```python
k = 5
targets = (
    grid.sort(["code", "date"])
    .with_columns(
        (pl.col("adj_close").shift(-k) / pl.col("adj_close")).log().alias(f"target_ret_{k}d")
    )
)
```

### CV（PurgedGroupTimeSeriesSplit）

`builder.validation.cv.PurgedGroupTimeSeriesSplit` を利用し、`purge_days` と `embargo_days=20` を設定する。標準化/スケーリングは fold の train 部分だけで fit。

---

## 自動品質チェック

`gogooku5/data/tools/check_dataset_quality.py` を使うと、次の検査を一括で実施できます。

- `(date, code)` 主キーの重複検出
- 指定ターゲット列の欠損カウント
- 未来日データの混入
- as-of ペア（例: `fs_disclosed_date<=date`）のリーク検査
- 結果の JSON レポート出力

例:

```bash
python gogooku5/data/tools/check_dataset_quality.py \
  --dataset data/output/ml_dataset_latest_full.parquet \
  --targets target_ret_5d target_ret_20d \
  --asof-check fs_disclosed_date<=date \
  --asof-check margin_published_date<=date \
  --report quality_report.json \
  --fail-on-warning
```

CI や `tools/project-health-check.sh` にこのスクリプトを組み込み、ビルドごとに自動で品質テストを走らせてください。

> **DatasetBuilder から自動実行するには?**
>
> `.env` で `ENABLE_DATASET_QUALITY_CHECK=1` を設定すると、チャンク出力とフルマージの両方で品質チェッカーが走ります。既定値は
>
> - `DATASET_QUALITY_TARGETS=ret_prev_1d,ret_prev_5d,ret_prev_20d,ret_prev_60d`
> - `DATASET_QUALITY_ASOF_CHECKS=DisclosedDate<=Date,earnings_event_date<=Date`
>
> で、主要なラベル列とイベント日付の健全性を監視します。`DATASET_QUALITY_FAIL_ON_WARNING` や `DATASET_QUALITY_ALLOW_FUTURE_DAYS` で厳しさを調整できます。

---

## 今後の推奨フロー

1. DatasetBuilder のステージごとに上記 10 ステップをマッピング（進捗管理）。
2. 新しい特徴量を追加する際は、as-of ユーティリティと quality checker にテストを追加。
3. parity / CV / 学習パイプラインでも quality checker を必須化。

これにより、「リークなし＋高品質」を維持したまま特徴量数・ターゲット数を拡張できます。
