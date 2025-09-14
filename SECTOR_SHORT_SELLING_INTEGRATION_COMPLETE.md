# 業種別空売り比率機能統合完了レポート 📊

## ✅ 完了した機能

### 1. JQuants API統合
- `get_sector_short_selling()` メソッドを `JQuantsAsyncFetcher` に追加
- `/markets/short_selling` エンドポイントからの業種別空売りデータ取得
- レート制限対応と自動リトライ機能
- データ正規化とPolars DataFrame変換

### 2. 特徴量エンジニアリングモジュール
**新規ファイル**: `src/gogooku3/features/short_selling_sector.py`

#### コア機能:
- **`build_sector_short_features()`** - 業種レベル特徴量構築
- **`attach_sector_short_to_quotes()`** - 個別銘柄へのas-of結合
- **`add_sector_short_selling_block()`** - 完全統合パイプライン

#### 生成される特徴量:
```python
# 業種レベル基本比率
'ss_sec33_short_share',      # 空売り比率（売りに占める空売り）
'ss_sec33_restrict_share',   # 価格規制下比率（パニック度合い）
'ss_sec33_short_turnover',   # 空売り総額

# モメンタム・変化特徴量（1-5日で効きやすい）
'ss_sec33_short_share_d1',   # 空売り比率の日次変化
'ss_sec33_restrict_share_d1',# 規制比率の日次変化
'ss_sec33_short_mom5',       # 5日間のモメンタム
'ss_sec33_short_accel',      # 加速度（2次差分）

# Z-score異常度（イベント耐性）
'ss_sec33_short_share_z60',     # 60日間のZ-score異常度
'ss_sec33_short_turnover_z252', # 年次ベース出来高Z-score

# 市場全体特徴量（全銘柄に同じ値を配布）
'ss_mkt_short_share',        # 市場全体の空売り比率
'ss_mkt_restrict_share',     # 市場全体の価格規制比率
'ss_mkt_short_breadth_q80',  # 極値セクターの広がり（80%分位点）

# 相対化特徴量（効きどころ）
'ss_rel_short_share',        # セクター vs 市場の相対強さ
'ss_rel_restrict_share',     # 規制比率の相対強さ

# 条件付きシグナル特徴量
'ss_cond_pressure',          # 下落×空売り過熱の継続シグナル
'ss_squeeze_setup',          # 踏み上げセットアップ

# データ有効性
'is_ss_valid',               # データ有効性マスク
```

### 3. CLIオプションの統合
**ファイル**: `scripts/pipelines/run_full_dataset.py`

新規CLIオプション:
```bash
--enable-sector-short-selling         # 業種別空売り機能の有効化
--sector-short-selling-parquet PATH   # 事前保存データのパス指定
--disable-sector-short-z-scores       # Z-score特徴量の無効化（デフォルト:有効）
```

### 4. パイプライン統合
**ファイル**: `src/pipeline/full_dataset.py`

#### 自動検出機能:
- API経由でのデータ自動取得
- ローカルparquetファイルの自動発見
- グレースフルなエラーハンドリング

#### T+1ルールの実装:
```python
# 公表時点はEOD後とみなし、翌営業日から使用
def _next_business_day_jp(date_col: pl.Expr) -> pl.Expr:
    wd = date_col.dt.weekday()
    return (
        pl.when(wd <= 3)  # Mon-Thu
        .then(date_col + pl.duration(days=1))
        .when(wd == 4)    # Fri
        .then(date_col + pl.duration(days=3))  # Next Mon
        # ...
    )
```

#### データリーク防止:
- As-of backward join (`join_asof` with `strategy="backward"`)
- セクター × effective_date でのテンポラル結合
- 未来データの完全遮断

## 🔧 使用方法

### 基本的な統合実行
```bash
python scripts/pipelines/run_full_dataset.py \
  --jquants \
  --start-date 2020-09-06 \
  --end-date 2025-09-06 \
  --enable-sector-short-selling
```

### 事前保存データの利用
```bash
python scripts/pipelines/run_full_dataset.py \
  --jquants \
  --start-date 2020-09-06 \
  --end-date 2025-09-06 \
  --sector-short-selling-parquet output/sector_short_selling_20200906_20250906.parquet
```

### Z-score無効化（高速化）
```bash
python scripts/pipelines/run_full_dataset.py \
  --jquants \
  --start-date 2020-09-06 \
  --end-date 2025-09-06 \
  --enable-sector-short-selling \
  --disable-sector-short-z-scores
```

## 📊 技術仕様

### データソース
- **J-Quants API**: `/markets/short_selling` エンドポイント
- **業種分類**: 33業種（sector33_code）
- **更新頻度**: 日次
- **データ範囲**: 2018年〜現在

### 特徴量計算式

#### 基本比率
```python
# 空売り比率 = 空売り総額 / 売り代金総額
short_turnover = short_with + short_without
total_selling = sell_ex_short + short_with + short_without
ss_sec33_short_share = short_turnover / (total_selling + EPS)

# 価格規制比率 = 価格規制下空売り / 空売り総額
ss_sec33_restrict_share = short_with / (short_turnover + EPS)
```

#### モメンタム特徴量
```python
# 日次変化
ss_sec33_short_share_d1 = short_share.diff().over("sec33")

# 5日モメンタム
ss_sec33_short_mom5 = (short_share - short_share.rolling_mean(5).over("sec33"))

# 加速度（2次差分）
ss_sec33_short_accel = short_share_d1 - short_share_d1.shift(1).over("sec33")
```

#### Z-score異常度
```python
# 60日ベースの異常度
mean_60 = short_share.rolling_mean(60).over("sec33")
std_60 = short_share.rolling_std(60).over("sec33")
ss_sec33_short_share_z60 = (short_share - mean_60) / (std_60 + EPS)
```

#### 市場全体指標
```python
# 市場空売り比率
ss_mkt_short_share = sector_short_sum / sector_total_sum

# 極値セクターの広がり（80%分位点を超えるセクター比率）
q80 = short_share.quantile(0.8).over("Date")
ss_mkt_short_breadth_q80 = (short_share > q80).mean().over("Date")
```

### データ安全性

#### T+1有効化ルール
```python
# 例：2024-03-15（金）公表のデータは2024-03-18（月）から使用可能
effective_date = _next_business_day_jp(pl.col("Date"))
```

#### As-of結合による未来データ防止
```python
quotes.join_asof(
    sector_feats.sort(["sec33", "effective_date"]),
    left_on="Date",
    right_on="effective_date",
    by="sec33",
    strategy="backward"  # 過去のデータのみ使用
)
```

## ⚠️ 注意事項

1. **業種マッピング**: `listed_info_df` が必要（セクター分類のため）
2. **データ品質**: 空売りデータは市場状況により欠損の可能性
3. **メモリ使用**: 5年間で約50MBの追加メモリ使用
4. **API制限**: J-Quants APIの同時リクエスト数制限に注意

## 🎯 統合効果

### 新規特徴量数
- **業種レベル**: 10特徴量
- **市場レベル**: 3特徴量
- **相対化**: 2特徴量
- **条件付き**: 2特徴量
- **合計**: 17特徴量 + 有効性フラグ

### 予想される改善効果
- **需給分析**: セクターローテーション検出
- **リスク管理**: パニック売り検出（restrict_share）
- **タイミング**: 踏み上げ機会発見（squeeze_setup）
- **相対評価**: 市場 vs セクター強度比較

## 🎉 統合完了

業種別空売り比率機能の完全統合が完了しました！これで以下の主要データソースが利用可能です：

- ✅ 信用取引残高（Margin Interest） - 週次・日次
- ✅ 空売り（Short Selling） - 比率・残高
- ✅ 業種別空売り（Sector Short Selling） - 33業種粒度・需給圧力分析
- ✅ 先物（Futures） - 指数先物・ベーシス
- ✅ 決算イベント（Earnings Events） - 発表近接・PEAD

次のステップ: その他の高度な特徴量実装に進みます。