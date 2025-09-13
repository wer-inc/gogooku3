# Weekly Margin Interest → 日次パネル反映 仕様 (v2.2) - 実装完了

目的: J-Quants の週次「信用取引残高」(/markets/weekly_margin_interest) を日次パネル
(Code, Date) にリーク無しで統合し、短期リターン (1–5d) に効く需給特徴量ブロックを提供する。

## 実装状況 ✅ 完了
- 特徴量生成ロジック: src/gogooku3/features/margin_weekly.py
- データ統合: scripts/data/ml_dataset_builder.py
- パイプライン統合: scripts/pipelines/run_full_dataset.py
- 設定管理: configs/features/margin_weekly.yaml
- テスト: tests/test_margin_weekly.py

## 設計原則
- 公表翌営業日以降のみ有効 (PublishedDate が無い場合は保守ラグ `lag_bdays_weekly` を噛ませる)
- 週次原系列で派生量 (差分・Z 等) を計算 → as‑of backward で日次へ貼付
- 規模正規化は調整済み出来高からの `ADV20_shares` を使用
- 結合キーは `(Code, Date)`、貼付は `join_asof(backward, by=Code)`
- 欠損は `is_margin_valid=0` を付与し安全に無効化

## 出力列 (Minimal)
- `margin_short_to_adv20`, `margin_long_to_adv20`
- `margin_credit_ratio`, `margin_imbalance`
- `margin_d_net_wow`, `margin_d_short_to_adv20`, `margin_d_long_to_adv20`
- `short_z52`, `long_z52`, `margin_gross_z52`, `ratio_z52`
- `margin_impulse`, `margin_days_since`, `is_margin_valid`
- `margin_issue_type`, `is_borrowable`

## 時間処理 (リーク防止)
- `effective_start`:
  - PublishedDate あり: `next_business_day(PublishedDate)`
  - なし: `next_business_day(Date) + (lag_bdays_weekly-1)` 営業日
- 日次への付与: as‑of backward (effective_start 以降にのみ値が出現)

## 実装詳細 (Polars)

### コア実装
- **メイン**: `src/gogooku3/features/margin_weekly.py`
  - `build_weekly_effective`: effective_start計算（営業日処理含む）
  - `add_weekly_core_features`: 週次特徴量生成（差分・Z-score・比率）
  - `attach_adv20_and_scale`: ADV20による規模正規化
  - `asof_attach_to_daily`: 日次グリッドへのas-of結合
  - `add_margin_weekly_block`: 高水準統合API

### 営業日計算の改善
- `_add_business_days_jp`: 日本の営業日（月-金）を考慮した日付計算
- 週末を適切にスキップしてlag_bdays_weeklyを正確に適用

### 統合ポイント
- **Dataset Builder**: `scripts/data/ml_dataset_builder.py`
  - `add_margin_weekly_block`メソッドでBuilderに統合
- **Pipeline**: `scripts/pipelines/run_full_dataset.py`
  - `--weekly-margin-parquet`オプションでの自動検出と統合
- **Full Dataset**: `src/pipeline/full_dataset.py`
  - `enrich_and_save`関数内での統合処理

## コンフィグ
`configs/features/margin_weekly.yaml`

```
features:
  margin_weekly:
    enabled: true
    lag_bdays_weekly: 3
    z_window_weeks: 52
    scale:
      use_adjustment_volume: true
      adv_window_days: 20
    outputs:
      minimal: true
      extended: false
```

## 使い方

### 1. run_full_dataset.pyでの使用（推奨）

```bash
# Margin機能付きで完全なデータセットを構築
python scripts/pipelines/run_full_dataset.py --jquants \
  --start-date 2020-09-06 --end-date 2025-09-06 \
  --weekly-margin-parquet output/weekly_margin_interest_*.parquet \
  --margin-weekly-lag 3 \
  --adv-window-days 20
```

### 2. プログラムでの直接使用

```python
from src.gogooku3.features.margin_weekly import add_margin_weekly_block

# 既存の quotes DataFrame (Code, Date, Volume/AdjustmentVolume) と週次 margin DF を用意
result_df = add_margin_weekly_block(
    quotes=quotes_df,
    weekly_df=weekly_margin_df,
    lag_bdays_weekly=3,  # PublishedDate欠損時の保守ラグ
    adv_window_days=20   # ADV計算窓
)
```

### 3. コマンドラインオプション

- `--weekly-margin-parquet`: 週次信用取引残高Parquetファイルのパス
- `--margin-weekly-lag`: PublishedDate欠損時の営業日ラグ（デフォルト: 3）
- `--adv-window-days`: ADV計算窓の日数（デフォルト: 20）

## バリデーション (推奨)
- `Date < effective_start` に margin_* が埋まっていないこと
- 主要列の非 NaN 率 (例: `margin_short_to_adv20`) > 60% (期間に依存)
- 0 割回避 (ADV20_shares=0 の場合は NaN → `is_margin_valid=0` で無効化)
- サンプル銘柄で `effective_start` 境界の値更新を確認

## 背景と効果
- `short_to_adv20`: days-to-cover 近似で踏み上げ余地を捉える
- `d_*_wow`: 需給の新規フローを捕捉 (転換点の同時/先行指標)
- `Z(52w)`: 過熱/異常水準を検出しイベント感応度を高める

