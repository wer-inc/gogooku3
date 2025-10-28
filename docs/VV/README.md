# 🚀 APEX-Ranker: 完全設計書 v1.0
## PatchTST × Advanced Ranking System with Modular Architecture

---

## 📋 目次

1. [プロジェクト概要](#プロジェクト概要)
2. [GPU活用戦略と数値目標](#gpu活用戦略と数値目標)
3. [データ仕様と制約](#データ仕様と制約)
   - [VVM 特徴アーキテクチャ](#vvm-特徴アーキテクチャ)
4. [アーキテクチャ設計](#アーキテクチャ設計)
5. [段階的実装ロードマップ](#段階的実装ロードマップ)
6. [詳細実装仕様](#詳細実装仕様)
7. [検証プロトコル](#検証プロトコル)
8. [運用・最適化](#運用最適化)

---

# プロジェクト概要

## 🎯 プロジェクト名候補

### 1. **APEX-Ranker** (Adaptive Predictive EXchange Ranker) ✨ 推奨
- **意味**: 適応的市場予測ランキングシステム / 頂点を目指すランカー
- **特徴**: 簡潔で覚えやすく、技術的かつビジネス的にも通用
- **ディレクトリ名**: `apex-ranker/`, パッケージ名: `apex_ranker`

### 2. **QuantumFlow** (Quantum-inspired Forecasting & Learning Optimizer Workflow)
- **意味**: 量子計算にインスパイアされた予測最適化ワークフロー
- **特徴**: 先進的・学術的な印象、複雑な相互作用を多次元で捉えるイメージ

### 3. **NEXUS-Alpha** (Neural EXtension for Universal Stock Alpha)
- **意味**: 普遍的な超過収益を目指すニューラル拡張システム
- **特徴**: 情報統合の核となるプラットフォームとしての位置づけ

---

## 🎯 基本方針

### コア設計思想

1. **段階的進化**: v0（最小可動）→ v5（最終形）まで、各段階で検証可能
2. **モジュール分離**: データ不足でも動作、後から無停止で機能追加可能
3. **GPU最適化**: Mixed Precision + DDP対応で実用的な計算時間を実現
4. **再現性重視**: 完全な再現性とablation studyによる効果検証

### 現状データ制約への対応

#### ✅ **利用可能なデータ**
- 価格データ（OHLCV）
- テクニカル指標（395特徴の一部）
- フロー指標（出来高、板、売買代金等）
- セクター横断特徴（業種、規模等）
- ボラティリティ（Yang-Zhang, VoV等の実現ボラ）
- グラフ特徴（K近傍、次数等）

#### ❌ **現時点で不足・今後追加予定**
- オプションIV/VIX系指標
- 外生マクロ経済データ（為替、金利、コモディティ）
- テキスト情報（ニュース、開示資料）
- 取引コスト/約定データ（スプレッド、マーケットインパクト）

#### 🔧 **対応戦略**
- **代替特徴で補完**: 実現ボラ、breadth指標、市場回帰残差で代用
- **拡張スロット設計**: 後から差し込める IF を用意
- **シナリオ分析**: コストは複数シナリオ（0/10/20/30bps）で評価

---

# GPU活用戦略と数値目標

## 💻 GPU活用戦略

### ハードウェア要件

#### **最小構成**
- GPU: NVIDIA GPU 16GB VRAM以上（Tesla T4, RTX 4060Ti等）
- RAM: 32GB以上
- ストレージ: SSD 500GB以上（データ用）

#### **推奨構成**
- GPU: NVIDIA V100 / A100 32-48GB VRAM × 1-4枚
- RAM: 64GB以上
- ストレージ: NVMe SSD 1TB以上

#### **クラウド推奨**
- AWS: p3.2xlarge (V100 16GB) / p4d.24xlarge (A100 40GB×8)
- GCP: n1-standard-8 + Tesla V100
- Azure: NCv3-series

### 計算量設計

#### **メモリ見積**

| コンポーネント | サイズ | 備考 |
|--------------|--------|------|
| **入力データ** | 284MB | [2000銘柄, 180日, 395特徴] @ FP16 |
| **パッチ埋め込み後** | 15MB | [2000, 20パッチ, 192dim] @ FP16 |
| **モデルパラメータ** | 50-100MB | depth=3-6の場合 |
| **勾配・オプティマイザ状態** | 150-300MB | AdamW使用時 |
| **総メモリ（ピーク）** | 8-12GB | 1バッチ（1日=2000銘柄）処理時 |

#### **学習速度見積**

| 設定 | GPU | 時間/エポック | 時間/fold |
|------|-----|--------------|----------|
| v0 (基本) | V100 16GB | 18分 | 12分 (40epoch) |
| v2 (KNN) | V100 16GB | 25分 | 18分 |
| v5 (フル) | V100 16GB | 45分 | 35分 |
| v5 (4×V100) | V100 16GB×4 | 12分 | 10分 (fold並列) |
| v5 (A100) | A100 40GB | 20分 | 15分 |

### 最適化技術スタック

#### **Mixed Precision (AMP)**
```python
# PyTorch Lightning推奨設定
trainer = pl.Trainer(
    precision="16-mixed",  # FP16 + FP32自動混在
    devices=[0],           # GPU ID
    accelerator="gpu",
)
```

#### **分散並列（DDP）**
```python
# マルチGPU設定
trainer = pl.Trainer(
    precision="16-mixed",
    devices=4,  # 4GPU使用
    strategy="ddp_find_unused_parameters_false",  # 高速化
    sync_batchnorm=True,  # BatchNorm同期
)
```

#### **Gradient Checkpointing（メモリ削減）**
- メモリ使用量: 約50%削減
- 学習速度: 約20%低下
- 適用タイミング: VRAM不足時のみ

#### **Flash Attention 2.0（速度向上）**
- 計算速度: 2-3倍高速化
- メモリ使用量: 約50%削減
- 適用: depth>4, n_heads>8の場合に効果大

---

## 🎯 数値目標

### パフォーマンス目標（10年バックテスト）

#### **v0: 最小可動版**（実装1-2週間）
| 指標 | 目標値 | 許容範囲 |
|------|--------|----------|
| **RankIC (5d)** | 0.055 | 0.05-0.06 |
| **ICIR (5d)** | 1.1 | 1.0-1.2 |
| **年率Sharpe** | 1.35 | 1.2-1.5 |
| **最大DD** | -22.5% | -20% ~ -25% |
| **Precision@50** | 53.5% | 52-55% |
| **年間収益率** | 13.5% | 12-15% |
| **勝率（日次）** | 54% | 53-55% |

#### **v2: 実用版**（実装3-4週間）
| 指標 | 目標値 | 許容範囲 |
|------|--------|----------|
| **RankIC (5d)** | 0.09 | 0.08-0.10 |
| **ICIR (5d)** | 1.75 | 1.5-2.0 |
| **年率Sharpe** | 2.0 | 1.8-2.2 |
| **最大DD** | -16.5% | -15% ~ -18% |
| **Precision@50** | 60% | 58-62% |
| **年間収益率** | 22.5% | 20-25% |
| **勝率（日次）** | 57% | 56-58% |

#### **v5: 最終形**（実装3-6ヶ月）
| 指標 | 目標値 | 許容範囲 |
|------|--------|----------|
| **RankIC (5d)** | 0.11 | 0.10-0.12 |
| **ICIR (5d)** | 2.25 | 2.0-2.5 |
| **年率Sharpe** | 2.75 | 2.5-3.0 |
| **最大DD** | -11% | -10% ~ -12% |
| **Precision@50** | 67.5% | 65-70% |
| **年間収益率** | 35% | 30-40% |
| **勝率（日次）** | 60% | 58-62% |

### 計算効率目標

| 指標 | 目標値 | 測定方法 |
|------|--------|----------|
| **学習時間/fold** | < 15分 | 単一V100、400日分 |
| **推論速度** | < 1秒 | 2000銘柄バッチ推論 |
| **GPU利用率** | > 85% | `nvidia-smi dmon` |
| **VRAM使用率** | < 80% | OOM回避 |
| **データロード時間** | < 5% | 総学習時間の5%未満 |

### 安定性・堅牢性目標

| 指標 | 目標値 | 検証方法 |
|------|--------|----------|
| **過学習制御** | val-train IC差 < 0.02 | 全fold平均 |
| **レジーム適応** | 危機時IC > 0.03 | 2020年3月等 |
| **分位校正** | P10超過率 9-11% | v1以降 |
| **再現性** | 完全一致 | seed固定時 |
| **ターンオーバー** | < 50%/月 | v2以降 |

---

# データ仕様と制約

## 📊 データ構造

### 入力形式

```
data/
├── train/
│   ├── features.parquet       # 長表: (Date, Code, 特徴群)
│   └── targets.parquet         # 長表: (Date, Code, ret_1d, ret_5d, ret_10d, ret_20d)
├── metadata/
│   ├── dataset_features_detail.json  # 特徴メタ情報
│   ├── sector_mapping.csv           # 業種マッピング
│   └── trading_calendar.csv         # 営業日カレンダー
└── config/
    └── data_config.yaml              # データ設定
```

### 設定ファイル（data_config.yaml）

```yaml
data:
  # 基本設定
  lookback: 180          # 過去何日分を入力とするか
  horizons: [1, 5, 10, 20]  # 予測ホライズン（日数）
  
  # 特徴選択（モジュール制御）
  features:
    # 使用する特徴カテゴリ（現状利用可能）
    include:
      - price              # OHLCV
      - technical          # テクニカル指標
      - flow              # 出来高・板・売買代金
      - sector_cross      # セクター横断特徴
      - volatility        # Yang-Zhang, VoV等
      - graph             # K近傍、次数等
    
    # 今は使わない（将来追加予定）
    exclude:
      - options_iv        # オプションIV/VIX
      - macro             # 外生マクロ（為替・金利等）
      - text              # テキスト埋め込み
      - execution         # 取引コスト・約定データ
  
  # 正規化戦略
  normalization:
    method: cross_section_z  # 当日内Z-score
    train_mode: fit          # 学習時は統計をfit
    infer_mode: transform    # 推論時は当日統計で変換
    clip_sigma: 5.0          # 外れ値クリップ（±5σ）
  
  # サンプリング
  sampling:
    strategy: day_batch      # 1バッチ=1日全銘柄
    min_stocks_per_day: 500  # 最低銘柄数（これ未満の日は除外）
    skip_same_return_days: true  # 全銘柄リターンが同値の日をスキップ

# 拡張スロット（将来の差し込み用）
extensions:
  iv_slot:
    enabled: false
    dim: 0
  macro_slot:
    enabled: false
    dim: 0
  text_slot:
    enabled: false
    dim: 0
  execution_slot:
    enabled: false
    dim: 0
```

### 特徴メタ情報（dataset_features_detail.json）

```json
{
  "feature_groups": {
    "price": {
      "columns": ["open", "high", "low", "close", "volume", "vwap"],
      "count": 6,
      "enabled": true
    },
    "technical": {
      "columns": ["rsi_14", "macd", "bb_upper", "bb_lower", "..."],
      "count": 50,
      "enabled": true
    },
    "flow": {
      "columns": ["bid_volume", "ask_volume", "trade_imbalance", "..."],
      "count": 20,
      "enabled": true
    },
    "sector_cross": {
      "columns": ["sector_id", "market_cap_rank", "relative_strength", "..."],
      "count": 15,
      "enabled": true
    },
    "volatility": {
      "columns": ["yang_zhang", "vov", "realized_vol_20d", "..."],
      "count": 10,
      "enabled": true
    },
    "graph": {
      "columns": ["knn_degree", "correlation_centrality", "..."],
      "count": 8,
      "enabled": true
    },
    "options_iv": {
      "columns": [],
      "count": 0,
      "enabled": false,
      "note": "将来追加予定"
    },
    "macro": {
      "columns": [],
      "count": 0,
      "enabled": false,
      "note": "将来追加予定"
    },
    "text": {
      "columns": [],
      "count": 0,
      "enabled": false,
      "note": "将来追加予定"
    },
    "execution": {
      "columns": [],
      "count": 0,
      "enabled": false,
      "note": "将来追加予定"
    }
  },
  "total_features_enabled": 109,
  "total_features_planned": 150
}
```

### コア特徴量バンドル（core50 / plus30）

2025-10 時点の実運用では、約 190 列の中から **短期で効きやすくリークを避けた列**を
明示的にグルーピングし、`configs/atft/feature_groups.yaml` で管理しています。
データセット設定 (`configs/atft/data/jpx_large_scale.yaml`) では `core50` を必須とし、
必要に応じて `plus30` を追加する構成になりました。

- **core50:** 短期モメンタム + 市場レジーム + セクター相対 + フロー + 財務基礎  
  (`returns_1d`, `volatility_20d`, `ema_5/20/200`, `mkt_ret_1d`, `alpha_1d`,  
  `flow_foreign_net_z`, `stmt_progress_op` など 50 列前後)
- **plus30:** 追加モメンタム・市場状態・フロー比率・財務詳細での上乗せ  
  (`returns_10d`, `volatility_60d`, `rsi_14`, `mkt_ret_20d`, `flow_activity_ratio`,  
  `stmt_roe` など 30 列前後)

カテゴリ別の代表的な内訳は次のとおりです（重複を除外した実在列を想定）：

| 区分                 | 主な列例                                                                                      |
|----------------------|------------------------------------------------------------------------------------------------|
| 価格・テクニカル      | `returns_1d`, `returns_5d`, `volatility_5d`, `ema_5`, `ma_gap_5_20`, `bb_position`             |
| 市場レジーム          | `mkt_ret_1d`, `mkt_gap_5_20`, `mkt_bull_200`, `mkt_vol_20d`, `mkt_ret_1d_z`                    |
| 銘柄×市場クロス       | `beta_60d`, `alpha_5d`, `rel_strength_5d`, `idio_vol_ratio`, `trend_align_mkt`                 |
| セクター相対          | `sec_mom_20`, `sec_ret_5d_eq`, `sec_vol_20_z`, `sec_gap_5_20`, `sec_member_cnt`                |
| フロー／需給          | `flow_foreign_net_z`, `flow_smart_idx`, `flow_impulse`, `flow_activity_ratio`, `flow_breadth_pos` |
| 財務イベント          | `stmt_yoy_sales`, `stmt_opm`, `stmt_progress_op`, `stmt_rev_fore_np`, `stmt_imp_statement`      |
| モメンタム補助 (plus) | `returns_10d`, `returns_20d`, `volatility_60d`, `rsi_2`, `macd_histogram`, `stoch_k`           |

マスク列は `is_flow_valid`, `is_stmt_valid`, `is_valid_ma`, `is_sec_cs_valid` を併用する想定で、
学習時は **Fold 内でのクロスセクション Z-score** を前処理として適用します。
この整理により、APEX-Ranker v0 / ATFT-GAT-FAN いずれのパイプラインでも同じ束を参照できるようになりました。

### VVM 特徴アーキテクチャ

**結論:** はい。現在の dataset は Volatility / Volume / Momentum（VVM）を中心に据えた 4 階層構成で、短期（1–5d）～中期（10–20d）予測にそのまま投入できます。

#### レイヤー構造とマスク運用
- 4 階層（銘柄・市場 TOPIX・セクター集約・銘柄×市場クロス）の同系統指標を用意
- `is_*_valid` 系フラグでウォームアップ未満のレコードを自動マスクし、リーク対策を徹底
- すべて当日時点までの履歴のみで算出し、15:00/T+1 ルールを遵守

#### Volatility（価格変動）
- **銘柄レベル:** `volatility_5d`, `volatility_10d`, `volatility_20d`, `volatility_60d`, `realized_volatility`, `bb_width`
- **市場レベル（TOPIX）:** `mkt_vol_20d`, `mkt_vol_20d_z`, `mkt_bb_bw`, `mkt_bb_bw_z`, `mkt_high_vol`
- **セクター集約:** `sec_vol_20`, `sec_vol_20_z`
- **銘柄×市場クロス:** `idio_vol_ratio`（= `volatility_20d / (mkt_vol_20d + ε)`）

#### Volume（出来高・フロー）
- **銘柄レベル:** `volume_ratio_5`, `volume_ratio_20`, `volume_ma_5`, `volume_ma_20`, `turnover_rate`, `dollar_volume`
- **市場/フロー補助:** `flow_activity_z`, `flow_activity_ratio`
- **Premium 機能:** `am_vol_ratio_20`, `bd_activity_ratio`, `bd_net_z_52`

#### Momentum（トレンド / 反転）
- **銘柄レベル（価格）:** `returns_1d/5d/10d/20d`, `log_returns_1d`, `ema_5/20/200`, `ma_gap_5_20`, `ema5_slope`, `dist_to_200ema`, `rsi_2`, `rsi_14`, `bb_position`, `close_to_high`, `close_to_low`, `macd_histogram`, `stoch_k`
- **市場レベル:** `mkt_ret_1d/5d/10d`, `mkt_gap_5_20`, `mkt_bull_200`, `mkt_trend_up`
- **セクター相対:** `sec_mom_20`, `sec_ret_5d_eq`, `rel_to_sec_5d`
- **銘柄×市場クロス:** `rel_strength_5d`, `alpha_1d`, `alpha_5d`, `trend_align_mkt`

#### VVM 強度の評価
- **Volatility:** 銘柄・市場・セクターの 3 階層を網羅
- **Volume:** 生出来高・移動平均比・回転率で水準と変化を両立
- **Momentum:** 1–20 日のリターン／EMA ギャップ／位置系／オシレーターを揃え、相対軸も確保
- **運用:** `is_*_valid` マスクと T+1 遵守で実務的なリーク防止

#### VVM をさらに厚くする最小差分
1. **Volatility の質感向上**
   - `downside_vol_20d = std(min(returns_1d, 0), 20) * sqrt(252)`（下方ボラの顕在化）
   - `range_pct = (High - Low) / (Close + ε)`（価格水準に依存しないレンジ正規化）
2. **Volume の乾湿指標**
   - `volume_ratio_60` と `liquidity_dry_flag = (volume_ratio_20 < 0.5).int8()` で流動性低下を検知
   - `tvr_ratio_20 = (Close * Volume) / mean(Close * Volume, 20)` で価格×出来高の交互作用を追跡
3. **Momentum の二相分離**
   - 反転系の補強に `ret_1d_z_in_cs`（当日クロスセクション Z、学習 fold 内で算出）
   - 追随系の可視化に `mom_20 = sum(returns_1d, 20)`

上記はいずれも派生数行で実装でき、既存パイプラインに追加してもリークは生じません。

#### VVM 健全性チェック（Polars ミニレシピ）
```python
import polars as pl

df = pl.read_parquet("output/ml_dataset_latest_enriched.parquet")

# 1. 列存在チェック
must_have = [
    "volatility_5d", "volatility_20d", "realized_volatility", "mkt_vol_20d", "sec_vol_20",
    "volume_ratio_5", "volume_ratio_20", "turnover_rate",
    "returns_1d", "returns_5d", "ema_5", "ema_20", "ma_gap_5_20", "bb_position", "rsi_2",
    "rel_strength_5d", "sec_mom_20", "rel_to_sec_5d", "idio_vol_ratio",
]
missing = [c for c in must_have if c not in df.columns]
print("missing:", missing)

# 2. 有効カバレッジ（必要に応じてマスク列を適用）
def coverage(col: str, valid: str | None = None) -> float:
    series = df[col]
    if valid and valid in df.columns:
        mask = df[valid].fill_null(0) == 1
        series = series.filter(mask)
    return 1.0 - series.null_count() / series.len()

print("vol20 coverage:", coverage("volatility_20d"))
print("volume_ratio_20 coverage:", coverage("volume_ratio_20"))
print("ma_gap_5_20 coverage (valid_ma):", coverage("ma_gap_5_20", "is_valid_ma"))
```

#### まとめ
- Dataset は VVM を多層・多表現でカバー済みで、短期～中期ホライズンに即応可能
- 下方ボラ / 流動性フラグ / 単純モメンタムを補うだけで更なる感度向上が見込める
- 学習前処理では当日クロスセクション Z と `is_*_valid` マスクを標準化することを推奨

### データフロー

```
Raw Data (Parquet)
    ↓
FeatureSelector (include/excludeでフィルタ)
    ↓
PanelIndexer (Code×Date の高速インデックス)
    ↓
DayPanelDataset (日単位でバッチ化)
    ↓
Normalization (当日内Z-score)
    ↓
Model Input [B, L, F]
```

---

# アーキテクチャ設計

## 🏗️ 全体アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    APEX-Ranker System                        │
├─────────────────────────────────────────────────────────────┤
│  Input: [Batch, Lookback=180, Features=109]                 │
│         (現状利用可能な特徴のみ)                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   v0: Base Encoder                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ PatchTST Encoder                                      │   │
│  │ - Patch Embedding (Conv1D)                            │   │
│  │ - Multi-Head Self-Attention × depth                   │   │
│  │ - Layer Norm + FFN + Residual                         │   │
│  │ Output: [B, d_model=192]                              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│          v1: Risk-Aware Enhancement (Optional)               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Quantile Head (τ=0.1, 0.5, 0.9)                      │   │
│  │ - P10, Median, P90予測                                │   │
│  │ - Adjusted Score = score - λ×P10                      │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│      v2: Cross-Sectional Context (Optional)                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Adaptive KNN Graph Attention                          │   │
│  │ - 市場状態に応じてK動的調整 (k_min=5, k_max=30)         │   │
│  │ - GAT 1層で銘柄間相互作用                              │   │
│  │ Output: [B, d_model] (context-enhanced)               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│               Multi-Horizon Prediction Head                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Linear Heads × 4 Horizons                             │   │
│  │ - 1日先: score_1d  [B]                                │   │
│  │ - 5日先: score_5d  [B]                                │   │
│  │ - 10日先: score_10d [B]                               │   │
│  │ - 20日先: score_20d [B]                               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Ranking Loss                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Composite Loss (同日バッチ内で計算)                     │   │
│  │ - ListNet (70%): Top-K重み付きKL divergence           │   │
│  │ - RankNet (30%): Pairwise logistic loss               │   │
│  │ - (Optional) MSE (初期のみ): 安定化用                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Metrics                        │
│  - RankIC / ICIR (Spearman相関)                              │
│  - Precision@K / Hit@K                                       │
│  - Top-K Portfolio (Sharpe, MDD, Turnover)                   │
│  - Cost Sensitivity Analysis (0/10/20/30bps)                │
└─────────────────────────────────────────────────────────────┘
```

## 🔌 拡張スロット設計（将来の差し込み用）

```python
class ExtensibleEncoder(nn.Module):
    """拡張可能なエンコーダ（将来のモジュール追加に対応）"""
    def __init__(self, config):
        super().__init__()
        
        # v0: 基本エンコーダ（常に有効）
        self.base_encoder = PatchTSTEncoder(
            in_feats=config.n_features_enabled,
            d_model=config.d_model,
            depth=config.depth
        )
        
        # v1: 分位点ヘッド（オプション）
        self.quantile_enabled = config.get('quantile_enabled', False)
        if self.quantile_enabled:
            self.quantile_head = QuantileHead(config.d_model, config.horizons)
        
        # v2: クロスセクション（オプション）
        self.crosssec_enabled = config.get('crosssec_enabled', False)
        if self.crosssec_enabled:
            self.crosssec_layer = AdaptiveKNNGraph(config.d_model)
        
        # 拡張スロット（将来用）
        self.iv_slot = ExtensionSlot(enabled=False)      # IV追加時に有効化
        self.macro_slot = ExtensionSlot(enabled=False)   # マクロ追加時に有効化
        self.text_slot = ExtensionSlot(enabled=False)    # テキスト追加時に有効化
        
    def forward(self, X, extensions=None):
        # v0: 基本エンコーディング
        z_base, tokens = self.base_encoder(X)  # [B, d_model]
        
        # v2: クロスセクション（有効時のみ）
        if self.crosssec_enabled:
            z = self.crosssec_layer(z_base)[0]
        else:
            z = z_base
        
        # 拡張スロット（将来の追加機能）
        if extensions is not None:
            if self.iv_slot.enabled and 'iv' in extensions:
                z = self.iv_slot.fuse(z, extensions['iv'])
            if self.macro_slot.enabled and 'macro' in extensions:
                z = self.macro_slot.fuse(z, extensions['macro'])
            if self.text_slot.enabled and 'text' in extensions:
                z = self.text_slot.fuse(z, extensions['text'])
        
        return z, tokens

class ExtensionSlot(nn.Module):
    """将来の機能追加用の拡張スロット"""
    def __init__(self, enabled=False, fusion_type='concat'):
        super().__init__()
        self.enabled = enabled
        self.fusion_type = fusion_type
        self.fusion_layer = None
        
    def enable(self, input_dim, output_dim):
        """スロットを有効化（モジュール追加時に呼ぶ）"""
        self.enabled = True
        if self.fusion_type == 'concat':
            self.fusion_layer = nn.Linear(input_dim + output_dim, output_dim)
        elif self.fusion_type == 'gate':
            self.fusion_layer = GatedFusion(input_dim, output_dim)
    
    def fuse(self, z, extension_features):
        """特徴融合"""
        if not self.enabled or self.fusion_layer is None:
            return z
        return self.fusion_layer(torch.cat([z, extension_features], dim=-1))
```

---

# 段階的実装ロードマップ

## 📅 開発マイルストーン

### **Phase 0: 環境構築**（3-5日）

#### タスク
- [ ] GPU環境セットアップ（Docker/Conda）
- [ ] データパイプライン構築
- [ ] 基本ユーティリティ実装

#### 成果物
```
apex-ranker/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── environment.yml
├── requirements.txt
├── data/
│   └── README.md
└── scripts/
    └── setup_env.sh
```

#### 検証基準
- [ ] GPU認識確認（`torch.cuda.is_available()`）
- [ ] データ読み込み成功（サンプル100日分）
- [ ] 基本テスト通過（pytest）

---

### **Phase 1: v0実装（最小可動版）**（1-2週間）

#### タスク
- [ ] PatchTSTエンコーダ実装
- [ ] マルチホライズンヘッド実装
- [ ] ListNet + RankNet損失実装
- [ ] DayBatchSampler実装
- [ ] 学習ループ構築（Lightning）
- [ ] 評価指標実装（RankIC, ICIR, P@K）

#### コア実装

##### 1. PatchTSTエンコーダ

```python
class PatchTSTEncoder(nn.Module):
    """
    時系列をパッチ分割してTransformerで処理
    
    Args:
        in_feats: 入力特徴数（現状109, 将来的に拡張可能）
        d_model: モデル次元
        depth: Transformerブロック数
        patch_len: パッチ長
        stride: パッチストライド
        n_heads: アテンションヘッド数
        dropout: ドロップアウト率
        channel_independent: 各特徴を独立処理するか
    """
    def __init__(self, 
                 in_feats=109,  # 現状利用可能な特徴数
                 d_model=192, 
                 depth=3, 
                 patch_len=16, 
                 stride=8,
                 n_heads=8,
                 dropout=0.1,
                 channel_independent=True):
        super().__init__()
        
        # パッチ埋め込み
        if channel_independent:
            # 各特徴独立でConv → 統合
            d_patch = 4
            self.patch_embed = nn.Sequential(
                nn.Conv1d(in_feats, in_feats * d_patch,
                         kernel_size=patch_len, stride=stride,
                         groups=in_feats, bias=False),
                Rearrange('b (c p) n -> b n (c p)', c=in_feats),
                nn.Linear(in_feats * d_patch, d_model)
            )
        else:
            # チャネル混合
            self.patch_embed = nn.Sequential(
                nn.Conv1d(in_feats, d_model,
                         kernel_size=patch_len, stride=stride,
                         bias=False),
                Rearrange('b c n -> b n c')
            )
        
        # Transformerブロック
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model*4, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [B, L, F] → [B, F, L]
        x = x.transpose(1, 2)
        
        # パッチ埋め込み: [B, F, L] → [B, Np, d_model]
        tokens = self.patch_embed(x)
        
        # Transformer処理
        for block in self.blocks:
            tokens = block(tokens)
        
        tokens = self.norm(tokens)
        
        # プーリング: [B, Np, d_model] → [B, d_model]
        z = tokens.mean(dim=1)
        
        return z, tokens


class TransformerBlock(nn.Module):
    """標準的なTransformerブロック"""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-Attention
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        
        # Feed-Forward
        h = self.ln2(x)
        h = self.ffn(h)
        x = x + h
        
        return x
```

##### 2. ランキング損失

```python
class ListNetLoss(nn.Module):
    """
    ListNet損失（温度付き、Top-K重み対応）
    
    Args:
        tau: 温度パラメータ（小さいほど上位重視）
        topk: 上位何件を重視するか（Noneで全体）
        eps: 数値安定化用の微小値
    """
    def __init__(self, tau=0.5, topk=None, eps=1e-12):
        super().__init__()
        self.tau = tau
        self.topk = topk
        self.eps = eps
    
    def forward(self, scores, labels):
        """
        Args:
            scores: [B] モデル予測スコア（同日バッチ）
            labels: [B] 真のリターン（同日バッチ）
        Returns:
            loss: スカラー
        """
        # 同値日のチェック
        if torch.isclose(labels.std(), torch.tensor(0., device=labels.device)):
            return torch.tensor(0., device=scores.device)
        
        # 予測分布
        p = torch.softmax(scores / self.tau, dim=0)
        
        # 教師分布
        q = torch.softmax(labels / self.tau, dim=0)
        
        # Top-K重み付け
        if self.topk is not None and self.topk < len(labels):
            _, topk_idx = torch.topk(q, self.topk, largest=True, sorted=False)
            w = torch.zeros_like(q)
            w[topk_idx] = 1.0
            w = w / (w.sum() + self.eps)
            loss = -(w * torch.log(p + self.eps)).sum()
        else:
            loss = -(q * torch.log(p + self.eps)).sum()
        
        return loss


class RankNetLoss(nn.Module):
    """
    RankNet損失（ペアワイズロジスティック）
    
    Args:
        neg_sample: ペア数削減用のサンプリング数（Noneで全ペア）
    """
    def __init__(self, neg_sample=None):
        super().__init__()
        self.neg_sample = neg_sample
    
    def forward(self, scores, labels):
        """
        Args:
            scores: [B] モデル予測スコア
            labels: [B] 真のリターン
        """
        B = labels.size(0)
        
        # 全ペア生成 or サンプリング
        if self.neg_sample is not None and B*(B-1)//2 > self.neg_sample:
            # ランダムサンプリング
            n_pairs = self.neg_sample
            idx_i = torch.randint(0, B, (n_pairs,), device=scores.device)
            idx_j = torch.randint(0, B, (n_pairs,), device=scores.device)
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]
        else:
            # 全ペア
            idx_i, idx_j = torch.triu_indices(B, B, offset=1, device=scores.device)
        
        # スコア差と教師差
        s_diff = scores[idx_i] - scores[idx_j]
        y_diff = labels[idx_i] - labels[idx_j]
        
        # ペアワイズロス
        # y_diff > 0 なら s_diff も正を望む
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            s_diff, (y_diff > 0).float(), reduction='mean'
        )
        
        return loss


class CompositeLoss(nn.Module):
    """複合損失（ListNet + RankNet + オプションでMSE）"""
    def __init__(self, config):
        super().__init__()
        self.listnet = ListNetLoss(
            tau=config.loss.listnet.tau,
            topk=config.loss.listnet.topk
        )
        self.ranknet = RankNetLoss(
            neg_sample=config.loss.ranknet.neg_sample
        )
        self.mse = nn.MSELoss()
        
        self.w_listnet = config.loss.listnet.weight
        self.w_ranknet = config.loss.ranknet.weight
        self.w_mse = config.loss.mse.weight
    
    def forward(self, scores, labels):
        loss = 0.0
        
        if self.w_listnet > 0:
            loss += self.w_listnet * self.listnet(scores, labels)
        
        if self.w_ranknet > 0:
            loss += self.w_ranknet * self.ranknet(scores, labels)
        
        if self.w_mse > 0:
            loss += self.w_mse * self.mse(scores, labels)
        
        return loss
```

##### 3. DayBatchSampler & Dataset

```python
class DayPanelDataset(torch.utils.data.Dataset):
    """
    日単位でバッチを返すデータセット
    
    1つの__getitem__呼び出しで、その日の全銘柄データを返す
    （ランキング損失は同日内で計算されるため）
    """
    def __init__(self, df, feature_cols, target_cols, 
                 lookback=180, min_stocks=500):
        """
        Args:
            df: パネルデータ（(Date, Code)でソート済み）
            feature_cols: 特徴列名リスト
            target_cols: ターゲット列名リスト
            lookback: 過去何日分を使うか
            min_stocks: 最低銘柄数（未満の日は除外）
        """
        self.df = df
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.lookback = lookback
        
        # 日付リスト作成（十分な履歴がある日のみ）
        all_dates = sorted(df['Date'].unique())
        self.dates = [
            d for i, d in enumerate(all_dates)
            if i >= lookback  # 十分な履歴がある
        ]
        
        # 各日の銘柄リスト
        self.stocks_by_date = {}
        for date in self.dates:
            stocks = df[df['Date'] == date]['Code'].unique()
            if len(stocks) >= min_stocks:
                self.stocks_by_date[date] = stocks
        
        # 最低銘柄数を満たす日のみ残す
        self.dates = [d for d in self.dates if d in self.stocks_by_date]
        
        # インデクサ構築（高速アクセス用）
        self.indexer = self._build_indexer()
    
    def _build_indexer(self):
        """Code×Date のインデクサを構築"""
        indexer = {}
        for code in self.df['Code'].unique():
            code_df = self.df[self.df['Code'] == code].reset_index(drop=True)
            date_to_idx = {
                date: idx for idx, date in enumerate(code_df['Date'])
            }
            indexer[code] = {
                'df': code_df,
                'date_to_idx': date_to_idx
            }
        return indexer
    
    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, idx):
        date = self.dates[idx]
        stocks = self.stocks_by_date[date]
        
        X_list = []
        y_list = []
        valid_stocks = []
        
        for code in stocks:
            # 過去lookback日分のデータ取得
            data = self._get_window(code, date)
            if data is not None:
                X, y = data
                X_list.append(X)
                y_list.append(y)
                valid_stocks.append(code)
        
        if len(X_list) == 0:
            # フォールバック（通常起きない）
            return None
        
        X = np.stack(X_list, axis=0)  # [B, L, F]
        y = np.stack(y_list, axis=0)  # [B, H]
        
        return {
            'X': X.astype(np.float32),
            'y': y.astype(np.float32),
            'codes': valid_stocks,
            'date': date
        }
    
    def _get_window(self, code, end_date):
        """指定銘柄・日付の過去lookback日分を取得"""
        if code not in self.indexer:
            return None
        
        info = self.indexer[code]
        df = info['df']
        date_to_idx = info['date_to_idx']
        
        if end_date not in date_to_idx:
            return None
        
        end_idx = date_to_idx[end_date]
        start_idx = end_idx - self.lookback + 1
        
        if start_idx < 0:
            return None
        
        window_df = df.iloc[start_idx:end_idx+1]
        
        if len(window_df) != self.lookback:
            return None  # 欠損がある
        
        X = window_df[self.feature_cols].values  # [L, F]
        y = window_df.iloc[-1][self.target_cols].values  # [H]
        
        return X, y


def collate_day_batch(batch):
    """
    バッチ内は1日分（サンプラーがday単位で返す想定）
    """
    if batch[0] is None:
        return None
    
    sample = batch[0]
    
    return {
        'X': torch.tensor(sample['X']),      # [B, L, F]
        'y': torch.tensor(sample['y']),      # [B, H]
        'codes': sample['codes'],
        'date': sample['date']
    }
```

##### 4. Lightningモジュール

```python
import pytorch_lightning as pl

class APEXRankerV0(pl.LightningModule):
    """
    v0: 最小可動版
    - PatchTSTエンコーダ
    - マルチホライズンヘッド
    - ListNet + RankNet損失
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # エンコーダ
        self.encoder = PatchTSTEncoder(
            in_feats=config.model.in_feats,
            d_model=config.model.d_model,
            depth=config.model.depth,
            patch_len=config.model.patch_len,
            stride=config.model.stride,
            n_heads=config.model.n_heads,
            dropout=config.model.dropout
        )
        
        # マルチホライズンヘッド
        self.heads = nn.ModuleDict({
            f'h{h}': nn.Linear(config.model.d_model, 1)
            for h in config.head.horizons
        })
        
        # 損失関数
        self.criterion = CompositeLoss(config)
        
        # 評価指標の累積用
        self.validation_outputs = []
    
    def forward(self, X):
        """
        Args:
            X: [B, L, F]
        Returns:
            scores: {horizon: [B]}
        """
        z, _ = self.encoder(X)  # [B, d_model]
        
        scores = {}
        for h, head in self.heads.items():
            scores[h] = head(z).squeeze(-1)  # [B]
        
        return scores
    
    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        
        X = batch['X']  # [B, L, F]
        Y = batch['y']  # [B, H]
        
        scores = self(X)  # {h: [B]}
        
        total_loss = 0.0
        logs = {}
        
        for i, h in enumerate(self.config.head.horizons):
            s = scores[f'h{h}']
            y = Y[:, i]
            
            # 同値日スキップ
            if torch.isclose(y.std(), torch.tensor(0., device=y.device)):
                continue
            
            loss_h = self.criterion(s, y)
            total_loss += loss_h
            logs[f'train_loss_h{h}'] = loss_h
        
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None
        
        X = batch['X']
        Y = batch['y']
        
        scores = self(X)
        
        metrics = {}
        for i, h in enumerate(self.config.head.horizons):
            s = scores[f'h{h}'].detach()
            y = Y[:, i].detach()
            
            if torch.isclose(y.std(), torch.tensor(0., device=y.device)):
                continue
            
            # RankIC (Spearman)
            ic = spearman_rank_correlation(s, y)
            
            # Precision@K
            k = min(50, len(s)//10)
            pk = precision_at_k(s, y, k=k)
            
            metrics[f'val_RankIC_h{h}'] = ic
            metrics[f'val_P@{k}_h{h}'] = pk
        
        self.validation_outputs.append(metrics)
        
        return metrics
    
    def on_validation_epoch_end(self):
        if len(self.validation_outputs) == 0:
            return
        
        # 平均集計
        avg_metrics = {}
        all_keys = set()
        for output in self.validation_outputs:
            all_keys.update(output.keys())
        
        for key in all_keys:
            values = [o[key] for o in self.validation_outputs if key in o]
            if len(values) > 0:
                avg_metrics[key] = np.mean(values)
        
        self.log_dict(avg_metrics, prog_bar=True)
        
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.train.epochs
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


# ヘルパー関数
def spearman_rank_correlation(pred, target):
    """Spearman順位相関係数"""
    pred_rank = torch.argsort(torch.argsort(pred, descending=True))
    target_rank = torch.argsort(torch.argsort(target, descending=True))
    
    n = len(pred)
    diff = (pred_rank - target_rank).float()
    
    rho = 1 - 6 * (diff**2).sum() / (n * (n**2 - 1))
    
    return rho.item()


def precision_at_k(scores, returns, k):
    """
    Precision@K: 上位K件のうち実際にリターンが正だった割合
    """
    _, topk_idx = torch.topk(scores, k=k, largest=True)
    topk_returns = returns[topk_idx]
    
    precision = (topk_returns > 0).float().mean()
    
    return precision.item()
```

#### 成果物
```
apex-ranker/
├── apex_ranker/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── patchtst.py       # PatchTSTエンコーダ
│   │   ├── heads.py          # 予測ヘッド
│   │   └── ranker.py         # APEXRankerV0
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── listnet.py
│   │   ├── ranknet.py
│   │   └── composite.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py        # DayPanelDataset
│   │   └── sampler.py        # DayBatchSampler
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py        # RankIC, Precision@K等
│       └── config.py
├── configs/
│   └── v0_base.yaml
├── scripts/
│   └── train_v0.py
└── tests/
    ├── test_encoder.py
    ├── test_losses.py
    └── test_dataset.py
```

#### 検証基準
- [ ] **形状テスト**: 入力`[B, L, F]` → 出力`[B, d_model]`正常
- [ ] **損失計算**: ListNet/RankNetが同値日でloss=0
- [ ] **勾配フロー**: `loss.backward()`後に勾配が非ゼロ
- [ ] **過学習テスト**: 1日×32銘柄でRankIC→1.0付近
- [ ] **学習完了**: 1 fold（400日）を15分以内で完了
- [ ] **目標達成**: val RankIC(5d) > 0.05, Sharpe > 1.2

---

### **Phase 2: v1実装（リスク管理）**（1週間）

#### タスク
- [ ] 分位点ヘッド実装（τ=0.1, 0.5, 0.9）
- [ ] ピンボール損失実装
- [ ] リスク調整スコア計算
- [ ] 分位校正評価

#### コア実装

```python
class QuantileHead(nn.Module):
    """
    分位点予測ヘッド
    
    各ホライズンに対して、複数の分位点（P10, P50, P90等）を予測
    """
    def __init__(self, d_model, horizons, taus=[0.1, 0.5, 0.9]):
        super().__init__()
        self.horizons = horizons
        self.taus = taus
        
        # 各ホライズン×各分位点のヘッド
        self.heads = nn.ModuleDict()
        for h in horizons:
            self.heads[f'h{h}'] = nn.ModuleDict({
                f'q{int(tau*100)}': nn.Linear(d_model, 1)
                for tau in taus
            })
    
    def forward(self, z):
        """
        Args:
            z: [B, d_model]
        Returns:
            quantiles: {horizon: {quantile: [B]}}
        """
        outputs = {}
        for h in self.horizons:
            outputs[f'h{h}'] = {}
            for tau in self.taus:
                q_name = f'q{int(tau*100)}'
                outputs[f'h{h}'][q_name] = self.heads[f'h{h}'][q_name](z).squeeze(-1)
        
        return outputs


class PinballLoss(nn.Module):
    """ピンボール損失（分位点回帰用）"""
    def __init__(self, tau):
        super().__init__()
        self.tau = tau
    
    def forward(self, pred, target):
        """
        Args:
            pred: [B] 分位点予測
            target: [B] 真の値
        """
        error = target - pred
        loss = torch.maximum(self.tau * error, (self.tau - 1) * error)
        return loss.mean()


class APEXRankerV1(APEXRankerV0):
    """
    v1: リスク管理版
    - v0 + 分位点ヘッド
    - リスク調整スコア = median - λ×P10
    """
    def __init__(self, config):
        super().__init__(config)
        
        # 分位点ヘッド追加
        self.quantile_head = QuantileHead(
            config.model.d_model,
            config.head.horizons,
            config.head.quantiles
        )
        
        # ピンボール損失
        self.pinball_losses = {
            tau: PinballLoss(tau)
            for tau in config.head.quantiles
        }
        
        # リスク調整係数
        self.risk_lambda = config.head.get('risk_lambda', 0.5)
    
    def forward(self, X, return_quantiles=False):
        z, _ = self.encoder(X)
        
        # 中央値予測（ランキング用）
        scores = {}
        for h in self.config.head.horizons:
            scores[f'h{h}'] = self.heads[f'h{h}'](z).squeeze(-1)
        
        if return_quantiles:
            # 分位点予測
            quantiles = self.quantile_head(z)
            
            # リスク調整スコア
            adjusted_scores = {}
            for h in self.config.head.horizons:
                median = quantiles[f'h{h}']['q50']
                p10 = quantiles[f'h{h}']['q10']
                adjusted_scores[f'h{h}'] = median - self.risk_lambda * p10
            
            return scores, quantiles, adjusted_scores
        
        return scores
    
    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        
        X = batch['X']
        Y = batch['y']
        
        scores, quantiles, _ = self(X, return_quantiles=True)
        
        total_loss = 0.0
        logs = {}
        
        for i, h in enumerate(self.config.head.horizons):
            y = Y[:, i]
            
            if torch.isclose(y.std(), torch.tensor(0., device=y.device)):
                continue
            
            # ランキング損失（中央値で）
            s = scores[f'h{h}']
            ranking_loss = self.criterion(s, y)
            total_loss += ranking_loss
            logs[f'train_ranking_h{h}'] = ranking_loss
            
            # 分位点損失
            for tau in self.config.head.quantiles:
                q_name = f'q{int(tau*100)}'
                q_pred = quantiles[f'h{h}'][q_name]
                pinball_loss = self.pinball_losses[tau](q_pred, y)
                total_loss += pinball_loss * 0.1  # 重み調整
                logs[f'train_pinball_{q_name}_h{h}'] = pinball_loss
        
        self.log_dict(logs, on_step=False, on_epoch=True)
        
        return total_loss
```

#### 検証基準
- [ ] **分位校正**: P10超過率 ≈ 10% (±1%)
- [ ] **リスク削減**: MDD改善（-25% → -20%以下）
- [ ] **Sharpe向上**: +15%以上（v0比）

---

### **Phase 3: v2実装（クロスセクション）**（1-2週間）

#### タスク
- [ ] AdaptiveKNNGraph実装
- [ ] GAT統合
- [ ] 市場レジーム検出

#### コア実装

```python
from torch_geometric.nn import GATConv

class AdaptiveKNNGraph(nn.Module):
    """
    市場状態に応じてKを動的調整するKNNグラフ注意
    
    Args:
        d_model: 特徴次元
        k_min: 最小近傍数
        k_max: 最大近傍数
        n_heads: GATヘッド数
    """
    def __init__(self, d_model=192, k_min=5, k_max=30, n_heads=8):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        
        # 市場レジーム検出器
        self.regime_detector = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 0-1で市場の結合度を推定
        )
        
        # Graph Attention
        self.gat = GATConv(
            d_model, d_model // n_heads,
            heads=n_heads,
            concat=True,
            dropout=0.1,
            add_self_loops=False
        )
        
        self.fusion = nn.Linear(d_model, d_model)
    
    def forward(self, z):
        """
        Args:
            z: [B, d_model] 各銘柄の埋め込み
        Returns:
            z_enhanced: [B, d_model] グラフ文脈を含む埋め込み
            k_used: int 使用した近傍数
        """
        B = z.size(0)
        
        # 1. 市場レジームの推定
        regime = self.regime_detector(z).mean()  # スカラー
        k = int(self.k_min + (self.k_max - self.k_min) * regime.item())
        k = min(k, B - 1)  # バッチサイズを超えないように
        
        # 2. KNN グラフ構築
        # コサイン類似度
        z_norm = F.normalize(z, p=2, dim=1)
        sim = torch.mm(z_norm, z_norm.t())  # [B, B]
        
        # 自己ループを除外
        sim = sim.fill_diagonal_(-float('inf'))
        
        # Top-K選択
        topk_sim, topk_idx = torch.topk(sim, k=k, dim=1)  # [B, k]
        
        # エッジリスト構築
        edge_list = []
        edge_attr_list = []
        for i in range(B):
            for j_local in range(k):
                j_global = topk_idx[i, j_local].item()
                edge_list.append([i, j_global])
                edge_attr_list.append(topk_sim[i, j_local].item())
        
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=z.device).t()  # [2, E]
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float, device=z.device).unsqueeze(-1)  # [E, 1]
        
        # 3. GAT適用
        z_gat = self.gat(z, edge_index, edge_attr=edge_attr)  # [B, d_model]
        
        # 4. 元の埋め込みと統合
        z_enhanced = self.fusion(z + z_gat)
        
        return z_enhanced, k


class APEXRankerV2(APEXRankerV1):
    """
    v2: クロスセクション注意版
    - v1 + AdaptiveKNNGraph
    """
    def __init__(self, config):
        super().__init__(config)
        
        # KNNグラフ層追加
        self.knn_graph = AdaptiveKNNGraph(
            d_model=config.model.d_model,
            k_min=config.model.get('k_min', 5),
            k_max=config.model.get('k_max', 30),
            n_heads=config.model.n_heads
        )
    
    def forward(self, X, return_quantiles=False):
        z_base, _ = self.encoder(X)
        
        # クロスセクション文脈追加
        z, k_used = self.knn_graph(z_base)
        
        # ランキングスコア
        scores = {}
        for h in self.config.head.horizons:
            scores[f'h{h}'] = self.heads[f'h{h}'](z).squeeze(-1)
        
        if return_quantiles:
            quantiles = self.quantile_head(z)
            adjusted_scores = {}
            for h in self.config.head.horizons:
                median = quantiles[f'h{h}']['q50']
                p10 = quantiles[f'h{h}']['q10']
                adjusted_scores[f'h{h}'] = median - self.risk_lambda * p10
            
            return scores, quantiles, adjusted_scores
        
        return scores
```

#### 検証基準
- [ ] **RankIC向上**: +0.01-0.02（v1比）
- [ ] **K値の妥当性**: 危機時→大きく、安定時→小さく
- [ ] **計算時間**: v1比で+20%以内

---

### **Phase 4: 評価・最適化**（1-2週間）

#### タスク
- [ ] Purged Walk-Forward CV実装
- [ ] 評価指標完全実装
- [ ] ハイパーパラメータ最適化（Optuna）
- [ ] バックテストフレームワーク

#### 成果物
- 完全な評価レポート
- 最適ハイパーパラメータ
- バックテスト結果（複数コストシナリオ）

---

### **Phase 5: v3-v5（高度化）**（2-6ヶ月）

詳細は後述の「高度化ロードマップ」参照

---

# 詳細実装仕様

## 🔧 主要コンポーネント詳細

### 1. データパイプライン

#### FeatureSelector

```python
class FeatureSelector:
    """
    設定ファイルに基づいて特徴を動的に選択
    
    不足カテゴリを自動スキップし、後から追加可能
    """
    def __init__(self, config_path, metadata_path):
        self.config = self._load_config(config_path)
        self.metadata = self._load_metadata(metadata_path)
        
        self.selected_cols = self._select_features()
    
    def _select_features(self):
        """include/excludeに基づいて特徴列を確定"""
        include = self.config['data']['features']['include']
        exclude = self.config['data']['features']['exclude']
        
        selected = []
        for group_name, group_info in self.metadata['feature_groups'].items():
            if group_info['enabled'] and group_name in include and group_name not in exclude:
                selected.extend(group_info['columns'])
        
        print(f"Selected {len(selected)} features from {len(include)} groups")
        return selected
    
    def get_feature_count(self):
        """有効特徴数を返す"""
        return len(self.selected_cols)
    
    def validate_data(self, df):
        """データに必要な列が存在するか検証"""
        missing = set(self.selected_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return True
```

#### 正規化

```python
class CrossSectionNormalizer:
    """
    当日内クロスセクションZ-score正規化
    
    学習時: train統計でfit
    推論時: 当日統計で変換（リーケージ回避）
    """
    def __init__(self, clip_sigma=5.0):
        self.clip_sigma = clip_sigma
        self.train_stats = None
    
    def fit(self, X_train):
        """
        学習データで統計計算
        
        Args:
            X_train: [N_days, N_stocks, L, F] または list of [N_stocks, L, F]
        """
        all_values = []
        for day_data in X_train:
            # 各日のデータを集約
            all_values.append(day_data.reshape(-1, day_data.shape[-1]))
        
        all_values = np.concatenate(all_values, axis=0)  # [N_total, F]
        
        self.train_stats = {
            'mean': np.mean(all_values, axis=0),
            'std': np.std(all_values, axis=0) + 1e-8
        }
    
    def transform(self, X, use_train_stats=False):
        """
        正規化実行
        
        Args:
            X: [B, L, F] 同日バッチ
            use_train_stats: Trueなら学習統計、Falseなら当日統計
        """
        if use_train_stats and self.train_stats is not None:
            # 学習統計で正規化（学習時）
            mean = self.train_stats['mean']
            std = self.train_stats['std']
        else:
            # 当日統計で正規化（推論時）
            mean = X.mean(dim=(0, 1), keepdim=True)  # [1, 1, F]
            std = X.std(dim=(0, 1), keepdim=True) + 1e-8
        
        X_norm = (X - mean) / std
        
        # 外れ値クリッピング
        X_norm = torch.clamp(X_norm, -self.clip_sigma, self.clip_sigma)
        
        return X_norm
```

### 2. 学習・評価フレームワーク

#### Purged Walk-Forward Split

```python
class PurgedWalkForwardSplit:
    """
    時系列を考慮した検証分割
    
    - Purging: テスト期間前のembargo日数を学習から除外
    - Embargo: テスト期間との重複回避
    """
    def __init__(self, n_splits=6, embargo_days=20, test_ratio=0.15):
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.test_ratio = test_ratio
    
    def split(self, dates):
        """
        Args:
            dates: 昇順のリスト
        Yields:
            (train_dates, val_dates, test_dates)
        """
        dates = sorted(dates)
        N = len(dates)
        
        # 分割点計算
        split_size = N // self.n_splits
        
        for i in range(self.n_splits):
            # テスト期間
            test_start = i * split_size
            test_end = min((i + 1) * split_size, N)
            test_dates = dates[test_start:test_end]
            
            # Embargo適用
            train_end = max(0, test_start - self.embargo_days)
            
            # 学習期間（テスト開始前まで）
            train_dates = dates[:train_end]
            
            # 検証期間（学習の末尾20%程度）
            val_size = int(len(train_dates) * 0.2)
            val_dates = train_dates[-val_size:] if val_size > 0 else []
            train_dates = train_dates[:-val_size] if val_size > 0 else train_dates
            
            if len(train_dates) < 100:  # 最低日数チェック
                continue
            
            yield {
                'fold': i,
                'train': train_dates,
                'val': val_dates,
                'test': test_dates
            }
```

#### 評価指標

```python
class RankingMetrics:
    """ランキング評価指標のコレクション"""
    
    @staticmethod
    def rank_ic(scores, returns):
        """Spearman Rank IC"""
        return spearmanr(scores, returns)[0]
    
    @staticmethod
    def rank_icir(daily_ics):
        """ICIR = mean(IC) / std(IC)"""
        return np.mean(daily_ics) / (np.std(daily_ics) + 1e-8)
    
    @staticmethod
    def precision_at_k(scores, returns, k):
        """上位K件の正解率"""
        topk_idx = np.argsort(scores)[-k:]
        topk_returns = returns[topk_idx]
        return (topk_returns > 0).mean()
    
    @staticmethod
    def top_k_sharpe(daily_portfolio_returns):
        """Top-Kポートフォリオの年率Sharpe"""
        mean_ret = np.mean(daily_portfolio_returns)
        std_ret = np.std(daily_portfolio_returns)
        sharpe = mean_ret / (std_ret + 1e-8) * np.sqrt(252)
        return sharpe
    
    @staticmethod
    def max_drawdown(cumulative_returns):
        """最大ドローダウン"""
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
```

---

# 検証プロトコル

## ✅ 多層検証戦略

### レベル1: ユニットテスト

```python
# tests/test_encoder.py
def test_patchtst_shape():
    """形状テスト"""
    B, L, F = 64, 180, 109
    model = PatchTSTEncoder(in_feats=F, d_model=192, depth=3)
    
    X = torch.randn(B, L, F)
    z, tokens = model(X)
    
    assert z.shape == (B, 192)
    # パッチ数 = floor((L - patch_len) / stride) + 1
    expected_patches = (180 - 16) // 8 + 1  # = 21
    assert tokens.shape == (B, expected_patches, 192)


def test_loss_invariance():
    """損失の置換不変性"""
    B = 100
    scores = torch.randn(B)
    labels = torch.randn(B)
    
    loss_fn = ListNetLoss()
    loss1 = loss_fn(scores, labels)
    
    # 同じ順序でシャッフル
    perm = torch.randperm(B)
    loss2 = loss_fn(scores[perm], labels[perm])
    
    assert torch.isclose(loss1, loss2, atol=1e-5)


def test_same_return_day():
    """同値日のスキップ"""
    scores = torch.randn(100)
    labels = torch.ones(100) * 3.14  # 全て同値
    
    loss_fn = ListNetLoss()
    loss = loss_fn(scores, labels)
    
    assert torch.isclose(loss, torch.tensor(0.0))
```

### レベル2: 統合テスト

```python
# tests/test_integration.py
def test_overfit_sanity():
    """過学習サニティチェック"""
    # 1日×32銘柄の小データで完全過学習できるか
    B, L, F = 32, 180, 109
    
    X = torch.randn(B, L, F)
    y = torch.randn(B)
    
    model = APEXRankerV0(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(1000):
        scores = model(X)['h5']
        loss = model.criterion(scores, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            ic = spearman_rank_correlation(scores, y)
            print(f"Epoch {epoch}: IC = {ic:.4f}")
    
    final_ic = spearman_rank_correlation(scores, y)
    assert final_ic > 0.9, f"Failed to overfit: IC={final_ic}"


def test_full_pipeline():
    """フルパイプラインテスト"""
    # データ読み込み → 学習 → 評価
    config = load_config('configs/v0_base.yaml')
    
    # データセット構築
    dataset = DayPanelDataset(...)
    train_loader = DataLoader(dataset, batch_size=None, collate_fn=collate_day_batch)
    
    # モデル
    model = APEXRankerV0(config)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=5,
        devices=1,
        accelerator='gpu',
        precision='16-mixed'
    )
    
    # 学習
    trainer.fit(model, train_loader)
    
    # 検証
    metrics = trainer.validate(model, train_loader)
    
    assert metrics[0]['val_RankIC_h5'] > 0.0
```

### レベル3: バックテストシミュレーション

```python
class BacktestEngine:
    """コスト考慮型バックテストエンジン"""
    
    def __init__(self, cost_bps=10, initial_capital=1e6):
        self.cost_bps = cost_bps
        self.initial_capital = initial_capital
    
    def run(self, predictions, returns, top_k=50):
        """
        Args:
            predictions: {date: {code: score}}
            returns: {date: {code: return}}
            top_k: ポートフォリオサイズ
        
        Returns:
            metrics: {'sharpe', 'mdd', 'total_return', ...}
        """
        daily_returns = []
        positions = {}  # 前日ポジション
        
        for date in sorted(predictions.keys()):
            # 今日のスコア
            scores = predictions[date]
            
            # Top-K選択
            sorted_codes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            new_positions = {code: 1.0/top_k for code, _ in sorted_codes[:top_k]}
            
            # ターンオーバー計算
            turnover = self._compute_turnover(positions, new_positions)
            
            # コスト
            cost = turnover * self.cost_bps / 10000
            
            # リターン
            port_return = sum(
                weight * returns[date].get(code, 0)
                for code, weight in new_positions.items()
            )
            
            # コスト控除後リターン
            net_return = port_return - cost
            daily_returns.append(net_return)
            
            positions = new_positions
        
        # 指標計算
        daily_returns = np.array(daily_returns)
        cumulative = np.cumprod(1 + daily_returns)
        
        metrics = {
            'sharpe': RankingMetrics.top_k_sharpe(daily_returns),
            'mdd': RankingMetrics.max_drawdown(cumulative),
            'total_return': cumulative[-1] - 1,
            'avg_turnover': np.mean([self._compute_turnover(...) for ...]),
        }
        
        return metrics, daily_returns
    
    def _compute_turnover(self, old_pos, new_pos):
        """ポートフォリオ変更率"""
        all_codes = set(old_pos.keys()) | set(new_pos.keys())
        turnover = sum(
            abs(new_pos.get(c, 0) - old_pos.get(c, 0))
            for c in all_codes
        )
        return turnover / 2  # 片道
```

---

# 運用・最適化

## 🚀 GPU最適化実践

### 1. メモリ最適化

```python
# Gradient Checkpointing
class MemoryEfficientPatchTST(PatchTSTEncoder):
    def forward(self, x):
        x = x.transpose(1, 2)
        tokens = self.patch_embed(x)
        
        # Transformerブロックでcheckpoint適用
        for block in self.blocks:
            tokens = checkpoint(block, tokens, use_reentrant=False)
        
        tokens = self.norm(tokens)
        z = tokens.mean(dim=1)
        
        return z, tokens


# Flash Attention（PyTorch 2.0+）
from torch.nn.functional import scaled_dot_product_attention

class FlashAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Flash Attention（PyTorch組み込み）
        out = scaled_dot_product_attention(q, k, v, is_causal=False)
        
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        
        return out
```

### 2. 分散学習（DDP）

```python
# scripts/train_ddp.py
import torch.distributed as dist
import torch.multiprocessing as mp

def main_worker(rank, world_size, config):
    # プロセス初期化
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://localhost:12355',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(rank)
    
    # モデル
    model = APEXRankerV0(config).cuda(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    # データローダー（各プロセスで異なるfoldを担当）
    fold_ids = list(range(6))
    my_fold = fold_ids[rank % len(fold_ids)]
    
    dataset = get_fold_dataset(my_fold)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=None, sampler=sampler)
    
    # Trainer
    trainer = pl.Trainer(
        devices=[rank],
        strategy='ddp',
        precision='16-mixed'
    )
    
    trainer.fit(model, loader)
    
    dist.destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(world_size, config), nprocs=world_size)
```

### 3. ハイパーパラメータ最適化

```python
import optuna

def objective(trial):
    # ハイパーパラメータ提案
    config = {
        'd_model': trial.suggest_categorical('d_model', [128, 192, 256]),
        'depth': trial.suggest_int('depth', 2, 6),
        'patch_len': trial.suggest_categorical('patch_len', [12, 16, 20]),
        'stride': trial.suggest_categorical('stride', [6, 8, 12]),
        'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
        'weight_decay': trial.suggest_loguniform('wd', 1e-5, 1e-3),
        'listnet_weight': trial.suggest_uniform('list_w', 0.5, 0.9),
    }
    
    # モデル構築
    model = APEXRankerV0(config)
    
    # 学習
    trainer = pl.Trainer(max_epochs=20, devices=1, precision='16-mixed')
    trainer.fit(model, train_loader, val_loader)
    
    # 評価
    metrics = trainer.validate(model, val_loader)
    val_ic = metrics[0]['val_RankIC_h5']
    
    return val_ic


# 最適化実行
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, n_jobs=4)

print("Best params:", study.best_params)
print("Best IC:", study.best_value)
```

---

## 📊 モニタリング・可視化

### TensorBoard統合

```python
from pytorch_lightning.loggers import TensorBoardLogger

logger = TensorBoardLogger('logs/', name='apex_ranker')

trainer = pl.Trainer(
    logger=logger,
    log_every_n_steps=10
)

# カスタムログ
class APEXRankerV0(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        # ... (損失計算)
        
        # カスタムログ
        self.log('gpu_memory', torch.cuda.memory_allocated() / 1e9)  # GB
        self.log('batch_size', len(batch['X']))
        
        return loss
```

### Weights & Biases統合

```python
import wandb
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger(project='apex-ranker', name='v0_baseline')

trainer = pl.Trainer(logger=wandb_logger)

# 予測分布の可視化
wandb.log({
    'score_distribution': wandb.Histogram(scores.cpu().numpy()),
    'ic_over_time': wandb.plot.line_series(
        xs=dates,
        ys=[daily_ics],
        keys=['RankIC'],
        title='Daily RankIC'
    )
})
```

---

## 🔒 再現性の確保

```python
def set_seed(seed=42):
    """完全な再現性のためのシード設定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 決定的アルゴリズム（速度低下あり）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch Lightningの設定
    pl.seed_everything(seed, workers=True)


# 実行時
set_seed(42)

trainer = pl.Trainer(
    deterministic=True,  # 完全再現モード
    ...
)
```

---

## 📦 デプロイメント

### 推論API（FastAPI）

```python
from fastapi import FastAPI
import onnxruntime as ort

app = FastAPI()

# モデルロード（ONNX変換済み）
session = ort.InferenceSession(
    'models/apex_ranker_v2.onnx',
    providers=['CUDAExecutionProvider']
)

@app.post('/predict')
async def predict(data: dict):
    """
    Args:
        data: {'stocks': [{'code': 'AAPL', 'features': [...]}]}
    Returns:
        {'predictions': [{'code': 'AAPL', 'score': 0.85}]}
    """
    # 前処理
    X = preprocess(data['stocks'])  # [B, L, F]
    
    # 推論
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X})
    
    scores = outputs[0]  # [B, H]
    
    # 後処理
    results = [
        {'code': stock['code'], 'score': float(score[2])}  # 5d先のスコア
        for stock, score in zip(data['stocks'], scores)
    ]
    
    return {'predictions': results}
```

---

## 🎯 まとめ

### 実装の段階的進化

| フェーズ | 期間 | 成果物 | 性能目標 |
|---------|------|--------|---------|
| **Phase 0** | 3-5日 | 環境構築 | GPU認識・データロード |
| **Phase 1** | 1-2週 | v0実装 | IC>0.05, Sharpe>1.2 |
| **Phase 2** | 1週 | v1実装 | MDD改善-15% |
| **Phase 3** | 1-2週 | v2実装 | IC>0.08 |
| **Phase 4** | 1-2週 | 評価・最適化 | 本番準備完了 |
| **Phase 5** | 2-6ヶ月 | v3-v5 | IC>0.10, Sharpe>2.5 |

### 重要な設計原則

1. **モジュール分離**: データ不足でも動作、後から拡張可能
2. **段階的検証**: 各フェーズで明確な目標と検証基準
3. **GPU最適化**: 実用的な計算時間を実現
4. **再現性**: 完全な再現と効果測定

### 次のステップ

```bash
# 環境構築
conda create -n apex-ranker python=3.10
conda activate apex-ranker
pip install -r requirements.txt

# データ準備
python scripts/prepare_data.py

# v0学習
python scripts/train_v0.py --config configs/v0_base.yaml

# 評価
python scripts/evaluate.py --checkpoint logs/version_0/checkpoints/best.ckpt
```

---

**実装支援が必要な場合は、各スクリプトの完全版、テストコード、Docker設定等を提供可能です！**
