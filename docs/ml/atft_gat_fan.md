# ATFT-GAT-FAN モデル仕様（Graph Attention Network 含む）

本ドキュメントは **ATFT-GAT-FAN** アーキテクチャのうち、Graph Attention Network (GAT) を含む推論パイプラインと学習設定を記録する。`docs/ml/dataset.md` がデータセット仕様を、`docs/ml/dataset_ingestion_strategy.md` が取得オペレーションの補足ノートを扱うのに対し、本稿は **モデル側の構成・入出力・訓練フロー・監視指標** をまとめる。

## 1. モデル全体像

| ブロック | 役割 | 主なハイパーパラメータ | 実装パス |
| --- | --- | --- | --- |
| Input Projection | 373次元の特徴量を `hidden_size=64` に射影 | LayerNorm, Dropout=0.1 | `ATFT_GAT_FAN._build_input_projection` |
| Adaptive Temporal Fusion Transformer (ATFT) | 時系列エンコーダ（sequence length=20） | LSTM layers=1, heads=4, dropout=0.1 | `TemporalFusionTransformer` |
| Graph Attention Network (GAT) | 銘柄間の相関を伝搬 | num_layers=2, heads=[4,2], edge_dim=3 | `MultiLayerGAT` (`gat_layer.py`) |
| Frequency / Slice Adaptive Normalization | 周波数ドメイン正則化 (FAN + SAN) | window_sizes=[5,10,20], num_slices=3 | `FrequencyAdaptiveNorm`, `SliceAdaptiveNorm` |
| Prediction Head | multi-horizon + quantile + Student-t | hidden=[32], dropout=0.2, quantiles=[0.1..0.9] | `PredictionHead` |

### 入力テンソル

```
batch = {
    "dynamic_features": Tensor[batch, seq_len=20, feature_dim=373],
    "static_features": Optional Tensor,
    "regime_features": Optional Tensor,
    "graph": {
        "edge_index": Tensor[2, E],
        "edge_attr": Tensor[E, 3],
        "node_mask": Tensor[num_nodes],
    }
}
```

### 出力

```
{
  "predictions": {
    "horizon_1d": Tensor[batch, targets],
    "horizon_5d": ...,
    "horizon_10d": ...,
    "horizon_20d": ...
  },
  "features": Tensor[batch, seq_len, hidden],
  "output_type": "multi_horizon"
}
```

## 2. グラフ構築パイプライン

- 実装: `src/data/utils/graph_builder.py` (`FinancialGraphBuilder`)
- 入力データ: 日次リターン `return_1d`, `feat_ret_1d` とメタデータ (`sector33`, `log_mktcap`)
- 主要パラメータ（デフォルトは `configs/model/atft_gat_fan.yaml`, `configs/atft/config_production_optimized.yaml`）:
  - `correlation_window`: 60 営業日
  - `min_observations`: 40
  - `correlation_threshold`: 0.25〜0.30（環境依存で変更可）
  - `max_edges_per_node`: 10〜15
  - `correlation_method`: `ewm_demean`（半減期 `ewm_halflife`=30）
  - `shrinkage_gamma`: 0.1（Ledoit-Wolf 風の収縮）
  - `symmetric`: true（双方向エッジ）
  - `edge_attr`: 3次元 `[corr_norm, market_similarity, sector_similarity]`
  - キャッシュ: `output/graph_cache/*.pkl`（gzip圧縮）
- 更新頻度: 週次（`update_frequency='weekly'`）。学習時は各バッチ日に対し最新グラフを呼び出す（ない場合は再計算）。

### エッジ抽出手順
1. 指定期間 (`date_end - correlation_window`) のリターン行列を作成。
2. 指数加重平均で demean / 分散調整。
3. 相関が閾値を超えた銘柄ペアを抽出。
4. 銘柄ごとに `max_edges_per_node` を保持（絶対相関の強い順）。
5. エッジ属性を計算:
   - `corr_norm = sign(corr) * min(|corr|, 0.99)`
   - `market_similarity = 1.0` if MarketCode 一致 else 0.0（または距離）
   - `sector_similarity = 1.0` if 33業種一致 else 0.0
6. `edge_index` (2×E) と `edge_attr` (E×3) を PyTorch Tensor に変換。

### グラフ特徴量と GAT への受け渡し
- グラフは PyG 互換形式で GAT 層へ渡される。
- ノード埋め込み: ATFT 出力（最終時間ステップ側、次元 `hidden_size=64`）。
- `MultiLayerGAT` が 2 層構成で処理し、最終層は `concat=False` により `[batch, num_nodes, hidden_size]` に再射影。
- 未接続ノードはゼロ埋めのまま `gat_features=None` として扱われ、最後の結合で zero pad される。

## 3. GAT ブロック詳細

| 層 | 入力チャネル | ヘッド数 | 出力 | Dropout | その他 |
| --- | --- | --- | --- | --- | --- |
| GAT Layer 1 | 64 | 4 | 64×4 (concat) | dropout=0.2 | edge_dropout=0.1, LeakyReLU (α=0.2) |
| GAT Layer 2 | 64×4 | 2 | 64 (concat=False) | dropout=0.2 | self-loop なし |

- `edge_dim=3` の属性を線形射影して attention に加算 (`edge_projection=linear`)。
- 正則化:
  - `edge_weight_penalty=0.01` L2 (edge attention のスパース化)
  - `attention_entropy_penalty=0.001`（エントロピーペナルティ）
- `gat_residual_gate`（学習可能スカラー）で TFT とのブレンド:
  - `alpha = sigmoid(gate)` 初期値 0.5
  - 出力: `alpha * projection(TFT+GAT) + (1-alpha) * GAT`
- 追加安全装置:
  - `EDGE_DROPOUT_INPUT_P=0.1` (環境変数) → 入力グラフのランダムスパース化
  - Attention 勾配監視フック（小さすぎる勾配を WARN ログに残す）

## 4. トレーニングフロー

### 4.1 データローダ設定
- `sequence_length=20`、`prediction_horizons=[1,5,10,20]`
- データソース: Parquet (`output/atft_data/*.parquet`)、オンライン正規化 (`PyArrow streaming`)
- ミニバッチ: 1,024〜2,048（A100 80GB 推奨）
- メモリ: `USE_GPU_ETL=1` + RAPIDS (`cudf`, `rmm`, `cupy`)

### 4.2 損失関数
- メイン: Pinball Loss（quantiles: 0.1, 0.25, 0.5, 0.75, 0.9）
- オプション: Student-t NLL（`USE_T_NLL=1`, `OUTPUT_NOISE_STD=0.02`）
- 補助:
  - Huber Loss δ=0.01（`improvements.huber_loss=true`）
  - Sharpe Loss（任意）
  - Pairwise Ranking Loss（任意、top-k 選択可）
  - Decision Layer（任意、alpha=2.0）

### 4.3 環境変数・ウォームアップ

| キー | 役割 | 既定値 |
| --- | --- | --- |
| `GAT_ALPHA_INIT` | GAT 残差ゲート初期値 | 0.3 |
| `GAT_ALPHA_MIN` | 下限値スケジューリング | 0.1 |
| `GAT_ALPHA_PENALTY` | ゲートの正則化 | 1e-3 |
| `EDGE_DROPOUT_INPUT_P` | 入力エッジの dropout | 0.1 |
| `DEGENERACY_GUARD` | attention 崩壊監視 | 1 |
| `USE_AMP` | Mixed Precision | 1 (`bf16`) |

- Phase1: `FUSE_FORCE_MODE=tft_only` で TFT にウォームアップ → Phase2 で GAT をブレンド。
- EMA Teacher (`decay=0.995`) が出力を滑らかにし、Sharpe 指標を安定化。
- Optimizer: AdamW (weight decay=1e-4), gradient clip=1.0。
- Learning rate: スケジューラ (OneCycle or Cosine) を Hydra config 経由で選択。

### 4.4 GPU 前提チェック
- `scripts/pipelines/run_full_dataset.py` は **RAPIDS (cuDF, CuPy, RMM)** を検知できない場合に即エラーで停止する。CPU フォールバックは無効。
- 本番ワークフローでも `USE_GPU_ETL=1` が要求される。

## 5. 推論パス

1. 直近 20 日の `dynamic_features` を正規化して取得 (`MemoryMapDataLoader`)。
2. グラフキャッシュから当日分の `edge_index`, `edge_attr` をロード（なければ再計算）。
3. Input Projection → ATFT → (GAT ブロック) → FAN/SAN → Prediction Head の順で前向き計算。
4. Multi-horizon 予測を Dictionary 形式で返却（Point + Quantile + Optional Student-t）。
5. 後段では Sharpe/Calmar 等の評価関数を経てポジション決定。

## 6. 監視・ロギング

| 指標 | ログ先 | 説明 |
| --- | --- | --- |
| `train/loss_total` | W&B, TensorBoard | Pinball + 補助損失の合計 |
| `train/gat_entropy` | W&B | attention entropy 正則化の監視 |
| `train/gat_alpha` | W&B | TFT/GAT ブレンド比率 |
| `train/graph_edge_coverage` | ログ | グラフにエッジが何本残っているか |
| `eval/sharpe`, `eval/hit_ratio` | W&B | 時系列評価メトリクス |
| `hardware/gpu_mem`, `hardware/gpu_util` | TensorBoard | GPU 資源の使用状況 |

## 7. 実装ファイル一覧

| コンポーネント | ファイル |
| --- | --- |
| ATFT-GAT-FAN main module | `src/atft_gat_fan/models/architectures/atft_gat_fan.py` |
| GAT Layer 定義 | `src/atft_gat_fan/models/components/gat_layer.py` |
| グラフビルダー (CPU) | `src/data/utils/graph_builder.py` |
| グラフビルダー (GPU) | `src/data/utils/graph_builder_gpu.py` |
| モデル設定 | `configs/model/atft_gat_fan.yaml`, `configs/atft/config_production_optimized.yaml` |
| 環境変数管理 | `src/gogooku3/training/atft/environment.py` |
| メタデータ・セクター結合 | `scripts/data/ml_dataset_builder.py` |

## 8. 今後の拡張 TODO

- [ ] GAT α の動的スケジューラ（市場 regime に応じたブレンド比率調整）
- [ ] Edge 属性の拡張（例えば β 係数差分、流動性距離など）
- [ ] Graph augmentation の強化（複数の相関モードを ensemble）
- [ ] GPU グラフ構築 (`graph_builder_gpu.py`) の安定化と A/B 検証
- [ ] 低流動銘柄向けに `edge_threshold` を銘柄ごとに最適化する実験

---

このドキュメントは GAT を含むモデル本体の仕様書として運用し、データセット定義 (`docs/ml/dataset.md`) と組み合わせることで **ATFT-GAT-FAN パイプライン全体** が追跡可能になる。追加のアーキテクチャ変更やハイパーパラメータ調整を行った際は本書も更新すること。***
