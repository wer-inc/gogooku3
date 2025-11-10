# ATFT-GAT-FAN: Advanced Trading with Financial Transformers

**最新のAI技術で強化された高性能金融予測システム**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.6+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Private-black.svg)]()

> Contributors: See the Repository Guidelines in [docs/development/agents.md](docs/development/agents.md). Enable hooks with `pre-commit install` and `pre-commit install -t commit-msg`.

## 🎯 概要

ATFT-GAT-FANは、最新の深層学習技術を活用した高性能金融時系列予測システムです。**ATFT-GAT-FAN**アーキテクチャにより、従来比**+20%**の予測精度向上を実現しています。

### 🚧 進行中の課題
- 市場センチメントを捉える外部ボラティリティ指数（例: 日経平均VI、TOPIXボラティリティ指数）の取得・統合を追加予定。現状のATRなど実現ボラティリティ指標のみでは期待ボラティリティ情報が不足しており、マクロモジュール強化のためデータ統合パイプラインを整備する。

### 🚀 主な特徴

- **🧠 最新アーキテクチャ**: ATFT-GAT-FAN (Adaptive Temporal Fusion Transformer + Graph Attention + Frequency Adaptive Normalization)
- **📈 性能向上**: RankIC@1d **+20.0%**, 学習時間**-6.7%**, 損失**-10.0%**
- **⚡ 高効率**: GPU最適化、メモリ効率化、並列処理
- **🛡️ 堅牢性**: 自動回復、OOM対策、継続監視
- **🔧 運用性**: 統合モニタリング、自動アラート、段階的デプロイ

### 🎯 最新の改善実装

#### ✅ 実装済み改善機能
- **出力ヘッド最適化**: Small-init + LayerScaleで予測安定性向上
- **FreqDropout**: 周波数ドメイン正則化による過学習防止
- **EMA Teacher**: 指数移動平均による学習安定化
- **Huber損失**: 外れ値耐性向上
- **ParamGroup最適化**: 層別学習率設定
- **監視システム**: W&B + TensorBoard統合
- **堅牢性向上**: OOM自動回復、緊急チェックポイント
- **Premiumデータ統合**: 前場四本値・売買内訳・配当・財務・先物・オプションの6系統データをPITで取り込み、`am_*` / `bd_*` / `div_*` / `fs_*` / `fut_*` / `opt_*` 特徴量を生成

#### 📊 性能改善結果
| 指標 | 改善前 | 改善後 | 改善率 | ステータス |
|------|--------|--------|--------|-----------|
| **RankIC@1d** | 0.150 | 0.180 | **+20.0%** | ✅ 目標達成 |
| **学習時間** | 10.5s | 9.8s | **+6.7%** | ✅ 目標達成 |
| **損失** | 0.050 | 0.045 | **+10.0%** | ✅ 目標達成 |
| **GPUスループット** | - | 5130 samples/sec | - | ✅ 高効率 |

### 🏆 チームレビュー結果
- **評価**: 4.5/5 (Excellent)
- **承認率**: 100%
- **ブロッキングイシュー**: 0件
- **本番デプロイ**: ✅ 承認済み

## 🚀 クイックスタート

### 1. 環境セットアップ（これだけでOK！）

```bash
# 全自動セットアップ - これ一つで完了
make setup
```

**自動的に実行される内容**:
- ✅ Python仮想環境作成
- ✅ 全依存パッケージインストール
- ✅ pre-commitフック設定
- ✅ .env設定ファイル作成
- ✅ GPU環境自動検出＆セットアップ（利用可能な場合）
- ✅ インストール検証

**セットアップ後の手動作業（1分）**:
```bash
# 1. 認証情報を編集
nano .env

# 2. 仮想環境を有効化
source venv/bin/activate
```

> ℹ️ `make setup`（内部では `scripts/setup_env.sh`）は、`gcloud` / `gsutil` が未導入の場合に Google Cloud SDK をローカルへダウンロードします。生成される `google-cloud-sdk/` フォルダとアーカイブは `.gitignore` 済みなので Git には追加しないでください。すでにシステムに `gcloud` が入っている環境ではダウンロード処理は自動的にスキップされます。

### 2. システム検証

```bash
# 🔍 スモークテスト（基本機能確認）
python scripts/smoke_test.py

# 📊 詳細性能検証
python scripts/validate_improvements.py --data output/ml_dataset_20250827_174908.parquet --detailed

# 🖥️ 監視ダッシュボード起動
python scripts/monitoring_dashboard.py --start-tensorboard
```

### 3. データセット生成

```bash
# SSH切断にも安全なバックグラウンド実行
make dataset-bg

# モニタ
tail -f _logs/dataset/*.log
```

#### 🔄 チャンク化パイプライン

長期間の再構築や増分更新は四半期チャンクで実行できます。ウォームアップは
85営業日固定で、自動的にカットされます。

```bash
# チャンク計画の確認（ドライラン）
make build-chunks START=2020-01-01 END=2020-12-31 DRY_RUN=1

# 完了済みをスキップしながら実行
make build-chunks START=2020-01-01 END=2020-12-31 RESUME=1

# 最新チャンクのみ（例: デイリー更新）
make build-chunks START=2024-01-01 END=2024-12-31 LATEST=1

# マージして最新データセットを更新
make merge-chunks

# 未完了チャンクを許容してマージする場合（明示的に指定）
make merge-chunks ALLOW_PARTIAL=1
# CLI を直接使う場合
python data/tools/merge_chunks.py --chunks-dir output/chunks --allow-partial
```

チャンクごとの `ml_dataset.parquet` / `metadata.json` / `status.json` は
`output/chunks/<chunk_id>/` に保存されます。詳細は
[docs/CHUNK_PIPELINE.md](docs/CHUNK_PIPELINE.md) を参照してください。
CI や自動ジョブからは `scripts/ci/run_chunked_build.sh` もしくは
`Chunked Dataset Build` ワークフロー（`workflow_dispatch`、self-hosted GPU）
で同じ手順を実行できます。マルチイヤーの実行前に `.env` で
`DATA_PREFETCH_THREADS=0` に切り替えておくと、余計なprefetchが抑制され
ピークメモリをさらに削減できます（短期間ジョブでは任意に戻してください）。

### 4. モデルトレーニング

```bash
# 🔥 推奨: 本番設定でのトレーニング
python -c "
from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN
import yaml
with open('configs/atft/config.yaml') as f:
    config = yaml.safe_load(f)
model = ATFT_GAT_FAN(config)
print('✅ ATFT-GAT-FAN model initialized successfully!')
"

# 📈 ハイパーパラメータチューニング
python scripts/hyperparameter_tuning.py --data output/ml_dataset_20250827_174908.parquet --method random --trials 20 --save-best

# 🏭 本番ワークロード検証
python scripts/production_validation.py --scenario medium_scale

# 📊 フィードバック収集
python scripts/collect_feedback.py --create-sample --summary

# 🌐 直接API取得
python main.py direct-api-dataset

# 🎯 完全ATFT学習（内製ルート）
python scripts/integrated_ml_training_pipeline.py

# 🗺 Hydra パイプライン実行例
# config 衝突を避けるため `--config-path ../configs/atft` を明示します
python scripts/integrated_ml_training_pipeline.py \
  --config-path ../configs/atft \
  --config-name config \
  --max-epochs 1

# CPU ベースライン（単一プロセス強制）
ACCELERATOR=cpu FORCE_SINGLE_PROCESS=1 \
  python scripts/integrated_ml_training_pipeline.py \
  --config-path ../configs/atft \
  --config-name config \
  --max-epochs 1

# 参考ログ:
#  - Hydra 衝突検証: output/reports/hydra_collision.log
#  - CPU ベンチ: output/reports/cpu_benchmark.log
#  - GPU ベンチ: output/reports/gpu_benchmark.log
```

## 🏗️ アーキテクチャ

### ATFT-GAT-FAN 概要

```
┌─────────────────────────────────────────────────────────┐
│                    ATFT-GAT-FAN                         │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Input      │  │  Temporal   │  │  Graph      │     │
│  │ Projection  │  │  Fusion     │  │  Attention  │     │
│  │             │  │  Transformer │  │  Network    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ Freq        │  │ EMA         │  │ Huber       │     │
│  │ Dropout     │  │ Teacher     │  │ Loss        │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ W&B         │  │ TensorBoard │  │ Auto        │     │
│  │ Monitor     │  │ Dashboard   │  │ Recovery    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### コアコンポーネント

#### 1. **Adaptive Temporal Fusion Transformer (ATFT)**
- 時系列データの適応的な融合
- 多変量特徴量の動的処理
- 短期・長期依存関係のモデル化

#### 2. **Graph Attention Network (GAT)**
- 銘柄間関係のモデル化
- 注意機構による重要度計算
- 市場構造の動的適応

#### 3. **Frequency Adaptive Normalization (FAN)**
- 周波数領域での特徴正則化
- FreqDropoutによる過学習防止
- スペクトル適応正則化

### 改善機能一覧

#### 🎯 モデル改善
- ✅ **Small-init + LayerScale**: 出力ヘッドの安定化
- ✅ **FreqDropout**: 周波数ドメイン正則化 (最適値: 0.2)
- ✅ **GAT温度パラメータ**: 注意機構最適化 (最適値: 0.8)
- ✅ **Edge Dropout**: グラフ構造のランダム化

#### 🔧 学習改善
- ✅ **Huber損失**: 外れ値耐性向上 (δ=0.01)
- ✅ **EMA Teacher**: 学習安定化 (decay=0.995)
- ✅ **ParamGroup最適化**: 層別学習率設定
- ✅ **Gradient Checkpointing**: メモリ効率化

#### 📊 データ処理改善
- ✅ **PyArrowストリーミング**: メモリマップ + オンライン正規化
- ✅ **ゼロコピーTensor変換**: CPUメモリ節約
- ✅ **チャンキング**: 大規模データ処理

#### 🛡️ 運用改善
- ✅ **W&B統合**: 実験追跡と可視化
- ✅ **TensorBoard統合**: リアルタイム監視
- ✅ **自動アラート**: 異常検知と通知
- ✅ **OOM自動回復**: メモリ不足時の自動対応

## 📊 性能仕様

### システム要件
- **Python**: 3.10+
- **PyTorch**: 2.0+
- **CUDA**: 12.6+
- **GPU**: A100/H100/V100 (推奨) または対応GPU
- **メモリ**: 16GB+ (推奨32GB+)

### 性能指標
- **モデルサイズ**: ~37Kパラメータ (軽量化)
- **GPUスループット**: 5,130 samples/sec (A100)
- **メモリ効率**: ピーク6.7GB (バッチ256)
- **学習安定性**: スコア0.748 (安定)

### スケーラビリティ
- **最小バッチ**: 256 samples
- **推奨バッチ**: 1024-2048 samples
- **最大バッチ**: 4096+ samples (GPU容量による)
- **並列処理**: DataLoader最適化済み

## 🔧 運用ガイド

### 監視システム

#### TensorBoard起動
```bash
# TensorBoard起動
python scripts/monitoring_dashboard.py --start-tensorboard --port 6006

# ブラウザでアクセス: http://localhost:6006
```

#### W&B設定
```bash
# W&B APIキー設定
export WANDB_API_KEY="your-api-key-here"

# W&Bログイン
python scripts/setup_monitoring.py --setup-wandb

# プロジェクトダッシュボード: https://wandb.ai/your-project/atft-gat-fan
```

#### 継続監視
```bash
# リアルタイム監視（バックグラウンド実行）
python scripts/monitoring_dashboard.py --continuous --interval 300 &

# アラートテスト
python scripts/alert_system.py --test
```

### トラブルシューティング

#### 🚨 よくある問題と解決法

##### 1. CUDAメモリ不足 (OOM)
```bash
# 解決策1: バッチサイズ削減
export CUDA_LAUNCH_BLOCKING=1
python scripts/production_validation.py --scenario small_scale

# 解決策2: メモリ最適化設定
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 解決策3: GPUメモリクリア
python -c "import torch; torch.cuda.empty_cache()"
```

##### 2. W&B接続エラー
```bash
# APIキー確認
echo $WANDB_API_KEY

# 再ログイン
python scripts/setup_monitoring.py --setup-wandb
```

##### 3. モデル初期化エラー
```bash
# 設定ファイル確認
python -c "
import yaml
with open('configs/atft/config.yaml') as f:
    config = yaml.safe_load(f)
    print('Config loaded successfully')
"

# 依存関係確認
python -c "import torch; print('PyTorch:', torch.__version__)"
```

##### 4. 学習不安定
```bash
# ハイパーパラメータ確認
python scripts/hyperparameter_tuning.py --data your_data.parquet --method random --trials 5 --save-best

# 学習率調整
# configs/atft/train/production.yaml の scheduler.gamma を調整
```

### 📈 パフォーマンス最適化

#### GPU最適化設定
```bash
# CUDA最適化
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# PyTorch最適化
export TORCH_USE_CUDA_DSA=1
```

#### メモリ最適化
```yaml
# configs/atft/config.yaml
improvements:
  memory_map: true
  compile_model: true  # PyTorch 2.0+
  gradient_checkpointing: true
```

#### 並列処理最適化
```yaml
# configs/atft/train/production.yaml
batch:
  num_workers: 8  # CPUコア数に合わせる
  prefetch_factor: 4
  persistent_workers: true
  pin_memory: true
```

## 📖 Documentation

### Training & Configuration Guides (v3.0)

- **[Training Commands Reference](docs/TRAINING_COMMANDS.md)** - Phase-by-phase execution commands with pre-flight checklists
- **[Model Input Dimensions Guide](docs/MODEL_INPUT_DIMS.md)** - Correct usage of `model.input_dims.*` parameters (avoid confusion!)
- **[Experiment Status & Evaluation Protocol](EXPERIMENT_STATUS.md)** - Weekly milestones, metrics calculation, escalation criteria

### Legacy Documentation

- **[Previous Experiments](EXPERIMENT_STATUS.md)** - Historical experiment results and findings

---

## 📚 API リファレンス

### 主要スクリプト

#### `scripts/smoke_test.py`
```bash
# 使用法
python scripts/smoke_test.py

# 説明: 基本機能の正常動作確認
# 出力: モデル初期化、フォワードパス、損失計算の検証結果
```

#### `scripts/validate_improvements.py`
```bash
# 使用法
python scripts/validate_improvements.py --data path/to/data.parquet --detailed

# 説明: 改善機能の性能検証
# 出力: before/after比較、RankIC、メモリ使用量など
```

#### `scripts/hyperparameter_tuning.py`
```bash
# 使用法
python scripts/hyperparameter_tuning.py --data path/to/data.parquet --method random --trials 20 --save-best

# 説明: ハイパーパラメータ最適化
# 出力: 最適パラメータ設定、最適スコア
```

#### `scripts/monitoring_dashboard.py`
```bash
# 使用法
python scripts/monitoring_dashboard.py --continuous --interval 300

# 説明: リアルタイムシステム監視
# 出力: CPU/GPU使用率、メモリ使用量、アラート
```

## 🤝 コントリビューション

### 開発ワークフロー
1. **ブランチ作成**: `git checkout -b feature/your-feature`
2. **テスト実行**: `python scripts/smoke_test.py`
3. **コードレビュー**: 改善点をレビュー
4. **マージ**: 承認後にmainブランチにマージ

### コーディング標準
- **Python**: PEP 8準拠
- **ドキュメント**: Googleスタイルdocstring
- **テスト**: pytest使用、coverage 80%以上
- **型ヒント**: 必須

## 📄 ライセンス

**Private License** - 社内利用限定

## 📞 サポート

### 連絡先
- **技術サポート**: tech-support@company.com
- **ドキュメント**: [内部Wikiリンク]
- **課題管理**: [JIRA/Issue Tracker]

### 緊急連絡
- **システム障害**: system-alert@company.com
- **セキュリティ問題**: security@company.com

---

## 🎯 次のステップ

1. **W&B APIキー設定** (必須)
2. **本番データでの検証**
3. **継続的チューニング**
4. **運用ドキュメント更新**

---

*最終更新: 2025-08-29*
*バージョン: v2.0.0*
*ステータス: 本番デプロイ完了 ✅*

# 出力例:
# 🚀 gogooku3-standalone - 壊れず・強く・速く
# 📈 金融ML システム統合実行環境
# Workflow: safe-training
# Mode: full
# ✅ 学習結果: エポック数: 10, 最終損失: 0.0234
```

## 📁 プロジェクト構造

```
gogooku3-standalone/
├── 🎬 main.py                          # メイン実行スクリプト
├── 📦 requirements.txt                 # 統合依存関係
├── 📋 README.md                        # このファイル
├── 🔧 scripts/                         # コア処理
│   ├── 🛡️ run_safe_training.py               # 7段階安全パイプライン
│   ├── 🎯 integrated_ml_training_pipeline.py  # ATFT完全統合（内製）
│   ├── 📊 data/
│   │   ├── ml_dataset_builder.py             # 強化データセット構築
│   │   └── direct_api_dataset_builder.py     # 直接API取得
│   ├── 🤖 models/                            # モデルコンポーネント
│   ├── 📈 monitoring_system.py               # 監視システム
│   ├── ⚡ performance_optimizer.py           # 性能最適化
│   └── ✅ quality/                           # 品質保証
├── 🏗️ src/                             # ソースモジュール
│   ├── data/          # データ処理コンポーネント
│   ├── models/        # モデルアーキテクチャ
│   ├── graph/         # グラフニューラルネット
│   └── features/      # 特徴量エンジニアリング
├── 🧪 tests/                           # テストスイート
├── ⚙️ configs/                         # 設定ファイル
└── 📈 output/                          # 結果・出力
```

## 🗃️ Archived Scripts（移管済み）
以下のスクリプトは保守対象外となり、`scripts/_archive/` へ移動しました。代替手順をご利用ください。

- apply_best_practices.py → 代替: `pre-commit run --all-files`、`ruff/black/mypy/bandit`
- benchmark_market_features.py → 代替: `python scripts/validate_improvements.py --detailed`
- complete_atft_training.sh → 代替: `python scripts/integrated_ml_training_pipeline.py`
- convert_4000_to_atft_format.py → 代替: `python scripts/data/ml_dataset_builder.py`
- create_full_historical_dataset.py / create_historical_dataset.py → 代替: `python scripts/pipelines/run_pipeline_v4_optimized.py`
- data_optimizer.py → 代替: `python scripts/run_safe_training.py --memory-limit 6`、`python scripts/validate_data.py`
- evaluate_atft_model.py → 代替: `python scripts/integrated_ml_training_pipeline.py`（評価内包）
- generate_full_dataset.py → 代替: `python scripts/pipelines/run_full_dataset.py`
- production_deployment.py → 代替: `python scripts/integrated_ml_training_pipeline.py`（Dockerスタック廃止）
- production_training.py → 代替: `python scripts/run_safe_training.py`
- run_jquants_pipeline.py → 代替: `python scripts/pipelines/run_pipeline_v4_optimized.py`
- test_optimized_pipeline.py → 代替: `pytest tests/integration/`、`python scripts/smoke_test.py`

アーカイブ版は互換のため残置されていますが、今後は上記の代替コマンドを使用してください。

## 🔧 ワークフロー詳細

### 1. 🛡️ 安全学習パイプライン (`safe-training`)

7段階の包括的学習パイプライン:

1. **データ読み込み**: 632銘柄, 155特徴量
2. **特徴量生成**: テクニカル・ファンダメンタル統合
3. **正規化**: CrossSectionalNormalizerV2 (robust_outlier_clip)
4. **Walk-Forward検証**: 20日エンバーゴ
5. **GBMベースライン**: LightGBM学習
6. **グラフ構築**: 相関ベースグラフ
7. **性能レポート**: 包括的結果出力

```bash
# フル学習 (推奨)
python main.py safe-training --mode full

# クイックテスト
python main.py safe-training --mode quick
```

### 2. 📊 MLデータセット構築 (`ml-dataset`)

gogooku2バッチデータから強化データセット作成:

- **入力**: gogooku2/output/batch タンicalアナリシス結果
- **出力**: 632銘柄 × 155特徴量
- **品質**: MIN_COVERAGE_FRAC=0.98
- **形式**: Parquet + メタデータJSON

```bash
python main.py ml-dataset
```

### 3. 🌐 直接API取得 (`direct-api-dataset`)

JQuants APIから全銘柄直接取得:

- **対象**: 3,803銘柄（TSE Prime/Standard/Growth）
- **期間**: 2021-01-01 ～ 現在
- **並列**: 50同時接続
- **フォールバック**: 期間短縮リトライ

```bash
python main.py direct-api-dataset
```

### 4. 🎯 完全ATFT学習 (`complete-atft`)

ATFT-GAT-FAN完全統合パイプライン:

- **目標**: Sharpe 0.849
- **パラメータ**: 5.6M
- **アーキテクチャ**: ATFT-GAT-FAN
- **学習**: PyTorch 2.0 + bf16

```bash
python main.py complete-atft
```

## 🔧 高度な使用方法

### 個別スクリプト実行

```bash
# 7段階安全パイプライン
python scripts/run_safe_training.py

# MLデータセット構築
python scripts/data/ml_dataset_builder.py

# 完全ATFT学習（内製ルート）
python scripts/integrated_ml_training_pipeline.py

# 互換エイリアス（ドキュメント互換の最終版エントリ）
python scripts/integrated_ml_training_pipeline_final.py
```

### 設定カスタマイズ

```python
# scripts/run_safe_training.py 内
MIN_COVERAGE_FRAC = 0.98  # 特徴量品質閾値
OUTLIER_CLIP_QUANTILE = 0.01  # 外れ値クリップ
WALK_FORWARD_EMBARGO_DAYS = 20  # エンバーゴ日数
```

### 統合パイプラインの高度な使用例

```bash
# 1) SafeTrainingPipeline を事前実行して検証（学習はスキップ）
python scripts/integrated_ml_training_pipeline.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --run-safe-pipeline --max-epochs 0

# 2) Hydraオーバーライドを透過的に適用（train.* 名前空間）
python scripts/integrated_ml_training_pipeline.py \
  train.optimizer.lr=2e-4 train.trainer.max_epochs=10

# 3) HPOメトリクスをJSONで出力
python scripts/integrated_ml_training_pipeline.py \
  hpo.output_metrics_json=tmp/hpo.json train.trainer.max_epochs=1

# 4) 高度グラフ学習を有効化（EWM+shrinkage, 既定補完あり）
python scripts/integrated_ml_training_pipeline.py --adv-graph-train
```

## 📊 データ仕様

### MLデータセット

- **サイズ**: 4,643,404行 × 395列
- **銘柄数**: 4,220
- **期間**: 2020-10-20 ～ 2025-10-17
- **特徴量**: 381 (価格・テクニカル・フロー・マージン・決算・先物など)
- **ターゲット**: 回帰 (1d,5d,10d,20d) + 分類 (バイナリ)

### 特徴量カテゴリ

1. **価格・出来高**: 6列 (`Open`,`High`,`Low`,`Close`,`Volume`,`TurnoverValue`)
2. **リターン**: 20列（生/対数リターン、マーケット・セクター調整含む）
3. **ボラティリティ**: 21列（5/10/20/60日、Yang-Zhang、VoV 等）
4. **移動平均/トレンド**: 22列（SMA/EMA、ギャップ、セクター相対）
5. **テクニカル指標**: 18列（RSI、MACD、ADX、ATR、ボリンジャー等）
6. **市場（TOPIX）指標**: 10列（市場リターン、ボラ指標、レジームフラグ）
7. **マクロセンチメント（VIX/為替）**: VIX/FX 合計最大20列（VIX水準・USD/JPYレート、各種リターン、Zスコア、スパイクフラグ等 ※ `--enable-vix`, `--enable-fx-usdjpy` 時）
8. **クロスマーケット**: 13列（β、α、相対強度、トレンド整合度など）
9. **セクター特徴量**: 12列（区分コード、セクター統計）
10. **フロー/資金動向**: 37列（投信・海外勢・スマートマネー指標等）
11. **マージン（週次/日次）**: 92列（信用残、貸借倍率、日次開示指標）
12. **決算・財務**: 20列（進捗率、利益率、YoY 等）
13. **ショートセリング**: 2列（空売り統計）
14. **先物派生**: 88列（TOPIX/NK225/JN400/REIT 先物のスプレッド・建玉）
15. **その他補助**: 118列（VALIDフラグ、ターゲット派生、補助メタ）

## 🚨 重要な制約

### システム要件

- **CPU**: 24コア推奨
- **メモリ**: 200GB+ (バッチ処理時)
- **ストレージ**: 100GB+
- **Python**: 3.9+

### データ品質制約

- **MIN_COVERAGE_FRAC**: 0.98 (特徴量品質)
- **最小データ点**: 200日以上
- **エンバーゴ**: 20日 (データリーク防止)

### API制限

- **JQuants**: 毎秒10リクエスト
- **並列数**: 50接続
- **認証**: 環境変数必須

## 🔍 トラブルシューティング

### よくある問題

1. **メモリ不足**
   ```bash
   # GPU/CPUメモリ使用量確認
   nvidia-smi
   htop
   ```

2. **API認証エラー**
   ```bash
   # 環境変数確認
   echo $JQUANTS_AUTH_EMAIL
   echo $JQUANTS_AUTH_PASSWORD
   ```

3. **依存関係エラー**
   ```bash
   # 依存関係再インストール
   pip install -r requirements.txt --force-reinstall
   ```

4. **DataLoaderハング問題**
   - 解決済み: ImportError による無限ハングを修正
   - 詳細: [docs/fixes/dataloader_hanging_fix.md](docs/fixes/dataloader_hanging_fix.md)
5. **マルチワーカー DataLoader 設定**
   - 既定で `ALLOW_UNSAFE_DATALOADER=auto` となり、CPUコア数から自動算出した `NUM_WORKERS` / `PREFETCH_FACTOR` / `PIN_MEMORY` を使用します。
   - シングルワーカーへ戻したい場合は `.env` で `ALLOW_UNSAFE_DATALOADER=0` または `FORCE_SINGLE_PROCESS=1` を設定してください。
   - より細かな調整が必要な場合は `NUM_WORKERS` / `PREFETCH_FACTOR` / `PIN_MEMORY` / `PERSISTENT_WORKERS` を個別に上書きできます。

### ログ確認

```bash
# メインログ
tail -f logs/main.log

# ML学習ログ
tail -f logs/ml_training.log

# 安全パイプライン
tail -f logs/safe_training.log
```

## 📈 パフォーマンス

### ベンチマーク結果

- **データ処理**: 605K行を30秒以下 (Polars)
- **特徴量生成**: 155特徴量を2分以下
- **学習時間**: ATFT-GAT-FAN 10エポック 45分
- **メモリ効率**: 99%+ Polars利用率

### 最適化Tips

1. **Polars並列化**: `n_threads=24`
2. **バッチサイズ**: 2048 (PyTorch)
3. **精度**: bf16混合精度
4. **キャッシュ**: 中間結果キャッシュ

## 🤝 貢献

このプロジェクトはプライベート開発中です。

## 📄 ライセンス

プライベートライセンス - 無断使用禁止

## 🔐 セキュリティ & 運用性

### 安全な起動手順

#### 1. 環境変数設定（必須）

```bash
# .env.example をコピーして編集
cp .env.example .env

# 必須の環境変数を設定
nano .env
```

**必須環境変数:**
```bash
# MinIO Storage
MINIO_ROOT_USER=your_secure_username
MINIO_ROOT_PASSWORD=your_secure_password_here
MINIO_DEFAULT_BUCKETS=gogooku,feast,mlflow,dagster

# ClickHouse Database
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=your_secure_ch_password_here
CLICKHOUSE_DB=gogooku3

# Redis Cache
REDIS_PASSWORD=your_secure_redis_password_here

# J-Quants API
JQUANTS_AUTH_EMAIL=your_email@example.com
JQUANTS_AUTH_PASSWORD=your_secure_api_password_here
```

#### 2. 周辺サービスの準備

> ℹ️ 旧Docker Composeスタックは撤去済みです。MinIO、ClickHouse、Redis などの補助サービスが必要な場合は、既存の社内インフラやマネージドサービスへ接続してください。`.env` で指定したホスト/ポートが到達可能であることを確認してください。

#### 3. セキュリティ検証

```bash
# セキュリティスキャン実行
python ops/health_check.py health

# ログセキュリティ確認
tail -f logs/main.log | grep -i security
```

### オプション機能（デフォルト無効）

#### パフォーマンス最適化

```bash
# Polarsストリーミング有効化
export PERF_POLARS_STREAM=1

# メモリ最適化有効化
export PERF_MEMORY_OPTIMIZATION=1
```

#### 監視・オブザーバビリティ

```bash
# メトリクス収集有効化
export OBS_METRICS_ENABLED=1

# データ品質チェック有効化
export DATA_QUALITY_ENABLED=1
```

### 監視エンドポイント

```bash
# ヘルスチェック
curl http://localhost:8000/healthz

# 詳細ヘルスチェック
python ops/health_check.py health --format json

# Prometheusメトリクス
curl http://localhost:8000/metrics

# 準備状況チェック
python ops/health_check.py ready
```

### ログ管理

```bash
# ログローテーション設定確認
cat ops/logrotate.conf

# ログローテーション実行
sudo logrotate -f /etc/logrotate.d/gogooku3

# ログアーカイブ確認
ls -la /var/log/gogooku3/archive/
```

## 🧪 テスト & 品質保証

### 利用可能なテストスイート

```bash
# 全テスト実行
pytest tests/ -v

# ユニットテストのみ
pytest tests/ -k "unit" -v

# 統合テストのみ
pytest tests/ -k "integration" -v

# ヘルスチェックテスト
pytest tests/test_health_check.py -v

# データ品質テスト
pytest tests/ -k "data_quality" -v

# パフォーマンステスト
pytest tests/ -k "performance" --benchmark-only
```

### CI/CD パイプライン

- **セキュリティスキャン**: Trivy, Gitleaks, Bandit, pip-audit
- **テスト自動化**: ユニット/統合/E2E/パフォーマンス/データ品質
- **依存関係監査**: pip-audit, 脆弱性チェック
- **パフォーマンス監視**: ベンチマーク自動実行
- **バックアップ検証**: 日次自動バックアップ検証
- **Semantic Release**: 自動バージョン管理とCHANGELOG生成

### データ品質チェック

```bash
# Great Expectations 統合データ品質チェック
export DATA_QUALITY_ENABLED=1
python data_quality/great_expectations_suite.py validate --input data/processed/dataset.parquet

# 品質チェックレポート確認
cat data_quality/results/validation_*.json
```

> ℹ️ DockerベースのE2Eテストはスタック撤去に伴い廃止しました。E2E検証が必要な場合は実運用環境に合わせた新しいフローを構築してください。

### パフォーマンス最適化

```bash
# パフォーマンス最適化有効化
export PERF_POLARS_STREAM=1
export PERF_PARALLEL_PROCESSING=1
export PERF_MEMORY_OPTIMIZATION=1
export PERF_CACHING_ENABLED=1

# 最適化適用で実行
python main.py safe-training --mode full

# パフォーマンスメトリクス確認
python ops/metrics_exporter.py --once | grep -E "(optimization|performance)"
```

## 🛠️ 運用・保守

### バックアップ & リカバリ

```bash
# データベースバックアップ
clickhouse-client \
  --host "$CLICKHOUSE_HOST" \
  --port "${CLICKHOUSE_PORT:-9000}" \
  --user "$CLICKHOUSE_USER" \
  --password "$CLICKHOUSE_PASSWORD" \
  --query "BACKUP DATABASE gogooku3 TO Disk('backups', 'backup_$(date +%Y%m%d)')"

# ファイルシステムバックアップ
tar -czf backups/data_$(date +%Y%m%d).tar.gz data/ output/

# リストア手順
tar -xzf backups/latest.tar.gz -C /
```

### ログ分析

```bash
# エラーログ確認
grep -i error logs/*.log

# パフォーマンスログ分析
grep -i "duration\|memory\|cpu" logs/*.log

# セキュリティイベント確認
grep -i "security\|auth\|access" logs/*.log
```

### パフォーマンス監視

```bash
# システムリソース監視
python ops/health_check.py health

# パフォーマンスベンチマーク
pytest tests/ -k "performance" --benchmark-only

# メモリプロファイリング
python -m memory_profiler main.py safe-training --mode quick
```

### 障害対応

参照: `ops/runbook.md`

```bash
# 緊急停止
# 例: systemctl stop gogooku3.service
systemctl stop gogooku3.service

# 安全再起動
# 例: systemctl start gogooku3.service
systemctl start gogooku3.service

# ログ確認
tail -f logs/main.log
```

## 📊 アーキテクチャ & ドキュメント

### システム構成図

```mermaid
graph TB
    A[Client] --> B[main.py]
    B --> C[Safe Training Pipeline]
    B --> D[ML Dataset Builder]
    B --> E[Direct API Dataset]

    C --> F[Data Processing]
    C --> G[Model Training]
    C --> H[Validation]

    F --> I[Polars Engine]
    G --> J[PyTorch ATFT-GAT-FAN]
    H --> K[Cross-Sectional Validation]

    L[External Services] --> M[Object Storage]
    L --> N[ClickHouse]
    L --> O[Redis]
    L --> P[MLflow]

    Q[Monitoring] --> R[Health Check]
    Q --> S[Metrics Exporter]
    Q --> T[Log Rotation]

    R --> U[/healthz]
    S --> V[/metrics]
    T --> W[Log Archive]
```

### 主要ドキュメント

- **運用Runbook**: `ops/runbook.md`
- **セキュリティガイド**: `security/sast.md`
- **アーキテクチャ図**: `docs/arch/`
- **APIドキュメント**: `docs/guides/`
- **トラブルシューティング**: `docs/faq.md`

## 🙏 謝辞

- **JQuants API**: 日本株データ提供
- **Polars**: 高速データ処理
- **PyTorch**: 深層学習フレームワーク
- **gogooku2**: ベースシステム

---

**🚀 gogooku3-standalone - 壊れず・強く・速く の実現**

## 📜 Log Files Overview

All dataset/logging output lives under `logs/`.

```
logs/
  chunks/      # chunk builder runs (timestamped) + latest.log symlink
  dataset/     # full-run logs + pid/pgid files per run (latest.log)
  health/      # health-check JSON snapshots
```

Helpers:
- `scripts/show_logs.sh chunk --tail 200`
- `scripts/show_logs.sh dataset`
- `scripts/show_logs.sh health`
