# 🌟 Gogooku3 はじめに

<!-- TOC -->

Gogooku3-standaloneは **「壊れず・強く・速く」** を実現する日本株式向けMLOpsシステムです。

## 🎯 システム概要

### 主要特徴
- **🛡️ 壊れず (Unbreakable)**: Walk-Forward分割・embargo・データリーク防止
- **💪 強く (Strong)**: ATFT-GAT-FAN（Sharpe 0.849目標）高性能モデル
- **⚡ 速く (Fast)**: Polars最適化・1.9秒パイプライン実行

### 成果指標
- **📊 データ処理**: 606K samples × 145 features を1.9秒で処理
- **💾 メモリ効率**: 7GB使用（目標<8GB達成）
- **🏆 ML性能**: 632銘柄対応・ATFT-GAT-FAN 5.6M parameters
- **🔒 安全性**: Walk-Forward + 20日embargo実装

---

## 🚀 5分でスタート

### 📋 前提条件
```bash
# 必要環境
- Python 3.11+
- CUDA 12.4 対応GPU (A100 80GB 推奨)
- 16GB+ RAM, 50GB+ disk
- MinIO / ClickHouse / Redis など周辺サービスへアクセス可能な環境（クラウド・社内インフラ等）
- JQuants API アカウント（データ取得用）
```

### ⚡ クイックセットアップ

#### 1. 環境準備
```bash
cd /home/ubuntu/gogooku3-standalone

# 依存関係インストール
make setup                        # Python venv + dependencies

# 環境設定
cp .env.example .env
vim .env                          # JQuants認証情報を設定
```

#### 2. 周辺サービスの接続確認
```bash
# MinIO接続確認（例）
aws --endpoint-url $MLFLOW_S3_ENDPOINT_URL s3 ls

# ClickHouse接続確認（例）
clickhouse-client --host $CLICKHOUSE_HOST --port ${CLICKHOUSE_PORT:-9000} --query "SELECT 1"

# Redis接続確認（例）
redis-cli -h $REDIS_HOST -p ${REDIS_PORT:-6379} -a $REDIS_PASSWORD PING
```

#### 3. スモークテスト実行
```bash
# 軽量テスト実行（推奨）
make smoke                        # 1-epoch 軽量学習でシステム確認

# 完全学習実行（時間要）
make train-cv                     # 5-fold cross-validation学習
```

---

## 🖥️ Web UI アクセス

> ℹ️ Docker Compose ベースのダッシュボードは廃止しました。各サービスを利用する場合はクラウドやオンプレでホストされている既存エンドポイントに接続してください。以下は参考構成です。

### 主要サービス
| サービス | URL | 用途 | 認証 |
|---------|-----|------|------|
| **Dagster** | https://your-dagster.example.com | オーケストレーション | 環境に依存 |
| **MLflow** | https://mlflow.your-domain.example | 実験追跡・モデル管理 | 環境に依存 |
| **Grafana** | https://grafana.your-domain.example | 監視ダッシュボード | 環境に依存 |
| **MinIO** | https://minio-console.your-domain.example | オブジェクトストレージ | 環境に依存 |

### 🔧 初回セットアップ確認

#### Dagster（パイプライン管理）
```bash
# アクセス: https://your-dagster.example.com
# 1. "Assets" タブ確認
# 2. "Materialize all" でパイプライン実行
# 3. 実行ステータス確認
```

#### MLflow（実験管理）
```bash
# アクセス: https://mlflow.your-domain.example
# 1. "Experiments" タブで実験確認
# 2. "Models" タブでモデル登録確認
# 3. メトリクス・ログ表示確認
```

#### Grafana（監視）
```bash
# アクセス: https://grafana.your-domain.example
# 認証情報は組織ポリシーに従う
# 1. ダッシュボード表示確認
# 2. メトリクス収集確認
# 3. アラート設定確認
```

---

## 💻 開発環境セットアップ

### 🧰 ローカル開発環境（コンテナレス）

#### 基本操作
```bash
# 依存関係と仮想環境セットアップ
make dev

# 環境リセット（仮想環境・キャッシュ削除）
make clean
```

> ℹ️ 従来の `docker-up` / `docker-down` ターゲットは削除しました。必要な周辺サービスは既存のインフラに接続してください。

### 🧪 テスト・品質管理

#### テスト実行
```bash
# 軽量スモークテスト（推奨）
make smoke                        # 1 epoch 軽量学習

# 包括テスト
make test                         # 単体・統合テスト実行
pytest tests/integration/ -v     # 統合テスト

# システム統合テスト
python scripts/run_safe_training.py --n-splits 1 --verbose
```

#### コード品質
```bash
# リント・フォーマット
make lint                         # ruff + mypy
ruff check src/ --fix             # 自動修正
mypy src/gogooku3                 # 型チェック

# pre-commit（自動実行）
pre-commit run --all-files        # 全品質チェック
```

---

## 📊 完全データセット構築

### 🚀 run_full_dataset.py（推奨）

#### 基本実行
```bash
# 5年間の完全統合データセット構築
python scripts/pipelines/run_full_dataset.py --jquants \
  --start-date 2020-09-06 --end-date 2025-09-06
```

#### Margin Weekly機能付き
```bash
# 信用取引残高特徴量を含むデータセット構築
python scripts/pipelines/run_full_dataset.py --jquants \
  --start-date 2020-09-06 --end-date 2025-09-06 \
  --weekly-margin-parquet output/weekly_margin_interest_*.parquet \
  --margin-weekly-lag 3 \
  --adv-window-days 20
```

#### 完全機能セット
```bash
# TOPIX・フロー・文書・Margin全機能統合
python scripts/pipelines/run_full_dataset.py --jquants \
  --start-date 2020-09-06 --end-date 2025-09-06 \
  --topix-parquet output/topix_history_*.parquet \
  --statements-parquet output/event_raw_statements_*.parquet \
  --weekly-margin-parquet output/weekly_margin_interest_*.parquet \
  --sector-onehot33 \
  --sector-te-targets target_5d,target_1d
```

#### セクター相対（Sector Cross‑Sectional）
```bash
# 例: rsi_14 と returns_10d に対して _vs_sec/_in_sec_z を追加
python scripts/pipelines/run_full_dataset.py --jquants \
  --start-date 2020-09-06 --end-date 2025-09-06 \
  --enable-sector-cs --sector-cs-cols "rsi_14,returns_10d"
```

#### グラフ特徴（相関ネットワーク）
```bash
# 窓60日、相関しきい値0.3、最大次数10、キャッシュディレクトリ指定
python scripts/pipelines/run_full_dataset.py --jquants \
  --start-date 2020-09-06 --end-date 2025-09-06 \
  --enable-graph-features --graph-window 60 --graph-threshold 0.3 \
  --graph-max-k 10 --graph-cache-dir output/graph_cache
```

#### Nikkei225 オプション市場アグリゲートの付与（T+1）
```bash
# 既存のraw/features parquetが無い場合はAPIから取得して構築
python scripts/pipelines/run_full_dataset.py --jquants \
  --start-date 2020-09-06 --end-date 2025-09-06 \
  --attach-nk225-option-market
```

#### YAML設定での一括指定（CLI優先）
```bash
# configs/pipeline/full_dataset.yaml を読み込み、セクター相対/グラフ等の既定を設定
python scripts/pipelines/run_full_dataset.py --jquants \
  --start-date 2020-09-06 --end-date 2025-09-06 \
  --config configs/pipeline/full_dataset.yaml
```

```bash
# 研究用途: フルインデックスセット（Topix-17, Sector33等すべて）を含む設定
python scripts/pipelines/run_full_dataset.py --jquants \
  --start-date 2020-09-06 --end-date 2025-09-06 \
  --config configs/pipeline/research_full_indices.yaml
```

### 📁 出力ファイル
```bash
# 結果確認
ls -la output/ml_dataset_latest_full.parquet
ls -la output/ml_dataset_latest_full_metadata.json

# データ概要
python -c "
import polars as pl
df = pl.read_parquet('output/ml_dataset_latest_full.parquet')
print(f'データ形状: {df.shape}')
print(f'Margin機能: {\"margin_short_to_adv20\" in df.columns}')
"
```

---

## 🧠 ML学習・実行

### 🎯 基本学習実行

#### Safe Training Pipeline（推奨）
```bash
# 完全統合パイプライン実行
python scripts/run_safe_training.py --verbose --n-splits 5

# カスタム設定
python scripts/run_safe_training.py \
  --data-dir data/raw/large_scale \
  --n-splits 3 \
  --embargo-days 20 \
  --memory-limit 8 \
  --experiment-name production
```

#### 個別コンポーネント使用
```python
# データ処理
from gogooku3.data.loaders import ProductionDatasetV3
loader = ProductionDatasetV3(
    data_files=["data/ml_dataset_full.parquet"],
    config={"batch_size": 1024}
)

# 特徴量生成
from gogooku3.features import QualityFinancialFeaturesGenerator
generator = QualityFinancialFeaturesGenerator()
enhanced_data = generator.generate_quality_features(loader.data)

# 安全な正規化
from gogooku3.data.scalers import CrossSectionalNormalizerV2
normalizer = CrossSectionalNormalizerV2(robust_clip=5.0)
normalized_data = normalizer.fit_transform(enhanced_data)
```

### 🎛️ Modern CLI使用

#### v2.0.0 新CLIインターフェース
```bash
# パッケージインストール後
pip install -e .

# 統一CLI使用
gogooku3 train --config configs/atft/train/production.yaml
gogooku3 data --build-dataset
gogooku3 infer --model-path models/best_model.pth
gogooku3 --version

# モジュール実行
python -m gogooku3.cli train
python -m gogooku3.cli --help
```

---

## 📊 特徴量・データ確認

### 📈 ML Dataset概要
```python
# データ概要確認
import polars as pl
df = pl.read_parquet("data/raw/large_scale/ml_dataset_full.parquet")

print(f"データ形状: {df.shape}")           # (606,127, 145)
print(f"期間: {df['Date'].min()} - {df['Date'].max()}")
print(f"銘柄数: {df['Code'].n_unique()}")   # 632銘柄
```

### 🔧 特徴量構成（145+列）
- **識別子** (2列): Code, Date
- **OHLCV** (6列): Open, High, Low, Close, Volume, row_idx
- **技術指標** (131列): SMA, EMA, MACD, RSI, Stoch, BB, ADX, etc.
- **品質特徴量** (+6列): Cross-sectional quantiles, sigma-threshold features
- **📊 Margin Weekly** (任意): 信用取引残高由来の需給特徴量（margin_short_to_adv20等）

### 🛡️ データ安全性確認
```python
# Walk-Forward分割確認
from gogooku3.data.scalers import WalkForwardSplitterV2

splitter = WalkForwardSplitterV2(n_splits=5, embargo_days=20)
validation = splitter.validate_split(df)

print(f"重複確認: {len(validation['overlaps'])} overlaps detected")
print(f"embargo確認: {validation['embargo_respected']}")
```

### 🧭 Indices Integration (Market/Sector)

```bash
# Attach indices (daily spreads/breadth, sector index features)
python scripts/pipelines/run_full_dataset.py --jquants \
  --start-date 2024-01-01 --end-date 2024-06-30 \
  --enable-indices \
  --indices-codes 0000,0040,0500,0501,0502,0075,8100,8200,0028,002D,8501,8502,8503

# Offline indices parquet (optional)
python scripts/pipelines/run_full_dataset.py --jquants \
  --start-date 2024-01-01 --end-date 2024-06-30 \
  --enable-indices \
  --indices-parquet output/indices_history_20240101_20240630.parquet

# Disable special halt-day (2020-10-01) masking for range-derived features
python scripts/pipelines/run_full_dataset.py --jquants \
  --start-date 2024-01-01 --end-date 2024-06-30 \
  --enable-indices \
  --indices-codes 0000,0040,0500,0501,0502,0075,8100,8200,0028,002D,8501,8502,8503 \
  --disable-halt-mask
```

Options:
- `--enable-indices`: Attach index features.
- `--indices-codes`: Comma-separated codes to fetch via API.
- `--indices-parquet`: Use a pre-saved indices parquet (Date, Code, OHLC).
- `--disable-halt-mask`: Disable 2020-10-01 halt-day masking for range features.

Quick reference tables for index codes (Sector 33, Topix‑17, Market Segments, Style/Size, REIT) are available in `docs/ml/dataset_new.md` under “Indices (Market & Sector) Features”.

#### 📝 Cheat Sheet — Common Index Codes

| Code | Family     | Name                         |
|------|------------|------------------------------|
| 0000 | MARKET     | TOPIX                        |
| 0500 | MARKET     | 東証プライム市場指数         |
| 0501 | MARKET     | 東証スタンダード市場指数     |
| 0502 | MARKET     | 東証グロース市場指数         |
| 0046 | SECTOR33   | 化学                         |
| 005A | SECTOR33   | 小売業                       |
| 8100 | STYLE      | TOPIX バリュー               |
| 8200 | STYLE      | TOPIX グロース               |
| 0028 | SIZE       | TOPIX Core30                 |
| 002D | SIZE       | TOPIX Small                  |
| 0088 | TOPIX‑17   | 電機・精密                   |
| 0085 | TOPIX‑17   | 自動車・輸送機               |
| 0075 | REIT       | 東証REIT（総合）             |
| 8501 | REIT SEG   | オフィス                     |
| 8502 | REIT SEG   | 住宅                         |
| 8503 | REIT SEG   | 商業・物流等                 |

---

## 🛑 停止・クリーンアップ

### 🚪 正常停止
```bash
# サービス停止（例：systemd）
systemctl stop gogooku3.service
```

### 🧹 完全クリーンアップ
```bash
# 全データ・ボリューム削除
make clean
rm -rf output/experiments/*      # 実験結果削除
```

---

## 🆘 トラブルシューティング

### 🚨 よくある問題

#### メモリ不足
```bash
# 現在メモリ使用量確認
nvidia-smi
free -h

# 解決策
# 1. GPU/システムメモリ割り当てを確認し必要に応じて調整
# 2. メモリ制限パラメータ使用
python scripts/run_safe_training.py --memory-limit 4
```

#### ポート競合
```bash
# ポート使用確認
lsof -i :3001  # Dagster
lsof -i :5000  # MLflow
lsof -i :9001  # MinIO

# 解決策
# サービス構成ファイルでポート変更、または競合プロセス停止
```

#### サービス起動失敗
```bash
# ログ確認
journalctl -u gogooku3.service --since "5 minutes ago"

# 個別サービス再起動
systemctl restart gogooku3.service

# 完全再構築
# 必要に応じて仮想環境再作成やキャッシュクリアを実施
make clean && make setup
```

#### データ・設定問題
```bash
# 環境設定確認
cat .env                          # JQuants認証情報確認

# データファイル確認
ls -la data/raw/large_scale/      # MLデータセット存在確認

# 権限問題修正
sudo chown -R $USER:$USER output/
chmod -R 755 output/
```

### 🩺 システムヘルスチェック
```bash
# 包括的ヘルスチェック
make check                        # 全体システム確認

# 個別確認
# 事前に MLFLOW_BASE_URL / DAGSTER_BASE_URL を環境に設定
curl "$MLFLOW_BASE_URL/health"   # MLflow
curl "$DAGSTER_BASE_URL/health"  # Dagster

# パッケージ確認
python -c "import gogooku3; print('✅ Package OK')"
python -c "from gogooku3.training import SafeTrainingPipeline; print('✅ Training OK')"
```

---

## 📚 次のステップ

### 🎓 学習リソース
1. **[👥 開発貢献ガイド](development/contributing.md)** - 詳細な開発フロー
2. **[🏗️ アーキテクチャ概要](architecture/overview.md)** - システム設計理解
3. **[🛡️ 安全性ガードレール](ml/safety-guardrails.md)** - データリーク防止詳細
4. **[📊 モデル学習/評価](ml/model-training.md)** - 学習・評価の概要

### 🔧 カスタマイズ
- **設定変更**: `configs/` 配下の設定ファイル編集
- **新機能追加**: `src/gogooku3/` パッケージ拡張
- **実験管理**: MLflow UI でハイパーパラメータ調整

### 🚀 本番運用
- **[📋 運用手順](operations/runbooks.md)** - 本番環境運用
- **[🔧 トラブルシューティング](operations/troubleshooting.md)** - 障害対応
- **[📈 監視設定](operations/observability.md)** - Grafana・アラート

---

## 🔗 サポート・コミュニケーション

### 📖 ドキュメント
- **[📋 メインポータル](index.md)** - 全体ナビゲーション
- **[❓ FAQ](faq.md)** - よくある質問
- **[📚 用語集](glossary.md)** - 専門用語解説
- **[🔄 移行ガイド](../architecture/migration.md)** - v1→v2移行手順

### 📞 問題解決
1. **ドキュメント検索**: 該当セクション参照
2. **ログ確認**: `journalctl -u gogooku3.service --since "10 minutes ago"` で詳細確認
3. **設定確認**: `.env` と `configs/` 設定検証
4. **システム再起動**: `make clean && make dev`

---

**🎉 セットアップ完了！**
**次は [👥 開発貢献ガイド](development/contributing.md) で詳細な開発フローを確認してください。**



*Gogooku3 - 壊れず・強く・速く*
