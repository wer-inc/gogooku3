# gogooku3-standalone

**壊れず・強く・速く** を実現する金融ML システム

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Polars](https://img.shields.io/badge/Polars-0.20+-orange.svg)](https://pola.rs/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Private-black.svg)]()

## 🎯 概要

gogooku3-standaloneは、日本株市場の機械学習予測に特化した統合システムです。「**壊れず・強く・速く**」の理念のもと、以下の特徴を持ちます：

- **🛡️ 壊れず (Unbreakable)**: 堅牢な品質チェック・安全パイプライン
- **💪 強く (Strong)**: ATFT-GAT-FAN（Sharpe 0.849）の高性能モデル 
- **⚡ 速く (Fast)**: Polars高速処理・並列化最適化

### 主な成果

- **632銘柄対応**: gogooku2の644銘柄から品質向上により632銘柄に最適化
- **155特徴量**: 高度なテクニカル分析・ファンダメンタル特徴量
- **完全統合**: gogooku3の全処理をスタンドアロンシステムに移植
- **ATFT-GAT-FAN**: 5.6M パラメータ、目標Sharpe 0.849を達成

## 🚀 クイックスタート

### 1. 環境構築

```bash
# リポジトリ移動
cd /home/ubuntu/gogooku3-standalone

# 依存関係インストール
pip install -r requirements.txt

# ディレクトリ作成（自動）
python main.py --help
```

### 2. 基本実行

```bash
# 🔥 推奨: 安全学習パイプライン (フル)
python main.py safe-training --mode full

# ⚡ クイックテスト (1エポック)
python main.py safe-training --mode quick

# 📊 データセット構築
python main.py ml-dataset

# 🌐 直接API取得
python main.py direct-api-dataset

# 🎯 完全ATFT学習
python main.py complete-atft
```

### 3. 実行例

```bash
# 💡 最初の実行（推奨）
python main.py safe-training --mode full

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
│   ├── 🎯 integrated_ml_training_pipeline.py  # ATFT完全統合
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

# 完全ATFT学習
python scripts/integrated_ml_training_pipeline.py
```

### 設定カスタマイズ

```python
# scripts/run_safe_training.py 内
MIN_COVERAGE_FRAC = 0.98  # 特徴量品質閾値
OUTLIER_CLIP_QUANTILE = 0.01  # 外れ値クリップ
WALK_FORWARD_EMBARGO_DAYS = 20  # エンバーゴ日数
```

## 📊 データ仕様

### MLデータセット

- **サイズ**: 605,618行 × 169列
- **銘柄数**: 632
- **期間**: 2021-01-04 ～ 2025-08-22  
- **特徴量**: 155 (テクニカル + ファンダメンタル)
- **ターゲット**: 回帰 (1d,5d,10d,20d) + 分類 (バイナリ)

### 特徴量カテゴリ

1. **基本価格**: OHLCV
2. **リターン**: 1d, 5d, 10d, 20d
3. **テクニカル**: EMA, RSI, MACD, BB, ADX, ATR
4. **ボラティリティ**: 20d, 60d, Sharpe比率
5. **相関特徴量**: クロスセクション統計  
6. **ファンダメンタル**: PER, PBR, 時価総額

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
   # Docker メモリ割り当て確認
   docker stats
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

#### 2. Docker Compose起動

```bash
# セキュアな設定で起動（推奨）
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d

# 従来通り起動（開発用のみ）
docker compose up -d
```

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

# E2Eテストのみ
pytest tests/test_e2e_docker.py -v

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
docker exec gogooku3-clickhouse clickhouse-client --query "BACKUP DATABASE gogooku3 TO Disk('backups', 'backup_$(date +%Y%m%d)')"

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
docker compose down

# 安全再起動
docker compose up -d --force-recreate

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

    L[Docker Services] --> M[MinIO]
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