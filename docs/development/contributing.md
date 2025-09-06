# 👥 Gogooku3 開発貢献ガイド

<!-- TOC -->

> **📍 注意**: この文書は `/CLAUDE.md` から移行されました。最新の開発情報はこちらを参照してください。

## 🏗️ リポジトリ概要

Gogooku3は日本株式市場分析向け次世代MLOpsバッチシステムです。JQuants API → 技術指標計算 → Graph Attention Network学習 → 特徴量ストア管理の完全な金融MLパイプラインです。

### 🏛️ アーキテクチャ

**データ処理パイプライン**:
- **JQuants API統合**: 非同期150並行接続によるデータ取得
- **技術指標**: 713指標から最適化された62+特徴量
- **ATFT-GAT-FAN モデル**: Adaptive Temporal Fusion Transformer + Graph Attention + Frequency Adaptive Normalization
  - **TFT Core**: LSTM多頭アテンション時系列融合
  - **適応正規化**: FAN (Frequency Adaptive) + SAN (Slice Adaptive)動的スケーリング
  - **Graph Attention**: 動的相関グラフ構築+GATレイヤー
  - **多期間予測**: 1d/2d/3d/5d/10d予測と分位回帰
- **特徴量ストア**: Feast+Redisオンライン特徴量管理

**MLOpsインフラ**:
- **Dagster**: オーケストレーション（8 assets, 7 jobs, 6 schedules, 7 sensors）
- **MLflow**: 実験追跡・モデルレジストリ
- **Docker Compose**: マルチサービス展開（12+サービス）
- **監視**: Grafana + Prometheus + カスタムメトリクス

**ストレージ層**:
- **MinIO**: S3互換オブジェクトストレージ
- **ClickHouse**: OLAP分析データベース
- **Redis**: キャッシュ・オンライン特徴量ストア
- **PostgreSQL**: MLflow/Dagster/Feastメタデータ

## 🚀 開発環境セットアップ

### 基本セットアップ
```bash
# 環境構築
make setup                    # Python venv作成・依存関係インストール
cp .env.example .env         # 環境変数設定
vim .env                     # JQuants API認証情報編集

# Dockerサービス
make docker-up               # 全サービス起動（MinIO, ClickHouse等）
make docker-down             # 全サービス停止
make docker-logs             # 全サービスログ表示

# 開発モード
make dev                     # 環境セットアップ+サービス起動
```

### 🧪 テスト・検証
```bash
# テスト実行
make test                    # 全テスト実行
make smoke                   # スモークテスト（1 epoch, 軽量）
make lint                    # コード品質チェック

# ML学習
make train-cv                # クロスバリデーション学習
make infer                   # 推論実行

# 品質管理
make clean                   # 環境リセット
make check                   # ヘルスチェック
```

## 🔧 開発ワークフロー

### 📋 開発規約

#### コミット規約
```bash
# 推奨コミット形式
feat: ATFT-GAT-FAN多期間予測対応
fix: Walk-Forward分割での時系列リーク修正  
docs: ML評価指標ドキュメント更新
refactor: 特徴量生成パフォーマンス最適化
test: 交差検証スモークテスト追加
```

#### コード品質
```bash
# Pre-commit hooks (自動実行)
ruff check src/ --fix                 # リント・フォーマット
mypy src/gogooku3                     # 型チェック
pytest tests/ -v                      # テスト実行
```

#### ブランチ戦略
- `main`: 本番安定版
- `develop`: 開発統合ブランチ
- `feature/*`: 機能開発ブランチ
- `hotfix/*`: 緊急修正ブランチ

### 🔄 プルリクエスト

#### PR作成前チェックリスト
- [ ] `make lint` 通過
- [ ] `make test` 全通過
- [ ] `make smoke` 動作確認
- [ ] ドキュメント更新（必要に応じて）
- [ ] CHANGELOG.md更新（機能追加時）

#### PR テンプレート
```markdown
## 📋 変更内容
- [x] ATFT-GAT-FANモデルの多期間予測対応
- [x] Walk-Forward分割でのembargo実装

## 🧪 テスト
- [x] スモークテスト通過
- [x] 単体テスト追加・通過
- [x] 統合テスト確認

## 📊 パフォーマンス
- メモリ使用量: 7GB → 6.5GB
- 実行時間: 1.9s → 1.7s

## 🔗 関連Issue
Closes #123, #124
```

## 📁 プロジェクト構造

### v2.0.0 モダンパッケージ構造
```
gogooku3-standalone/
├── src/gogooku3/              # 🆕 メインパッケージ
│   ├── cli.py                 # コマンドライン
│   ├── data/                  # データ処理
│   │   ├── loaders/           # ProductionDatasetV3
│   │   └── scalers/           # CrossSectionalNormalizerV2
│   ├── features/              # 特徴量エンジニアリング
│   ├── models/                # ML模型
│   │   ├── atft_gat_fan.py    # ATFT-GAT-FAN
│   │   └── lightgbm_baseline.py
│   ├── graph/                 # グラフニューラル
│   ├── training/              # 学習パイプライン
│   └── compat/                # 後方互換レイヤー
├── scripts/                   # レガシー（互換モード）
├── configs/                   # 設定ファイル
├── tests/                     # テストスイート
└── docs/                      # ドキュメント（このファイル含む）
```

### 重要ファイル
- **pyproject.toml**: setuptools モダン設定
- **.pre-commit-config.yaml**: コード品質自動化
- **MIGRATION.md**: v2.0.0移行ガイド
- **src/gogooku3/compat/**: 後方互換性レイヤー

## 🎯 開発ベストプラクティス

### 🛡️ データ安全性
```python
# ✅ 正しい実装
normalizer = CrossSectionalNormalizerV2()
normalizer.fit(train_data)  # trainデータのみでfitting
train_norm = normalizer.transform(train_data)
test_norm = normalizer.transform(test_data)  # 同じ統計を適用

# ❌ 危険な実装  
normalizer.fit(all_data)  # 未来情報リーク
```

### ⚡ パフォーマンス
```python
# ✅ Polars使用（3-5x高速）
import polars as pl
df = pl.scan_parquet("data.parquet").lazy()  # 遅延読み込み

# ✅ メモリ制限
pipeline.run_pipeline(memory_limit_gb=8.0)
```

### 🧪 テスト
```python
# 統合テスト例
def test_safe_training_pipeline():
    pipeline = SafeTrainingPipeline(experiment_name="test")
    results = pipeline.run_pipeline(n_splits=1, embargo_days=20)
    
    assert results['total_duration'] < 30.0  # 30秒以内
    assert len(results['warnings']) == 0     # 警告なし
```

## 🔧 トラブルシューティング

### よくある問題

#### メモリ不足
```bash
# 解決策
make docker-down              # サービス停止
docker system prune -f        # 不要コンテナ削除
python scripts/train.py --memory-limit 4  # 制限設定
```

#### インポートエラー
```bash
# 解決策
pip install -e .              # パッケージ再インストール
export PYTHONPATH=/path/to/gogooku3-standalone/src:$PYTHONPATH
```

#### ポート競合
```bash
# 確認
make docker-logs | grep port
netstat -tulpn | grep :8000

# 解決
make docker-down
# docker-compose.ymlポート変更
make docker-up
```

### 🩺 ヘルスチェック
```bash
# システム状態確認
make check                    # 全体ヘルスチェック
docker ps                     # コンテナ状態
curl http://localhost:8000/health  # MLflow確認

# コンポーネント個別確認
python -c "import gogooku3; print('✅ Package OK')"
python -c "from gogooku3.training import SafeTrainingPipeline; print('✅ Training OK')"
```

## 📈 継続的改善

### パフォーマンス目標
- **パイプライン実行時間**: <2秒（606K件処理）
- **メモリ使用量**: <8GB
- **Walk-Forward安全性**: embargo>=20日
- **テストカバレッジ**: >85%

### 技術負債管理
1. **優先度P0**: セキュリティ・データリーク
2. **優先度P1**: パフォーマンス劣化  
3. **優先度P2**: コード保守性

### ロードマップ
- **Q1 2025**: リアルタイム推論API
- **Q2 2025**: 多市場対応（US/EU株式）
- **Q3 2025**: AutoML パイプライン

## 🔗 関連リソース

### 内部ドキュメント
- [📈 モデル学習/評価](../ml/model-training.md) - 学習・評価の概要
- [🛡️ 安全性ガードレール](../ml/safety-guardrails.md) - データリーク防止
- [🏗️ アーキテクチャ概要](../architecture/overview.md) - システム設計詳細
- [⚙️ 運用手順](../operations/runbooks.md) - 本番運用ガイド

### 外部参考文献
- [JQuants API仕様](https://jpx-jquants.com/?) - データソース
- [Feast公式](https://feast.dev/) - 特徴量ストア
- [Dagster公式](https://dagster.io/) - MLOpsオーケストレーション

---

**📝 更新履歴**: 
- 2025-08-28: CLAUDE.md から移行・再構成
- 2025-08-28: v2.0.0 モダンパッケージ対応

 

*Gogooku3 - 壊れず・強く・速く*
