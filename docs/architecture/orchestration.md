# 🔄 ワークフロー・オーケストレーション (作成中)

このドキュメントは現在作成中です。

## 📋 予定内容

- **Dagster設計**: Assets・Jobs・Schedules
- **ワークフロー定義**: データパイプライン・ML学習フロー
- **依存関係管理**: タスク間依存・データ系譜
- **スケジューリング**: バッチ実行・リアルタイム処理
- **監視・アラート**: ワークフロー監視・失敗時対応

## 🔄 現在のワークフロー

現在の実行方法（手動・スクリプトベース）：

### 学習パイプライン
```bash
# SafeTrainingPipeline による統合実行
gogooku3 train --config configs/atft/train/production.yaml

# または従来方式
python scripts/run_safe_training.py --n-splits 5
```

### データ処理
```bash
# データ取得・前処理
python scripts/data/ml_dataset_builder.py

# 特徴量生成
python -c "from gogooku3.features import QualityFinancialFeaturesGenerator; ..."
```

### 推論実行
```bash
# モデル推論
gogooku3 infer --model-path models/best_model.pth --data-path data/latest.parquet
```

## 🎯 計画中のオーケストレーション

### Dagster Assets
```python
# 計画中の実装例
@asset
def raw_stock_data():
    """JQuants APIからの生データ取得"""
    pass

@asset  
def processed_features(raw_stock_data):
    """特徴量エンジニアリング"""
    pass

@asset
def trained_model(processed_features):
    """ATFT-GAT-FAN学習"""
    pass

@asset
def predictions(trained_model, processed_features):
    """推論実行"""
    pass
```

### スケジュール設定
```python
# 日次バッチ実行
daily_schedule = ScheduleDefinition(
    job=ml_training_job,
    cron_schedule="0 2 * * *",  # 毎日 2:00 AM JST
    execution_timezone="Asia/Tokyo"
)

# リアルタイム処理
realtime_sensor = SensorDefinition(
    job=inference_job,
    evaluation_fn=market_data_sensor
)
```

## 📊 予定機能

### ワークフロー管理
- **依存関係の可視化**: データ系譜・タスク依存グラフ
- **実行履歴**: 成功/失敗率・実行時間履歴
- **リトライ機能**: 失敗時の自動再実行
- **部分実行**: 特定のタスクのみ実行

### 監視・アラート
- **SLA監視**: 実行時間・成功率の監視
- **データ品質**: 入力データの品質チェック
- **モデル性能**: 予測精度の継続監視
- **リソース使用量**: CPU・メモリ・GPU使用率

### 実験管理
- **MLflow統合**: 実験・モデル管理
- **A/B テスト**: モデルバージョン比較
- **ハイパーパラメータ最適化**: Optuna統合
- **自動デプロイ**: 性能改善時の自動本番反映

## 🔗 関連ドキュメント

現在利用可能：
- [データパイプライン](data-pipeline.md) - データ処理フロー詳細
- [モデル学習](../ml/model-training.md) - ML学習パイプライン
- [運用手順](../operations/runbooks.md) - 現在の手動運用手順

計画中：
- **observability.md** - 監視・ダッシュボード設計
- **feature-store.md** - Feast統合設計

---

**🚧 作成予定日**: 2025年10月  
**👥 担当**: DevOps・データエンジニアリングチーム
