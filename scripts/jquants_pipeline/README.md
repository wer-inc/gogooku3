# JQuants Pipeline v4.0

営業日ベースのスケジューリングとハイブリッドアプローチを実装した最適化パイプライン

## 特徴

- **自動軸選択**: Date-axis vs Code-axisを自動判定
- **イベント検知**: 上場/廃止/市場変更イベントを自動検出
- **効率的なAPI利用**: 14回のAPIコールで5営業日のデータ取得
- **高速処理**: 従来比51%高速化

## 使用方法

### 基本的な実行

```bash
# 新しいパッケージ構造での実行
cd /home/ubuntu/gogooku3-standalone
python -m scripts.jquants_pipeline.pipeline \
  --start-date 2024-12-16 \
  --end-date 2024-12-20
```

### Pythonコードからの利用

```python
import sys
sys.path.append('/home/ubuntu/gogooku3-standalone/scripts')

from jquants_pipeline import JQuantsPipelineV4

# パイプラインの初期化と実行
pipeline = JQuantsPipelineV4()
df, metadata = await pipeline.run(
    use_jquants=True,
    start_date="2024-12-16", 
    end_date="2024-12-20"
)
```

## ディレクトリ構造

```
jquants_pipeline/
├── __init__.py         # パッケージ初期化
├── pipeline.py         # メインパイプライン
├── components/         # 最適化コンポーネント
│   ├── axis_decider.py
│   ├── event_detector.py
│   ├── listed_info_manager.py
│   ├── daily_quotes_by_code.py
│   ├── trading_calendar_fetcher.py
│   └── market_code_filter.py
└── data/
    └── ml_dataset_builder.py
```

## パフォーマンス

- APIコール: 14回（最適）
- 処理速度: 2,143 rows/秒
- メモリ使用: ~250MB
- 実行時間: ~27秒（5営業日）

## 主要コンポーネント

### AxisDecider
Date-axis vs Code-axisの効率性を測定し、自動選択

### EventDetector  
日次のCode集合変化から市場イベントを検知

### ListedInfoManager
月次スナップショット＋バイナリサーチでイベント日を特定

### DailyQuotesByCode
市場所属期間を考慮したコード軸での効率的取得

## 旧バージョンとの互換性

旧バージョンは`_archive/`ディレクトリに保存されています：
- `_archive/run_pipeline.py` (v1)
- `_archive/run_pipeline_v2.py` (v2)  
- `_archive/run_pipeline_v3.py` (v3)

## 更新履歴

- v4.0 (2025-09-02): 最適化版リリース
  - AxisDecider実装
  - EventDetector追加
  - パフォーマンス51%向上