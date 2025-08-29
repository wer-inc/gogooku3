# 📚 Gogooku3 用語集

Gogooku3で使用される主要な技術用語・概念の解説集です。

## 🏗️ アーキテクチャ用語

### **ATFT-GAT-FAN**
**Adaptive Temporal Fusion Transformer + Graph Attention Network + Frequency Adaptive Normalization**

Gogooku3の主力MLモデル。3つのコンポーネントを組み合わせた次世代時系列予測モデル：
- **ATFT**: 適応的時間融合トランスフォーマー（時系列アテンション）
- **GAT**: グラフアテンションネットワーク（銘柄間相関モデリング）  
- **FAN**: 周波数適応正規化（動的特徴量スケーリング）
- **パラメータ数**: 5.6M
- **目標Sharpe**: 0.849

### **SafeTrainingPipeline**
データリーク防止とモデル学習を統合した7段階安全パイプライン：
1. データローディング（ProductionDatasetV3）
2. 特徴量エンジニアリング（QualityFinancialFeaturesGenerator）
3. 横断面正規化（CrossSectionalNormalizerV2）
4. Walk-Forward分割（WalkForwardSplitterV2）
5. ベースライン学習（LightGBMFinancialBaseline）
6. グラフ構築（FinancialGraphBuilder）
7. ATFT-GAT-FAN学習

### **ProductionDatasetV3**
Polars最適化による高速データローダー：
- **Lazy Loading**: 必要時のみデータ読み込み
- **Column Projection**: 必要列のみ選択的ロード
- **Memory Efficiency**: メモリ使用量を59%削減（17GB→7GB）
- **Processing Speed**: 606K samples を0.1秒で処理

## 📊 機械学習・評価用語

### **IC (Information Coefficient)**
予測値と実際のリターンの相関係数。金融ML評価の基本指標：
- **範囲**: -1 ～ +1（高いほど良い）
- **有意水準**: IC > 0.05 で統計的に有意
- **計算式**: `pearson_correlation(predictions, actual_returns)`

### **RankIC (Rank Information Coefficient)**
予測順位と実際リターン順位のスピアマン相関。外れ値に頑健：
- **特徴**: ICより外れ値の影響を受けにくい
- **本番目標**: RankIC > 0.10
- **用途**: デシル分析・ロングショート戦略評価

### **Sharpe Ratio**
リスク調整済みリターンの指標：
- **計算式**: `(平均リターン - リスクフリーレート) / リターン標準偏差`
- **ATFT-GAT-FAN目標**: 0.849
- **用途**: 戦略の効率性評価

### **Walk-Forward Validation**
時系列データ専用の交差検証手法。未来情報リークを防ぐ：
- **Embargo Period**: 学習・検証間の空白期間（20日）
- **時系列順序保持**: 過去→現在→未来の順序を厳密に維持
- **分割数**: 5分割（5-fold）
- **最小学習期間**: 252日（約1年）

### **Cross-Sectional Normalization**
日次横断面Z-score正規化。市場全体効果を除去：
- **目的**: 相対的パフォーマンス（銘柄間比較）に注力
- **手法**: 各日の銘柄群をZ-score正規化（平均0、標準偏差1）
- **安全性**: 学習時統計量のみ使用（未来情報リーク防止）

## 🛡️ 安全性・品質管理用語

### **Embargo Period**
学習データと検証データ間の意図的な空白期間：
- **期間**: 20日（最大予測期間と一致）
- **目的**: 情報リークの完全防止
- **実装**: WalkForwardSplitterV2で自動適用

### **Data Leakage (データリーク)**
未来の情報が過去の予測に使用される問題：
- **典型例**: 未来データでの正規化、BatchNormの使用
- **防止策**: CrossSectionalNormalizerV2、Walk-Forward分割
- **検出**: 自動重複検知システム

### **Quality Features**
高品質特徴量エンジニアリング：
- **横断面分位数特徴量**: 銘柄間相対位置
- **頑健外れ値検出**: 2σ閾値でのクリッピング
- **品質向上**: +6特徴量追加（139→145列）

## 📈 金融・市場用語

### **JQuants API**
日本取引所グループが提供する金融データAPI：
- **データソース**: 株価、財務、企業情報
- **銘柄数**: 4000+ 日本株
- **更新頻度**: 日次（T+1）
- **認証**: メールアドレス・パスワード認証

### **Technical Indicators**
テクニカル指標。価格・出来高データから計算される指標：
- **現在実装**: 26指標（RSI、MACD、ボリンジャーバンド等）
- **元実装**: 713指標（96%削減で効率化）
- **選択基準**: 低相関・高予測力・計算安定性

### **Multi-Horizon Prediction**
複数期間での同時予測：
- **予測期間**: 1日、5日、10日、20日
- **重み付け**: 1日>5日>10日=20日
- **用途**: 短期・中期戦略の両立

### **Financial Graph**
銘柄間相関に基づく金融グラフネットワーク：
- **ノード**: 株式銘柄（50銘柄）
- **エッジ**: 相関係数（266エッジ）
- **更新頻度**: 月次（相関期間：60日）
- **負相関**: 分散投資効果のため含有

## ⚙️ 技術・実装用語

### **Polars**
Rust実装の高速データフレームライブラリ：
- **性能**: pandas比3-5倍高速
- **メモリ効率**: Lazy Evaluationによる最適化
- **並列処理**: マルチスレッド自動最適化

### **Hydra**
Facebook製の設定管理フレームワーク：
- **設定階層**: data, model, training の構造化
- **動的合成**: 実行時設定の組み合わせ
- **実験管理**: パラメータスイープ・ハイパーパラメータ調整

### **Mixed Precision Training**
bfloat16による高速学習：
- **精度**: bf16（学習） + fp32（損失計算）
- **速度向上**: A100/H100で最大2倍高速
- **メモリ削減**: 約50%のメモリ使用量削減

### **Graph Attention Network (GAT)**
グラフニューラルネットワークの一種：
- **アテンション機構**: 隣接ノード重要度の動的計算
- **マルチヘッド**: 8ヘッドアテンション
- **更新機構**: エポック毎のグラフ動的更新

## 🔧 運用・DevOps用語

### **Pre-commit Hooks**
コミット前自動品質チェック：
- **ruff**: Python linting・formatting
- **mypy**: 型チェック
- **bandit**: セキュリティチェック
- **markdownlint**: Markdown品質チェック

### **Package Structure (v2.0.0)**
現代的Pythonパッケージ構造：
```
src/gogooku3/
├── __init__.py          # Public API
├── cli.py              # CLI interface
├── training/           # SafeTrainingPipeline
├── models/             # ATFT-GAT-FAN
├── data/               # loaders, scalers
└── compat/             # 後方互換レイヤー
```

### **Settings Management**
Pydantic設定管理システム：
- **環境変数**: `.env`ファイル自動読み込み
- **型安全性**: Pydanticバリデーション
- **階層設定**: 開発・本番環境分離

## 📊 パフォーマンス・ベンチマーク用語

### **壊れず・強く・速く**
Gogooku3設計哲学：
- **🛡️ 壊れず**: データリーク防止・品質チェック・例外処理
- **💪 強く**: 高性能モデル・多期間予測・Graph Attention
- **⚡ 速く**: Polars最適化・並列処理・メモリ効率化

### **Target Metrics**
本番運用目標値：
```yaml
実行性能:
  パイプライン実行時間: <2秒 (現在: 1.9秒) ✅
  メモリ使用量: <8GB (現在: 7.0GB) ✅
  
ML性能:
  Sharpe Ratio: 0.849
  IC (1日): >0.05  
  RankIC (5日): >0.10
  
安全性:
  データリーク: 0件
  Embargo: 20日間
```

## 🔄 移行・互換性用語

### **Compatibility Layer**
後方互換性レイヤー：
- **場所**: `src/gogooku3/compat/`
- **機能**: 旧API → 新API自動変換
- **警告**: Deprecation warning表示
- **移行期間**: v2.0.0 → v3.0.0

### **Script Migration**
スクリプト → パッケージ移行：
- **Before**: `python scripts/run_safe_training.py`
- **After**: `gogooku3 train` または `python -m gogooku3.cli train`
- **利点**: モジュール化・型安全性・テスタビリティ

## ❓ 略語・アクロニム

| 略語 | 正式名称 | 意味 |
|------|---------|------|
| **IC** | Information Coefficient | 情報係数 |
| **GAT** | Graph Attention Network | グラフアテンションネットワーク |
| **ATFT** | Adaptive Temporal Fusion Transformer | 適応的時間融合トランスフォーマー |
| **FAN** | Frequency Adaptive Normalization | 周波数適応正規化 |
| **CV** | Cross Validation | 交差検証 |
| **API** | Application Programming Interface | アプリケーションプログラミングインターフェース |
| **CLI** | Command Line Interface | コマンドラインインターフェース |
| **ML** | Machine Learning | 機械学習 |
| **DL** | Deep Learning | 深層学習 |
| **NLP** | Natural Language Processing | 自然言語処理 |
| **GPU** | Graphics Processing Unit | グラフィック処理装置 |
| **CPU** | Central Processing Unit | 中央処理装置 |
| **RAM** | Random Access Memory | ランダムアクセスメモリ |
| **SSD** | Solid State Drive | ソリッドステートドライブ |
| **JSON** | JavaScript Object Notation | JSONデータ形式 |
| **YAML** | YAML Ain't Markup Language | YAML設定形式 |
| **CSV** | Comma-Separated Values | カンマ区切り値 |
| **SQL** | Structured Query Language | 構造化問い合わせ言語 |

## 🔗 関連ドキュメント

- [📖 メインドキュメント](index.md) - ドキュメントポータル
- [🚀 はじめに](getting-started.md) - セットアップガイド
- [🏗️ アーキテクチャ](architecture/data-pipeline.md) - 技術仕様
- [🧠 モデル学習](ml/model-training.md) - ATFT-GAT-FAN詳細
- [❓ FAQ](faq.md) - よくある質問

---

**💡 Tip**: このページは検索機能（Ctrl+F / Cmd+F）を活用してご利用ください。不明な用語がございましたら、[FAQ](faq.md)もご確認ください。