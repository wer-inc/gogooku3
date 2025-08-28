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

## 🙏 謝辞

- **JQuants API**: 日本株データ提供
- **Polars**: 高速データ処理
- **PyTorch**: 深層学習フレームワーク
- **gogooku2**: ベースシステム

---

**🚀 gogooku3-standalone - 壊れず・強く・速く の実現**