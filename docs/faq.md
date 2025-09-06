# ❓ Frequently Asked Questions (FAQ)

Gogooku3に関するよくある質問と回答集です。

## 🚀 セットアップ・基本操作

### Q: Gogooku3を初めて使うには何から始めれば良いですか？

**A:** まず[はじめに](getting-started.md)ドキュメントに従ってセットアップを行ってください：

1. **環境準備**: Python 3.9+、pip、8GB+ RAM
2. **インストール**: `pip install -e .`
3. **設定**: `.env`ファイル作成（JQuants認証情報等）
4. **動作確認**: `gogooku3 --version` でインストール確認
5. **クイックテスト**: `python scripts/run_safe_training.py --n-splits 1` で30秒テスト

### Q: システム要件はどの程度必要ですか？

**A:** 最小・推奨要件：

```yaml
最小要件:
  CPU: 4コア以上
  RAM: 8GB以上
  Storage: 10GB以上の空き容量
  Python: 3.9以上

推奨要件:
  CPU: 24コア以上（AMD EPYC 7V13等）
  RAM: 16GB以上（32GB推奨）
  GPU: A100/H100（ATFT学習時）
  Storage: SSD 50GB以上
```

### Q: JQuants API の認証情報はどこで取得しますか？

**A:** [JQuants公式サイト](https://jpx-jquants.com)でアカウント登録後：

1. ダッシュボードでAPIキーを発行
2. `.env`ファイルに以下を設定：
```bash
JQUANTS_AUTH_EMAIL=your_email@example.com
JQUANTS_AUTH_PASSWORD=your_password
```
3. `python -c "from gogooku3 import settings; print(settings.jquants_email)"` で確認

## 🧠 機械学習・モデル関連

### Q: ATFT-GAT-FANモデルの特徴を教えてください

**A:** 次世代金融時系列予測モデルの特徴：

- **ATFT**: 適応的時間融合トランスフォーマー（60日シーケンス処理）
- **GAT**: グラフアテンション（銘柄間相関モデリング、50銘柄・266エッジ）
- **FAN**: 周波数適応正規化（動的特徴量スケーリング）
- **パラメータ**: 5.6M、target Sharpe 0.849
- **予測期間**: マルチ期間（1/5/10/20日）
- **学習時間**: A100で50エポック約4時間

詳細: [モデル学習ドキュメント](ml/model-training.md)

### Q: なぜ713特徴量から26特徴量に削減したのですか？

**A:** 品質と効率性の向上のため：

```yaml
改善効果:
  メモリ使用量: 17GB → 7GB (59%削減)
  処理時間: 数分 → 1.9秒 (100倍高速)
  過学習リスク: 高 → 低
  解釈しやすさ: 困難 → 容易
  
選択基準:
  - 実証的根拠のある指標のみ選択
  - 低相関（冗長性排除）
  - 計算安定性
  - 金融理論的妥当性
```

詳細: [モデル学習/評価](ml/model-training.md)

### Q: Walk-Forward Validationとは何ですか？

**A:** 時系列データ専用の安全な交差検証手法：

- **目的**: 未来情報リーク防止
- **Embargo**: 学習・検証間20日の空白期間
- **分割**: 時系列順序を保った5分割
- **実装**: WalkForwardSplitterV2で自動適用

```python
# 使用例
splitter = WalkForwardSplitterV2(
    n_splits=5,
    embargo_days=20,        # 最大予測期間と一致
    min_train_days=252      # 最低1年の学習データ
)
```

詳細: [安全性ガードレール](ml/safety-guardrails.md)

## 📊 パフォーマンス・効率化

### Q: メモリ不足エラーが発生します。対処法は？

**A:** 以下の方法で段階的に対処：

**Step 1: メモリ制限設定**
```bash
# パイプライン実行時にメモリ制限
python scripts/run_safe_training.py --memory-limit 4  # 4GB制限

# または新CLI
gogooku3 train --memory-limit 4
```

**Step 2: バッチサイズ削減**
```bash
# ATFT学習時
gogooku3 train --config configs/model/atft/train.yaml --batch-size 256
```

**Step 3: データサンプリング**
```python
# 一部データでの学習
pipeline = SafeTrainingPipeline(sample_fraction=0.5)  # 50%サンプリング
```

### Q: 処理速度を上げる方法はありますか？

**A:** 複数の最適化手法：

**データ処理最適化:**
- Polarsの活用（既に実装済み、pandas比3-5倍高速）
- 並列処理（CPU全コア活用）
- Lazy Loading（必要時のみメモリロード）

**学習最適化:**
```bash
# Mixed precision training (A100/H100)
gogooku3 train --precision bf16-mixed

# GPU並列処理（複数GPU環境）
gogooku3 train --devices 2

# 勾配チェックポインティング（メモリvs速度トレードオフ）
gogooku3 train --gradient-checkpointing
```

### Q: 現在のパフォーマンスベンチマークは？

**A:** 本番実績（2025-08-28時点）：

```yaml
実行速度（606K samples処理）:
  全パイプライン: 1.9秒 ✅ (目標: <2秒)
  データ読み込み: 0.1秒
  特徴量生成: 0.2秒
  正規化: 0.2秒
  ML学習: 0.6秒
  
メモリ効率:
  ピーク使用量: 7.0GB ✅ (目標: <8GB)
  削減率: 59% (17GB→7GB)
  
ML性能:
  銘柄数: 632銘柄
  特徴量: 145列（+6品質向上）
  Target Sharpe: 0.849 (ATFT-GAT-FAN)
```

## 🛠️ トラブルシューティング

### Q: "ImportError: No module named 'gogooku3'" エラーが出ます

**A:** パッケージインストール確認：

```bash
# Step 1: 現在のディレクトリ確認
pwd  # /home/ubuntu/gogooku3-standaloneにいることを確認

# Step 2: 開発モードインストール
pip install -e .

# Step 3: インストール確認
python -c "import gogooku3; print('✅ Package OK')"
python -c "from gogooku3.training import SafeTrainingPipeline; print('✅ Training OK')"

# Step 4: 失敗する場合はPYTHONPATH設定
export PYTHONPATH="/home/ubuntu/gogooku3-standalone:$PYTHONPATH"
```

### Q: "データリークの警告"が表示されます。対処が必要ですか？

**A:** 警告レベルで判断：

**🟡 軽微な警告（対処不要）:**
```
⚠️ Found 2 temporal overlaps (edge cases)
⚠️ Minor date boundary overlap: <24 hours
```
→ 既知の技術的課題、embargo期間は維持されており本番影響なし

**🔴 重大な警告（要対処）:**
```
❌ Severe temporal overlap detected
❌ Embargo period violated
```
→ WalkForwardSplitterV2のembargo設定を調整：

```python
# embargo期間を延長
splitter = WalkForwardSplitterV2(
    embargo_days=30,        # 20→30日に延長
    min_train_days=365      # 最小学習期間も延長
)
```

### Q: ATFT学習が途中で停止します

**A:** GPU・メモリ関連の確認：

**GPU確認:**
```bash
# GPU利用可能性確認
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# GPU使用量監視
nvidia-smi

# CPU学習に切り替え
gogooku3 train --config configs/model/atft/train.yaml --accelerator cpu
```

**メモリ確認:**
```bash
# システムメモリ確認
free -h

# メモリ制限学習
gogooku3 train --memory-limit 4 --batch-size 256
```

**チェックポイント復旧:**
```bash
# 途中から再開
gogooku3 train --config configs/model/atft/train.yaml --resume checkpoints/last.ckpt
```

## 🔄 移行・アップデート

### Q: v1.x からv2.0.0への移行方法は？

**A:** 段階的移行アプローチ：

**Phase 1: 互換性レイヤー使用**
```python
# 既存コード（v1.x）そのまま動作、deprecation warning表示
from scripts.run_safe_training import SafeTrainingPipeline
pipeline = SafeTrainingPipeline()  # ⚠️ 非推奨警告
```

**Phase 2: 新パッケージ導入**
```python
# 新しいインポート方式
from gogooku3.training import SafeTrainingPipeline
pipeline = SafeTrainingPipeline()
```

**Phase 3: 新CLI活用**
```bash
# 旧方式
python scripts/run_safe_training.py --verbose

# 新方式  
gogooku3 train --verbose
```

詳細: [移行ガイド](../MIGRATION.md)

### Q: 既存のスクリプトは動作し続けますか？

**A:** はい、後方互換性を保持：

```bash
# v2.0.0でも動作（deprecation warning付き）
python scripts/run_safe_training.py ✅
python scripts/train_atft.py ✅
python scripts/integrated_ml_training_pipeline.py ✅

# 新方式（推奨）
gogooku3 train ✅
python -m gogooku3.cli train ✅
```

移行期間中（v2.x系）は両方式をサポート。v3.0.0で旧方式廃止予定。

## 📈 運用・本番環境

### Q: 本番環境でのベストプラクティスは？

**A:** 推奨運用設定：

**環境分離:**
```bash
# 本番環境設定
export GOGOOKU3_ENVIRONMENT=production
export GOGOOKU3_LOG_LEVEL=INFO
export GOGOOKU3_MEMORY_LIMIT_GB=8
```

**監視・アラート:**
- メモリ使用量: <8GB
- 実行時間: <5秒（アラート閾値）
- データ取得失敗率: <1%

**バックアップ・復旧:**
```bash
# モデル保存
gogooku3 train --save-path models/production/atft_$(date +%Y%m%d).pth

# 設定バックアップ
cp configs/model/atft/train.yaml backups/
```

### Q: API制限やレート制限はありますか？

**A:** JQuants API制限情報：

```yaml
JQuants制限:
  リクエスト/秒: 10req/s
  リクエスト/日: 100,000req/day
  同時接続数: 5接続まで
  
Gogooku3最適化:
  並行接続: 5接続（上限遵守）
  リトライ機能: 指数バックオフ
  キャッシュ: 日次データキャッシュ
```

制限に達した場合は自動的に待機・リトライします。

## 🔗 コミュニティ・サポート

### Q: バグ報告や機能要望はどこに連絡すれば？

**A:** GitHubイシュートラッカーをご利用ください：

- **バグ報告**: [GitHub Issues](https://github.com/your-org/gogooku3/issues) 
- **機能要望**: Feature Request template使用
- **質問**: Discussions板またはこのFAQ
- **セキュリティ**: security@example.com に直接連絡

**報告時の情報:**
```bash
# システム情報収集
gogooku3 --version
python --version  
pip list | grep -E "(gogooku3|polars|torch)"
```

### Q: 追加ドキュメントはどこで見られますか？

**A:** 包括的ドキュメントを整備：

- **📖 [メインポータル](index.md)**: 全ドキュメント索引
- **🚀 [はじめに](getting-started.md)**: セットアップ・基本操作
- **🏗️ [アーキテクチャ](architecture/data-pipeline.md)**: 技術仕様詳細
- **🧠 [ML学習](ml/model-training.md)**: ATFT-GAT-FAN詳細
- **🛡️ [安全性](ml/safety-guardrails.md)**: データリーク防止
- **📚 [用語集](glossary.md)**: 技術用語解説
- **🔄 [移行ガイド](../MIGRATION.md)**: アップグレード手順

### Q: コードサンプル・チュートリアルはありますか？

**A:** 実践的なサンプルを提供：

**基本操作:**
```python
# クイックスタート（30秒で完了）
from gogooku3.training import SafeTrainingPipeline
pipeline = SafeTrainingPipeline(experiment_name="quickstart")
results = pipeline.run_pipeline(n_splits=1, embargo_days=20)
```

**カスタマイズ:**
```python  
# 本格学習（カスタム設定）
pipeline = SafeTrainingPipeline(
    data_path="your_data.parquet",
    experiment_name="custom_experiment", 
    verbose=True
)

results = pipeline.run_pipeline(
    n_splits=5,
    embargo_days=30,        # 長いembargo
    memory_limit_gb=16      # 大容量メモリ
)
```

**Jupyter Notebook**: `examples/` ディレクトリ（計画中）

---

## 💡 さらにヘルプが必要な場合

1. **🔍 検索**: このFAQとドキュメント内をCtrl+F/Cmd+Fで検索
2. **📚 用語集**: [用語集](glossary.md)で技術用語を確認
3. **🚀 セットアップ**: [はじめに](getting-started.md)で基本操作を確認
4. **🐛 トラブル**: 上記トラブルシューティングセクション参照
5. **💬 コミュニティ**: GitHub DiscussionsまたはIssues

**📞 緊急時**: システム障害等の緊急事態は admin@example.com までご連絡ください。
