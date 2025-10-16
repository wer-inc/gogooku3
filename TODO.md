# TODO.md - gogooku3

**最終更新**: 2025-10-16 07:20 (HPO Sweep実行中)
**前バージョン**: GAT勾配ゼロ問題解決版（2025-10-07）

---

## 📌 現在の状況 (2025-10-16)

### ✅ PyTorch Lightning依存関係削除 - 完了

**状態**: PyTorch Lightning完全削除、純粋なPyTorchモデルに移行完了

**問題**:
- PyTorch Lightning import時に `std::bad_alloc` エラー発生
- 2TiB RAM環境でもインポート不可能
- v2.5.5, v2.0.9, v1.9.5 全てのバージョンで失敗

**解決方法**: PyTorch Lightningを完全削除し、純粋なnn.Moduleに移行
- `ATFT_GAT_FAN` を `pl.LightningModule` → `torch.nn.Module` に変更
- 全ての `self.log()` 呼び出しを削除（15箇所以上）
- `on_train_epoch_start()` コールバックを無効化
- `configure_optimizers()` を学習スクリプトに移動
- `training_step()` / `validation_step()` の引数を調整

**検証結果**:
- ✅ モデルインポート成功
- ✅ 1エポック学習テスト成功（Loss=0.3588, Val Loss=0.3626）
- ✅ HPO sweep開始成功

### 🚀 HPO Sweep実行中 (2025-10-16)

**状態**: 20トライアルHPO sweep正常稼働中

**開始時刻**: 2025-10-16 06:43 (UTC)
**プロセスID**: 159175
**完了予定**: 2025-10-17 午前1-3時頃（約20-30時間）

**設定**:
- Study名: `atft_hpo_production_20251016`
- トライアル数: 20
- エポック/トライアル: 10
- 出力先: `output/hpo_production/`
- ログ: `/tmp/hpo_production.log`

**最適化パラメータ**:
- Learning Rate: 1e-5 ~ 1e-3 (log scale)
- Batch Size: 512, 1024, 2048, 4096
- Hidden Size: 128, 256, 384, 512
- GAT Dropout: 0.1 ~ 0.4
- GAT Layers: 2 ~ 4

**目標**: Sharpe Ratio 最大化

**環境**:
- GPU: NVIDIA A100 80GB PCIe
- CPU: 256-core AMD EPYC
- RAM: 2.0 TiB (前回の10倍)
- Dataset: 8.9M rows, 4,484 stocks, 112 columns

---

## ⏳ 次のタスク（優先順）

### 1. HPO Sweep完了待ち 🕐
- [x] HPO sweep開始（2025-10-16 06:43）
- [ ] 中間結果確認（4-6時間後: 今日 10:00-12:00頃）
- [ ] 半分完了確認（12時間後: 今日 18:00頃）
- [ ] 最終結果確認（20-30時間後: 明日 08:00-10:00頃）

### 2. HPO結果分析 📊 (Sweep完了後)
- [ ] 最良パラメータの特定
- [ ] 全20トライアルの結果比較
- [ ] パフォーマンス傾向の分析
- [ ] 最適パラメータでの本番学習実行

### 3. モデルパフォーマンス評価 📈
- [ ] 最良モデルでのSharpe Ratio評価
- [ ] RankIC/IC改善度の測定
- [ ] バックテスト実行
- [ ] 予測精度の検証

---

## ✅ 完了済みタスク

### HPO環境セットアップ (2025-10-16)
- [x] 環境セットアップと検証
- [x] 前回のsweep結果確認（W&B/ローカル）
- [x] メモリ最適化設定の適用（2TiB RAM用）
- [x] GCSからデータセットダウンロード（8.9M rows）
- [x] W&Bインストールと認証（v0.22.2）
- [x] HPO環境セットアップ（Optuna v4.5.0）
- [x] PyTorch Lightning依存関係削除
- [x] 1エポック学習テスト実行
- [x] HPO sweep実行（1トライアルテスト）
- [x] 20トライアル本格的なHPO sweep開始

### PyTorch Lightning削除作業 (2025-10-16)
- [x] `import pytorch_lightning as pl` コメントアウト
- [x] クラス継承変更（`pl.LightningModule` → `nn.Module`）
- [x] `self.log()` 呼び出し削除（全箇所）
- [x] `on_train_epoch_start()` 無効化
- [x] `configure_optimizers()` 移動
- [x] `training_step()` / `validation_step()` 引数調整
- [x] `einops` パッケージインストール
- [x] モデルインポートテスト成功
- [x] 学習動作確認テスト成功

### GAT勾配ゼロ問題解決 (2025-10-06～2025-10-07)
- [x] Phase 1-6の調査完了
- [x] 根本原因特定（設定ファイル不完全）
- [x] GAT regularization設定追加
- [x] ゼロパディング実装
- [x] 検証ログでの動作確認
- [x] 診断ログDEBUG化（Phase 7-1）

---

## 🔍 監視方法（HPO Sweep）

### リアルタイム進捗確認
```bash
# HPO進捗（Optunaログ）
tail -f /tmp/hpo_production.log

# 学習詳細ログ
tail -f logs/ml_training.log

# プロセス状態
ps aux | grep 159175 | grep -v grep

# 完了トライアル数
ls output/hpo_production/trial_* 2>/dev/null | wc -l
```

### 中間結果確認
```bash
# 全トライアル結果
cat output/hpo_production/all_trials.json | jq '.[] | {trial: .number, sharpe: .value, params: .params}'

# 現在の最良パラメータ
cat output/hpo_production/best_params.json
```

### GPU/CPU使用状況
```bash
# GPU監視
watch -n 5 nvidia-smi

# CPU/メモリ監視
htop
```

---

## 📁 重要なファイル

### HPO関連
```
scripts/hpo/run_optuna_atft.py                # HPO実行スクリプト
output/hpo_production/best_params.json        # 最良パラメータ
output/hpo_production/all_trials.json         # 全トライアル結果
output/hpo_production/trial_*/metrics.json    # 各トライアルメトリクス
/tmp/hpo_production.log                       # HPOログ
```

### モデル・設定
```
src/atft_gat_fan/models/architectures/atft_gat_fan.py  # モデル（PyTorch Lightning削除済み）
configs/atft/config_production_optimized.yaml           # 本番設定
configs/atft/model/atft_gat_fan.yaml                    # モデル設定
```

### ログ
```
logs/ml_training.log                          # 学習ログ
/tmp/hpo_production.log                       # HPOログ
```

---

## 📚 学んだ教訓

### PyTorch Lightning依存関係問題
- **問題**: 大規模RAM環境でもインポート時にメモリエラー
- **原因**: PyTorch Lightning内部のメモリ割り当て問題
- **解決**: 完全削除し、純粋なPyTorchに移行
- **教訓**: 重要なフレームワーク依存は避け、コア機能のみ使用すべき

### HPO設定の重要性
- **発見**: 適切なハイパーパラメータ探索は性能向上に不可欠
- **方法**: Optunaによる自動最適化（TPE sampler + Median pruner）
- **メトリクス**: Sharpe Ratio最大化を目標に設定

---

## 🎯 今後の目標

### 短期目標 (1-2日)
- [ ] ✅ HPO sweep完了（20トライアル）
- [ ] 📊 最良パラメータ特定
- [ ] 🚀 最適パラメータで本番学習実行（120 epochs）
- [ ] 📈 モデルパフォーマンス評価

### 中期目標 (1週間)
- [ ] 🔍 バックテスト実行
- [ ] 📝 結果レポート作成
- [ ] ⚙️ 本番デプロイ準備
- [ ] 🧪 Ablation study（GAT有無、layers数など）

### 長期目標 (1ヶ月)
- [ ] 🌐 他のGNNアーキテクチャ検証
- [ ] 🔄 Multi-hop attention mechanisms
- [ ] 📈 Dynamic graph learning
- [ ] 🏆 Production deployment

---

## 🚨 トラブルシューティング

### HPO Sweepが停止した場合
```bash
# プロセス確認
ps aux | grep 159175

# 停止していた場合は再開（Optunaは途中から再開可能）
nohup python scripts/hpo/run_optuna_atft.py \
  --data-path output/ml_dataset_latest_full.parquet \
  --n-trials 20 \
  --max-epochs 10 \
  --study-name atft_hpo_production_20251016 \
  --output-dir output/hpo_production \
  > /tmp/hpo_production.log 2>&1 &
```

### メモリ不足エラー
- 2TiB RAM環境では発生しないはずだが、発生した場合:
  - バッチサイズを512に固定
  - `NUM_WORKERS=4` に削減

### GPU OOMエラー
- A100 80GBでは発生しないはずだが、発生した場合:
  - Mixed precision training確認（bf16使用中）
  - `RMM_POOL_SIZE=40GB` に削減

---

## 📊 環境情報（新環境）

```
GPU: NVIDIA A100 80GB PCIe
CPU: 256-core AMD EPYC
Memory: 2.0 TiB RAM (前回の10倍)
Storage: SSD
CUDA: 12.x
PyTorch: 2.9.0+cu128
Python: 3.12.3
Optuna: 4.5.0
wandb: 0.22.2
```

---

## 🎉 まとめ

### 達成したこと
- ✅ PyTorch Lightning依存関係を完全削除
- ✅ 純粋なPyTorchモデルに移行成功
- ✅ 新環境（2TiB RAM）でHPO sweep開始成功
- ✅ 20トライアルの自動ハイパーパラメータ最適化実行中

### 進行中
- 🚀 HPO sweep実行中（PID: 159175）
- ⏱️ 完了予定: 2025-10-17 午前1-3時頃

### 次のステップ
- 📊 HPO結果分析（明日朝）
- 🎯 最適パラメータで本番学習
- 📈 モデルパフォーマンス評価
- 🏆 Production deployment準備

---

**注**: 詳細な過去の履歴（GAT勾配ゼロ問題など）は以下に記録済み:
- Phase 1-6の調査プロセス
- 各種修正ファイル
- 検証ログ

現在はHPO sweepに集中。明日朝に最適パラメータを取得し、本番学習に進みます。
