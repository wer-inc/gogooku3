# Claude Code Skills ガイド

ATFT-GAT-FAN プロジェクト用の Claude Code Skills セットアップと使用方法

## 概要

Claude Code Skills は、プロジェクト固有のタスクを自動化する強力な機能です。このプロジェクトでは、以下の5つの専門スキルを提供しています。

## インストール方法

リポジトリ内の `claude/skills/` に最新のスキル定義を保存しています。最初に以下を実行してローカル環境へ同期してください:

```bash
mkdir -p ~/.claude/skills
rsync -av claude/skills/ ~/.claude/skills/
ls -la ~/.claude/skills/
# atft-pipeline/
# atft-training/
# atft-research/
# atft-code-quality/
# atft-autonomy/
```

Codex 連携を利用する場合は OpenAI Codex CLI を導入し、初期設定を済ませます:

```bash
npm install -g @openai/codex
./tools/codex.sh --no-check   # .mcp.json / AGENTS.md を自動生成
```

Claude Code を再起動するとスキルが読み込まれます。

## スキル一覧

### 1. atft-pipeline (データパイプライン)

**用途**: データセット生成、キャッシュ管理、JQuants API 連携

**自動起動するケース**:
- "データセットを生成して"
- "過去5年分のデータを取得"
- "キャッシュを確認"
- "データパイプラインを実行"

**提供するコマンド/ワークフロー**:
- `make dataset-check-strict` → GPU/認証プリフライト
- `make dataset-bg` → 5年分データセットを GPU + SSH-safe 背景実行
- `make dataset-gpu-refresh` → キャッシュをスキップし強制再取得
- `make dataset-safe-resume` → チャンク生成の失敗箇所を再開
- `make cache-stats` / `make cache-prune` → グラフキャッシュの健全性維持
- `python scripts/pipelines/run_full_dataset.py --inspect-graph` → 特徴量/エッジ検査
- `tools/project-health-check.sh --section dataset` → 全体診断

**詳細**: `~/.claude/skills/atft-pipeline/SKILL.md`

### 2. atft-training (モデルトレーニング)

**用途**: モデル学習、ハイパーパラメータ調整、実験管理

**自動起動するケース**:
- "モデルをトレーニング"
- "Safe mode で学習"
- "120エポック実行"
- "ハイパーパラメータ最適化"

**提供するコマンド/ワークフロー**:
- `make train-optimized` → TorchInductor + FlashAttention2 を有効にした本番訓練
- `make train-quick` → 3エポックのクイックスモーク
- `make train-safe` → シングルワーカー/非コンパイルの安全モード
- `make train-monitor` / `make train-status` → 背景プロセス監視
- `make hpo-run HPO_TRIALS=24 ...` → Optuna ハイパーパラメータ探索
- `python scripts/integrated_ml_training_pipeline.py --profile` → プロファイリング
- `tools/project-health-check.sh --section training` → 事前ヘルスチェック

**詳細**: `~/.claude/skills/atft-training/SKILL.md`

### 3. atft-research (研究・分析)

**用途**: 特徴量分析、予測性能評価、実験結果可視化

**自動起動するケース**:
- "特徴量の重要度を分析"
- "予測性能を評価"
- "ベースラインと比較"
- "研究レポートを生成"

**提供するコマンド/ワークフロー**:
- `make research-plus` → 指標集計 + 可視化 + レポート生成
- `make research-baseline RUN=...` → 最新ランと基準モデルの比較
- `python scripts/research/factor_drift.py` → 特徴量ドリフト検査
- `python scripts/research/regime_detector.py` → レジーム分析
- `python scripts/research/plot_metrics.py` → KPI グラフ生成
- `make research-report FACTORS=... HORIZONS=...` → レポートテンプレート出力

**詳細**: `~/.claude/skills/atft-research/SKILL.md`

### 4. atft-code-quality (コード品質)

**用途**: Linting、フォーマット、型チェック、テスト実行

**自動起動するケース**:
- "コードをフォーマット"
- "リントエラーを修正"
- "型チェックを実行"
- "テストを走らせて"

**提供するコマンド/ワークフロー**:
- `tools/project-health-check.sh --section quality` → 事前診断
- `ruff check src/ --fix` / `ruff format src/ tests/` → Lint & フォーマット
- `mypy src/gogooku3 scripts/` → 型チェック
- `pytest tests/unit -n auto` / `pytest tests/integration -m "not slow"` → テスト実行
- `pre-commit run --all-files` → フックのフル実行
- `bandit -qr src/` / `detect-secrets scan` → セキュリティ診断

**詳細**: `~/.claude/skills/atft-code-quality/SKILL.md`

### 5. atft-autonomy (Claude × Codex 連携)

**用途**: Claude Skills と OpenAI Codex を組み合わせ、完全自律運用フローを構築

**自動起動するケース**:
- "Claude と Codex を連携してメンテして"
- "完全自律モードで日次メンテナンス"
- "インシデント対応をエージェント協調で"

**提供するコマンド/ワークフロー**:
- `./tools/claude-code.sh` → 高速メンテナンス
- `./tools/codex.sh --max --exec "..."` → 深い最適化/調査
- `tools/project-health-check.sh --summary` → 事前スナップショット
- `_logs/codex-sessions/` と `_logs/claude-code/` を集約し `docs/ops/autonomy_log.md` に記録
- cron 連携例: `0 2 * * * cd /workspace/gogooku3 && ./tools/codex.sh --max`

**詳細**: `~/.claude/skills/atft-autonomy/SKILL.md`

## 実践例

### 例1: 初回プロジェクトセットアップ

```
あなた: "プロジェクトをセットアップして、データを準備し、クイック訓練したい"

Claude の自動処理:
1. [atft-pipeline] 環境チェック
   → make dataset-check

2. [atft-pipeline] データセット生成 (5年分)
   → make dataset-bg

3. [atft-training] クイック訓練 (3エポック)
   → make train-quick

4. [atft-research] 動作確認
   → python scripts/smoke_test.py --max-epochs 1

5. 結果レポート
```

### 例2: ハイパーパラメータ実験

```
あなた: "バッチサイズ4096、学習率2e-4で120エポック訓練したい"

Claude の自動処理:
1. [atft-training] データセット確認
   → ls -la output/ml_dataset_latest_full.parquet

2. [atft-training] カスタム訓練実行
   → make train BATCH_SIZE=4096 LR=2e-4 EPOCHS=120

3. [atft-training] 進捗監視
   → make train-monitor

4. [atft-research] 結果分析
   → make research-baseline
```

### 例3: デバッグセッション

```
あなた: "訓練が途中で止まる。原因を調査して修正したい"

Claude の自動処理:
1. [atft-training] ログ確認
   → make train-status
   → tail -f _logs/training/*.log

2. [atft-training] プロセス状態確認
   → ps aux | grep train_atft

3. [atft-pipeline] データ検証
   → python -c "import polars as pl; df = pl.read_parquet(...); print(df.shape)"

4. [atft-code-quality] コード品質チェック
   → ruff check src/gogooku3/training/

5. 問題特定 + 修正提案
   → Safe mode での再実行を提案
```

### 例4: コミット前品質チェック

```
あなた: "コミット前に全品質チェックして"

Claude の自動処理:
1. [atft-code-quality] Linting
   → ruff check src/ --fix

2. [atft-code-quality] フォーマット
   → ruff format src/

3. [atft-code-quality] 型チェック
   → mypy src/gogooku3

4. [atft-code-quality] ユニットテスト
   → pytest tests/unit/ -v

5. [atft-code-quality] Pre-commit
   → pre-commit run --all-files

6. 結果サマリー
```

## スキルの仕組み

### Proactive Mode (自動起動)

すべてのスキルは `proactive: true` に設定されており、関連するタスクを自動検出します。

**動作フロー**:
1. ユーザーがタスクを依頼
2. Claude がタスクの内容を分析
3. 関連するスキルを自動識別
4. スキルの知識を読み込み
5. タスクを実行

### Manual Mode (手動起動)

特定のスキルを明示的に指定することも可能:

```
"atft-pipeline スキルを使ってデータセットを生成"
"atft-training スキルで Safe mode トレーニング"
```

## カスタマイズ

### スキルの編集

各スキルの動作をカスタマイズできます:

```bash
# 例: デフォルトエポック数を変更
vi ~/.claude/skills/atft-training/SKILL.md
```

### 新しいスキルの追加

```bash
# 1. ディレクトリ作成
mkdir ~/.claude/skills/my-custom-skill

# 2. SKILL.md を作成
cat > ~/.claude/skills/my-custom-skill/SKILL.md << 'EOF'
---
name: my-custom-skill
description: カスタムスキルの説明
proactive: true
---

# My Custom Skill

スキルの詳細...
EOF

# 3. Claude Code 再起動
```

## トラブルシューティング

### スキルが認識されない

**確認事項**:
1. ディレクトリ構造が正しいか
   ```bash
   ls -la ~/.claude/skills/atft-*/SKILL.md
   ```

2. YAML フロントマターが正しいか
   ```bash
   head -n 5 ~/.claude/skills/atft-pipeline/SKILL.md
   ```

3. Claude Code を再起動したか

### スキルが自動起動しない

**原因と対処**:
1. `proactive: false` になっている
   → `proactive: true` に変更

2. トリガーワードが不明確
   → description を具体的に記述

3. 複数スキルが競合
   → 明示的にスキル名を指定

### スキルの動作確認

```bash
# スキルメタデータ確認
for skill in ~/.claude/skills/*/SKILL.md; do
  echo "=== $(dirname $skill | xargs basename) ==="
  head -n 5 "$skill"
  echo
done
```

## ベストプラクティス

### 1. タスクは具体的に依頼

❌ 悪い例: "何かやって"
✅ 良い例: "過去5年分のデータセットを生成して、Safe mode で3エポック訓練したい"

### 2. 関連スキルを連携させる

```
"データセット生成 → 訓練 → 結果分析までやって"
→ pipeline → training → research の3スキル連携
```

### 3. エラー時は詳細情報を提供

```
"訓練が失敗する。ログはこれです: [ログ内容]"
→ Claude が適切なスキルで原因分析
```

### 4. カスタムパラメータは明示

```
"BATCH_SIZE=4096、LR=2e-4、EPOCHS=120 で訓練"
→ training スキルが正確にパラメータを適用
```

## 参考リンク

- **Claude Code Documentation**: https://docs.claude.com/en/docs/claude-code/
- **Skills Marketplace**: https://github.com/anthropics/skills
- **プロジェクト CLAUDE.md**: [CLAUDE.md](../CLAUDE.md)
- **プロジェクト README**: [README.md](../README.md)
- **スキル定義リポジトリ版**: [claude/skills/README.md](../claude/skills/README.md)
- **Codex 自律運用ガイド**: [tools/README.md](../tools/README.md)
- **Codex MCP 設定**: [docs/guides/codex_mcp.md](../docs/guides/codex_mcp.md)

## バージョン情報

- **Claude Code**: 2.0.22
- **Skills Version**: 1.0.0
- **プロジェクト**: ATFT-GAT-FAN v2.0.0
- **作成日**: 2025-10-18

---

これらのスキルを活用して、ATFT-GAT-FAN プロジェクトの開発を加速させましょう！
