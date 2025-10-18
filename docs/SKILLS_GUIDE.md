# Claude Code Skills ガイド

ATFT-GAT-FAN プロジェクト用の Claude Code Skills セットアップと使用方法

## 概要

Claude Code Skills は、プロジェクト固有のタスクを自動化する強力な機能です。このプロジェクトでは、以下の4つの専門スキルを提供しています。

## インストール方法

スキルは既に `~/.claude/skills/` にインストールされています:

```bash
ls -la ~/.claude/skills/
# atft-pipeline/
# atft-training/
# atft-research/
# atft-code-quality/
```

Claude Code を起動すると自動的に認識されます。

## スキル一覧

### 1. atft-pipeline (データパイプライン)

**用途**: データセット生成、キャッシュ管理、JQuants API 連携

**自動起動するケース**:
- "データセットを生成して"
- "過去5年分のデータを取得"
- "キャッシュを確認"
- "データパイプラインを実行"

**提供するコマンド**:
- `make dataset-bg` - バックグラウンドでデータセット生成
- `make cache-verify` - キャッシュ検証
- `make dataset-check` - 環境チェック

**詳細**: `~/.claude/skills/atft-pipeline/SKILL.md`

### 2. atft-training (モデルトレーニング)

**用途**: モデル学習、ハイパーパラメータ調整、実験管理

**自動起動するケース**:
- "モデルをトレーニング"
- "Safe mode で学習"
- "120エポック実行"
- "ハイパーパラメータ最適化"

**提供するコマンド**:
- `make train` - 最適化トレーニング
- `make train-safe` - 安定トレーニング
- `make train-status` - 進捗確認

**詳細**: `~/.claude/skills/atft-training/SKILL.md`

### 3. atft-research (研究・分析)

**用途**: 特徴量分析、予測性能評価、実験結果可視化

**自動起動するケース**:
- "特徴量の重要度を分析"
- "予測性能を評価"
- "ベースラインと比較"
- "研究レポートを生成"

**提供するコマンド**:
- `make research-plus` - 完全研究バンドル
- `make research-baseline` - ベースライン比較
- `python scripts/smoke_test.py` - スモークテスト

**詳細**: `~/.claude/skills/atft-research/SKILL.md`

### 4. atft-code-quality (コード品質)

**用途**: Linting、フォーマット、型チェック、テスト実行

**自動起動するケース**:
- "コードをフォーマット"
- "リントエラーを修正"
- "型チェックを実行"
- "テストを走らせて"

**提供するコマンド**:
- `ruff check src/ --fix` - Linting
- `ruff format src/` - フォーマット
- `mypy src/gogooku3` - 型チェック
- `pytest tests/ -v` - テスト実行

**詳細**: `~/.claude/skills/atft-code-quality/SKILL.md`

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

## バージョン情報

- **Claude Code**: 2.0.22
- **Skills Version**: 1.0.0
- **プロジェクト**: ATFT-GAT-FAN v2.0.0
- **作成日**: 2025-10-18

---

これらのスキルを活用して、ATFT-GAT-FAN プロジェクトの開発を加速させましょう！
