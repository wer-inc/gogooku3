# AI CLI Tools - Autonomous Mode

このディレクトリには、AI CLIツール（Claude Code / Codex）を**完全自律モード**で起動し、プロジェクトを自動改善するためのツールが含まれています。

## 🎯 主な機能

1. **自動診断**: 起動時にプロジェクトの健全性を自動チェック
2. **自律修正**: 検出された問題を完全自動で修正
3. **継続監視**: 定期的な監視と自動修復（オプション）
4. **プロアクティブ最適化**: 問題がなくても改善の機会を発見

## 📋 ツール一覧

| ファイル | 説明 | 自律度 |
|---------|------|--------|
| **claude-code.sh** | Claude Code（高速・実用的）| ★★★★★ 完全自律 |
| **codex.sh** | Codex（深い推論）| ★★★★★ 完全自律 |
| **autonomous-monitor.sh** | 継続的な監視とメンテナンス | ★★★★★ 完全自律 |
| **project-health-check.sh** | プロジェクト診断（手動）| ★☆☆☆☆ 診断のみ |
| **brave-search-demo.sh** | Web検索デモ | ☆☆☆☆☆ デモのみ |

## 🚀 基本的な使い方

### Claude Code（推奨 - 高速実用的）

```bash
# 完全自律モード：診断 → 問題検出 → 自動修正
./tools/claude-code.sh

# ヘルスチェックをスキップして直接起動
./tools/claude-code.sh --no-check

# 通常の対話モード（自律機能なし）
./tools/claude-code.sh --interactive

# 特定のタスクを指示
./tools/claude-code.sh "Optimize the training pipeline for A100 GPU"
```

**自動設定:**
- ✅ **完全自律**: すべての操作を自動承認（`--permission-mode bypassPermissions`）
- ✅ **自動診断**: 起動時に健全性チェック実行
- ✅ **プロジェクト認識**: ATFT-GAT-FANプロジェクト固有のコンテキスト
- ✅ **MCPサーバー**: playwright, filesystem, git, brave-search
- ✅ **タスク追跡**: TodoWrite自動使用

### Codex（深い推論が必要な場合）

```bash
# 完全自律モード：診断 → 深い分析 → 最適解を実装（非対話）
./tools/codex.sh

# ヘルスチェックをスキップ（非対話）
./tools/codex.sh --no-check

# 複雑な問題を解決（非対話）
./tools/codex.sh "Redesign the attention mechanism for better long-term predictions"
```

**自動設定:**
- ✅ **非対話モード**: `codex exec`で自律実行（プロンプトがある場合）
- ✅ **深い推論**: o1モデルの推論能力を活用
- ✅ **完全自律**: Web検索 + フルオート（`--search --full-auto`）
- ✅ **自動診断**: 起動時に健全性チェック実行
- ✅ **プロジェクト認識**: システムコンテキスト自動付与
- ✅ **MCPサーバー**: playwright, filesystem, git, brave-search

**注意**: 引数なしで実行すると対話モード、プロンプトありで実行すると非対話モード（`codex exec`）になります。

### 継続的自律監視

```bash
# 一度だけチェック実行
./tools/autonomous-monitor.sh

# 問題を見つけたら即座に自動修正
./tools/autonomous-monitor.sh --fix-now

# 1時間ごとに監視（デーモンモード）
./tools/autonomous-monitor.sh --daemon

# 5分ごとに監視（watch モード）
./tools/autonomous-monitor.sh --watch
```

**cron設定例（毎時0分に自動チェック＆修正）:**
```bash
# crontab -e で以下を追加
0 * * * * cd /workspace/gogooku3 && ./tools/autonomous-monitor.sh --fix-now >> _logs/autonomous-monitor/cron.log 2>&1
```

## 🎭 自律モードの動作フロー

### 1. 起動時の自動診断

```
./tools/claude-code.sh を実行
    ↓
🔍 プロジェクト健全性チェック実行
    ↓
📊 診断結果を分析（8カテゴリ）
    ↓
🤖 問題の重大度に応じて自動プロンプト生成
    ↓
🚀 Claude起動 + 自律的に修正開始
```

### 2. 問題の優先度付け

| 優先度 | 状態 | AIの動作 |
|-------|------|---------|
| **P0（Critical）** | 重大な問題（例：パッケージ未インストール） | 即座に修正開始 |
| **P1（High）** | 警告（例：キャッシュ未設定） | 調査して修正 |
| **P2（Medium）** | 推奨事項（例：最適化の機会） | プロアクティブに改善 |
| **Healthy** | 問題なし | 技術的負債や最適化を探索 |

### 3. 自律的な作業フロー

AIは以下の手順で自律的に作業します：

1. **TodoWrite**でタスクリスト作成（必須）
2. **Read**でファイルを読んで現状を理解
3. **Edit/Write**で問題を修正
4. **Bash**でテストや検証を実行
5. **健全性チェック**で修正を確認
6. **完了報告**

## 📊 健全性チェックの詳細

### チェック項目（8カテゴリ）

1. **環境設定**: `.env`, API認証情報, キャッシュ設定
2. **依存関係**: Python, gogooku3パッケージ, GPU
3. **データパイプライン**: データセット, 価格キャッシュ
4. **トレーニング**: 実行中のプロセス, ログ, モデル
5. **コード品質**: TODO/FIXME, pre-commit, Git状態
6. **パフォーマンス**: DataLoader, torch.compile, RankIC
7. **ディスク容量**: 使用率
8. **設定ファイル**: Hydra configs

### 単独実行

```bash
./tools/project-health-check.sh
```

**出力:**
- ターミナル: カラフルな視覚的レポート
- JSON: `_logs/health-checks/health-check-YYYYMMDD-HHMMSS.json`

**終了コード:**
- `0`: すべて正常
- `1`: 警告あり
- `2`: 重大な問題あり

## 🎯 使い分けガイド

| シーン | 推奨ツール | 理由 |
|-------|----------|------|
| 毎日のメンテナンス | `claude-code.sh` | 高速で実用的 |
| 複雑なアーキテクチャ変更 | `codex.sh` | 深い推論が可能 |
| バグ修正 | `claude-code.sh` | 即座に対応 |
| パフォーマンス最適化 | `codex.sh` | 非自明な改善を発見 |
| 継続的な監視 | `autonomous-monitor.sh --daemon` | 常時監視 |
| 緊急対応 | `autonomous-monitor.sh --fix-now` | 即座に修正 |

## 🔧 高度な使用例

### 1. プロジェクトの完全自動メンテナンス

```bash
# 朝: Claude Codeで高速メンテナンス（5-10分）
./tools/claude-code.sh

# 問題がなければ、Codexで深い最適化（30-60分）
./tools/codex.sh
```

### 2. CI/CD統合

```bash
# GitLab CI / GitHub Actions
script:
  - ./tools/project-health-check.sh
  - |
    if [ $? -eq 2 ]; then
      # Critical issues detected
      ./tools/autonomous-monitor.sh --fix-now
    fi
```

### 3. 定期的な自動改善（cron）

```bash
# 毎日午前2時に自動最適化
0 2 * * * cd /workspace/gogooku3 && ./tools/codex.sh >> _logs/daily-optimization.log 2>&1

# 毎時間チェック＆修正
0 * * * * cd /workspace/gogooku3 && ./tools/autonomous-monitor.sh --fix-now >> _logs/hourly-check.log 2>&1
```

### 4. トレーニング失敗時の自動対応

```bash
# トレーニングスクリプトの最後に追加
make train || {
    echo "Training failed - triggering autonomous debugging"
    ./tools/claude-code.sh "Training failed. Analyze logs in _logs/training/, diagnose the issue, and fix it."
}
```

## 🔒 セキュリティと制限

### 安全性

- ✅ ファイルシステムアクセスは `/workspace/gogooku3` に制限
- ✅ MCPサーバーもワークスペーススコープ
- ✅ Git操作はプロジェクトリポジトリのみ
- ⚠️ **完全自律モード**: すべての操作が自動承認される（危険な操作も含む）

### 推奨される安全対策

1. **バックアップ**: 重要な変更前にGitコミット
2. **レビュー**: 自律修正後は変更内容を確認
3. **テスト環境**: 本番環境で初めて使う前にテスト
4. **監視**: `_logs/autonomous-monitor/` を定期確認

## 📦 MCP サーバー（自動設定）

両方のスクリプトは `.mcp.json` を自動生成し、以下を有効化：

1. **playwright**: Webブラウザ自動化、スクレイピング
2. **filesystem**: ファイルシステムアクセス（ワークスペーススコープ）
3. **git**: Git操作（コミット、ブランチ、ログ等）
4. **brave-search**: Web検索API（最新情報の取得）

## 📝 ログとトラブルシューティング

### ログファイル

```bash
# 健全性チェック履歴
_logs/health-checks/health-check-*.json

# 自律監視ログ
_logs/autonomous-monitor/auto-fix-*.log
_logs/autonomous-monitor/alerts.log

# トレーニングログ
_logs/training/train_*.log
```

### トラブルシューティング

**問題**: AIが起動しない

```bash
# 依存関係を確認
which claude    # Claude Codeがインストールされているか
which codex     # Codexがインストールされているか
which jq        # jqがインストールされているか（health check用）
```

**問題**: 自律修正が動作しない

```bash
# ヘルスチェックを手動実行して確認
./tools/project-health-check.sh

# 最新のレポートを確認
cat _logs/health-checks/health-check-*.json | jq .
```

**問題**: 権限エラー

```bash
# スクリプトを実行可能にする
chmod +x tools/*.sh
```

## 🌟 ベストプラクティス

1. **毎日の習慣**:
   ```bash
   # 朝一番で実行
   ./tools/claude-code.sh
   ```

2. **大きな変更前**:
   ```bash
   # Gitコミットしてから実行
   git add . && git commit -m "Before autonomous fixes"
   ./tools/claude-code.sh
   ```

3. **週次の深い最適化**:
   ```bash
   # 週末にCodexで深い分析
   ./tools/codex.sh
   ```

4. **常時監視**:
   ```bash
   # cronで継続的監視を設定
   crontab -e
   # 追加: 0 * * * * cd /workspace/gogooku3 && ./tools/autonomous-monitor.sh --fix-now
   ```

## 📚 その他のドキュメント

- [プロジェクト全体のドキュメント](../CLAUDE.md)
- [Brave Search使用例](brave-search-demo.sh)
- [健全性チェック詳細](project-health-check.sh)

---

**🤖 完全自律化されたAI開発環境 - あなたのプロジェクトは24/7自己改善します**
