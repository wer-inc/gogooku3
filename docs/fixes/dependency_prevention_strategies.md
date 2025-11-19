# 依存関係欠落問題の予防策

**作成日**: 2025-11-15
**背景**: yfinance未インストールにより40列の特徴量が欠落
**対策**: 今後同様の問題を防ぐための5つの予防策

---

## 📋 実施済み対策

### ✅ 1. 依存関係の明示的な管理

**実施内容**:
- `gogooku5/data/pyproject.toml`に`yfinance>=0.2.0`を追加済み
- 今後の環境構築時に自動インストールされる

**ファイル**: `gogooku5/data/pyproject.toml:23`

```toml
dependencies = [
    "polars>=0.20.0",
    "pyarrow>=12.0.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.0.3",
    "python-dotenv>=1.0.0",
    "requests>=2.31.0",
    "numpy>=1.26.0",
    "aiohttp>=3.9.0",
    "nest_asyncio>=1.6.0",
    "yfinance>=0.2.0",  # 🆕 追加
]
```

**インストール方法**:
```bash
# 開発環境セットアップ時
pip install -e gogooku5/data

# または全依存関係を明示的に
pip install polars pyarrow pydantic requests numpy aiohttp nest_asyncio yfinance
```

---

### ✅ 2. ビルド前依存関係チェックスクリプト

**実施内容**:
- `gogooku5/data/scripts/validate_dependencies.py`を作成
- 必須およびオプション依存関係を自動チェック
- 欠落時にimpactを明示的に表示

**使用方法**:
```bash
# 基本チェック（オプション依存関係は警告のみ）
python gogooku5/data/scripts/validate_dependencies.py

# Strictモード（オプション依存関係も必須扱い）
python gogooku5/data/scripts/validate_dependencies.py --strict
```

**出力例**:
```
================================================================================
🔍 Dependency Validation Check
================================================================================

✅ polars               v1.35.2          (Core DataFrame operations)
✅ pyarrow              v22.0.0          (Parquet I/O)
✅ yfinance             v0.2.66          (40 macro/VIX features)
...

================================================================================
📊 Summary
================================================================================
Required:  6 passed, 0 failed
Optional:  1 passed, 0 failed

✅ ALL DEPENDENCIES VALIDATED
```

**exitコード**:
- `0`: すべて正常
- `1`: 必須依存関係欠落
- `2`: Strictモード + オプション依存関係欠落

---

## 🔄 推奨運用フロー

### パターン A: 新規環境セットアップ時

```bash
# 1. リポジトリクローン
git clone <repo>
cd gogooku3

# 2. 依存関係インストール
pip install -e gogooku5/data

# 3. 依存関係検証
python gogooku5/data/scripts/validate_dependencies.py

# 4. ビルド実行
python gogooku5/data/scripts/build_chunks.py --start 2025-01-01 --end 2025-12-31
```

### パターン B: 既存環境での定期チェック

```bash
# 毎週月曜日に依存関係をチェック（cron例）
0 9 * * 1 cd /workspace/gogooku3 && python gogooku5/data/scripts/validate_dependencies.py >> /var/log/dependency_check.log 2>&1
```

### パターン C: CI/CD統合（将来）

```yaml
# .github/workflows/build_dataset.yml (例)
name: Dataset Build
on: [push, pull_request]

jobs:
  validate_and_build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -e gogooku5/data
      - name: Validate dependencies
        run: python gogooku5/data/scripts/validate_dependencies.py --strict
      - name: Build dataset
        run: python gogooku5/data/scripts/build_chunks.py ...
```

---

## 🛡️ 追加予防策（実施推奨）

### 3. build_chunks.pyに事前チェックを統合

**提案内容**:
`build_chunks.py`の先頭に依存関係チェックを組み込む

```python
# gogooku5/data/scripts/build_chunks.py (提案)

import sys
from pathlib import Path

# 依存関係チェックを最初に実行
validator_path = Path(__file__).parent / "validate_dependencies.py"
result = subprocess.run([sys.executable, str(validator_path)], capture_output=True)
if result.returncode != 0:
    print("❌ 依存関係チェック失敗。ビルドを中止します。")
    print(result.stdout.decode())
    sys.exit(1)

# 通常のビルド処理
...
```

**効果**:
- ビルド実行前に自動チェック
- 欠落時に即座にエラー終了（数時間後のスキーマミスマッチを防ぐ）

---

### 4. requirements.txtスナップショット（推奨）

**提案内容**:
動作確認済みの環境をrequirements.txtで固定

```bash
# 動作確認後にスナップショット作成
pip freeze > gogooku5/data/requirements_frozen_20251115.txt

# 新環境でスナップショットから復元
pip install -r gogooku5/data/requirements_frozen_20251115.txt
```

**メリット**:
- バージョン違いによる互換性問題を防止
- 再現性の高い環境構築

**デメリット**:
- メンテナンスコスト増加
- セキュリティパッチ適用が遅れる可能性

**推奨**:
- プロダクション環境向けには有効
- 開発環境ではpyproject.tomlの範囲指定を使用

---

### 5. ビルド完了後の自動検証

**提案内容**:
ビルド完了時にスキーマハッシュを自動検証

```python
# build_chunks.py の最後に追加（提案）

# チャンク保存後
chunk_path = output_dir / f"{quarter}/ml_dataset.parquet"
chunk_df = pl.read_parquet(chunk_path)

# スキーマ検証
expected_hash = manifest["schema_hash"]
actual_hash = compute_schema_hash(chunk_df.schema)

if actual_hash != expected_hash:
    print(f"⚠️  スキーマハッシュ不一致検出！")
    print(f"   期待: {expected_hash}")
    print(f"   実際: {actual_hash}")
    print(f"   列数差: {len(manifest['columns']) - len(chunk_df.columns)} columns")

    # 欠落列を即座に表示
    missing_cols = set(manifest['columns']) - set(chunk_df.columns)
    if missing_cols:
        print(f"   欠落列 ({len(missing_cols)}):")
        for col in sorted(missing_cols)[:10]:  # 最初の10列のみ表示
            print(f"      - {col}")
```

**効果**:
- ビルド直後に問題検出（数時間の無駄を防ぐ）
- 欠落列を即座に特定

---

## 📊 対策の優先度

| 対策 | 状態 | 優先度 | コスト | 効果 | 推奨度 |
|------|------|--------|--------|------|--------|
| 1. pyproject.toml更新 | ✅ 完了 | 最高 | 低 | 高 | ⭐⭐⭐⭐⭐ |
| 2. 依存関係チェックスクリプト | ✅ 完了 | 高 | 低 | 高 | ⭐⭐⭐⭐⭐ |
| 3. build_chunks.pyに統合 | 提案 | 中 | 低 | 中 | ⭐⭐⭐⭐ |
| 4. requirements.txt固定 | 提案 | 低 | 中 | 中 | ⭐⭐⭐ |
| 5. ビルド後自動検証 | 提案 | 高 | 中 | 高 | ⭐⭐⭐⭐ |

---

## ✅ チェックリスト（新環境構築時）

環境セットアップ時に以下を確認:

- [ ] `pip install -e gogooku5/data`を実行
- [ ] `python gogooku5/data/scripts/validate_dependencies.py`でyfinanceを確認
- [ ] テストビルド（1四半期）を実行
- [ ] `scripts/check_chunk_status.py`でスキーマハッシュを確認
- [ ] 2767列であることを確認（2727列の場合はyfinance欠落）

---

## 📚 関連ドキュメント

- **根本原因分析**: `docs/fixes/gogooku5_missing_macro_columns_20251115.md`
- **スキーママニフェスト**: `gogooku5/data/schema/feature_schema_manifest.json`
- **ビルドスクリプト**: `gogooku5/data/scripts/build_chunks.py`
- **依存関係定義**: `gogooku5/data/pyproject.toml`

---

## 🔍 今回の教訓

### 問題点
1. **Silent failure**: yfinance欠落時にエラーが出ず、空のDataFrameを返す設計
2. **遅延検出**: ビルド完了後（数時間後）にschema validationで初めて検出
3. **不明確な依存関係**: pyproject.tomlにyfinanceが未記載

### 改善点
1. **明示的な依存関係**: すべてpyproject.tomlに記載
2. **早期検出**: ビルド前に依存関係チェック
3. **即座のフィードバック**: 欠落時に具体的なimpactを表示

### 設計変更検討（将来）
```python
# 現状（graceful degradation）
yf = get_yfinance_module(raise_on_missing=False)
if yf is None:
    LOGGER.warning("yfinance not available")
    return pl.DataFrame()  # Silent failure

# 提案（fail-fast）
yf = get_yfinance_module(raise_on_missing=True)  # Explicit failure
```

**メリット**: 問題を即座に検出
**デメリット**: 柔軟性が低下
**推奨**: オプション依存関係は現状維持、ビルド前チェックで対応

---

**まとめ**: 対策1,2は完了。対策3,5を実装することで完全な予防が可能。
