# 包括的Macro特徴量欠落予防策（2025-11-15版）

**目的**: yfinance欠落により40列のmacro特徴量が欠落する問題を完全に防止
**参照**: ユーザーフィードバックに基づく6層防御戦略

---

## 実装済み

### ✅ 1. 依存関係の明示的管理
- **ファイル**: `gogooku5/data/pyproject.toml:23`
- **内容**: `yfinance>=0.2.0`を依存関係に追加
- **効果**: 新規環境で自動インストール

### ✅ 2. 依存関係検証スクリプト
- **ファイル**: `gogooku5/data/scripts/validate_dependencies.py`
- **使用法**:
  ```bash
  python gogooku5/data/scripts/validate_dependencies.py
  python gogooku5/data/scripts/validate_dependencies.py --strict
  ```
- **出力**: 欠落依存関係のimpactを明示

### ✅ 3. Macroキャッシュウォーマー（Safeguard 4 先行実装）
- **ファイル**: `gogooku5/data/scripts/warm_macro_cache.py`
- **機能**:
  1. yfinance可用性チェック
  2. 2日間窓での検証モード (`--validate`)
  3. VIX + VVMD全データプリフェッチ
  4. ヘルスマーカー保存 (`output/cache/macro/vix_health.json`)
  5. 詳細exit code (0=成功, 1=依存関係, 2=ネットワーク, 3=空データ, 4=キャッシュエラー)

**使用法**:
```bash
# クイック検証（ビルド前必須）
python gogooku5/data/scripts/warm_macro_cache.py --validate

# 全期間プリフェッチ
python gogooku5/data/scripts/warm_macro_cache.py --start 2020-01-01 --end 2025-12-31

# 強制リフレッシュ
python gogooku5/data/scripts/warm_macro_cache.py --start 2020-01-01 --end 2025-12-31 --force-refresh
```

---

## 実装推奨（優先順位順）

### 🔴 Safeguard 1: Makefileラッパーに自動依存関係チェック

**実装場所**: `gogooku5/data/scripts/build_chunks.py` 先頭

**コード例**:
```python
# build_chunks.py の先頭に追加（imports前）
import sys
import subprocess
from pathlib import Path

def preflight_dependency_check():
    """Verify yfinance is installed before build."""
    print("="  * 80)
    print("🔍 Preflight Dependency Check")
    print("=" * 80)

    # Check yfinance
    try:
        import yfinance
        print(f"✅ yfinance v{yfinance.__version__} detected")
    except ImportError:
        print("❌ FATAL: yfinance not installed")
        print("   Install with: pip install yfinance")
        print("   Or: pip install -e gogooku5/data")
        print("\n⚠️  Build ABORTED: Missing dependency will cause 40 column drop")
        sys.exit(1)

    # Warm macro cache (2-day validation)
    print("\n🔥 Warming macro cache (validation mode)...")
    script_path = Path(__file__).parent / "warm_macro_cache.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--validate"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ FATAL: Macro cache warming failed")
        print(result.stdout)
        print(result.stderr)
        print("\n⚠️  Build ABORTED: Macro features unavailable (will drop 40 columns)")
        sys.exit(1)

    print("✅ Preflight checks passed - build can proceed\n")

# Run before any imports
preflight_dependency_check()
```

**効果**:
- ビルド開始前にyfinance検証
- 2日間窓でmacro API到達性確認
- 問題検出時に即座にabort（数時間後のschema mismatch防止）

---

### 🔴 Safeguard 2: スキーマゲート（macro列数検証）

**実装場所**: `gogooku5/data/scripts/build_chunks.py` チャンク保存後

**コード例**:
```python
def validate_macro_columns(df: pl.DataFrame, chunk_id: str):
    """Assert expected macro column count after chunk build."""
    macro_cols = [c for c in df.columns if c.startswith("macro_")]
    expected_macro_count = 40  # 10 VIX + 30 VVMD

    if len(macro_cols) < expected_macro_count:
        print(f"\n❌ FATAL: Macro column count mismatch in {chunk_id}")
        print(f"   Expected: {expected_macro_count} macro_* columns")
        print(f"   Actual:   {len(macro_cols)} columns")
        print(f"   Missing:  {expected_macro_count - len(macro_cols)} columns")

        # List missing features
        all_macro_features = set([
            # VIX features (10)
            "macro_vix_close", "macro_vix_log_close",
            "macro_vix_ret_1d", "macro_vix_ret_5d", "macro_vix_ret_10d", "macro_vix_ret_20d",
            "macro_vix_sma_ratio_5_20", "macro_vix_spike",
            "macro_vix_vol_20", "macro_vix_vol_z",
            # VVMD features (30) - list key ones for debugging
            "macro_vvmd_vol_spy_rv20", "macro_vvmd_vol_spy_drv_20_63",
            "macro_vvmd_vlm_spy_surge20", "macro_vvmd_mmt_spy_mom_20",
            # ... add all 40 here
        ])
        missing_features = all_macro_features - set(macro_cols)
        if missing_features:
            print(f"\n   Missing features (first 10):")
            for feat in sorted(missing_features)[:10]:
                print(f"      - {feat}")

        print("\n💡 Troubleshooting:")
        print("   1. Run: python gogooku5/data/scripts/warm_macro_cache.py --validate")
        print("   2. Check yfinance: python -c 'import yfinance'")
        print("   3. Verify network: curl https://finance.yahoo.com")

        raise RuntimeError(f"Macro column validation failed for {chunk_id}")

    print(f"✅ Macro columns validated: {len(macro_cols)}/{expected_macro_count}")


# チャンク保存後に呼び出し
chunk_df = build_chunk(...)  # existing code
save_chunk(chunk_df, chunk_path)  # existing code

# 🆕 Add this line
validate_macro_columns(chunk_df, chunk_id=f"{year}Q{quarter}")
```

**効果**:
- チャンク保存直後にmacro列数を検証
- 40列未満の場合は即座にRuntimeError
- 数時間後のvalidation failureを防止

---

### 🟡 Safeguard 3: check_chunk_status.pyに拡張検証

**実装場所**: `gogooku5/data/scripts/check_chunk_status.py`

**追加コード**:
```python
def validate_macro_features(chunk_path: Path) -> dict:
    """Extended validation: macro column count."""
    df = pl.read_parquet(chunk_path)
    macro_cols = [c for c in df.columns if c.startswith("macro_")]

    return {
        "macro_column_count": len(macro_cols),
        "expected": 40,
        "status": "ok" if len(macro_cols) >= 40 else "macro_feature_missing",
        "missing_count": max(0, 40 - len(macro_cols))
    }

# メインループに追加
for chunk_dir in chunk_dirs:
    ...
    # Existing validation
    status = validate_chunk(...)

    # 🆕 Add macro validation
    if status == "completed":
        macro_status = validate_macro_features(chunk_path)
        if macro_status["status"] != "ok":
            status = "failed_macro_missing"
            errors.append(f"Missing {macro_status['missing_count']} macro columns")
```

**効果**:
- `python scripts/check_chunk_status.py`で自動macro検証
- Schema hash不一致の前にmacro欠落を検出

---

### 🟡 Safeguard 4: ログレベルをERRORに変更

**実装場所**: `gogooku5/data/src/builder/features/macro/vix.py:43-46`

**現状**:
```python
if yf is None:
    LOGGER.warning("yfinance not available; VIX history unavailable")
    return pl.DataFrame()
```

**変更後**:
```python
if yf is None:
    LOGGER.error("yfinance not available; VIX history unavailable")
    LOGGER.error("   Install with: pip install yfinance")
    LOGGER.error("   Build will produce 2727 columns (missing 10 VIX features)")
    # Still return empty to allow graceful degradation in dev
    # But ERROR level ensures monitoring dashboards catch this
    return pl.DataFrame()
```

**同様の変更**: `global_regime.py:86-89`

**効果**:
- WARNINGをERRORに昇格
- Monitoring dashboardで即座に検出可能
- ログ解析でmacro欠落を早期発見

---

### 🟡 Safeguard 5: Macro列数モニタリング

**実装場所**: `gogooku5/data/scripts/build_chunks.py` 最後

**追加コード**:
```python
def save_macro_status_report(chunks_info: list, output_path: Path):
    """Save macro column counts for monitoring."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "chunks": {}
    }

    for chunk in chunks_info:
        report["chunks"][chunk["id"]] = {
            "macro_column_count": chunk["macro_cols"],
            "expected": 40,
            "status": "ok" if chunk["macro_cols"] >= 40 else "missing"
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📊 Macro status report saved: {output_path}")

# ビルド完了後
save_macro_status_report(
    chunks_info=all_chunks,
    output_path=Path("_logs/macro_feature_status.json")
)
```

**監視アラート**:
```bash
# cron job (毎時実行)
*/60 * * * * python -c "
import json
with open('/workspace/gogooku3/_logs/macro_feature_status.json') as f:
    data = json.load(f)
    for chunk_id, info in data['chunks'].items():
        if info['status'] != 'ok':
            print(f'ALERT: {chunk_id} missing {40 - info[\"macro_column_count\"]} macro columns')
"
```

**効果**:
- macro列数を常時追跡
- Dashboardで可視化可能
- 偏差検出時に即座にアラート

---

### 🟢 Safeguard 6: Unit/Integration Test

**実装場所**: `gogooku5/data/tests/integration/test_macro_features.py` (新規)

**テストコード**:
```python
import pytest
import polars as pl
from builder.pipelines.dataset_builder import build_chunk

def test_macro_features_generated():
    """Assert macro_vix_* and macro_vvmd_* columns exist in 3-day chunk."""
    # Build minimal chunk
    chunk_df = build_chunk(
        start_date="2025-11-13",
        end_date="2025-11-15"
    )

    # Assert macro columns
    macro_cols = [c for c in chunk_df.columns if c.startswith("macro_")]

    assert len(macro_cols) >= 40, (
        f"Expected 40+ macro columns, got {len(macro_cols)}. "
        f"Check yfinance installation."
    )

    # Assert specific key features
    assert "macro_vix_close" in chunk_df.columns, "VIX close missing"
    assert "macro_vvmd_vol_spy_rv20" in chunk_df.columns, "VVMD SPY vol missing"

    print(f"✅ Test passed: {len(macro_cols)} macro features generated")


def test_yfinance_importable():
    """Fail fast if yfinance missing."""
    try:
        import yfinance
    except ImportError:
        pytest.fail("yfinance not installed - run: pip install yfinance")
```

**CI統合** (`.github/workflows/test.yml`):
```yaml
name: Dataset Build Tests
on: [push, pull_request]

jobs:
  test-macro-features:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -e gogooku5/data
      - name: Test macro features
        run: pytest gogooku5/data/tests/integration/test_macro_features.py -v
```

**効果**:
- PR時に自動テスト
- yfinance欠落を即座に検出
- 本番デプロイ前に防止

---

## 実装優先順位

| Safeguard | 優先度 | 実装時間 | 効果 | 状態 |
|-----------|--------|----------|------|------|
| 1. Makefileラッパー依存チェック | 🔴 最高 | 30分 | 即座abort | 実装推奨 |
| 2. スキーマゲート（macro列数） | 🔴 最高 | 20分 | ビルド直後検出 | 実装推奨 |
| 3. check_chunk_status拡張 | 🟡 高 | 15分 | 検証強化 | 実装推奨 |
| 4. ログレベルERROR化 | 🟡 中 | 5分 | Dashboard可視化 | 実装推奨 |
| 5. Macro列数モニタリング | 🟡 中 | 30分 | 継続監視 | 実装推奨 |
| 6. Unit/Integration Test | 🟢 低 | 45分 | CI/CD統合 | 任意 |
| ✅ warm_macro_cache.py | ✅ 完了 | - | プリフェッチ検証 | **実装済み** |
| ✅ validate_dependencies.py | ✅ 完了 | - | 依存関係検証 | **実装済み** |
| ✅ pyproject.toml更新 | ✅ 完了 | - | 自動インストール | **実装済み** |

---

## クイックスタートガイド

### 新規環境セットアップ

```bash
# 1. 依存関係インストール
cd gogooku5/data
pip install -e .

# 2. 依存関係検証
python scripts/validate_dependencies.py

# 3. Macroキャッシュウォーム（必須！）
python scripts/warm_macro_cache.py --validate

# 4. 全期間プリフェッチ（推奨）
python scripts/warm_macro_cache.py --start 2020-01-01 --end 2025-12-31

# 5. テストビルド
python scripts/build_chunks.py --start 2025-01-01 --end 2025-03-31

# 6. 検証
python scripts/check_chunk_status.py
```

### 本番ビルド前チェックリスト

- [ ] `pip install -e gogooku5/data` 実行
- [ ] `python scripts/validate_dependencies.py` → ALL PASS
- [ ] `python scripts/warm_macro_cache.py --validate` → exit 0
- [ ] `python scripts/warm_macro_cache.py --start YYYY-MM-DD --end YYYY-MM-DD` → 全期間キャッシュ
- [ ] テストチャンク（1四半期）をビルド
- [ ] `python scripts/check_chunk_status.py` → "completed" & 2767列
- [ ] Macro列数: `ls output_g5/chunks/*/ml_dataset.parquet | xargs -I {} python -c "import polars as pl; df=pl.read_parquet('{}'); print(len([c for c in df.columns if c.startswith('macro_')]))"`

---

## トラブルシューティング

### Q: warm_macro_cache.py が exit 1 (yfinance not available)

**解決**:
```bash
pip install yfinance
python -c "import yfinance; print(yfinance.__version__)"
```

### Q: warm_macro_cache.py が exit 2 (Network error)

**解決**:
```bash
# Yahoo Finance APIの到達性確認
curl -I https://finance.yahoo.com

# プロキシ設定確認
echo $HTTP_PROXY
echo $HTTPS_PROXY

# タイムアウト増加
# vix.py, global_regime.py内のyf.download(..., timeout=30) → timeout=60
```

### Q: warm_macro_cache.py が exit 3 (Empty data)

**可能性**:
1. 日付範囲が未来すぎる（Yahoo Financeにデータなし）
2. APIレート制限
3. 株式市場休日

**解決**:
```bash
# 既知の良い期間で再テスト
python scripts/warm_macro_cache.py --start 2024-01-01 --end 2024-12-31
```

### Q: ビルド後にmacro列が36列しかない（4列足りない）

**調査**:
```bash
# ログで欠落feature特定
grep "WARNING.*macro" _logs/chunk_*.log

# 手動でVIX取得テスト
python -c "
import yfinance as yf
vix = yf.download('^VIX', start='2024-01-01', end='2024-12-31')
print(vix.head())
print(f'Rows: {len(vix)}')
"
```

---

## まとめ

### 完全な予防には

1. **✅ 実装済み**:
   - pyproject.toml更新
   - validate_dependencies.py
   - warm_macro_cache.py

2. **🔴 最優先実装**:
   - Safeguard 1: build_chunks.py preflight check
   - Safeguard 2: validate_macro_columns() gate

3. **🟡 推奨実装**:
   - Safeguard 3: check_chunk_status.py拡張
   - Safeguard 4: ログレベルERROR化
   - Safeguard 5: Macro列数モニタリング

4. **🟢 任意**:
   - Safeguard 6: Unit/Integration Test

### 効果予測

| 予防策 | 検出タイミング | 時間節約 | 自動化 |
|--------|----------------|----------|--------|
| pyproject.toml | 環境構築時 | - | ✅ pip |
| validate_dependencies.py | ビルド前 | 0秒 | ⚠️ 手動 |
| warm_macro_cache.py | ビルド前 | 0秒 | ⚠️ 手動 |
| Safeguard 1 (preflight) | ビルド開始時 | **数秒** | ✅ 自動 |
| Safeguard 2 (schema gate) | チャンク保存時 | **数時間** | ✅ 自動 |
| Safeguard 3 (check拡張) | 検証時 | **即座** | ✅ 自動 |

**合計削減時間**: チャンク再ビルド回避により **数十時間～数日** 節約可能

---

**文責**: gogooku5 migration team
**最終更新**: 2025-11-15
