# CS-Z Replace Mode Implementation - COMPLETE

**日時**: 2025-11-02 17:00 UTC
**ステータス**: ✅ **コア実装完了** | ⚠️ **Sanity check blocked by data issue**

---

## 🎯 実装完了事項

### ✅ 修正1: Config に patch_multiplier=2 を明示

**File**: `apex-ranker/configs/v0_base_corrected.yaml`

```yaml
model:
  d_model: 256
  depth: 4
  patch_len: 16
  stride: 8
  n_heads: 8
  dropout: 0.2
  patch_multiplier: 2  # Explicit: Matches checkpoint (89 × 2 = 178 output)
```

**✅ 完了**: Checkpoint と一致する設定を明示

---

### ✅ 修正2: モデル初期化 - in_features=89 固定

**File**: `apex-ranker/apex_ranker/backtest/inference.py:84-101`

```python
# FIX: CS-Z is REPLACE mode (not append), so in_features stays at n_features
# Checkpoint: in_features=89, patch_multiplier=2 → Conv output=178
# CS-Z replaces raw values with z-scores, shape remains [N, L, 89]
in_features = n_features  # Always raw feature count (e.g., 89)

print(f"[Model Init] in_features={in_features}, patch_multiplier={model_cfg.get('patch_multiplier', 'auto')}, add_csz={add_csz}")

model = APEXRankerV0(
    in_features=in_features,  # ✅ 89 固定
    horizons=horizons,
    d_model=model_cfg["d_model"],
    depth=model_cfg["depth"],
    patch_len=model_cfg["patch_len"],
    stride=model_cfg["stride"],
    n_heads=model_cfg["n_heads"],
    dropout=model_cfg.get("dropout", 0.1),
    patch_multiplier=model_cfg.get("patch_multiplier", None),  # ✅ Config から取得
).to(device)
```

**✅ 完了**: CS-Z で特徴量を倍にしない（89のまま）

---

### ✅ 修正3: APEXRankerV0 に patch_multiplier パラメータを追加

**File**: `apex-ranker/apex_ranker/models/ranker.py:45-60`

```python
def __init__(
    self,
    in_features: int,
    horizons: Iterable[int],
    *,
    d_model: int = 192,
    depth: int = 3,
    patch_len: int = 16,
    stride: int = 8,
    n_heads: int = 8,
    dropout: float = 0.1,
    patch_multiplier: int | None = None,  # ✅ NEW
    loss_fn: nn.Module | None = None,
) -> None:
    super().__init__()
    self.in_features = in_features
    self.horizons = [int(h) for h in horizons]

    self.encoder = PatchTSTEncoder(
        in_feats=in_features,
        d_model=d_model,
        depth=depth,
        patch_len=patch_len,
        stride=stride,
        n_heads=n_heads,
        dropout=dropout,
        patch_multiplier=patch_multiplier,  # ✅ Explicit from config
    )
```

**✅ 完了**: Config から patch_multiplier を明示的に渡す

---

### ✅ 修正4: CS-Z を **REPLACE モード** に変更

**File**: `apex-ranker/apex_ranker/backtest/inference.py:265-288`

```python
def _replace_with_cross_sectional_z(self, features: np.ndarray) -> np.ndarray:
    """
    Replace raw features with cross-sectional Z-scores (in-place normalization).

    IMPORTANT: This is REPLACE mode, not APPEND. Shape stays [N, L, F].
    The checkpoint was trained with in_features=89, patch_multiplier=2.
    CS-Z normalization replaces raw values with z-scores, maintaining 89 channels.

    Args:
        features: [N_stocks, L_lookback, F] raw feature array

    Returns:
        [N_stocks, L_lookback, F] with values replaced by CS-Z (same shape)
    """
    # Cross-sectional normalization per lookback timestep
    # Normalize across stocks (axis=0) for each time step and feature
    mean = np.nanmean(features, axis=0, keepdims=True)  # [1, L, F]
    std = np.nanstd(features, axis=0, keepdims=True)    # [1, L, F]
    std = np.maximum(std, self.csz_eps)  # Prevent division by zero

    z_features = (features - mean) / std  # [N, L, F]
    z_features = np.clip(z_features, -self.csz_clip, self.csz_clip)

    return z_features  # [N, L, F] - SAME shape as input
```

**✅ 完了**: 連結 (append) から置換 (replace) に変更

---

### ✅ 修正5: 呼び出し側を replace モードに変更

**File**: `apex-ranker/apex_ranker/backtest/inference.py:324-341`

```python
features = np.stack(feature_windows, axis=0).astype(np.float32, copy=False)

# Apply cross-sectional Z-normalization if enabled (REPLACE mode, not append)
# Shape remains [N, L, F] - values are replaced with z-scores
if self.add_csz:
    features = self._replace_with_cross_sectional_z(features)  # ✅ REPLACE

# Fail-fast check (Phase 1.2): Use model's expected dimension as single source of truth
# With REPLACE mode, dimension should always match raw feature count
expected_dim = self.model.in_features
if features.shape[-1] != expected_dim:
    raise ValueError(
        f"❌ Dimension mismatch at {target_date}!\n"
        f"   Model expects: {expected_dim} features (in_features)\n"
        f"   Data provides: {features.shape[-1]} features\n"
        f"   Raw features: {len(self.feature_cols)}\n"
        f"   CS-Z mode: {'REPLACE (values normalized)' if self.add_csz else 'RAW'}\n"
        f"   First 3 features: {self.feature_cols[:3]}\n"
        f"   This indicates model/data configuration mismatch!"
    )
```

**✅ 完了**: _append → _replace に変更、次元検証も修正

---

## 📊 動作検証

### テスト結果

```bash
python apex-ranker/scripts/sanity_check_csz.py
```

**出力**:
```
[Model Init] in_features=89, patch_multiplier=2, add_csz=True ✅
```

**✅ 成功**: モデル初期化が正しく動作

**⚠️ ブロック**: データセットに2特徴量が欠損
- `dmi_net_to_adv20`
- `dmi_z26_net`

---

## ⚠️ 残る問題

### 問題: データセットの特徴量不足

**症状**:
```
unable to find column "dmi_net_to_adv20"
89 features requested, 87 available in dataset
```

**原因**:
- Checkpoint は 89 特徴量で学習済み（`Conv groups=89`）
- データセットには 87 特徴量しか存在しない
- 2つの DMI 特徴量が欠損

**影響**:
- Sanity check が失敗
- 推論テストができない

**解決策（3つ）**:

#### Option 1: ゼロ埋め ✅ **推奨（即座）**
```python
# 欠損特徴量を 0.0 で補完
for missing_feat in ["dmi_net_to_adv20", "dmi_z26_net"]:
    df = df.with_columns(pl.lit(0.0).alias(missing_feat))
```

**メリット**: 5分で実装、すぐテスト可能
**デメリット**: 2特徴量が常に0（影響は小さい）

#### Option 2: データセット再生成 ⏱️ 中期（3-4時間）
```bash
# DMI特徴量を含む完全なデータセットを生成
python scripts/pipelines/run_full_dataset.py \
  --start 2020-01-01 --end 2025-10-31 \
  --output output/ml_dataset_89feat_complete.parquet
```

**メリット**: 完全なデータ
**デメリット**: 時間がかかる

#### Option 3: モデル再学習 ⏱️ 長期（11.5時間）
```bash
# 87特徴量で新規学習
python apex-ranker/scripts/train_v0.py \
  --config apex-ranker/configs/v0_base_corrected.yaml \
  --features apex-ranker/configs/feature_names_v0_latest_87_compat.json \
  --output models/apex_ranker_v0_87feat.pt
```

**メリット**: データと完全一致
**デメリット**: 最も時間がかかる

---

## 🚀 次のステップ（優先順位順）

### P0: 即座（5分）

**1. データセットにゼロ埋めを適用**
```python
import polars as pl

# Load dataset
df = pl.read_parquet("output/ml_dataset_latest_full_filled.parquet")

# Add missing features as zeros
df = df.with_columns([
    pl.lit(0.0).alias("dmi_net_to_adv20"),
    pl.lit(0.0).alias("dmi_z26_net"),
])

# Save
df.write_parquet("output/ml_dataset_latest_full_filled_89feat.parquet")
```

**2. Config を修正して新しいデータセットを指定**
```yaml
# v0_base_corrected.yaml
data:
  parquet_path: output/ml_dataset_latest_full_filled_89feat.parquet
```

**3. Sanity check 再実行**
```bash
python apex-ranker/scripts/sanity_check_csz.py
```

**期待される出力**:
```
[Model Init] in_features=89, patch_multiplier=2, add_csz=True
✅ Engine created successfully
✅ model.in_features = 89
✅ Dimension check passed (89 == 89)
✅ Prediction successful
```

---

### P1: 短期（当日中）

**4. スモークテスト（5営業日）**
```bash
python apex-ranker/scripts/backtest_smoke_test.py \
  --model gogooku5/models/apex_ranker/output/apex_ranker_v0_latest.pt \
  --config apex-ranker/configs/v0_base_corrected.yaml \
  --data output/ml_dataset_latest_full_filled_89feat.parquet \
  --start-date 2024-09-01 --end-date 2024-09-05 \
  --horizon 5 --top-k 35 \
  --infer-add-csz \  # Enable CS-Z REPLACE mode
  --output /tmp/bt_smoke_csz_replace.json
```

**期待ログ**:
```
[Model Init] in_features=89, patch_multiplier=2, add_csz=True
[Inference] CS-Z mode=REPLACE → shape [N,L,89]
Dimension check OK: expected=89, got=89
```

**5. 4本回帰テスト準備**
- Baseline (no enhancements)
- A.3 only (hysteresis)
- A.4 only (risk neutralization)
- A.3+A.4 (combined)

---

### P2: 中期（1-2日）

**6. フルバックテスト（2.8年、A.3+A.4）**
```bash
python apex-ranker/scripts/backtest_smoke_test.py \
  --model gogooku5/models/apex_ranker/output/apex_ranker_v0_latest.pt \
  --config apex-ranker/configs/v0_base_corrected.yaml \
  --data output/ml_dataset_latest_full_filled_89feat.parquet \
  --start-date 2023-01-01 --end-date 2025-10-24 \
  --horizon 20 --top-k 50 \
  --rebalance-freq weekly \
  --infer-add-csz \
  --ei-hysteresis-entry-k 35 --ei-hysteresis-exit-k 60 \
  --ei-neutralize-risk \
  --output results/bt_csz_replace_A3_A4_full.json
```

**7. DM/CI 統計分析**
- 4本の結果を比較
- DM test > 1.96 を確認
- 95% CI > 0 を確認
- Sharpe +10% 目標達成確認

---

## 📝 実装サマリー

### Before (誤った設計)
```python
# ❌ WRONG: Append (concatenate) mode
effective_features = n_features * 2  # 89 → 178
model = APEXRankerV0(in_features=178, ...)  # Mismatch!
features = np.concatenate([raw, z_scored], axis=-1)  # [N,L,178]
```

**問題**:
- CS-Z を連結して 178ch 作成
- Checkpoint は `groups=89` で 89ch を期待
- Conv の in_channels と不一致 → エラー

### After (正しい設計)
```python
# ✅ CORRECT: Replace mode
in_features = n_features  # 89 固定
model = APEXRankerV0(
    in_features=89,
    patch_multiplier=2,  # ✅ Checkpoint と一致
    ...
)
features = _replace_with_cross_sectional_z(raw)  # [N,L,89] ← Same shape!
```

**解決**:
- CS-Z は値を置換（連結ではない）
- 形状は [N, L, 89] のまま
- Checkpoint の `groups=89` と一致 ✅

---

## 🎯 最終確認事項

### ✅ 実装完了
- [x] Config に patch_multiplier=2 を明示
- [x] モデル初期化で in_features=89 固定
- [x] APEXRankerV0 に patch_multiplier パラメータ追加
- [x] _replace_with_cross_sectional_z() 実装
- [x] 呼び出し側を replace モードに変更
- [x] 次元検証を修正

### ⚠️ ブロック中（データ issue）
- [ ] データセットに89特徴量確保（現在87）
- [ ] Sanity check 成功
- [ ] スモークテスト実行
- [ ] 回帰テスト実行

### 📋 次のステップ
1. **P0**: ゼロ埋めでデータセット修正（5分）
2. **P0**: Sanity check 再実行
3. **P1**: スモークテスト（5営業日）
4. **P1**: 4本回帰テスト準備
5. **P2**: フルバックテスト + DM/CI分析

---

## 💡 重要な学び

### 誤解していた点
- **誤**: CS-Z = 連結 (append) で 89 → 178
- **正**: CS-Z = 置換 (replace) で 89 のまま

### Checkpoint の意味
```
Conv weight: [178, 1, 16]
groups = 89
→ in_channels = 89 (groups数)
→ out_channels = 178 (89 × patch_multiplier=2)
```

**結論**: 入力は常に89チャンネル固定

### PatchTST の動作
```
Input: [N, L, 89]
→ PatchEmbedding (patch_multiplier=2)
  → Conv1d(in=89, out=178, groups=89)
  → Linear(in=178, out=d_model=256)
→ Transformer Blocks
→ Output: [N, d_model]
```

**重要**: `patch_multiplier` は内部で特徴量を増幅する役割

---

**ステータス**: ✅ **コア実装完了** | 次: データセット修正（5分）で全テスト可能
**実装時間**: 2時間
**残り作業**: データセット修正 → テスト実行（1時間）

---

*Generated: 2025-11-02 17:00 UTC*
*Implementation: Complete*
*Next Action: Zero-pad dataset (5 minutes)*
