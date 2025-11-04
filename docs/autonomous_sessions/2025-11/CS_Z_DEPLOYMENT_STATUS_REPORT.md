# CS-Z Deployment Status Report - 2025-11-02

**ステータス**: ⚠️ **BLOCKED - Checkpoint Compatibility Issue**

---

## 📋 Executive Summary

CS-Z堅牢化の **4つのコア修正は全て実装完了** しましたが、既存checkpoint (`apex_ranker_v0_latest.pt`) との互換性問題により、sanity checkが失敗しています。

### ✅ 完了した実装
1. ✅ **モデル初期化修正**: `load_model_checkpoint` に `add_csz` パラメータを渡す
2. ✅ **キャッシュ鍵改善**: CS-Zフラグ（raw/csz）を含めて衝突防止
3. ✅ **モデル属性追加**: `APEXRankerV0.in_features` を保存
4. ✅ **次元検証堅牢化**: `model.in_features` を真実の情報源として使用

### ❌ ブロッカー
**Patch Multiplier Mismatch**: Checkpoint と新モデルで次元が不一致
- Checkpoint: 178 features (89 raw × patch_multiplier=2)
- New model with `add_csz=True`: 356 features (178 effective × patch_multiplier=2)

---

## 🔍 根本原因の詳細分析

### 問題の構造

```python
# 現在の動作（❌ 失敗）
n_features = 89
add_csz = True
effective_features = 89 × 2 = 178  # ✅ 正しい

# しかし PatchTST 内で:
model = APEXRankerV0(in_features=178, ...)
  └─ PatchTSTEncoder(in_feats=178, ...)
       └─ patch_multiplier = max(2, d_model // max(1, in_feats))
          = max(2, 256 // 178) = max(2, 1) = 2  # デフォルト
       └─ Conv1d(in_channels=178, out_channels=178×2=356, ...)  # ❌ 不一致
```

### Checkpoint の実際の構成

```
Checkpoint Analysis (from weight shapes):
✅ Conv weight: torch.Size([178, 1, 16])
   → out_channels = 178 = in_features × patch_multiplier
   → in_features = 89, patch_multiplier = 2

✅ Proj weight: torch.Size([256, 178])
   → d_model = 256, in_features × patch_multiplier = 178

結論: Checkpoint は (in_features=89, patch_multiplier=2) で学習されている
```

### 現在のモデル作成

```
New Model Creation (with add_csz=True):
effective_features = 89 × 2 = 178
model = APEXRankerV0(in_features=178, d_model=256, ...)
  └─ PatchTSTEncoder(in_feats=178, patch_multiplier=2)  # ❌ Auto-calculated
       └─ Conv out_channels = 178 × 2 = 356  # ❌ Mismatch!
```

---

## 🔧 解決策の選択肢

### Option 1: Config に patch_multiplier を明示 ✅ **推奨**

**実装**:
```python
# configs/v0_base_corrected.yaml に追加
model:
  d_model: 256
  depth: 4
  patch_len: 16
  stride: 8
  n_heads: 8
  dropout: 0.2
  patch_multiplier: 1  # NEW: CS-Z使用時は1に固定
```

```python
# APEXRankerV0.__init__ を修正
def __init__(self, ..., patch_multiplier: int | None = None):
    ...
    self.encoder = PatchTSTEncoder(
        in_feats=in_features,
        d_model=d_model,
        ...,
        patch_multiplier=patch_multiplier,  # Config から明示的に渡す
    )
```

**メリット**:
- 最小修正で解決
- 既存checkpointとの互換性維持
- 将来の混乱を防止

**デメリット**:
- Config変更が必要

---

### Option 2: Checkpoint を再学習 ⏱️ 時間がかかる

**実装**:
```bash
# 178 features (89 raw + CS-Z) で新規学習
python apex-ranker/scripts/train_v0.py \
  --config apex-ranker/configs/v0_base_corrected.yaml \
  --data output/ml_dataset_latest_full_filled.parquet \
  --add-csz-to-data \  # データ側でCS-Z追加
  --output models/apex_ranker_v0_csz.pt
```

**メリット**:
- クリーンな解決
- 将来の拡張性

**デメリット**:
- 学習に11.5時間必要
- 既存checkpointが使えない

---

### Option 3: 推論時に patch_multiplier を調整 🔧 Hack的

**実装**:
```python
# load_model_checkpoint内で動的に調整
if add_csz:
    # CS-Zの場合、patch_multiplier を半分に
    model = APEXRankerV0(
        in_features=effective_features,  # 178
        ...
        patch_multiplier=1,  # Hardcode
    )
else:
    model = APEXRankerV0(
        in_features=n_features,  # 89
        ...
        patch_multiplier=2,  # Checkpoint default
    )
```

**メリット**:
- すぐに動く

**デメリット**:
- Hack的で脆弱
- Config との不整合

---

## 📝 推奨アクション（優先順位順）

### P0: 即座に実行（5分）

**1. patch_multiplier を Config に明示化**
```yaml
# apex-ranker/configs/v0_base_corrected.yaml
model:
  d_model: 256
  depth: 4
  patch_len: 16
  stride: 8
  n_heads: 8
  dropout: 0.2
  patch_multiplier: 1  # ADD THIS
```

**2. APEXRankerV0 を修正してConfig から patch_multiplier を受け取る**
```python
# apex_ranker/models/ranker.py
class APEXRankerV0(nn.Module):
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
        patch_multiplier: int | None = None,  # NEW
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
            patch_multiplier=patch_multiplier,  # Pass explicitly
        )
        ...
```

**3. load_model_checkpoint を修正して patch_multiplier を渡す**
```python
# apex_ranker/backtest/inference.py
model = APEXRankerV0(
    in_features=effective_features,
    horizons=horizons,
    d_model=model_cfg["d_model"],
    depth=model_cfg["depth"],
    patch_len=model_cfg["patch_len"],
    stride=model_cfg["stride"],
    n_heads=model_cfg["n_heads"],
    dropout=model_cfg.get("dropout", 0.1),
    patch_multiplier=model_cfg.get("patch_multiplier", None),  # NEW
).to(device)
```

**4. Sanity check 再実行**
```bash
python apex-ranker/scripts/sanity_check_csz.py
```

---

### P1: 短期（1-2時間）

**5. 既存 checkpoint の互換性を確認**
- `add_csz=False` で checkpoint が正しくロードできるか
- 予測が reasonable か

**6. Option 1で解決しない場合、Option 3（hardcode）を試す**

---

### P2: 中期（1日）

**7. 新規学習の準備**
- Core62 完成後に、CS-Z込みで再学習
- 178 features で学習し直す

**8. Checkpoint メタデータの追加**
```python
# Training時
torch.save({
    "model_state_dict": model.state_dict(),
    "config": config,
    "in_features": 89,  # or 178
    "add_csz": False,   # or True
    "patch_multiplier": 2,  # or 1
    "feature_names": feature_list,
    ...
}, checkpoint_path)
```

---

## 🎯 今後の設計改善

### 1. Checkpoint メタデータの標準化
```python
{
    "model_state_dict": {...},
    "config": {...},
    "feature_abi": {
        "raw_features": 89,
        "cs_z_applied": False,
        "effective_features": 178,  # With patch_multiplier
        "feature_names": [...],
        "feature_hash": "...",
    },
    "model_config": {
        "d_model": 256,
        "patch_multiplier": 2,
        "in_features": 89,
        ...
    },
    "training_info": {
        "dataset": "...",
        "epochs": 50,
        "best_metric": {...},
    }
}
```

### 2. 自動互換性チェック
```python
def load_model_with_autodetect(checkpoint_path, add_csz):
    ckpt = torch.load(checkpoint_path)

    # Extract config from checkpoint
    ckpt_in_features = ckpt["model_config"]["in_features"]
    ckpt_patch_mult = ckpt["model_config"]["patch_multiplier"]
    ckpt_cs_z = ckpt["feature_abi"]["cs_z_applied"]

    # Validate compatibility
    if add_csz and ckpt_cs_z:
        raise ValueError("Checkpoint already has CS-Z, don't apply again")

    # Calculate effective features
    effective_features = ckpt_in_features
    if add_csz and not ckpt_cs_z:
        effective_features *= 2

    # Create model with correct config
    model = APEXRankerV0(
        in_features=effective_features,
        patch_multiplier=ckpt_patch_mult if not add_csz else 1,
        ...
    )

    return model
```

### 3. Config Schema Validation
```python
from pydantic import BaseModel

class ModelConfig(BaseModel):
    d_model: int = 256
    depth: int = 4
    patch_len: int = 16
    stride: int = 8
    n_heads: int = 8
    dropout: float = 0.2
    patch_multiplier: int = 1  # Required field

    @validator("patch_multiplier")
    def validate_patch_multiplier(cls, v):
        if v not in [1, 2, 4]:
            raise ValueError("patch_multiplier must be 1, 2, or 4")
        return v
```

---

## 📊 Current Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| **CS-Z Core Fixes** | ✅ Complete | 4/4 fixes implemented |
| **Model Init** | ✅ Complete | `add_csz` parameter added |
| **Cache Key** | ✅ Complete | CS-Z flag included |
| **Dimension Validation** | ✅ Complete | Uses `model.in_features` |
| **Config d_model** | ✅ Fixed | 192 → 256 |
| **patch_multiplier** | ❌ **BLOCKER** | Not specified in config |
| **Sanity Checks** | ❌ Blocked | Waiting for patch_multiplier fix |
| **Regression Tests** | ⏸️ Pending | Blocked by sanity checks |

---

## 🚀 Next Steps

**Immediate (今すぐ)**:
1. Config に `patch_multiplier: 1` を追加
2. APEXRankerV0 と load_model_checkpoint を修正
3. Sanity check 再実行

**Short-term (1-2時間)**:
4. Checkpoint 互換性確認
5. 回帰テスト準備

**Medium-term (1日)**:
6. Core62 学習完了後、CS-Z込みで再学習
7. Checkpoint メタデータ標準化

---

## 📚 References

- **Fix Summary**: `apex-ranker/CS_Z_ROBUSTNESS_FIX_SUMMARY.md`
- **Implementation Report**: `APEX_RANKER_CS_Z_FIX_REPORT.md`
- **Sanity Check Script**: `apex-ranker/scripts/sanity_check_csz.py`
- **Feature List**: `apex-ranker/configs/feature_names_v0_latest_89.json`

---

**ステータス**: ⚠️ **Patch multiplier fix required before deployment**
**ETA to Resolution**: ~15 minutes (Option 1) or ~12 hours (Option 2)
**Recommended Action**: **Option 1 (Config + Code fix)**

---

*Generated: 2025-11-02*
*Last Updated: 2025-11-02 16:00 UTC*
*Report Version: 1.0*
