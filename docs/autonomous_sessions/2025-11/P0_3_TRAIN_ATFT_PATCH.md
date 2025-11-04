# P0-3: train_atft.py RFI-5/6ロギング統合パッチ

**目的**: RFI-5/6メトリクスをtrain_atft.pyのvalidation loopに統合

**適用方法**: 貼り付け可能（2箇所の変更）

---

## パッチ1: Import追加（ファイル先頭付近）

**場所**: `scripts/train_atft.py` の既存import部分（880行付近）に追加

```python
# ==== P0-3: RFI-5/6 Metrics Import ====
from src.gogooku3.utils.rfi_metrics import log_rfi_56_metrics
```

**完全なdiff**:
```diff
from src.data.validation.normalization_check import NormalizationValidator  # noqa: E402
+# P0-3: RFI-5/6 Metrics
+from src.gogooku3.utils.rfi_metrics import log_rfi_56_metrics
```

---

## パッチ2: Validation Loop内でRFI-5/6ログ出力

**場所**: `scripts/train_atft.py` の validation loop内（5556行付近）

**挿入位置**: `loss_result = criterion(predictions, tdict, batch_metadata=batch)` の直後

```python
                    loss_result = criterion(predictions, tdict, batch_metadata=batch)

                    # ==== P0-3: RFI-5/6 Metrics Logging (first batch only) ====
                    if batch_idx == 0 and epoch % 1 == 0:  # 毎epoch、最初のバッチのみ
                        try:
                            # Extract point and quantile forecasts
                            y_point = None
                            y_q = None

                            if isinstance(predictions, dict):
                                # Multi-horizon prediction dict
                                # Use horizon=1 for point forecast
                                y_point = predictions.get(1, predictions.get("point_forecast", None))
                                y_q = predictions.get("quantile_forecast", None)

                                # If no quantile forecast, synthesize from multi-horizon
                                if y_q is None and len(predictions) > 0:
                                    # Stack all horizons as pseudo-quantiles [B, H, num_horizons]
                                    horizon_preds = [predictions[h] for h in sorted(predictions.keys()) if isinstance(h, int)]
                                    if len(horizon_preds) > 0:
                                        y_q = torch.stack(horizon_preds, dim=-1)
                            else:
                                # Single tensor output
                                y_point = predictions
                                # Synthesize dummy quantiles
                                y_q = predictions.unsqueeze(-1).repeat(1, 1, 5)  # [B, H, 5]

                            # Extract ground truth (use horizon=1)
                            y_true = tdict.get(1, tdict.get("target", None))
                            if y_true is None and len(tdict) > 0:
                                # Use first available target
                                y_true = list(tdict.values())[0]

                            # Ensure tensors are valid
                            if y_point is not None and y_true is not None and y_q is not None:
                                # Prepare batch dict for graph stats
                                batch_for_stats = {
                                    "dynamic_features": batch.get("features", None),
                                    "edge_index": batch.get("edge_index", None),
                                    "edge_attr": batch.get("edge_attr", None),
                                }

                                # Log RFI-5/6
                                log_rfi_56_metrics(
                                    logger=logger,
                                    model=model,
                                    batch=batch_for_stats,
                                    y_point=y_point,
                                    y_q=y_q,
                                    y_true=y_true,
                                    epoch=epoch
                                )
                        except Exception as e:
                            logger.warning(f"[RFI-5/6] Logging failed: {e}")
                    # ==== End P0-3 RFI-5/6 ====

                    # Handle both single value and tuple return from criterion
```

**完全なdiff**:
```diff
                    loss_result = criterion(predictions, tdict, batch_metadata=batch)

+                    # P0-3: RFI-5/6 Metrics Logging (first batch only)
+                    if batch_idx == 0 and epoch % 1 == 0:
+                        try:
+                            y_point = None
+                            y_q = None
+                            if isinstance(predictions, dict):
+                                y_point = predictions.get(1, predictions.get("point_forecast", None))
+                                y_q = predictions.get("quantile_forecast", None))
+                                if y_q is None and len(predictions) > 0:
+                                    horizon_preds = [predictions[h] for h in sorted(predictions.keys()) if isinstance(h, int)]
+                                    if len(horizon_preds) > 0:
+                                        y_q = torch.stack(horizon_preds, dim=-1)
+                            else:
+                                y_point = predictions
+                                y_q = predictions.unsqueeze(-1).repeat(1, 1, 5)
+                            y_true = tdict.get(1, tdict.get("target", None))
+                            if y_true is None and len(tdict) > 0:
+                                y_true = list(tdict.values())[0]
+                            if y_point is not None and y_true is not None and y_q is not None:
+                                batch_for_stats = {
+                                    "dynamic_features": batch.get("features", None),
+                                    "edge_index": batch.get("edge_index", None),
+                                    "edge_attr": batch.get("edge_attr", None),
+                                }
+                                log_rfi_56_metrics(
+                                    logger=logger, model=model, batch=batch_for_stats,
+                                    y_point=y_point, y_q=y_q, y_true=y_true, epoch=epoch
+                                )
+                        except Exception as e:
+                            logger.warning(f"[RFI-5/6] Logging failed: {e}")
+
                    # Handle both single value and tuple return from criterion
```

---

## 適用確認

### 1. Import確認
```bash
grep "from src.gogooku3.utils.rfi_metrics import" scripts/train_atft.py
```

期待: マッチあり

### 2. ログ呼び出し確認
```bash
grep "log_rfi_56_metrics" scripts/train_atft.py
```

期待: 2箇所マッチ（import + 呼び出し）

### 3. 実行テスト
```bash
USE_GAT_SHIM=1 BATCH_SIZE=1024 make train-quick EPOCHS=1 2>&1 | grep "RFI56"
```

期待:
```
RFI56 | epoch=1 gat_gate_mean=0.4523 gat_gate_std=0.1234 ...
```

---

## トラブルシューティング

### Issue: `RFI56` ログが出ない

**原因1**: Import失敗
```bash
# エラーログ確認
grep -i "importerror\|modulenotfounderror" _logs/training/train_*.log
```

**対処**:
```python
# Import パス確認
python -c "from src.gogooku3.utils.rfi_metrics import log_rfi_56_metrics; print('✅ Import OK')"
```

**原因2**: Exception発生
```bash
# RFI-5/6エラー確認
grep "\[RFI-5/6\] Logging failed" _logs/training/train_*.log
```

**対処**: エラーメッセージに従って修正

### Issue: `predictions` / `tdict` 構造が想定と違う

**デバッグ**:
```python
# Validation loop内に追加（一時的）
logger.info(f"[DEBUG] predictions type: {type(predictions)}, keys: {predictions.keys() if isinstance(predictions, dict) else 'N/A'}")
logger.info(f"[DEBUG] tdict type: {type(tdict)}, keys: {tdict.keys()}")
```

**観察**: ログで構造を確認し、extraction logic を調整

---

## 代替方法（train_atft.py修正が難しい場合）

### Option A: 後処理スクリプト

RFI-5/6メトリクスを既存ログから後計算:

```python
# scripts/compute_rfi_from_logs.py
from src.gogooku3.utils.rfi_metrics import rank_ic, quantile_crossing_rate, ...

# ログからpredictions/targetsを抽出
# RFI-5/6メトリクスを計算
# 結果をCSV出力
```

### Option B: Callbackフック

PyTorch Lightning等を使っている場合:

```python
class RFI56LoggerCallback(Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            log_rfi_56_metrics(...)
```

---

## 期待される出力例

```
[INFO] RFI56 | epoch=1 gat_gate_mean=0.4523 gat_gate_std=0.1234 deg_avg=25.67 isolates=0.012 corr_mean=0.345 corr_std=0.234 RankIC=0.0234 WQL=0.123456 CRPS=0.098765 qx_rate=0.0234 grad_ratio=0.87
[INFO] RFI56 | epoch=2 gat_gate_mean=0.4612 gat_gate_std=0.1198 deg_avg=26.12 isolates=0.011 corr_mean=0.351 corr_std=0.228 RankIC=0.0289 WQL=0.119872 CRPS=0.095123 qx_rate=0.0198 grad_ratio=0.92
[INFO] RFI56 | epoch=3 gat_gate_mean=0.4701 gat_gate_std=0.1167 deg_avg=25.98 isolates=0.010 corr_mean=0.348 corr_std=0.231 RankIC=0.0312 WQL=0.116543 CRPS=0.091234 qx_rate=0.0176 grad_ratio=0.95
```

---

**作成**: 2025-11-02
**ステータス**: 貼り付け可能（2箇所のパッチ）
**想定所要時間**: 5分（パッチ適用） + 1分（確認）
