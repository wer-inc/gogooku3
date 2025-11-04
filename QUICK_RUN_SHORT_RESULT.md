=== 短縮版 Quick Run 完了レポート ===

**実行時刻**: Mon Nov  3 14:27:37 UTC 2025
**所要時間**: 14541.23秒 (~242分 / 4時間)

## 検証メトリクス (Validation)

[2025-11-03 11:36:03,378][root][INFO] - Val metrics (z) h=1: MAE=0.1232 RMSE=0.1497 R2=-19.1542 IC=0.0284 NAIVE_RMSE=0.0248 SCALE(yhat/y)=4.40 CAL=+0.000+1.000*yhat
[2025-11-03 11:36:03,436][root][INFO] - Val metrics (z) h=5: MAE=0.1804 RMSE=0.2004 R2=-1.7266 IC=0.0262 NAIVE_RMSE=0.0556 SCALE(yhat/y)=1.33 CAL=+0.004+0.015*yhat
[2025-11-03 11:36:03,500][root][INFO] - Val metrics (z) h=10: MAE=0.0728 RMSE=0.0999 R2=-0.6580 IC=-0.0158 NAIVE_RMSE=0.0767 SCALE(yhat/y)=0.80 CAL=+0.000+1.000*yhat
[2025-11-03 11:36:03,567][root][INFO] - Val metrics (z) h=20: MAE=0.0972 RMSE=0.1358 R2=-0.5895 IC=0.0159 NAIVE_RMSE=0.1070 SCALE(yhat/y)=0.78 CAL=+0.003+0.014*yhat

## 最終Sharpe Ratio
2025-11-03 11:48:23,410 - __main__ - INFO - 🎯 Achieved Sharpe Ratio: 0.08177260619898637

## 早期停止トリガー確認
早期停止回数: 0

## 判定

### 合格基準チェック:
- ✅ Loss 非ゼロ: Yes (Val loss=0.7377)
- ✅ IC 非ゼロ: Yes (h=1: 0.0284, h=5: 0.0262, h=20: 0.0159)
- ⚠️  Sharpe Ratio: 0.082 (目標 0.849 の ~10%)
- ❌ RFI-5/6 ログ: 未出力（実装未確認）
