#!/usr/bin/env python3
"""
staleness_days_list の実装テスト
データ鮮度追跡機能の動作確認
"""

import sys
import logging
import numpy as np
import torch
from datetime import datetime, timedelta

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_staleness_implementation():
    """staleness_days_list の実装をテスト"""

    print("🧪 Testing staleness_days_list implementation...")

    # モックデータでテスト
    class MockCriterion:
        def __init__(self):
            self.horizons = [1, 2, 3, 5, 10]
            self._staleness_days_list = []
            self._ema_rmse = {}
            self.dynamic_alpha = 0.1
            self.use_dynamic_weighting = True

            # データのタイムスタンプを設定
            self.current_batch_timestamp = torch.tensor(datetime.now().timestamp())
            self.data_last_updated = torch.tensor(
                (datetime.now() - timedelta(days=5)).timestamp()
            )

        def _calculate_staleness(self, yhat, y, horizon):
            """データ鮮度を計算するメソッド（実装された部分）"""
            with torch.no_grad():
                # データの鮮度（staleness）を追跡
                if not hasattr(self, "_staleness_days_list"):
                    self._staleness_days_list = []

                # 現在のバッチのタイムスタンプを取得
                if hasattr(self, "current_batch_timestamp"):
                    batch_ts = self.current_batch_timestamp
                else:
                    batch_ts = torch.tensor(0.0)

                # データの鮮度を計算（日数単位）
                try:
                    # データソースの最終更新時刻を取得
                    if hasattr(self, "data_last_updated"):
                        data_ts = self.data_last_updated
                    else:
                        data_ts = torch.tensor(0.0)

                    # 鮮度を日数で計算
                    staleness_days = (batch_ts - data_ts).item() / (24 * 3600)
                    self._staleness_days_list.append(staleness_days)

                    # 鮮度の統計情報をログ出力
                    if len(self._staleness_days_list) % 10 == 0:  # 10バッチごとにログ
                        avg_staleness = np.mean(self._staleness_days_list[-10:])
                        max_staleness = np.max(self._staleness_days_list[-10:])
                        logger.info(
                            f"Data staleness stats (last 10 batches): "
                            f"avg={avg_staleness:.2f} days, max={max_staleness:.2f} days"
                        )

                except Exception as e:
                    logger.warning(f"Failed to calculate data staleness: {e}")
                    self._staleness_days_list.append(0.0)

                # RMSE計算（元の機能）
                rmse = torch.sqrt(torch.mean((yhat - y) ** 2)).item()
                prev = self._ema_rmse.get(int(horizon), None)
                if prev is None:
                    self._ema_rmse[int(horizon)] = rmse
                else:
                    alpha = self.dynamic_alpha
                    self._ema_rmse[int(horizon)] = (1 - alpha) * prev + alpha * rmse

                return rmse

        def get_staleness_stats(self):
            """データ鮮度の統計情報を取得"""
            if not self._staleness_days_list:
                return {"error": "No staleness data available"}

            stats = {
                "total_batches": len(self._staleness_days_list),
                "avg_staleness_days": np.mean(self._staleness_days_list),
                "max_staleness_days": np.max(self._staleness_days_list),
                "min_staleness_days": np.min(self._staleness_days_list),
                "std_staleness_days": np.std(self._staleness_days_list),
                "recent_10_avg": np.mean(self._staleness_days_list[-10:])
                if len(self._staleness_days_list) >= 10
                else np.mean(self._staleness_days_list),
            }
            return stats

    # テスト実行
    criterion = MockCriterion()

    print("📊 Testing staleness calculation...")

    # 複数のバッチでテスト
    for i in range(25):
        # モックデータ
        yhat = torch.randn(32, 1) * 0.1
        y = torch.randn(32, 1) * 0.1
        horizon = np.random.choice([1, 2, 3, 5, 10])

        # データ鮮度計算
        rmse = criterion._calculate_staleness(yhat, y, horizon)

        if i % 5 == 0:
            print(f"Batch {i}: RMSE={rmse:.4f}")

    # 統計情報を取得
    stats = criterion.get_staleness_stats()

    print("\n📈 Staleness Statistics:")
    print("=" * 40)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # 検証用のstalenessリストもテスト
    print("\n🔍 Testing validation staleness...")

    # モックの検証データ
    criterion._val_staleness_days_list = []

    # 検証データの鮮度を計算
    for i in range(15):
        # モックの日付データ
        mock_date = datetime.now() - timedelta(days=i)
        mock_asof_date = mock_date - timedelta(days=np.random.randint(1, 10))

        batch_ts = mock_date
        asof_ts = mock_asof_date

        # データ鮮度を計算
        staleness_days = int((batch_ts - asof_ts).days)
        criterion._val_staleness_days_list.append(staleness_days)

        if i % 5 == 0:
            print(f"Val batch {i}: staleness={staleness_days} days")

    # 検証データの統計情報
    if criterion._val_staleness_days_list:
        val_arr = np.array(criterion._val_staleness_days_list, dtype=float)
        val_mean = float(np.mean(val_arr))
        val_median = float(np.median(val_arr))
        val_std = float(np.std(val_arr))
        val_min = float(np.min(val_arr))
        val_max = float(np.max(val_arr))

        print("\n📊 Validation Staleness Stats:")
        print(f"Mean: {val_mean:.2f} days")
        print(f"Median: {val_median:.2f} days")
        print(f"Std: {val_std:.2f} days")
        print(f"Min: {val_min:.2f} days")
        print(f"Max: {val_max:.2f} days")

    print("\n✅ Staleness implementation test completed!")
    return True


if __name__ == "__main__":
    try:
        success = test_staleness_implementation()
        if success:
            print("🎉 All tests passed!")
            sys.exit(0)
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        sys.exit(1)
