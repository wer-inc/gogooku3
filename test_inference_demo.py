#!/usr/bin/env python3
"""
ATFT-GAT-FAN推論システムの実用的な動作確認
"""

import polars as pl
import numpy as np
from scripts.models.atft_inference import ATFTInference
import time


def create_realistic_stock_data():
    """現実的な株価データを作成"""
    n_stocks = 5
    n_days = 50

    data = []
    for stock_id in range(n_stocks):
        stock_code = f"STOCK_{stock_id:04d}"
        base_price = 1000 + stock_id * 100

        for day in range(n_days):
            # 現実的な価格変動
            price_change = np.random.normal(0, 0.02)  # 2%の日次変動
            base_price *= 1 + price_change

            # 基本価格データ
            open_price = base_price * (1 + np.random.normal(0, 0.005))
            high_price = max(open_price, base_price) * (
                1 + abs(np.random.normal(0, 0.01))
            )
            low_price = min(open_price, base_price) * (
                1 - abs(np.random.normal(0, 0.01))
            )
            close_price = base_price
            volume = np.random.randint(1000, 10000)

            # テクニカル指標
            returns_1d = np.random.normal(0, 0.02)
            returns_5d = np.random.normal(0, 0.05)
            returns_20d = np.random.normal(0, 0.10)
            rsi = np.random.uniform(30, 70)
            macd = np.random.normal(0, 10)
            bb_upper = close_price * (1 + np.random.uniform(0.01, 0.05))
            atr = close_price * np.random.uniform(0.01, 0.03)
            obv = np.random.normal(0, 10000)

            row = {
                "Code": stock_code,
                "Date": f"2024-01-{day+1:02d}",
                "Open": open_price,
                "High": high_price,
                "Low": low_price,
                "Close": close_price,
                "Volume": volume,
                "returns_1d": returns_1d,
                "returns_5d": returns_5d,
                "returns_20d": returns_20d,
                "rsi": rsi,
                "macd": macd,
                "bb_upper": bb_upper,
                "atr": atr,
                "obv": obv,
            }
            data.append(row)

    return pl.DataFrame(data)


def test_inference_performance():
    """推論性能テスト"""
    print("🚀 ATFT-GAT-FAN推論システム性能テスト")
    print("=" * 50)

    # 推論システム初期化
    print("📋 推論システム初期化中...")
    start_time = time.time()
    atft = ATFTInference()
    init_time = time.time() - start_time
    print(f"✅ 初期化完了: {init_time:.2f}秒")

    # テストデータ作成
    print("📊 テストデータ作成中...")
    test_data = create_realistic_stock_data()
    print(f"✅ テストデータ作成完了: {test_data.shape}")

    # 推論実行
    print("🔮 推論実行中...")
    start_time = time.time()
    predictions = atft.predict_from_dataframe(test_data)
    inference_time = time.time() - start_time

    print(f"✅ 推論完了: {inference_time:.2f}秒")
    print(f"📈 予測結果: {predictions.shape}")

    # 結果分析
    print("\n📊 予測結果分析:")
    print(f"  銘柄数: {predictions['Code'].n_unique()}")
    print(
        f"  予測期間: {len([col for col in predictions.columns if 'prediction' in col])}期間"
    )
    prediction_cols = [col for col in predictions.columns if "prediction" in col]
    if prediction_cols:
        # 予測値の統計を計算
        pred_values = []
        for col in prediction_cols:
            pred_values.extend(predictions[col].to_list())

        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)
        print(f"  平均予測値: {mean_pred:.4f}")
        print(f"  予測値標準偏差: {std_pred:.4f}")
    else:
        print("  予測値が見つかりません")

    # サンプル予測結果表示
    print("\n📋 サンプル予測結果:")
    sample_predictions = predictions.head(10)
    print(sample_predictions)

    return {
        "init_time": init_time,
        "inference_time": inference_time,
        "predictions_shape": predictions.shape,
        "sample_predictions": sample_predictions,
    }


def test_batch_inference():
    """バッチ推論テスト"""
    print("\n🔄 バッチ推論テスト")
    print("=" * 30)

    atft = ATFTInference()

    # 異なるサイズのデータでテスト
    batch_sizes = [10, 50, 100, 250]

    results = {}
    for batch_size in batch_sizes:
        print(f"📊 バッチサイズ {batch_size} でテスト中...")

        # テストデータ作成
        test_data = create_realistic_stock_data().head(batch_size)

        # 推論実行
        start_time = time.time()
        predictions = atft.predict_from_dataframe(test_data)
        inference_time = time.time() - start_time

        results[batch_size] = {
            "inference_time": inference_time,
            "predictions_count": len(predictions),
            "throughput": batch_size / inference_time,
        }

        print(
            f"  ✅ 完了: {inference_time:.2f}秒 ({batch_size/inference_time:.1f} サンプル/秒)"
        )

    # 結果サマリー
    print("\n📈 バッチ推論性能サマリー:")
    for batch_size, result in results.items():
        print(f"  バッチサイズ {batch_size}: {result['throughput']:.1f} サンプル/秒")

    return results


def main():
    """メイン実行関数"""
    print("🎯 ATFT-GAT-FAN推論システム実用的動作確認")
    print("=" * 60)

    try:
        # 基本推論テスト
        basic_results = test_inference_performance()

        # バッチ推論テスト
        batch_results = test_batch_inference()

        # 総合結果
        print("\n🎉 推論システム動作確認完了!")
        print("=" * 40)
        print(f"✅ 初期化時間: {basic_results['init_time']:.2f}秒")
        print(f"✅ 推論時間: {basic_results['inference_time']:.2f}秒")
        print(f"✅ 予測結果: {basic_results['predictions_shape']}")
        print(
            f"✅ 最大スループット: {max(r['throughput'] for r in batch_results.values()):.1f} サンプル/秒"
        )

        return True

    except Exception as e:
        print(f"❌ 推論システムテスト失敗: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
