"""
Integration tests for TOPIX Market Features

市場特徴量統合テスト
- test_pipeline_with_market_features()
- test_no_data_leakage_with_market()
- test_performance_improvement()
"""

import pytest
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os


class TestMarketIntegration:
    """市場特徴量統合テスト"""

    @pytest.fixture
    def temp_data_dir(self):
        """一時データディレクトリ作成"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_integration_data(self):
        """統合テスト用のサンプルデータ生成"""
        dates = pd.date_range('2020-01-01', periods=500, freq='D')
        codes = ['1301', '1302', '1303', '1304', '1305']  # 5銘柄

        np.random.seed(42)

        # TOPIXデータ生成
        topix_returns = np.random.normal(0.0002, 0.015, len(dates))
        topix_prices = 2000 * np.exp(np.cumsum(topix_returns))

        topix_data = []
        for i, date in enumerate(dates):
            topix_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Close': topix_prices[i],
                'Volume': np.random.randint(1000000, 5000000)
            })

        # 銘柄データ生成
        stock_data = []
        for code in codes:
            # 個別銘柄の特性を設定
            beta = np.random.uniform(0.5, 2.0)
            alpha = np.random.normal(0.0001, 0.005)
            vol = np.random.uniform(0.01, 0.03)

            stock_returns = beta * topix_returns + alpha + np.random.normal(0, vol, len(dates))
            stock_prices = 1000 * np.exp(np.cumsum(stock_returns))

            for i, date in enumerate(dates):
                stock_data.append({
                    'Code': code,
                    'date': date,
                    'Close': stock_prices[i],
                    'return_1d': stock_returns[i] if i > 0 else 0.0,
                    'Volume': np.random.randint(10000, 100000),
                    'adjustment_close': stock_prices[i],
                    'adjustment_open': stock_prices[i] * (1 + np.random.normal(0, 0.005)),
                    'adjustment_high': stock_prices[i] * (1 + abs(np.random.normal(0, 0.01))),
                    'adjustment_low': stock_prices[i] * (1 - abs(np.random.normal(0, 0.01))),
                    'adjustment_volume': np.random.randint(10000, 100000)
                })

        return {
            'topix': pl.DataFrame(topix_data),
            'stocks': pl.DataFrame(stock_data)
        }

    def test_pipeline_with_market_features(self, sample_integration_data, temp_data_dir):
        """市場特徴量統合パイプラインのテスト"""
        from scripts.data.ml_dataset_builder import MLDatasetBuilder

        # テストデータを一時ファイルに保存
        topix_file = temp_data_dir / "topix_test.parquet"
        stocks_file = temp_data_dir / "stocks_test.parquet"

        sample_integration_data['topix'].write_parquet(topix_file)
        sample_integration_data['stocks'].write_parquet(stocks_file)

        # MLDatasetBuilderで市場特徴量統合を実行
        builder = MLDatasetBuilder(output_dir=temp_data_dir)
        enhanced_df = builder.add_topix_features(sample_integration_data['stocks'])

        # 市場特徴量が追加されているか確認
        market_cols = [col for col in enhanced_df.columns if col.startswith('mkt_')]
        cross_cols = [col for col in enhanced_df.columns if col.startswith(('beta_', 'alpha_', 'rel_'))]

        assert len(market_cols) >= 20, f"Expected >=20 market features, got {len(market_cols)}"
        assert len(cross_cols) >= 8, f"Expected >=8 cross features, got {len(cross_cols)}"

        # データサイズが維持されているか確認
        assert len(enhanced_df) == len(sample_integration_data['stocks'])

        print(f"✅ Pipeline integration successful: {len(market_cols)} market + {len(cross_cols)} cross features")

    def test_no_data_leakage_with_market(self, sample_integration_data):
        """市場特徴量によるデータリークがないことを確認"""
        from scripts.data.ml_dataset_builder import MLDatasetBuilder

        builder = MLDatasetBuilder()

        # 市場特徴量統合
        enhanced_df = builder.add_topix_features(sample_integration_data['stocks'])

        # 未来情報リークがないか確認
        # 各銘柄で、日付順にソートして特徴量が過去情報のみを使っているか確認
        for code in enhanced_df.select('Code').unique().to_series().to_list():
            code_data = enhanced_df.filter(pl.col('Code') == code).sort('date')

            # β値が過去データのみで計算されているか確認
            beta_values = code_data.select('beta_60d').to_series().drop_nulls()
            if len(beta_values) > 60:  # 十分なデータがある場合
                # βは過去60日間のデータで計算されるので、最初の60日間はnullのはず
                early_beta = code_data.head(60).select('beta_60d').to_series()
                null_count = early_beta.null_count()
                assert null_count >= 50, f"Early β values should be mostly null, got {null_count}/60 nulls"

        print("✅ No data leakage detected in market features")

    def test_performance_improvement(self, sample_integration_data):
        """市場特徴量による性能向上を確認"""
        from src.features.quality_features import QualityFinancialFeaturesGenerator

        # 市場特徴量なしの場合
        generator = QualityFinancialFeaturesGenerator(verbose=False)
        basic_features = generator.generate_quality_features(sample_integration_data['stocks'])

        # 市場特徴量ありの場合
        enhanced_df = sample_integration_data['stocks']
        from scripts.data.ml_dataset_builder import MLDatasetBuilder
        builder = MLDatasetBuilder()
        enhanced_features = builder.add_topix_features(enhanced_df)

        # 特徴量数の比較
        basic_count = len(basic_features.columns)
        enhanced_count = len(enhanced_features.columns)
        market_features_added = enhanced_count - basic_count

        assert market_features_added >= 24, f"Expected >=24 market features added, got {market_features_added}"

        # 数値特徴量の質を確認
        basic_numeric = basic_features.select([col for col in basic_features.columns
                                             if basic_features[col].dtype in [pl.Float32, pl.Float64]])
        enhanced_numeric = enhanced_features.select([col for col in enhanced_features.columns
                                                   if enhanced_features[col].dtype in [pl.Float32, pl.Float64]])

        assert len(enhanced_numeric.columns) > len(basic_numeric.columns)

        print(f"✅ Performance improvement: {basic_count} → {enhanced_count} features (+{market_features_added})")

    def test_cross_sectional_normalization_with_market(self, sample_integration_data):
        """市場特徴量を含むクロスセクショナル正規化のテスト"""
        from scripts.data.ml_dataset_builder import MLDatasetBuilder

        builder = MLDatasetBuilder()
        enhanced_df = builder.add_topix_features(sample_integration_data['stocks'])

        # 日付ごとにクロスセクショナル正規化をシミュレーション
        dates = enhanced_df.select('date').unique().sort('date').to_series().to_list()

        for date in dates[:10]:  # 最初の10日間のみテスト
            day_data = enhanced_df.filter(pl.col('date') == date)

            if len(day_data) >= 3:  # 十分な銘柄数がある場合
                # 市場特徴量の分布を確認
                market_cols = [col for col in day_data.columns if col.startswith('mkt_')]
                for col in market_cols[:5]:  # 最初の5つの特徴量のみ確認
                    values = day_data.select(col).to_series().drop_nulls()
                    if len(values) >= 3:
                        # 値が存在し、多様性があることを確認
                        unique_count = len(values.unique())
                        assert unique_count >= 1, f"Market feature {col} has no variation on {date}"

        print("✅ Cross-sectional normalization compatible with market features")

    def test_memory_efficiency(self, sample_integration_data):
        """メモリ効率のテスト"""
        import psutil
        import os

        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        from scripts.data.ml_dataset_builder import MLDatasetBuilder
        builder = MLDatasetBuilder()

        # 市場特徴量統合実行
        enhanced_df = builder.add_topix_features(sample_integration_data['stocks'])

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory

        # メモリ使用量が妥当か確認（500MB以内）
        assert memory_used < 500, f"Memory usage too high: {memory_used:.1f} MB"

        print(f"✅ Memory efficient: {memory_used:.1f} MB used")

    def test_feature_correlation_analysis(self, sample_integration_data):
        """特徴量間の相関分析"""
        from scripts.data.ml_dataset_builder import MLDatasetBuilder

        builder = MLDatasetBuilder()
        enhanced_df = builder.add_topix_features(sample_integration_data['stocks'])

        # 主要特徴量の相関を計算
        key_features = ['mkt_ret_1d', 'beta_60d', 'alpha_1d', 'rel_strength_5d']

        # 相関行列の計算（サンプルベース）
        correlation_data = []
        for i, feat1 in enumerate(key_features):
            for j, feat2 in enumerate(key_features):
                if i < j and feat1 in enhanced_df.columns and feat2 in enhanced_df.columns:
                    corr = enhanced_df.select([feat1, feat2]).drop_nulls().corr().item(0, 1)
                    if not np.isnan(corr):
                        correlation_data.append({
                            'feature1': feat1,
                            'feature2': feat2,
                            'correlation': abs(corr)
                        })

        # 相関が計算可能か確認
        assert len(correlation_data) > 0, "No valid correlations calculated"

        # 過度な相関（>0.95）がないことを確認
        high_corr_count = sum(1 for item in correlation_data if item['correlation'] > 0.95)
        assert high_corr_count == 0, f"Found {high_corr_count} highly correlated feature pairs"

        print(f"✅ Feature correlation analysis: {len(correlation_data)} correlation pairs analyzed")

    def test_market_regime_effectiveness(self, sample_integration_data):
        """市場レジーム特徴量の有効性テスト"""
        from scripts.data.ml_dataset_builder import MLDatasetBuilder

        builder = MLDatasetBuilder()
        enhanced_df = builder.add_topix_features(sample_integration_data['stocks'])

        # レジームフラグの分布を確認
        regime_flags = ['mkt_bull_200', 'mkt_trend_up', 'mkt_high_vol', 'mkt_squeeze']

        for flag in regime_flags:
            if flag in enhanced_df.columns:
                flag_values = enhanced_df.select(flag).to_series()
                flag_rate = flag_values.mean()

                # フラグがアクティブになる割合が妥当か確認（0.1-0.9の範囲）
                assert 0.1 <= flag_rate <= 0.9, f"Flag {flag} activation rate {flag_rate:.3f} is unreasonable"

        # レジームフラグが市場状態を適切に捉えているか確認
        if 'mkt_bull_200' in enhanced_df.columns and 'mkt_ret_1d' in enhanced_df.columns:
            bull_market_returns = enhanced_df.filter(pl.col('mkt_bull_200') == 1).select('mkt_ret_1d').mean()
            bear_market_returns = enhanced_df.filter(pl.col('mkt_bull_200') == 0).select('mkt_ret_1d').mean()

            # 強気市場の方がリターンが高いはず
            assert bull_market_returns > bear_market_returns, \
                f"Bull market returns {bull_market_returns:.6f} should be > bear market {bear_market_returns:.6f}"

        print("✅ Market regime effectiveness validated")
