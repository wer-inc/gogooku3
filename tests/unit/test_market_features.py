"""
Unit tests for TOPIX Market Features

市場特徴量生成器の単体テスト
- test_topix_features_generation()
- test_cross_features_calculation()
- test_beta_calculation_accuracy()
"""

import pytest
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestMarketFeaturesGenerator:
    """TOPIX市場特徴量生成器のテスト"""

    @pytest.fixture
    def sample_topix_data(self):
        """サンプルTOPIXデータを生成"""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        np.random.seed(42)

        # TOPIX価格の生成（ランダムウォーク）
        returns = np.random.normal(0.0002, 0.015, len(dates))
        prices = 2000 * np.exp(np.cumsum(returns))

        # データフレーム作成
        df = pl.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in dates],
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, len(dates))
        })

        return df

    @pytest.fixture
    def sample_stock_data(self):
        """サンプル銘柄データを生成"""
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        codes = ['1301', '1302', '1303']  # サンプル銘柄コード

        data = []
        np.random.seed(123)

        for code in codes:
            # 個別銘柄のリターンを生成
            returns = np.random.normal(0.0001, 0.02, len(dates))
            prices = 1000 * np.exp(np.cumsum(returns))

            for i, date in enumerate(dates):
                data.append({
                    'Code': code,
                    'date': date,
                    'Close': prices[i],
                    'return_1d': returns[i] if i > 0 else 0.0,
                    'Volume': np.random.randint(10000, 100000)
                })

        return pl.DataFrame(data)

    def test_topix_features_generation(self, sample_topix_data):
        """TOPIX特徴量生成のテスト"""
        from src.features.market_features import MarketFeaturesGenerator

        generator = MarketFeaturesGenerator()
        result_df = generator.build_topix_features(sample_topix_data)

        # 必須の市場特徴量が存在するか確認
        expected_features = [
            'mkt_ret_1d', 'mkt_ret_5d', 'mkt_ret_10d', 'mkt_ret_20d',
            'mkt_ema_5', 'mkt_ema_20', 'mkt_ema_60', 'mkt_ema_200',
            'mkt_vol_20d', 'mkt_atr_14', 'mkt_bb_pct_b', 'mkt_bb_bw',
            'mkt_bull_200', 'mkt_trend_up', 'mkt_high_vol', 'mkt_squeeze'
        ]

        for feature in expected_features:
            assert feature in result_df.columns, f"Missing feature: {feature}"

        # データサイズが正しいか確認
        assert len(result_df) == len(sample_topix_data)

        # Zスコア特徴量が存在するか確認
        zscore_features = ['mkt_ret_1d_z', 'mkt_vol_20d_z', 'mkt_bb_bw_z', 'mkt_dd_z']
        for feature in zscore_features:
            assert feature in result_df.columns, f"Missing Z-score feature: {feature}"

        print(f"✅ Generated {len(result_df.columns)} market features")

    def test_cross_features_calculation(self, sample_stock_data, sample_topix_data):
        """クロス特徴量計算のテスト"""
        from src.features.market_features import MarketFeaturesGenerator, CrossMarketFeaturesGenerator

        # 市場特徴量生成
        market_gen = MarketFeaturesGenerator()
        market_features_df = market_gen.build_topix_features(sample_topix_data)

        # クロス特徴量生成
        cross_gen = CrossMarketFeaturesGenerator()
        result_df = cross_gen.attach_market_and_cross(sample_stock_data, market_features_df)

        # 必須のクロス特徴量が存在するか確認
        expected_cross_features = [
            'beta_60d', 'alpha_1d', 'alpha_5d', 'beta_stability_60d',
            'rel_strength_5d', 'trend_align_mkt', 'alpha_vs_regime', 'idio_vol_ratio'
        ]

        for feature in expected_cross_features:
            assert feature in result_df.columns, f"Missing cross feature: {feature}"

        # 市場特徴量も統合されているか確認
        market_features = [col for col in result_df.columns if col.startswith('mkt_')]
        assert len(market_features) > 20, f"Expected >20 market features, got {len(market_features)}"

        # 銘柄別の計算が正しいか確認
        unique_codes = result_df.select('Code').unique().to_series().to_list()
        assert len(unique_codes) == 3, f"Expected 3 unique codes, got {len(unique_codes)}"

        print(f"✅ Generated {len([col for col in result_df.columns if col.startswith(('mkt_', 'beta_', 'alpha_', 'rel_'))])} cross/market features")

    def test_beta_calculation_accuracy(self, sample_stock_data, sample_topix_data):
        """β計算の正確性テスト"""
        from src.features.market_features import CrossMarketFeaturesGenerator

        # シンプルなテストデータ作成
        dates = pd.date_range('2020-01-01', periods=100, freq='D')

        # 市場リターンを固定
        market_returns = np.random.normal(0.001, 0.01, len(dates))

        # β=1.5の銘柄リターンを生成
        beta = 1.5
        stock_returns = beta * market_returns + np.random.normal(0, 0.005, len(dates))

        # データフレーム作成
        market_df = pl.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in dates],
            'Close': 2000 * np.exp(np.cumsum(market_returns)),
            'mkt_ret_1d': market_returns
        })

        stock_df = pl.DataFrame({
            'Code': ['TEST'] * len(dates),
            'date': dates,
            'Close': 1000 * np.exp(np.cumsum(stock_returns)),
            'return_1d': stock_returns
        })

        # β計算実行
        cross_gen = CrossMarketFeaturesGenerator(beta_window=60)
        result_df = cross_gen.attach_market_and_cross(stock_df, market_df)

        # βが期待値に近いか確認
        calculated_beta = result_df.select('beta_60d').to_series().drop_nulls().mean()
        assert abs(calculated_beta - beta) < 0.3, f"β calculation inaccurate: expected ~{beta}, got {calculated_beta}"

        print(".3f"
    def test_regime_flags_calculation(self, sample_topix_data):
        """レジームフラグ計算のテスト"""
        from src.features.market_features import MarketFeaturesGenerator

        generator = MarketFeaturesGenerator()
        result_df = generator.build_topix_features(sample_topix_data)

        # レジームフラグの範囲を確認
        regime_flags = ['mkt_bull_200', 'mkt_trend_up', 'mkt_high_vol', 'mkt_squeeze']

        for flag in regime_flags:
            values = result_df.select(flag).to_series()
            unique_vals = values.unique().to_list()
            assert all(val in [0, 1] for val in unique_vals), f"Flag {flag} should be 0 or 1, got {unique_vals}"

        # 少なくとも1つのレジームがアクティブになっているか確認
        total_regime_signals = 0
        for flag in regime_flags:
            total_regime_signals += result_df.select(flag).sum().item()

        assert total_regime_signals > 0, "At least one regime should be active"

        print(f"✅ Regime flags calculated: {total_regime_signals} total signals across {len(regime_flags)} flags")

    def test_zscore_normalization(self, sample_topix_data):
        """Zスコア正規化のテスト"""
        from src.features.market_features import MarketFeaturesGenerator

        generator = MarketFeaturesGenerator(z_score_window=100)
        result_df = generator.build_topix_features(sample_topix_data)

        # Zスコア特徴量の統計を確認
        zscore_features = ['mkt_ret_1d_z', 'mkt_vol_20d_z', 'mkt_bb_bw_z', 'mkt_dd_z']

        for feature in zscore_features:
            if feature in result_df.columns:
                values = result_df.select(feature).to_series().drop_nulls()
                if len(values) > 10:
                    mean_val = values.mean()
                    std_val = values.std()

                    # Zスコアの平均は0に近く、標準偏差は1に近いはず
                    assert abs(mean_val) < 0.5, f"Z-score mean should be ~0, got {mean_val} for {feature}"
                    assert abs(std_val - 1.0) < 0.5, f"Z-score std should be ~1, got {std_val} for {feature}"

        print("✅ Z-score normalization working correctly")

    def test_data_validation(self, sample_topix_data):
        """データ妥当性検証のテスト"""
        from src.features.market_features import validate_market_data

        # 有効なデータ
        is_valid, errors = validate_market_data(sample_topix_data)
        assert is_valid, f"Valid data should pass validation: {errors}"

        # 無効なデータ（空のデータ）
        empty_df = pl.DataFrame({'Date': [], 'Close': []})
        is_valid, errors = validate_market_data(empty_df)
        assert not is_valid
        assert len(errors) > 0

        # 欠損値の多いデータ
        bad_data = sample_topix_data.clone()
        bad_data = bad_data.with_columns(
            pl.when(pl.col('Close').cum_count() > len(bad_data) // 2)
            .then(None)
            .otherwise(pl.col('Close'))
            .alias('Close')
        )
        is_valid, errors = validate_market_data(bad_data)
        assert not is_valid

        print("✅ Data validation working correctly")

    def test_market_features_pipeline(self, sample_topix_data):
        """市場特徴量生成パイプラインのテスト"""
        from src.features.market_features import create_market_features_pipeline

        result_df = create_market_features_pipeline(sample_topix_data)

        # 必要な特徴量が全て生成されているか確認
        market_cols = [col for col in result_df.columns if col.startswith('mkt_')]
        assert len(market_cols) >= 24, f"Expected at least 24 market features, got {len(market_cols)}"

        # データが正しく処理されているか確認
        assert len(result_df) == len(sample_topix_data)
        assert 'Date' in result_df.columns

        print(f"✅ Market features pipeline generated {len(market_cols)} features successfully")
