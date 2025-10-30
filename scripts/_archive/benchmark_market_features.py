#!/usr/bin/env python3
# DEPRECATED: Moved to scripts/_archive on 2025-09-04
"""
TOPIX市場特徴量ベンチマーク評価スクリプト

市場特徴量の性能改善を評価:
- RankIC@1d, 3d の改善率測定
- α系列の有効性確認（相関分析）
- レジームフラグの予測力検証

使用方法:
python scripts/benchmark_market_features.py --data-path data/processed/ml_dataset_enhanced.parquet
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MarketFeaturesBenchmark:
    """市場特徴量ベンチマーク評価クラス"""

    def __init__(self, data_path: Path, output_dir: Path = None):
        """
        Args:
            data_path: 評価対象データのパス
            output_dir: 出力ディレクトリ
        """
        self.data_path = Path(data_path)
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 結果保存用
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'data_info': {},
            'rank_ic_analysis': {},
            'feature_importance': {},
            'regime_effectiveness': {},
            'performance_summary': {}
        }

    def run_full_benchmark(self) -> dict[str, Any]:
        """フルベンチマーク実行"""
        logger.info("🚀 Starting TOPIX Market Features Benchmark")

        # 1. データ読み込み
        df = self.load_data()
        self.results['data_info'] = self.analyze_data_info(df)

        # 2. RankIC分析
        self.results['rank_ic_analysis'] = self.analyze_rank_ic(df)

        # 3. 特徴量重要度分析
        self.results['feature_importance'] = self.analyze_feature_importance(df)

        # 4. レジーム有効性分析
        self.results['regime_effectiveness'] = self.analyze_regime_effectiveness(df)

        # 5. パフォーマンスサマリー
        self.results['performance_summary'] = self.create_performance_summary()

        # 6. 結果保存
        self.save_results()

        logger.info("✅ Benchmark completed successfully")
        return self.results

    def load_data(self) -> pl.DataFrame:
        """データ読み込み"""
        logger.info(f"Loading data from {self.data_path}")

        if self.data_path.suffix == '.parquet':
            df = pl.read_parquet(self.data_path)
        elif self.data_path.suffix == '.csv':
            df = pl.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def analyze_data_info(self, df: pl.DataFrame) -> dict[str, Any]:
        """データ情報分析"""
        info = {
            'total_rows': len(df),
            'total_features': len(df.columns),
            'date_range': {
                'start': df.select('date').min().item().isoformat(),
                'end': df.select('date').max().item().isoformat()
            },
            'unique_stocks': df.select('Code').nunique(),
            'market_features_count': len([col for col in df.columns if col.startswith('mkt_')]),
            'cross_features_count': len([col for col in df.columns if col.startswith(('beta_', 'alpha_', 'rel_'))]),
            'target_features': [col for col in df.columns if 'target_' in col]
        }

        logger.info(f"Data analysis: {info['unique_stocks']} stocks, {info['market_features_count']} market features")
        return info

    def analyze_rank_ic(self, df: pl.DataFrame) -> dict[str, Any]:
        """RankIC分析"""
        logger.info("Analyzing RankIC performance")

        results = {}

        # 利用可能なターゲットを確認
        target_cols = [col for col in df.columns if col.startswith('target_')]
        market_features = [col for col in df.columns if col.startswith(('mkt_', 'beta_', 'alpha_', 'rel_'))]

        if not target_cols:
            logger.warning("No target columns found for RankIC analysis")
            return results

        # 各ターゲットに対してRankICを計算
        for target_col in target_cols:
            if target_col not in df.columns:
                continue

            target_results = {}

            # 市場特徴量とのRankIC
            market_ic_results = []
            for feature in market_features:
                if feature in df.columns:
                    ic = self.calculate_rank_ic(df, feature, target_col)
                    if ic is not None:
                        market_ic_results.append({
                            'feature': feature,
                            'ic': ic,
                            'abs_ic': abs(ic)
                        })

            # 最も相関の高い特徴量トップ10
            market_ic_results.sort(key=lambda x: x['abs_ic'], reverse=True)
            target_results['top_market_features'] = market_ic_results[:10]

            # RankIC統計
            if market_ic_results:
                ic_values = [r['ic'] for r in market_ic_results]
                target_results['market_ic_stats'] = {
                    'mean': np.mean(ic_values),
                    'std': np.std(ic_values),
                    'max': max(ic_values),
                    'min': min(ic_values),
                    'significant_count': sum(1 for ic in ic_values if abs(ic) > 0.05)
                }

            results[target_col] = target_results

        return results

    def calculate_rank_ic(self, df: pl.DataFrame, feature_col: str, target_col: str,
                         min_periods: int = 50) -> float | None:
        """RankIC計算"""
        try:
            # 日付ごとにRankICを計算
            ic_values = []

            # 日付でグループ化
            for date_group in df.group_by('date'):
                date_data = date_group[1]

                # 十分なデータがあるか確認
                if len(date_data) < min_periods:
                    continue

                # 欠損値除去
                valid_data = date_data.select([feature_col, target_col]).drop_nulls()
                if len(valid_data) < min_periods:
                    continue

                # ランク相関係数計算
                feature_vals = valid_data.select(feature_col).to_series().to_numpy()
                target_vals = valid_data.select(target_col).to_series().to_numpy()

                # ランクに変換
                feature_ranks = pd.Series(feature_vals).rank(method='average')
                target_ranks = pd.Series(target_vals).rank(method='average')

                # 相関係数
                corr = feature_ranks.corr(target_ranks)
                if not np.isnan(corr):
                    ic_values.append(corr)

            # 平均RankICを返す
            if ic_values:
                return np.mean(ic_values)
            else:
                return None

        except Exception as e:
            logger.error(f"Error calculating RankIC for {feature_col}: {e}")
            return None

    def analyze_feature_importance(self, df: pl.DataFrame) -> dict[str, Any]:
        """特徴量重要度分析"""
        logger.info("Analyzing feature importance")

        results = {}

        # 利用可能なターゲットを確認
        target_cols = [col for col in df.columns if col.startswith('target_')]
        market_features = [col for col in df.columns if col.startswith(('mkt_', 'beta_', 'alpha_', 'rel_'))]

        if not target_cols or not market_features:
            return results

        # 各ターゲットに対する特徴量重要度
        for target_col in target_cols:
            if target_col not in df.columns:
                continue

            # 特徴量重要度の推定（相関係数の絶対値を使用）
            importance_scores = []
            for feature in market_features:
                if feature in df.columns:
                    ic = self.calculate_rank_ic(df, feature, target_col)
                    if ic is not None:
                        importance_scores.append({
                            'feature': feature,
                            'importance': abs(ic),
                            'ic': ic
                        })

            # 重要度でソート
            importance_scores.sort(key=lambda x: x['importance'], reverse=True)

            results[target_col] = {
                'top_features': importance_scores[:20],  # トップ20
                'importance_distribution': {
                    'high_importance': sum(1 for s in importance_scores if s['importance'] > 0.1),
                    'medium_importance': sum(1 for s in importance_scores if 0.05 <= s['importance'] <= 0.1),
                    'low_importance': sum(1 for s in importance_scores if s['importance'] < 0.05)
                }
            }

        return results

    def analyze_regime_effectiveness(self, df: pl.DataFrame) -> dict[str, Any]:
        """レジーム有効性分析"""
        logger.info("Analyzing regime effectiveness")

        results = {}

        # レジームフラグを確認
        regime_flags = ['mkt_bull_200', 'mkt_trend_up', 'mkt_high_vol', 'mkt_squeeze']
        available_regimes = [flag for flag in regime_flags if flag in df.columns]

        if not available_regimes:
            logger.warning("No regime flags found")
            return results

        # 各レジームの有効性を分析
        for regime in available_regimes:
            regime_results = {}

            # レジーム別のリターン分布
            regime_data = df.select([regime, 'return_1d', 'mkt_ret_1d']).drop_nulls()

            if len(regime_data) == 0:
                continue

            # レジームON/OFFでの統計
            regime_on = regime_data.filter(pl.col(regime) == 1)
            regime_off = regime_data.filter(pl.col(regime) == 0)

            if len(regime_on) > 0 and len(regime_off) > 0:
                regime_results['return_stats'] = {
                    'regime_on': {
                        'mean_return': regime_on.select('return_1d').mean().item(),
                        'volatility': regime_on.select('return_1d').std().item(),
                        'count': len(regime_on)
                    },
                    'regime_off': {
                        'mean_return': regime_off.select('return_1d').mean().item(),
                        'volatility': regime_off.select('return_1d').std().item(),
                        'count': len(regime_off)
                    }
                }

                # レジームの予測力（平均リターンの差）
                return_diff = (regime_results['return_stats']['regime_on']['mean_return'] -
                             regime_results['return_stats']['regime_off']['mean_return'])

                regime_results['predictive_power'] = {
                    'return_difference': return_diff,
                    'effect_size': abs(return_diff) / regime_off.select('return_1d').std().item()
                }

            results[regime] = regime_results

        return results

    def create_performance_summary(self) -> dict[str, Any]:
        """パフォーマンスサマリー作成"""
        summary = {
            'market_features_count': self.results['data_info'].get('market_features_count', 0),
            'cross_features_count': self.results['data_info'].get('cross_features_count', 0),
            'rank_ic_insights': {},
            'regime_insights': {},
            'recommendations': []
        }

        # RankIC分析のインサイト
        rank_ic_data = self.results.get('rank_ic_analysis', {})
        if rank_ic_data:
            summary['rank_ic_insights'] = {
                'targets_analyzed': len(rank_ic_data),
                'significant_features_found': sum(
                    len(target_data.get('market_ic_stats', {}).get('significant_count', 0))
                    for target_data in rank_ic_data.values()
                )
            }

        # レジーム分析のインサイト
        regime_data = self.results.get('regime_effectiveness', {})
        if regime_data:
            predictive_regimes = [
                regime for regime, data in regime_data.items()
                if data.get('predictive_power', {}).get('effect_size', 0) > 0.1
            ]
            summary['regime_insights'] = {
                'effective_regimes': predictive_regimes,
                'effectiveness_count': len(predictive_regimes)
            }

        # レコメンデーション生成
        if summary['market_features_count'] >= 24:
            summary['recommendations'].append("✅ 市場特徴量が十分に実装されています")
        else:
            summary['recommendations'].append("⚠️ 市場特徴量の数が不足しています")

        if summary['cross_features_count'] >= 8:
            summary['recommendations'].append("✅ クロス特徴量が十分に実装されています")
        else:
            summary['recommendations'].append("⚠️ クロス特徴量の数が不足しています")

        significant_features = summary['rank_ic_insights'].get('significant_features_found', 0)
        if significant_features > 10:
            summary['recommendations'].append(f"✅ {significant_features}個の有意な特徴量が見つかりました")
        else:
            summary['recommendations'].append(f"⚠️ 有意な特徴量が少ないです（{significant_features}個）")

        return summary

    def save_results(self):
        """結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.output_dir / f"market_features_benchmark_{timestamp}.json"

        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Results saved to {result_file}")

        # 簡易レポート出力
        self.print_summary_report()

    def print_summary_report(self):
        """サマリーレポート出力"""
        print("\n" + "="*70)
        print("📊 TOPIX市場特徴量ベンチマーク結果")
        print("="*70)

        data_info = self.results.get('data_info', {})
        print("📈 データ情報:")
        print(f"   • 総行数: {data_info.get('total_rows', 0):,}")
        print(f"   • 特徴量数: {data_info.get('total_features', 0)}")
        print(f"   • 銘柄数: {data_info.get('unique_stocks', 0)}")
        print(f"   • 市場特徴量: {data_info.get('market_features_count', 0)}")
        print(f"   • クロス特徴量: {data_info.get('cross_features_count', 0)}")

        rank_ic = self.results.get('rank_ic_analysis', {})
        if rank_ic:
            print("\n🎯 RankIC分析:")
            for target, data in rank_ic.items():
                stats = data.get('market_ic_stats', {})
                if stats:
                    print(f"   • {target}: 平均IC={stats.get('mean', 0):.4f}, 有意特徴量={stats.get('significant_count', 0)}")

        regime = self.results.get('regime_effectiveness', {})
        if regime:
            print("\n⚡ レジーム有効性:")
            for regime_name, data in regime.items():
                power = data.get('predictive_power', {})
                effect_size = power.get('effect_size', 0)
                if effect_size > 0.1:
                    print(f"   • {regime_name}: 効果量={effect_size:.3f} ✅")
                else:
                    print(f"   • {regime_name}: 効果量={effect_size:.3f}")

        recommendations = self.results.get('performance_summary', {}).get('recommendations', [])
        if recommendations:
            print("\n💡 レコメンデーション:")
            for rec in recommendations:
                print(f"   • {rec}")

        print("="*70)


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description='TOPIX市場特徴量ベンチマーク評価')
    parser.add_argument('--data-path', type=str, required=True,
                       help='評価対象データのパス')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='出力ディレクトリ')

    args = parser.parse_args()

    # ベンチマーク実行
    benchmark = MarketFeaturesBenchmark(
        data_path=args.data_path,
        output_dir=Path(args.output_dir)
    )

    benchmark.run_full_benchmark()

    logger.info("Benchmark completed successfully!")


if __name__ == "__main__":
    main()
