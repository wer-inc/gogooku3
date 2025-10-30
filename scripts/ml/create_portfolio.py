"""
金融ML予測結果からポートフォリオ構築スクリプト
予測に基づいて投資ポートフォリオを作成
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioConstructor:
    """ポートフォリオ構築クラス"""

    def __init__(self, predictions_path: str):
        """初期化"""
        self.predictions_path = predictions_path
        self.predictions_df = None
        self._load_predictions()

    def _load_predictions(self):
        """予測結果読み込み"""
        logger.info(f"📁 予測結果読み込み: {self.predictions_path}")

        try:
            self.predictions_df = pd.read_csv(self.predictions_path)
            logger.info(f"✅ 予測結果読み込み完了: {len(self.predictions_df)}銘柄")

            # データ型変換
            self.predictions_df['predicted_return'] = pd.to_numeric(self.predictions_df['predicted_return'], errors='coerce')
            self.predictions_df['predicted_return_pct'] = pd.to_numeric(self.predictions_df['predicted_return_pct'], errors='coerce')

        except Exception as e:
            logger.error(f"❌ 予測結果読み込みエラー: {e}")
            raise

    def create_long_short_portfolio(self, top_n: int = 50,
                                  long_weight: float = 0.5,
                                  short_weight: float = 0.5) -> dict:
        """ロング・ショートポートフォリオ作成"""
        logger.info(f"📊 ロング・ショートポートフォリオ作成 (各{top_n}銘柄)")

        # ロング候補（上昇予測トップ）
        long_candidates = (self.predictions_df[self.predictions_df['prediction_direction'] == 'UP']
                          .sort_values('predicted_return_pct', ascending=False)
                          .head(top_n))

        # ショート候補（下降予測トップ）
        short_candidates = (self.predictions_df[self.predictions_df['prediction_direction'] == 'DOWN']
                           .sort_values('predicted_return_pct', ascending=True)
                           .head(top_n))

        # 等加重ポートフォリオ
        long_weight_per_stock = long_weight / len(long_candidates) if len(long_candidates) > 0 else 0
        short_weight_per_stock = short_weight / len(short_candidates) if len(short_candidates) > 0 else 0

        portfolio = {
            'long_positions': [],
            'short_positions': [],
            'summary': {}
        }

        # ロングポジション
        for _, stock in long_candidates.iterrows():
            portfolio['long_positions'].append({
                'code': stock['Code'],
                'weight': long_weight_per_stock,
                'predicted_return_pct': stock['predicted_return_pct'],
                'prediction_strength': stock['prediction_strength']
            })

        # ショートポジション
        for _, stock in short_candidates.iterrows():
            portfolio['short_positions'].append({
                'code': stock['Code'],
                'weight': short_weight_per_stock,
                'predicted_return_pct': stock['predicted_return_pct'],
                'prediction_strength': stock['prediction_strength']
            })

        # サマリー計算
        portfolio['summary'] = {
            'total_stocks': len(long_candidates) + len(short_candidates),
            'long_stocks': len(long_candidates),
            'short_stocks': len(short_candidates),
            'expected_long_return': long_candidates['predicted_return_pct'].mean() if len(long_candidates) > 0 else 0,
            'expected_short_return': short_candidates['predicted_return_pct'].mean() if len(short_candidates) > 0 else 0,
            'portfolio_expected_return': (long_candidates['predicted_return_pct'].mean() * long_weight +
                                        short_candidates['predicted_return_pct'].mean() * short_weight)
        }

        logger.info("✅ ポートフォリオ作成完了")
        return portfolio

    def create_long_only_portfolio(self, top_n: int = 30) -> dict:
        """ロングオンリーポートフォリオ作成"""
        logger.info(f"📈 ロングオンリーポートフォリオ作成 ({top_n}銘柄)")

        # 上昇予測トップ銘柄
        long_candidates = (self.predictions_df[self.predictions_df['prediction_direction'] == 'UP']
                          .sort_values('predicted_return_pct', ascending=False)
                          .head(top_n))

        portfolio = {
            'positions': [],
            'summary': {}
        }

        # 等加重
        weight_per_stock = 1.0 / len(long_candidates) if len(long_candidates) > 0 else 0

        for _, stock in long_candidates.iterrows():
            portfolio['positions'].append({
                'code': stock['Code'],
                'weight': weight_per_stock,
                'predicted_return_pct': stock['predicted_return_pct'],
                'prediction_strength': stock['prediction_strength']
            })

        # サマリー
        portfolio['summary'] = {
            'total_stocks': len(long_candidates),
            'expected_return': long_candidates['predicted_return_pct'].mean() if len(long_candidates) > 0 else 0,
            'avg_prediction_strength': long_candidates['prediction_strength'].mean() if len(long_candidates) > 0 else 0
        }

        logger.info("✅ ロングオンリーポートフォリオ作成完了")
        return portfolio

    def create_sector_neutral_portfolio(self, sector_mapping: dict = None, top_n: int = 20) -> dict:
        """セクター中立ポートフォリオ作成"""
        logger.info(f"⚖️ セクター中立ポートフォリオ作成 ({top_n}銘柄)")

        # デフォルトのセクターマッピング（実際には企業情報から取得）
        if sector_mapping is None:
            # 簡易的なセクターマッピング（銘柄コードの最初の文字で分類）
            sector_mapping = {}
            for _, stock in self.predictions_df.iterrows():
                code = str(stock['Code'])
                if code.startswith('1'): sector_mapping[code] = '銀行・金融'
                elif code.startswith('2'): sector_mapping[code] = '証券・投資'
                elif code.startswith('3'): sector_mapping[code] = '建設・不動産'
                elif code.startswith('4'): sector_mapping[code] = '機械・精密機器'
                elif code.startswith('5'): sector_mapping[code] = '自動車・輸送機器'
                elif code.startswith('6'): sector_mapping[code] = '小売・サービス'
                elif code.startswith('7'): sector_mapping[code] = '情報・通信'
                elif code.startswith('8'): sector_mapping[code] = '商社'
                elif code.startswith('9'): sector_mapping[code] = 'エネルギー・素材'
                else: sector_mapping[code] = 'その他'

        # 予測結果にセクター情報を追加
        self.predictions_df['sector'] = self.predictions_df['Code'].astype(str).map(sector_mapping)

        # 各セクターのトップ銘柄を選択
        portfolio = {
            'positions': [],
            'sector_allocation': {},
            'summary': {}
        }

        # セクターごとに予測が良い銘柄を選択
        sectors = self.predictions_df['sector'].unique()

        for sector in sectors:
            sector_stocks = self.predictions_df[self.predictions_df['sector'] == sector]
            top_stocks = (sector_stocks[sector_stocks['prediction_direction'] == 'UP']
                         .sort_values('predicted_return_pct', ascending=False)
                         .head(min(top_n // len(sectors) + 1, len(sector_stocks))))

            portfolio['positions'].extend([{
                'code': stock['Code'],
                'sector': sector,
                'weight': 0,  # 後で再計算
                'predicted_return_pct': stock['predicted_return_pct']
            } for _, stock in top_stocks.iterrows()])

        # 重み付けの再計算（等加重）
        if portfolio['positions']:
            weight_per_stock = 1.0 / len(portfolio['positions'])
            for position in portfolio['positions']:
                position['weight'] = weight_per_stock

        # セクター別配分
        sector_df = pd.DataFrame(portfolio['positions'])
        portfolio['sector_allocation'] = sector_df.groupby('sector')['weight'].sum().to_dict()

        # サマリー
        portfolio['summary'] = {
            'total_stocks': len(portfolio['positions']),
            'sectors': len(sectors),
            'expected_return': np.mean([p['predicted_return_pct'] for p in portfolio['positions']])
        }

        logger.info("✅ セクター中立ポートフォリオ作成完了")
        return portfolio

    def save_portfolio(self, portfolio: dict, portfolio_type: str, output_dir: str = 'results'):
        """ポートフォリオ保存"""
        output_path = Path(output_dir) / f'portfolio_{portfolio_type}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.json'

        try:
            import json

            # NumPy型の変換
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(portfolio, f, indent=2, default=convert_numpy, ensure_ascii=False)

            logger.info(f"💾 ポートフォリオ保存: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"❌ ポートフォリオ保存エラー: {e}")
            return None

def display_portfolio(portfolio: dict, portfolio_type: str):
    """ポートフォリオ表示"""
    print(f"\n🏆 {portfolio_type.upper()} ポートフォリオ")
    print("=" * 60)

    if 'summary' in portfolio:
        summary = portfolio['summary']
        print("📊 サマリー:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(".4f")
            else:
                print(f"   {key}: {value}")

    if 'long_positions' in portfolio and portfolio['long_positions']:
        print(f"\n📈 ロングポジション ({len(portfolio['long_positions'])}銘柄):")
        for _pos in portfolio['long_positions'][:10]:  # トップ10のみ表示
            print(".4f")

    if 'short_positions' in portfolio and portfolio['short_positions']:
        print(f"\n📉 ショートポジション ({len(portfolio['short_positions'])}銘柄):")
        for _pos in portfolio['short_positions'][:10]:  # トップ10のみ表示
            print(".4f")

    if 'positions' in portfolio and portfolio['positions']:
        print(f"\n📈 ポジション ({len(portfolio['positions'])}銘柄):")
        for _pos in portfolio['positions'][:10]:  # トップ10のみ表示
            print(".4f")

    if 'sector_allocation' in portfolio and portfolio['sector_allocation']:
        print("\n⚖️ セクター配分:")
        for _sector, _weight in portfolio['sector_allocation'].items():
            print(".1%")

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("📊 金融MLポートフォリオ構築システム")
    print("=" * 80)

    # 予測結果ファイル
    predictions_path = 'results/predictions_20250829_140153.csv'

    try:
        # ポートフォリオ構築器初期化
        print("\\n🤖 Step 1: 予測結果読み込み")
        constructor = PortfolioConstructor(predictions_path)

        # 1. ロング・ショートポートフォリオ
        print("\\n📊 Step 2: ロング・ショートポートフォリオ作成")
        ls_portfolio = constructor.create_long_short_portfolio(top_n=50)
        display_portfolio(ls_portfolio, "Long-Short")

        # 2. ロングオンリーポートフォリオ
        print("\\n📈 Step 3: ロングオンリーポートフォリオ作成")
        long_portfolio = constructor.create_long_only_portfolio(top_n=30)
        display_portfolio(long_portfolio, "Long-Only")

        # 3. セクター中立ポートフォリオ
        print("\\n⚖️ Step 4: セクター中立ポートフォリオ作成")
        sector_portfolio = constructor.create_sector_neutral_portfolio(top_n=20)
        display_portfolio(sector_portfolio, "Sector-Neutral")

        # ポートフォリオ保存
        print("\\n💾 Step 5: ポートフォリオ保存")
        constructor.save_portfolio(ls_portfolio, 'long_short')
        constructor.save_portfolio(long_portfolio, 'long_only')
        constructor.save_portfolio(sector_portfolio, 'sector_neutral')

        print("\\n" + "=" * 80)
        print("🎉 ポートフォリオ構築完了！")
        print("=" * 80)
        print("\\n💡 次のステップ:")
        print("   1. バックテスト実行")
        print("   2. リスク分析")
        print("   3. パフォーマンス評価")
        print("   4. 実運用への適用")

    except Exception as e:
        logger.error(f"❌ ポートフォリオ構築エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
