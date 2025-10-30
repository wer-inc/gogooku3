"""
金融MLモデル予測スクリプト
学習済みモデルを使って実際の株価予測を実行
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockPredictor:
    """株価予測クラス"""

    def __init__(self, model_dir: str):
        """初期化"""
        self.model_dir = Path(model_dir)
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.feature_info = None

        self._load_model_components()

    def _load_model_components(self):
        """モデルコンポーネントの読み込み"""
        logger.info("📁 モデルコンポーネント読み込み中...")

        try:
            # LightGBMモデル読み込み
            model_path = self.model_dir / 'lightgbm_model.txt'
            self.model = lgb.Booster(model_file=str(model_path))
            logger.info(f"✅ モデル読み込み: {model_path}")

            # スケーラー読み込み
            scaler_path = self.model_dir / 'scaler.pkl'
            self.scaler = joblib.load(scaler_path)
            logger.info(f"✅ スケーラー読み込み: {scaler_path}")

            # 特徴量情報読み込み
            info_path = self.model_dir / 'feature_info.json'
            with open(info_path, encoding='utf-8') as f:
                self.feature_info = json.load(f)

            self.feature_cols = self.feature_info['feature_columns']
            logger.info(f"✅ 特徴量情報読み込み: {len(self.feature_cols)}個の特徴量")

        except Exception as e:
            logger.error(f"❌ モデルコンポーネント読み込みエラー: {e}")
            raise

    def load_prediction_data(self, data_path: str, n_recent_days: int = 60):
        """予測用データの読み込み"""
        logger.info(f"📊 予測用データ読み込み: {data_path}")

        try:
            # データ読み込み
            df = pd.read_parquet(data_path)
            logger.info(f"✅ データ読み込み完了: {len(df):,}行 × {len(df.columns)}列")

            # 最新の日付を取得
            latest_date = df['date'].max()
            cutoff_date = latest_date - pd.Timedelta(days=n_recent_days)

            # 最近のデータを選択（特徴量計算に必要な期間）
            recent_df = df[df['date'] >= cutoff_date].copy()
            logger.info(f"📅 予測対象期間: {cutoff_date} ~ {latest_date}")

            # 銘柄ごとに最新のデータを選択
            latest_data = []
            for code in recent_df['Code'].unique():
                stock_data = recent_df[recent_df['Code'] == code].copy()
                stock_data = stock_data.sort_values('date')

                # 最低10日以上のデータがある銘柄のみ
                if len(stock_data) >= 10:
                    latest_record = stock_data.iloc[-1].copy()
                    latest_data.append(latest_record)

            if not latest_data:
                raise ValueError("予測可能なデータがありません")

            prediction_df = pd.DataFrame(latest_data)
            logger.info(f"🎯 予測対象銘柄数: {len(prediction_df):,}銘柄")

            return prediction_df

        except Exception as e:
            logger.error(f"❌ データ読み込みエラー: {e}")
            raise

    def calculate_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """予測用特徴量の計算"""
        logger.info("🔧 予測用特徴量計算を開始...")

        # 必要な特徴量のみ抽出
        features_df = df[['Code', 'date'] + self.feature_cols].copy()

        # 欠損値処理
        for col in self.feature_cols:
            if features_df[col].isnull().any():
                features_df[col] = features_df[col].fillna(method='ffill')

        # 特徴量スケーリング
        logger.info("📊 特徴量スケーリング実行...")
        scaled_features = self.scaler.transform(features_df[self.feature_cols])
        scaled_df = pd.DataFrame(scaled_features, columns=self.feature_cols,
                               index=features_df.index)

        # 元のデータと統合
        result_df = features_df[['Code', 'date']].copy()
        result_df[self.feature_cols] = scaled_df

        logger.info("✅ 特徴量計算完了")
        return result_df

    def predict_returns(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """リターン予測実行"""
        logger.info("🎯 リターン予測を開始...")

        # 予測実行
        predictions = self.model.predict(features_df[self.feature_cols])

        # 結果データフレーム作成
        results_df = features_df[['Code', 'date']].copy()
        results_df['predicted_return'] = predictions

        # 予測をパーセント表記に変換
        results_df['predicted_return_pct'] = results_df['predicted_return'] * 100

        # 予測方向の判定
        results_df['prediction_direction'] = np.where(
            results_df['predicted_return'] > 0, 'UP', 'DOWN'
        )

        # 予測強度の計算（絶対値）
        results_df['prediction_strength'] = np.abs(results_df['predicted_return'])

        logger.info("✅ リターン予測完了")
        return results_df

    def get_top_predictions(self, results_df: pd.DataFrame,
                          top_n: int = 20,
                          direction: str = 'both') -> pd.DataFrame:
        """トップ予測結果を取得"""
        if direction == 'up':
            filtered = results_df[results_df['prediction_direction'] == 'UP']
            sorted_results = filtered.sort_values('predicted_return', ascending=False)
        elif direction == 'down':
            filtered = results_df[results_df['prediction_direction'] == 'DOWN']
            sorted_results = filtered.sort_values('predicted_return', ascending=True)
        else:  # both
            sorted_results = results_df.sort_values('prediction_strength', ascending=False)

        return sorted_results.head(top_n)

    def analyze_predictions(self, results_df: pd.DataFrame) -> dict:
        """予測結果の分析"""
        analysis = {}

        # 全体統計
        analysis['total_stocks'] = len(results_df)
        analysis['up_predictions'] = (results_df['prediction_direction'] == 'UP').sum()
        analysis['down_predictions'] = (results_df['prediction_direction'] == 'DOWN').sum()
        analysis['avg_prediction'] = results_df['predicted_return'].mean()
        analysis['std_prediction'] = results_df['predicted_return'].std()

        # 予測分布
        analysis['prediction_ranges'] = {
            'strong_up': (results_df['predicted_return'] > 0.02).sum(),
            'moderate_up': ((results_df['predicted_return'] > 0.01) & (results_df['predicted_return'] <= 0.02)).sum(),
            'weak_up': ((results_df['predicted_return'] > 0) & (results_df['predicted_return'] <= 0.01)).sum(),
            'weak_down': ((results_df['predicted_return'] < 0) & (results_df['predicted_return'] >= -0.01)).sum(),
            'moderate_down': ((results_df['predicted_return'] < -0.01) & (results_df['predicted_return'] >= -0.02)).sum(),
            'strong_down': (results_df['predicted_return'] < -0.02).sum()
        }

        return analysis

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("🎯 金融MLモデル予測システム")
    print("=" * 80)

    # モデルディレクトリ
    model_dir = 'models/ml_model_20250829_135853'
    data_path = 'data/processed/ml_dataset_20250829_135002.parquet'

    try:
        # 予測器初期化
        print("\\n🤖 Step 1: モデル読み込み")
        predictor = StockPredictor(model_dir)

        # データ読み込み
        print("\\n📊 Step 2: 予測用データ準備")
        prediction_data = predictor.load_prediction_data(data_path, n_recent_days=60)

        # 特徴量計算
        print("\\n🔧 Step 3: 特徴量計算")
        features_df = predictor.calculate_prediction_features(prediction_data)

        # 予測実行
        print("\\n🎯 Step 4: リターン予測実行")
        results_df = predictor.predict_returns(features_df)

        # 結果分析
        print("\\n📈 Step 5: 予測結果分析")
        analysis = predictor.analyze_predictions(results_df)

        # 結果表示
        print("\\n" + "=" * 80)
        print("🎉 予測完了！")
        print("=" * 80)

        print("\\n📊 全体統計:")
        print(f"   予測対象銘柄数: {analysis['total_stocks']:,}")
        print(f"   上昇予測: {analysis['up_predictions']:,} ({analysis['up_predictions']/analysis['total_stocks']*100:.1f}%)")
        print(f"   下降予測: {analysis['down_predictions']:,} ({analysis['down_predictions']/analysis['total_stocks']*100:.1f}%)")
        print(f"   平均予測リターン: {analysis['avg_prediction']:.6f}")
        print(f"   予測リターン標準偏差: {analysis['std_prediction']:.6f}")

        print("\\n📈 予測強度分布:")
        ranges = analysis['prediction_ranges']
        print(f"   強気上昇 (>2%): {ranges['strong_up']:,}")
        print(f"   中上昇 (1-2%): {ranges['moderate_up']:,}")
        print(f"   弱上昇 (0-1%): {ranges['weak_up']:,}")
        print(f"   弱下降 (0-1%): {ranges['weak_down']:,}")
        print(f"   中下降 (1-2%): {ranges['moderate_down']:,}")
        print(f"   強気下降 (>2%): {ranges['strong_down']:,}")

        # トップ予測表示
        print("\\n🏆 トップ20予測（強度順）:")
        top_predictions = predictor.get_top_predictions(results_df, top_n=20, direction='both')
        for _i, row in top_predictions.head(10).iterrows():
            direction_icon = "📈" if row['prediction_direction'] == 'UP' else "📉"
            print(f"   {direction_icon} {row['Code']} | 予測: {row['predicted_return_pct']:.4f}% | 強度: {row['prediction_strength']:.4f}")
        # 結果保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'results/predictions_{timestamp}.csv'
        Path('results').mkdir(exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"\\n💾 結果保存: {output_path}")

        # 追加分析
        print("\\n🔍 詳細分析:")
        print(f"   予測期間: {results_df['date'].min()} ~ {results_df['date'].max()}")

        if analysis['up_predictions'] > analysis['down_predictions']:
            print("   📊 全体的に強気相場予測")
        elif analysis['up_predictions'] < analysis['down_predictions']:
            print("   📊 全体的に弱気相場予測")
        else:
            print("   📊 中立的相場予測")

        print("\\n✅ 予測完了！")

    except Exception as e:
        logger.error(f"❌ 予測エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
