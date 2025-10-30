"""
金融MLモデル学習スクリプト
大規模株価データセット（3,977銘柄×5年）を使用したLightGBMベースの学習
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialMLTrainer:
    """金融時系列データ向けML学習クラス"""

    def __init__(self, data_path: str):
        """初期化"""
        self.data_path = data_path
        self.df = None
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.results = {}

    def load_and_preprocess_data(self):
        """データ読み込みと前処理"""
        logger.info("📁 データ読み込みを開始...")

        # データ読み込み
        self.df = pd.read_parquet(self.data_path)
        logger.info(f"✅ データ読み込み完了: {len(self.df):,}行 × {len(self.df.columns)}列")

        # 日付ソート
        self.df = self.df.sort_values(['date', 'Code']).reset_index(drop=True)

        # 特徴量カラム定義
        self.feature_cols = [col for col in self.df.columns
                           if col not in ['Code', 'date', 'target', 'year', 'month']]

        logger.info(f"🔧 特徴量数: {len(self.feature_cols)}")
        logger.info(f"📋 特徴量: {self.feature_cols}")

        # 前処理
        self._preprocess_features()

        # 目的変数の外れ値処理
        self._process_target_outliers()

        logger.info("✅ データ前処理完了")
        return self.df

    def _preprocess_features(self):
        """特徴量の前処理"""
        logger.info("🔧 特徴量前処理を開始...")

        # 欠損値補完（前値補完）
        for col in self.feature_cols:
            if self.df[col].isnull().any():
                null_count = self.df[col].isnull().sum()
                self.df[col] = self.df[col].fillna(method='ffill')
                logger.info(f"   {col}: {null_count}件の欠損値を補完")

        # 無限値の処理
        for col in self.feature_cols:
            inf_mask = np.isinf(self.df[col])
            if inf_mask.any():
                inf_count = inf_mask.sum()
                self.df.loc[inf_mask, col] = np.nan
                self.df[col] = self.df[col].fillna(method='ffill')
                logger.info(f"   {col}: {inf_count}件の無限値を補完")

        # 特徴量スケーリング（銘柄ごとに）
        logger.info("📊 特徴量スケーリングを開始...")
        self.scaler = RobustScaler()  # 外れ値に強いスケーラー

        # 銘柄ごとにスケーリング
        scaled_features = []
        for code in self.df['Code'].unique():
            stock_data = self.df[self.df['Code'] == code].copy()
            if len(stock_data) > 1:
                scaled_values = self.scaler.fit_transform(stock_data[self.feature_cols])
                scaled_df = pd.DataFrame(scaled_values, columns=self.feature_cols,
                                       index=stock_data.index)
                scaled_features.append(scaled_df)

        if scaled_features:
            scaled_features_df = pd.concat(scaled_features)
            self.df[self.feature_cols] = scaled_features_df

        logger.info("✅ 特徴量前処理完了")

    def _process_target_outliers(self):
        """目的変数の外れ値処理"""
        logger.info("🎯 目的変数外れ値処理を開始...")

        # 極端な値の統計
        original_mean = self.df['target'].mean()
        original_std = self.df['target'].std()

        # 3σ以上の値をクリッピング（ただし±20%以内に制限）
        upper_limit = min(0.20, original_mean + 3 * original_std)
        lower_limit = max(-0.20, original_mean - 3 * original_std)

        outlier_mask = (self.df['target'] > upper_limit) | (self.df['target'] < lower_limit)
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            logger.info(f"   外れ値検出: {outlier_count:,}件 ({outlier_count/len(self.df)*100:.2f}%)")
            self.df.loc[outlier_mask, 'target'] = np.clip(
                self.df.loc[outlier_mask, 'target'], lower_limit, upper_limit
            )

        processed_mean = self.df['target'].mean()
        processed_std = self.df['target'].std()

        logger.info(f"   処理後 - 平均: {processed_mean:.6f}, 標準偏差: {processed_std:.6f}")
        logger.info("✅ 目的変数処理完了")

    def create_time_series_splits(self, n_splits=5):
        """時系列クロスバリデーション用データ分割"""
        logger.info("📅 時系列クロスバリデーション分割を作成...")

        # 日付でソート
        self.df = self.df.sort_values('date').reset_index(drop=True)

        # 時系列分割
        tscv = TimeSeriesSplit(n_splits=n_splits)

        splits = []
        dates = self.df['date'].unique()

        for i, (train_index, test_index) in enumerate(tscv.split(dates)):
            train_dates = dates[train_index]
            test_dates = dates[test_index]

            train_mask = self.df['date'].isin(train_dates)
            test_mask = self.df['date'].isin(test_dates)

            X_train = self.df.loc[train_mask, self.feature_cols]
            y_train = self.df.loc[train_mask, 'target']
            X_test = self.df.loc[test_mask, self.feature_cols]
            y_test = self.df.loc[test_mask, 'target']

            splits.append({
                'fold': i + 1,
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'train_dates': (train_dates.min(), train_dates.max()),
                'test_dates': (test_dates.min(), test_dates.max())
            })

            logger.info(f"   Fold {i+1}: 学習={len(X_train):,}, テスト={len(X_test):,}")

        return splits

    def train_lightgbm_model(self, splits):
        """LightGBMモデルの学習"""
        logger.info("🚀 LightGBMモデル学習を開始...")

        # LightGBMパラメータ（金融時系列データ向け）
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }

        fold_results = []

        for split in splits:
            logger.info(f"📊 Fold {split['fold']} 学習開始...")

            # LightGBMデータセット作成
            train_data = lgb.Dataset(split['X_train'], label=split['y_train'])
            valid_data = lgb.Dataset(split['X_test'], label=split['y_test'], reference=train_data)

            # モデル学習（LightGBM 4.x対応）
            model = lgb.train(
                params,
                train_data,
                num_boost_round=1000,
                valid_sets=[train_data, valid_data],
                valid_names=['train', 'valid'],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50),
                    lgb.log_evaluation(period=0)  # verbose_eval=False相当
                ]
            )

            # 予測
            y_pred = model.predict(split['X_test'], num_iteration=model.best_iteration)

            # 評価指標計算
            metrics = self._calculate_metrics(split['y_test'], y_pred)

            fold_result = {
                'fold': split['fold'],
                'model': model,
                'metrics': metrics,
                'train_dates': split['train_dates'],
                'test_dates': split['test_dates']
            }

            fold_results.append(fold_result)

            logger.info(f"   Fold {split['fold']} 完了 - RMSE: {metrics['rmse']:.6f}")

        # 最適なモデルを選択（最後のFoldを使用）
        self.model = fold_results[-1]['model']

        return fold_results

    def _calculate_metrics(self, y_true, y_pred):
        """評価指標計算"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 金融特化指標
        returns = y_true

        # Sharpe比（単純化）
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)  # 年間化
        else:
            sharpe = 0

        # 最大ドローダウン
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # 勝率
        win_rate = (np.sign(y_pred) == np.sign(y_true)).mean()

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'sample_size': len(y_true)
        }

    def save_model_and_results(self, fold_results):
        """モデルと結果の保存"""
        logger.info("💾 モデルと結果を保存...")

        # ディレクトリ作成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = Path('models') / f'ml_model_{timestamp}'
        model_dir.mkdir(parents=True, exist_ok=True)

        # モデル保存
        model_path = model_dir / 'lightgbm_model.txt'
        self.model.save_model(str(model_path))
        logger.info(f"   モデル保存: {model_path}")

        # スケーラー保存
        scaler_path = model_dir / 'scaler.pkl'
        import joblib
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"   スケーラー保存: {scaler_path}")

        # 特徴量情報保存
        feature_info = {
            'feature_columns': self.feature_cols,
            'n_features': len(self.feature_cols),
            'n_stocks': self.df['Code'].nunique(),
            'date_range': (self.df['date'].min(), self.df['date'].max()),
            'total_samples': len(self.df)
        }

        feature_path = model_dir / 'feature_info.json'
        import json
        with open(feature_path, 'w', encoding='utf-8') as f:
            json.dump(feature_info, f, indent=2, default=str)
        logger.info(f"   特徴量情報保存: {feature_path}")

        # 評価結果保存
        results_df = pd.DataFrame([{
            'fold': r['fold'],
            'rmse': r['metrics']['rmse'],
            'mae': r['metrics']['mae'],
            'r2': r['metrics']['r2'],
            'sharpe': r['metrics']['sharpe'],
            'max_drawdown': r['metrics']['max_drawdown'],
            'win_rate': r['metrics']['win_rate'],
            'train_start': r['train_dates'][0],
            'train_end': r['train_dates'][1],
            'test_start': r['test_dates'][0],
            'test_end': r['test_dates'][1]
        } for r in fold_results])

        results_path = model_dir / 'evaluation_results.csv'
        results_df.to_csv(results_path, index=False)
        logger.info(f"   評価結果保存: {results_path}")

        # 集計結果表示
        print("\n🎯 学習結果サマリー:")
        print(f"   平均RMSE: {results_df['rmse'].mean():.6f}")
        print(f"   平均MAE: {results_df['mae'].mean():.6f}")
        print(f"   平均R²: {results_df['r2'].mean():.6f}")
        print(f"   平均勝率: {results_df['win_rate'].mean():.4f}")
        print(f"   平均Sharpe比: {results_df['sharpe'].mean():.4f}")

        return model_dir

    def plot_feature_importance(self, model_dir):
        """特徴量重要度の可視化"""
        logger.info("📊 特徴量重要度を可視化...")

        # 特徴量重要度取得
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()

        # データフレーム作成
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        # プロット
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature')
        plt.title('Top 20 Feature Importance (LightGBM)')
        plt.xlabel('Importance (Gain)')
        plt.ylabel('Features')
        plt.tight_layout()

        # 保存
        plot_path = model_dir / 'feature_importance.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"   特徴量重要度プロット保存: {plot_path}")

        return importance_df

def main():
    """メイン実行関数"""
    print("=" * 80)
    print("🚀 金融MLモデル学習システム")
    print("=" * 80)

    # データセットパス
    data_path = 'data/processed/ml_dataset_20250829_135002.parquet'

    # トレーナー初期化
    trainer = FinancialMLTrainer(data_path)

    try:
        # 1. データ読み込みと前処理
        print("\\n📁 Step 1: データ読み込みと前処理")
        trainer.load_and_preprocess_data()

        # 2. 時系列クロスバリデーション分割
        print("\\n📅 Step 2: 時系列クロスバリデーション")
        splits = trainer.create_time_series_splits(n_splits=3)  # 計算時間を考慮して3分割

        # 3. モデル学習
        print("\\n🚀 Step 3: LightGBMモデル学習")
        fold_results = trainer.train_lightgbm_model(splits)

        # 4. 結果保存と可視化
        print("\\n💾 Step 4: 結果保存")
        model_dir = trainer.save_model_and_results(fold_results)

        # 5. 特徴量重要度
        print("\\n📊 Step 5: 特徴量重要度分析")
        importance_df = trainer.plot_feature_importance(model_dir)

        # 結果表示
        print("\\n" + "=" * 80)
        print("🎉 学習完了！")
        print(f"📁 モデル保存先: {model_dir}")
        print(f"📊 学習データ: {trainer.df['Code'].nunique():,}銘柄 × {len(trainer.df):,}サンプル")
        print(f"🔧 特徴量数: {len(trainer.feature_cols)}")
        print("=" * 80)

        # トップ特徴量表示
        print("\\n🏆 トップ10特徴量:")
        for i, row in importance_df.head(10).iterrows():
            print(f"   {i+1:2d}. {row['feature']:<15} {row['importance']:.4f}")
    except Exception as e:
        logger.error(f"❌ 学習エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
