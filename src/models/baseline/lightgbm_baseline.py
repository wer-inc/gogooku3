"""
LightGBM Baseline for Financial Time Series Prediction
金融時系列予測のためのLightGBMベースライン

主な機能:
- 多ホライズン予測（1/5/10/20日）
- Walk-Forward検証対応
- IC/RankIC/Decile分析
- 特徴量重要度分析
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm

# 自作モジュール
from src.data.safety.cross_sectional_v2 import CrossSectionalNormalizerV2
from src.data.safety.walk_forward_v2 import WalkForwardSplitterV2
from src.metrics.financial_metrics import FinancialMetrics

logger = logging.getLogger(__name__)


class LightGBMFinancialBaseline:
    """
    LightGBM金融ベースライン
    
    特徴:
    - 多ホライズン回帰（1/5/10/20日）
    - 安全なWalk-Forward検証
    - 金融メトリクス（IC/RankIC/Decile）
    - 特徴量重要度分析
    """

    def __init__(
        self,
        prediction_horizons: list[int] = [1, 5, 10, 20],
        lgb_params: dict | None = None,
        n_splits: int = 5,
        embargo_days: int = 20,
        feature_columns: list[str] | None = None,
        target_columns: list[str] | None = None,
        normalize_features: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            prediction_horizons: 予測ホライズン
            lgb_params: LightGBMパラメータ
            n_splits: Walk-Forward分割数
            embargo_days: embargo期間
            feature_columns: 特徴量列
            target_columns: ターゲット列
            normalize_features: 特徴量正規化するか
            verbose: 詳細ログ
        """
        self.prediction_horizons = prediction_horizons
        self.n_splits = n_splits
        self.embargo_days = embargo_days
        self.feature_columns = feature_columns
        self.target_columns = target_columns or [f'return_{h}d' for h in prediction_horizons]
        self.normalize_features = normalize_features
        self.verbose = verbose

        # LightGBMパラメータ
        self.lgb_params = lgb_params or self._get_default_lgb_params()

        # モデル・結果格納
        self.models: dict[int, dict[int, lgb.LGBMRegressor]] = {}  # {horizon: {fold: model}}
        self.feature_importance: dict[int, np.ndarray] = {}
        self.normalizers: dict[int, CrossSectionalNormalizerV2] = {}
        self.results: dict[str, Any] = {}

        # 分割器・メトリクス計算器
        self.splitter = WalkForwardSplitterV2(
            n_splits=n_splits,
            embargo_days=embargo_days,
            verbose=verbose
        )
        self.metrics_calc = FinancialMetrics(
            min_stocks_per_day=20,
            decile_count=10
        )

        if self.verbose:
            logger.info(f"LightGBMFinancialBaseline: {len(prediction_horizons)} horizons, {n_splits} splits")

    def _get_default_lgb_params(self) -> dict:
        """デフォルトLightGBMパラメータ"""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbosity': -1,
            'seed': 42,
            'n_jobs': -1
        }

    def _prepare_features(self, data: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        """特徴量を準備"""
        # Pandasに変換
        if isinstance(data, pl.DataFrame):
            df = data.to_pandas()
        else:
            df = data.copy()

        # 必要列の存在確認
        required_cols = ['date', 'code']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # 特徴量列を決定
        if self.feature_columns is None:
            # 自動検出: 数値列から除外すべき列を除く
            exclude_cols = set(['date', 'code'] + self.target_columns)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = [col for col in numeric_cols if col not in exclude_cols]

        # 存在しない特徴量列を除外
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]

        if len(self.feature_columns) == 0:
            raise ValueError("No valid feature columns found")

        if self.verbose:
            logger.info(f"Using {len(self.feature_columns)} feature columns")

        # 欠損値処理
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        for col in self.target_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0.0)

        return df

    def _compute_financial_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        dates: np.ndarray
    ) -> dict[str, float]:
        """金融メトリクスを計算"""
        try:
            # IC (Information Coefficient)
            ic = self.metrics_calc.compute_information_coefficient(
                predictions, targets, dates
            )

            # RankIC (Rank Information Coefficient)
            rank_ic = self.metrics_calc.compute_rank_ic(
                predictions, targets, dates
            )

            # Decile分析
            decile_result = self.metrics_calc.compute_decile_analysis(
                predictions, targets, dates
            )

            return {
                'ic': float(ic) if not np.isnan(ic) else 0.0,
                'rank_ic': float(rank_ic) if not np.isnan(rank_ic) else 0.0,
                'long_short_spread': float(decile_result.get('long_short_spread', 0.0)),
                'valid_days': decile_result.get('valid_days', 0),
                'mse': float(mean_squared_error(targets, predictions)),
                'mae': float(mean_absolute_error(targets, predictions))
            }

        except Exception as e:
            logger.warning(f"Financial metrics computation failed: {e}")
            return {
                'ic': 0.0,
                'rank_ic': 0.0,
                'long_short_spread': 0.0,
                'valid_days': 0,
                'mse': float(mean_squared_error(targets, predictions)),
                'mae': float(mean_absolute_error(targets, predictions))
            }

    def fit(self, data: pd.DataFrame | pl.DataFrame) -> LightGBMFinancialBaseline:
        """
        Walk-Forward検証でモデルを学習
        
        Args:
            data: 学習データ
        """
        if self.verbose:
            logger.info("Starting LightGBM baseline training with Walk-Forward validation")

        # データ準備
        df = self._prepare_features(data)

        # Walk-Forward分割
        self.splitter.fit(df)
        splits = list(self.splitter.split(df))

        if len(splits) == 0:
            raise ValueError("No valid splits generated")

        # 各ホライズンで学習
        for i, horizon in enumerate(self.prediction_horizons):
            # Use configured target columns instead of hardcoded pattern
            if i < len(self.target_columns):
                target_col = self.target_columns[i]
            else:
                target_col = f'return_{horizon}d'  # fallback to default pattern

            if target_col not in df.columns:
                logger.warning(f"Target column {target_col} not found, skipping horizon {horizon}")
                continue

            self.models[horizon] = {}
            fold_results = []

            if self.verbose:
                logger.info(f"Training models for horizon {horizon}d")

            # 各フォールドで学習
            for fold, (train_idx, test_idx) in enumerate(tqdm(splits, desc=f"Horizon {horizon}d")):
                try:
                    # 学習・テストデータを分割
                    train_data = df.iloc[train_idx]
                    test_data = df.iloc[test_idx]

                    # 特徴量正規化
                    if self.normalize_features:
                        if fold not in self.normalizers:
                            normalizer = CrossSectionalNormalizerV2(
                                feature_columns=self.feature_columns
                            )
                            normalizer.fit(train_data)
                            self.normalizers[fold] = normalizer
                        else:
                            normalizer = self.normalizers[fold]

                        # 正規化実行
                        train_data_norm = normalizer.transform(train_data).to_pandas()
                        test_data_norm = normalizer.transform(test_data).to_pandas()

                        # 正規化された特徴量列名を更新
                        norm_feature_cols = [f"{col}_z" for col in self.feature_columns]
                        X_train = train_data_norm[norm_feature_cols].fillna(0.0)
                        X_test = test_data_norm[norm_feature_cols].fillna(0.0)
                    else:
                        X_train = train_data[self.feature_columns].fillna(0.0)
                        X_test = test_data[self.feature_columns].fillna(0.0)

                    y_train = train_data[target_col].fillna(0.0)
                    y_test = test_data[target_col].fillna(0.0)

                    # 無効な値をチェック
                    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
                        logger.warning(f"Invalid values in fold {fold}, horizon {horizon}")
                        continue

                    # LightGBMモデル作成
                    model = lgb.LGBMRegressor(**self.lgb_params)

                    # 学習実行
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_test, y_test)],
                        eval_names=['test'],
                        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
                    )

                    # 予測実行
                    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

                    # メトリクス計算
                    dates_test = test_data['date'].values
                    metrics = self._compute_financial_metrics(y_pred, y_test.values, dates_test)
                    metrics['fold'] = fold
                    metrics['horizon'] = horizon
                    metrics['n_train'] = len(X_train)
                    metrics['n_test'] = len(X_test)

                    fold_results.append(metrics)
                    self.models[horizon][fold] = model

                    if self.verbose:
                        logger.info(
                            f"Fold {fold}, Horizon {horizon}d: "
                            f"IC={metrics['ic']:.3f}, RankIC={metrics['rank_ic']:.3f}, "
                            f"MSE={metrics['mse']:.6f}"
                        )

                except Exception as e:
                    logger.error(f"Fold {fold}, Horizon {horizon}d failed: {e}")
                    continue

            # 特徴量重要度を集計
            if self.models[horizon]:
                importance_list = []
                for model in self.models[horizon].values():
                    importance_list.append(model.feature_importances_)

                if importance_list:
                    self.feature_importance[horizon] = np.mean(importance_list, axis=0)

            # フォールド結果を保存
            self.results[f'horizon_{horizon}d'] = fold_results

        if self.verbose:
            logger.info("LightGBM baseline training completed")

        return self

    def predict(
        self,
        data: pd.DataFrame | pl.DataFrame,
        horizon: int,
        fold: int = 0
    ) -> np.ndarray:
        """予測実行"""
        if horizon not in self.models:
            raise ValueError(f"Horizon {horizon} not trained")

        if fold not in self.models[horizon]:
            raise ValueError(f"Fold {fold} not available for horizon {horizon}")

        # データ準備
        df = self._prepare_features(data)

        # 特徴量正規化
        if self.normalize_features and fold in self.normalizers:
            normalizer = self.normalizers[fold]
            df_norm = normalizer.transform(df).to_pandas()
            norm_feature_cols = [f"{col}_z" for col in self.feature_columns]
            X = df_norm[norm_feature_cols].fillna(0.0)
        else:
            X = df[self.feature_columns].fillna(0.0)

        # 予測実行
        model = self.models[horizon][fold]
        predictions = model.predict(X, num_iteration=model.best_iteration)

        return predictions

    def get_feature_importance(self, horizon: int, top_k: int = 20) -> pd.DataFrame:
        """特徴量重要度を取得"""
        if horizon not in self.feature_importance:
            return pd.DataFrame()

        importance = self.feature_importance[horizon]
        feature_names = self.feature_columns

        # 重要度でソート
        sorted_indices = np.argsort(importance)[::-1]
        top_indices = sorted_indices[:top_k]

        return pd.DataFrame({
            'feature': [feature_names[i] for i in top_indices],
            'importance': importance[top_indices]
        })

    def get_results_summary(self) -> pd.DataFrame:
        """結果サマリーを取得"""
        summary_data = []

        for horizon_key, fold_results in self.results.items():
            if not fold_results:
                continue

            horizon = fold_results[0]['horizon']

            # fold別結果を集計
            metrics = ['ic', 'rank_ic', 'long_short_spread', 'mse', 'mae']

            for metric in metrics:
                values = [result.get(metric, 0.0) for result in fold_results]

                summary_data.append({
                    'horizon': horizon,
                    'metric': metric,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'n_folds': len(values)
                })

        return pd.DataFrame(summary_data)

    def evaluate_performance(self) -> dict[str, Any]:
        """性能評価サマリー"""
        summary = {}

        for horizon in self.prediction_horizons:
            if f'horizon_{horizon}d' not in self.results:
                continue

            fold_results = self.results[f'horizon_{horizon}d']
            if not fold_results:
                continue

            # メトリクス集計
            ic_values = [r.get('ic', 0.0) for r in fold_results]
            rank_ic_values = [r.get('rank_ic', 0.0) for r in fold_results]

            summary[f'{horizon}d'] = {
                'mean_ic': np.mean(ic_values),
                'std_ic': np.std(ic_values),
                'mean_rank_ic': np.mean(rank_ic_values),
                'std_rank_ic': np.std(rank_ic_values),
                'ic_positive_rate': np.mean([ic > 0 for ic in ic_values]),
                'rank_ic_positive_rate': np.mean([ric > 0 for ric in rank_ic_values]),
                'n_folds': len(fold_results)
            }

        return summary

    def save_models(self, save_dir: str | Path):
        """モデルを保存"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # モデル保存
        for horizon, fold_models in self.models.items():
            horizon_dir = save_dir / f"horizon_{horizon}d"
            horizon_dir.mkdir(exist_ok=True)

            for fold, model in fold_models.items():
                model_path = horizon_dir / f"fold_{fold}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

        # その他情報を保存
        metadata = {
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns,
            'prediction_horizons': self.prediction_horizons,
            'lgb_params': self.lgb_params,
            'results': self.results,
            'feature_importance': {k: v.tolist() for k, v in self.feature_importance.items()}
        }

        with open(save_dir / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        # 正規化器も保存
        if self.normalizers:
            norm_dir = save_dir / 'normalizers'
            norm_dir.mkdir(exist_ok=True)

            for fold, normalizer in self.normalizers.items():
                norm_path = norm_dir / f"fold_{fold}.pkl"
                with open(norm_path, 'wb') as f:
                    pickle.dump(normalizer, f)

        if self.verbose:
            logger.info(f"Models saved to {save_dir}")

    def load_models(self, load_dir: str | Path):
        """モデルを読み込み"""
        load_dir = Path(load_dir)

        # メタデータ読み込み
        with open(load_dir / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        self.feature_columns = metadata['feature_columns']
        self.target_columns = metadata['target_columns']
        self.prediction_horizons = metadata['prediction_horizons']
        self.lgb_params = metadata['lgb_params']
        self.results = metadata['results']
        self.feature_importance = {k: np.array(v) for k, v in metadata['feature_importance'].items()}

        # モデル読み込み
        self.models = {}
        for horizon in self.prediction_horizons:
            horizon_dir = load_dir / f"horizon_{horizon}d"
            if horizon_dir.exists():
                self.models[horizon] = {}

                for model_path in horizon_dir.glob("fold_*.pkl"):
                    fold = int(model_path.stem.split('_')[1])
                    with open(model_path, 'rb') as f:
                        self.models[horizon][fold] = pickle.load(f)

        # 正規化器読み込み
        norm_dir = load_dir / 'normalizers'
        if norm_dir.exists():
            self.normalizers = {}
            for norm_path in norm_dir.glob("fold_*.pkl"):
                fold = int(norm_path.stem.split('_')[1])
                with open(norm_path, 'rb') as f:
                    self.normalizers[fold] = pickle.load(f)

        if self.verbose:
            logger.info(f"Models loaded from {load_dir}")

        return self
