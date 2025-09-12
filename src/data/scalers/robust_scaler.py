"""
Robust target scaler with fallback for zero-std cases
ターゲット標準化の修正版
"""

import os
import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RobustZScaler:
    """
    ロバストなZ-score正規化
    std≈0の場合はIQRベースの代替手法に自動フォールバック
    """

    def __init__(self, min_std: float = 1e-6, fallback_std: float = 1.0):
        """
        Args:
            min_std: 最小許容標準偏差
            fallback_std: ロバスト手法でも失敗した場合のデフォルト標準偏差
        """
        self.min_std = min_std
        self.fallback_std = fallback_std
        self.mu: float = 0.0
        self.sigma: float = 1.0
        self.robust: bool = False
        self.fitted: bool = False

    def fit(self, y: np.ndarray, horizon: Optional[int] = None) -> "RobustZScaler":
        """
        訓練データでスケーラーをフィット

        Args:
            y: ターゲット配列
            horizon: Optional horizon number for horizon-specific handling

        Returns:
            self
        """
        y = np.asarray(y, dtype=np.float64)
        y_clean = y[np.isfinite(y)]  # NaN/Infを除外

        # Apply winsorization if enabled
        winsor_pct = float(os.getenv("TARGET_WINSOR_PCT", "0.0"))

        # Special handling for horizon 2 if needed
        if horizon == 2:
            extra_winsor = float(os.getenv("HORIZON_2_EXTRA_WINSOR", "0.0"))
            if extra_winsor > 0:
                winsor_pct = max(winsor_pct, extra_winsor)
                logger.info(f"Horizon 2: Using enhanced winsorization {winsor_pct:.2%}")

        if winsor_pct > 0 and len(y_clean) > 0:
            # Always use percentile-based winsorization for consistency
            lower = np.percentile(y_clean, winsor_pct * 100)
            upper = np.percentile(y_clean, 100 - winsor_pct * 100)

            original_std = np.std(y_clean)
            y_clean = np.clip(y_clean, lower, upper)
            new_std = np.std(y_clean)
            logger.info(
                f"Applied winsorization: {winsor_pct:.2%} on each tail, bounds=[{lower:.4f}, {upper:.4f}]"
            )
            logger.info(f"Std reduction: {original_std:.4f} → {new_std:.4f}")

        if len(y_clean) == 0:
            logger.warning("All values are NaN/Inf, using defaults")
            self.mu = 0.0
            self.sigma = self.fallback_std
            self.robust = True
            self.fitted = True
            return self

        # Check if we should force MAD-based scaling
        force_mad = int(os.getenv("TARGET_SCALER_USE_MAD", "0"))
        use_iqr_for_h2 = horizon == 2 and int(os.getenv("HORIZON_2_USE_IQR", "0"))

        if force_mad or use_iqr_for_h2:
            # Skip normal calculation and go straight to robust methods
            logger.info(
                f"Using robust scaling for horizon {horizon} (force_mad={force_mad}, h2_iqr={use_iqr_for_h2})"
            )
            std = 0.0  # Force robust path
            mu = float(np.median(y_clean))
        else:
            # 通常のmean/std計算（不偏推定量）
            mu = float(np.mean(y_clean))
            std = float(
                np.std(y_clean, ddof=1) if len(y_clean) > 1 else np.std(y_clean)
            )

        # std検証とロバスト代替
        if not np.isfinite(std) or std < self.min_std or force_mad or use_iqr_for_h2:
            logger.warning(
                f"Standard deviation too small ({std:.6f}), "
                f"trying MAD-based robust method first"
            )

            # MAD (Median Absolute Deviation) ベースの推定を最初に試す
            median = float(np.median(y_clean))
            mad = float(np.median(np.abs(y_clean - median)))
            # MAD → 標準偏差換算（正規分布の場合 σ ≈ 1.4826 * MAD）
            mad_std = 1.4826 * mad

            if np.isfinite(mad_std) and mad_std >= self.min_std:
                mu = median
                std = mad_std
                logger.info(f"Using MAD scaling: median={mu:.6f}, mad_std={std:.6f}")
            else:
                # MADが失敗したらIQRベースのロバスト推定
                q1, q3 = np.percentile(y_clean, [25, 75])
                mu = float(np.median(y_clean))
                iqr = float(q3 - q1)

                # IQR → 標準偏差換算（正規分布なら IQR ≈ 1.349σ）
                std = iqr / 1.349

                if np.isfinite(std) and std >= self.min_std:
                    logger.info(
                        f"Using IQR scaling: median={mu:.6f}, iqr_std={std:.6f}"
                    )
                else:
                    # それでも小さすぎる場合は最小値を使用
                    logger.warning(
                        f"Both MAD and IQR failed (mad_std={mad_std:.6f}, iqr_std={std:.6f}), "
                        f"using fallback={self.fallback_std}"
                    )
                    std = self.fallback_std

            self.robust = True
        else:
            self.robust = False

        self.mu = mu
        self.sigma = std
        self.fitted = True

        logger.info(
            f"Scaler fitted: mean={self.mu:.6f}, std={self.sigma:.6f}, "
            f"robust={self.robust}"
        )

        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        正規化を適用

        Args:
            y: 変換するデータ

        Returns:
            正規化されたデータ
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        y = np.asarray(y, dtype=np.float32)
        z = (y - self.mu) / self.sigma
        # NaN/Inf をゼロに吸収（後段の損失側でもマスクするためダブルガード）
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return z

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        """
        フィットと変換を同時に実行
        """
        self.fit(y)
        return self.transform(y)

    def validate_transform(self, y: np.ndarray, name: str = "data") -> Dict[str, float]:
        """
        変換後のデータを検証

        Args:
            y: 変換後のデータ
            name: データ名（ログ用）

        Returns:
            統計情報の辞書
        """
        y_clean = y[np.isfinite(y)]

        if len(y_clean) == 0:
            logger.error(f"[{name}] All values are NaN/Inf after transform!")
            return {"mean": np.nan, "std": np.nan, "valid": False}

        mean = float(np.mean(y_clean))
        std = float(np.std(y_clean, ddof=1))

        # Check for extreme normalization failure
        max_std = float(os.getenv("TARGET_MAX_STD", "10.0"))
        if std > max_std:
            logger.error(
                f"[{name}] CRITICAL: Normalization failure detected! "
                f"std={std:.2f} exceeds max={max_std}"
            )
            # Check if we should apply emergency clipping
            if int(os.getenv("NORM_FAILURE_FALLBACK", "0")):
                logger.warning(f"[{name}] Applying emergency clipping at ±{max_std}σ")
                y_clean = np.clip(y_clean, -max_std, max_std)
                mean = float(np.mean(y_clean))
                std = float(np.std(y_clean, ddof=1))
                logger.info(f"[{name}] After clipping: mean={mean:.4f}, std={std:.4f}")

        # 期待値チェック（z-score後は mean≈0, std≈1）
        # Relaxed for horizon-specific issues
        valid = abs(mean) < 0.5 and 0.5 < std < 2.0

        if not valid:
            logger.warning(
                f"[{name}] Transformed stats out of range: "
                f"mean={mean:.4f}, std={std:.4f}"
            )
        else:
            logger.info(f"[{name}] Transform OK: mean={mean:.4f}, std={std:.4f}")

        return {"mean": mean, "std": std, "valid": valid}


def create_target_scalers(
    train_targets: Dict[int, np.ndarray],
    val_targets: Optional[Dict[int, np.ndarray]] = None,
    test_targets: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[int, RobustZScaler]:
    """
    各ホライズンのターゲットスケーラーを作成

    Args:
        train_targets: 訓練ターゲット {horizon: array}
        val_targets: 検証ターゲット（検証用）
        test_targets: テストターゲット（検証用）

    Returns:
        {horizon: scaler} の辞書
    """
    scalers = {}

    for horizon, y_train in train_targets.items():
        logger.info(f"\n=== Fitting scaler for horizon {horizon} ===")

        # スケーラーをフィット（訓練データのみ）
        scaler = RobustZScaler()
        scaler.fit(y_train, horizon=horizon)  # Pass horizon for special handling
        scalers[horizon] = scaler

        # 変換と検証
        logger.info(f"Validating transformations for horizon {horizon}:")

        # 訓練データの検証
        y_train_norm = scaler.transform(y_train)
        train_stats = scaler.validate_transform(y_train_norm, f"train_h{horizon}")

        # 検証データの検証
        if val_targets and horizon in val_targets:
            y_val_norm = scaler.transform(val_targets[horizon])
            val_stats = scaler.validate_transform(y_val_norm, f"val_h{horizon}")
            logger.debug(
                f"Validation data normalized for horizon {horizon}: {val_stats}"
            )

        # テストデータの検証
        if test_targets and horizon in test_targets:
            y_test_norm = scaler.transform(test_targets[horizon])
            test_stats = scaler.validate_transform(y_test_norm, f"test_h{horizon}")
            logger.debug(f"Test data normalized for horizon {horizon}: {test_stats}")

        # 訓練データで正規化が失敗した場合はエラー（unless skip configured）
        if not train_stats["valid"]:
            if int(os.getenv("TARGET_NORM_CHECK_SKIP", "0")):
                logger.error(
                    f"Target normalization failed for horizon {horizon}: "
                    f"train mean={train_stats['mean']:.4f}, std={train_stats['std']:.4f} "
                    f"(continuing due to TARGET_NORM_CHECK_SKIP=1)"
                )
            else:
                raise ValueError(
                    f"Target normalization failed for horizon {horizon}: "
                    f"train mean={train_stats['mean']:.4f}, std={train_stats['std']:.4f}"
                )

    return scalers
