"""
Curriculum Learning Scheduler for Financial Time Series
金融時系列予測のカリキュラム学習スケジューラー
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CurriculumPhase:
    """カリキュラム学習のフェーズ設定"""
    name: str
    start_epoch: int
    end_epoch: int
    sequence_length: int
    prediction_horizons: List[int]
    horizon_weights: Dict[int, float]
    difficulty_features: Dict[str, Any]


class CurriculumScheduler:
    """
    金融時系列予測用カリキュラム学習スケジューラー
    
    段階的複雑性増加:
    1. 短期シーケンス → 長期シーケンス (40 → 60)
    2. 単一ホライズン → マルチホライズン ([1] → [1,5] → [1,2,3,5,10,20])
    3. 単純特徴量 → 複雑特徴量（peer、グラフ情報）
    4. 低ボラティリティ → 高ボラティリティ期間の段階導入
    """

    def __init__(
        self,
        max_epochs: int = 100,
        base_sequence_length: int = 40,
        target_sequence_length: int = 60,
        base_horizons: Optional[List[int]] = None,
        target_horizons: Optional[List[int]] = None,
        enable_difficulty_sampling: bool = True
    ):
        """
        Args:
            max_epochs: 最大エポック数
            base_sequence_length: 初期シーケンス長
            target_sequence_length: 最終シーケンス長
            base_horizons: 初期予測ホライズン
            target_horizons: 最終予測ホライズン
            enable_difficulty_sampling: 難易度ベースサンプリング
        """
        self.max_epochs = max_epochs
        self.base_sequence_length = base_sequence_length
        self.target_sequence_length = target_sequence_length
        
        self.base_horizons = base_horizons or [1]
        self.target_horizons = target_horizons or [1, 2, 3, 5, 10, 20]
        
        self.enable_difficulty_sampling = enable_difficulty_sampling
        
        # フェーズ定義を構築
        self.phases = self._build_curriculum_phases()
        self.current_phase_idx = 0
        
        logger.info(f"Initialized CurriculumScheduler with {len(self.phases)} phases")
        for i, phase in enumerate(self.phases):
            logger.info(f"Phase {i}: {phase.name} (epochs {phase.start_epoch}-{phase.end_epoch})")

    def _build_curriculum_phases(self) -> List[CurriculumPhase]:
        """カリキュラムフェーズを構築"""
        phases = []
        
        # Phase 1: シンプルベースライン (0-25%)
        phase1_end = int(self.max_epochs * 0.25)
        phases.append(CurriculumPhase(
            name="simple_baseline",
            start_epoch=0,
            end_epoch=phase1_end,
            sequence_length=self.base_sequence_length,
            prediction_horizons=self.base_horizons,
            horizon_weights={h: 1.0 for h in self.base_horizons},
            difficulty_features={
                'peer_features': False,
                'graph_features': False,
                'volatility_sampling': False,
                'augmentation': False
            }
        ))
        
        # Phase 2: シーケンス拡張 (25-50%)
        phase2_end = int(self.max_epochs * 0.5)
        intermediate_seq_len = int((self.base_sequence_length + self.target_sequence_length) / 2)
        phases.append(CurriculumPhase(
            name="sequence_extension",
            start_epoch=phase1_end,
            end_epoch=phase2_end,
            sequence_length=intermediate_seq_len,
            prediction_horizons=[1, 5],  # ホライズン追加
            horizon_weights={1: 1.0, 5: 0.8},
            difficulty_features={
                'peer_features': True,
                'graph_features': False,
                'volatility_sampling': False,
                'augmentation': False
            }
        ))
        
        # Phase 3: マルチホライズン展開 (50-75%)
        phase3_end = int(self.max_epochs * 0.75)
        mid_horizons = [1, 2, 3, 5, 10]
        phases.append(CurriculumPhase(
            name="multi_horizon",
            start_epoch=phase2_end,
            end_epoch=phase3_end,
            sequence_length=self.target_sequence_length,
            prediction_horizons=mid_horizons,
            horizon_weights={1: 1.0, 2: 0.9, 3: 0.8, 5: 0.7, 10: 0.6},
            difficulty_features={
                'peer_features': True,
                'graph_features': True,
                'volatility_sampling': True,
                'augmentation': False
            }
        ))
        
        # Phase 4: フル複雑性 (75-100%)
        phases.append(CurriculumPhase(
            name="full_complexity",
            start_epoch=phase3_end,
            end_epoch=self.max_epochs,
            sequence_length=self.target_sequence_length,
            prediction_horizons=self.target_horizons,
            horizon_weights={1: 1.0, 2: 0.9, 3: 0.8, 5: 0.7, 10: 0.6, 20: 0.5},
            difficulty_features={
                'peer_features': True,
                'graph_features': True,
                'volatility_sampling': True,
                'augmentation': True
            }
        ))
        
        return phases

    def get_phase_config(self, epoch: int) -> CurriculumPhase:
        """指定エポックのフェーズ設定を取得"""
        for i, phase in enumerate(self.phases):
            if phase.start_epoch <= epoch < phase.end_epoch:
                self.current_phase_idx = i
                return phase
        
        # 最終フェーズを返す
        self.current_phase_idx = len(self.phases) - 1
        return self.phases[-1]

    def should_update_dataset(self, epoch: int) -> bool:
        """データセット更新が必要かチェック"""
        current_phase = self.get_phase_config(epoch)
        
        # フェーズ切り替わりエポックで更新
        phase_starts = [phase.start_epoch for phase in self.phases]
        return epoch in phase_starts

    def get_sequence_length(self, epoch: int) -> int:
        """現在のシーケンス長を取得"""
        phase = self.get_phase_config(epoch)
        return phase.sequence_length

    def get_horizon_weights(self, epoch: int) -> Dict[int, float]:
        """現在のホライズン重みを取得"""
        phase = self.get_phase_config(epoch)
        return phase.horizon_weights.copy()

    def get_active_horizons(self, epoch: int) -> List[int]:
        """現在のアクティブホライズンを取得"""
        phase = self.get_phase_config(epoch)
        return phase.prediction_horizons.copy()

    def get_difficulty_features(self, epoch: int) -> Dict[str, Any]:
        """現在の難易度特徴量設定を取得"""
        phase = self.get_phase_config(epoch)
        return phase.difficulty_features.copy()

    def get_sampling_weights(
        self, 
        epoch: int, 
        volatilities: Optional[np.ndarray] = None,
        returns: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        難易度ベースサンプリング重みを取得
        
        Args:
            epoch: 現在エポック
            volatilities: ボラティリティ配列
            returns: リターン配列
            
        Returns:
            サンプリング重み（None=均等重み）
        """
        if not self.enable_difficulty_sampling:
            return None
            
        phase = self.get_phase_config(epoch)
        if not phase.difficulty_features.get('volatility_sampling', False):
            return None
            
        if volatilities is None:
            return None
        
        # ボラティリティベースの重みを計算
        weights = self._compute_volatility_weights(epoch, volatilities)
        
        # 極端リターンの重み調整
        if returns is not None:
            return_weights = self._compute_return_weights(epoch, returns)
            weights = weights * return_weights
            
        return weights / weights.sum()  # 正規化

    def _compute_volatility_weights(self, epoch: int, volatilities: np.ndarray) -> np.ndarray:
        """ボラティリティベースの重み計算"""
        # フェーズに応じた重み調整
        phase_progress = (epoch - self.phases[self.current_phase_idx].start_epoch) / \
                        max(self.phases[self.current_phase_idx].end_epoch - 
                            self.phases[self.current_phase_idx].start_epoch, 1)
        
        # 初期は低ボラティリティを重視、後期は高ボラティリティも含める
        vol_percentile = np.percentile(volatilities, [25, 50, 75])
        
        weights = np.ones_like(volatilities)
        
        if phase_progress < 0.3:
            # 低ボラティリティ重視
            weights[volatilities > vol_percentile[1]] *= 0.5
            weights[volatilities > vol_percentile[2]] *= 0.2
        elif phase_progress < 0.7:
            # 中程度のバランス
            weights[volatilities < vol_percentile[0]] *= 0.8
            weights[volatilities > vol_percentile[2]] *= 0.8
        else:
            # 高ボラティリティも積極的に
            weights[volatilities > vol_percentile[1]] *= 1.2
            weights[volatilities > vol_percentile[2]] *= 1.5
        
        return weights

    def _compute_return_weights(self, epoch: int, returns: np.ndarray) -> np.ndarray:
        """リターンベースの重み計算（極端値の段階導入）"""
        abs_returns = np.abs(returns)
        return_percentile = np.percentile(abs_returns, [90, 95, 99])
        
        weights = np.ones_like(returns)
        
        # 極端リターンの段階導入
        phase_progress = (epoch - self.phases[self.current_phase_idx].start_epoch) / \
                        max(self.phases[self.current_phase_idx].end_epoch - 
                            self.phases[self.current_phase_idx].start_epoch, 1)
        
        if phase_progress < 0.5:
            # 極端値を抑制
            weights[abs_returns > return_percentile[0]] *= 0.5
            weights[abs_returns > return_percentile[1]] *= 0.2
        else:
            # 極端値も学習に含める
            weights[abs_returns > return_percentile[0]] *= 1.1
            weights[abs_returns > return_percentile[1]] *= 1.2
        
        return weights

    def get_augmentation_config(self, epoch: int) -> Dict[str, Any]:
        """現在のデータ拡張設定を取得"""
        phase = self.get_phase_config(epoch)
        
        if not phase.difficulty_features.get('augmentation', False):
            return {'enabled': False}
        
        # フェーズ進行度に応じた拡張強度
        phase_progress = (epoch - phase.start_epoch) / max(phase.end_epoch - phase.start_epoch, 1)
        
        return {
            'enabled': True,
            'noise_scale': 0.002 + 0.003 * phase_progress,  # 0.002 → 0.005
            'feature_dropout_p': 0.05 + 0.05 * phase_progress,  # 0.05 → 0.10
            'time_warping': phase_progress > 0.5,
            'magnitude_warping': phase_progress > 0.7
        }

    def get_learning_rate_multiplier(self, epoch: int) -> float:
        """フェーズに応じた学習率調整"""
        phase = self.get_phase_config(epoch)
        
        # フェーズ開始時は学習率を少し上げる
        if epoch == phase.start_epoch and epoch > 0:
            return 1.2
        
        return 1.0

    def log_phase_info(self, epoch: int):
        """現在のフェーズ情報をログ出力"""
        phase = self.get_phase_config(epoch)
        
        if epoch == phase.start_epoch:
            logger.info(f"Starting curriculum phase: {phase.name}")
            logger.info(f"  Sequence length: {phase.sequence_length}")
            logger.info(f"  Prediction horizons: {phase.prediction_horizons}")
            logger.info(f"  Horizon weights: {phase.horizon_weights}")
            logger.info(f"  Difficulty features: {phase.difficulty_features}")

    def get_phase_summary(self) -> Dict[str, Any]:
        """カリキュラム全体のサマリー"""
        return {
            'total_phases': len(self.phases),
            'current_phase': self.current_phase_idx,
            'max_epochs': self.max_epochs,
            'sequence_progression': f"{self.base_sequence_length} → {self.target_sequence_length}",
            'horizon_progression': f"{self.base_horizons} → {self.target_horizons}",
            'phases': [
                {
                    'name': phase.name,
                    'epochs': f"{phase.start_epoch}-{phase.end_epoch}",
                    'sequence_length': phase.sequence_length,
                    'horizons': phase.prediction_horizons
                }
                for phase in self.phases
            ]
        }


def create_simple_curriculum(max_epochs: int = 50) -> CurriculumScheduler:
    """
    シンプルなカリキュラムを作成（少ないエポック用）
    
    Args:
        max_epochs: 最大エポック数
        
    Returns:
        シンプル設定のCurriculumScheduler
    """
    return CurriculumScheduler(
        max_epochs=max_epochs,
        base_sequence_length=40,
        target_sequence_length=60,
        base_horizons=[1, 5],
        target_horizons=[1, 5, 10],
        enable_difficulty_sampling=True
    )


def create_research_curriculum(max_epochs: int = 200) -> CurriculumScheduler:
    """
    研究用の詳細カリキュラムを作成（多エポック研究用）
    
    Args:
        max_epochs: 最大エポック数
        
    Returns:
        詳細設定のCurriculumScheduler
    """
    return CurriculumScheduler(
        max_epochs=max_epochs,
        base_sequence_length=20,
        target_sequence_length=60,
        base_horizons=[1],
        target_horizons=[1, 2, 3, 5, 10, 20],
        enable_difficulty_sampling=True
    )