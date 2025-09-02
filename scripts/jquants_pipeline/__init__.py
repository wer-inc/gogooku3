"""
JQuants Pipeline - 最適化されたデータ取得パイプライン

営業日ベースのスケジューリングとハイブリッドアプローチを実装した
J-Quants API用の最適化パイプライン。

Version: 4.0 (Optimized)
Author: Gogooku3 Engineering Team
"""

from .pipeline import JQuantsPipelineV4, PerformanceTracker

__version__ = "4.0.0"
__all__ = ["JQuantsPipelineV4", "PerformanceTracker"]