"""Graph neural network components.

This module contains:
- Graph attention networks (GAT)
- Graph construction utilities
- Financial graph representations
- Graph-based feature extraction
"""

from .financial_graph_builder import FinancialGraphBuilder

__all__ = [
    "FinancialGraphBuilder"
]