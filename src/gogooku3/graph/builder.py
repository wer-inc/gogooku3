"""Financial graph builder for creating stock relationship graphs."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logger.warning("NetworkX not available, using simplified graph implementation")


class FinancialGraphBuilder:
    """
    Financial Graph Builder for creating stock relationship graphs.
    
    Builds graphs representing relationships between stocks based on:
    - Sector/industry relationships
    - Correlation patterns
    - Market cap similarities
    - Trading volume patterns
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.3,
        sector_weight: float = 1.0,
        correlation_weight: float = 0.5,
        market_cap_weight: float = 0.3,
        max_connections: int = 50,
        **kwargs
    ):
        """
        Initialize Financial Graph Builder.
        
        Args:
            correlation_threshold: Minimum correlation for edge creation
            sector_weight: Weight for sector-based connections
            correlation_weight: Weight for correlation-based connections
            market_cap_weight: Weight for market cap similarity connections
            max_connections: Maximum connections per node
        """
        self.correlation_threshold = correlation_threshold
        self.sector_weight = sector_weight
        self.correlation_weight = correlation_weight
        self.market_cap_weight = market_cap_weight
        self.max_connections = max_connections
        self.kwargs = kwargs
        
        self.graph = None
        self.node_features = {}
        self.edge_features = {}
        self.adjacency_matrix = None
        
        logger.info(f"Initialized FinancialGraphBuilder with "
                   f"correlation_threshold={correlation_threshold}")
    
    def build_graph(
        self,
        stock_data: pd.DataFrame,
        sector_data: Optional[pd.DataFrame] = None,
        market_cap_data: Optional[pd.DataFrame] = None,
        price_column: str = 'close'
    ) -> Union[nx.Graph, Dict[str, Any]]:
        """
        Build financial graph from stock data.
        
        Args:
            stock_data: Stock price/return data with stocks as columns
            sector_data: Sector information for stocks
            market_cap_data: Market capitalization data
            price_column: Column name for price data
            
        Returns:
            NetworkX graph or dictionary representation
        """
        logger.info("Building financial relationship graph...")
        
        try:
            stocks = list(stock_data.columns)
            if price_column in stocks:
                stocks.remove(price_column)
            
            if NETWORKX_AVAILABLE:
                self.graph = nx.Graph()
                self.graph.add_nodes_from(stocks)
            else:
                self.graph = {"nodes": stocks, "edges": []}
            
            correlation_matrix = self._build_correlation_matrix(stock_data, stocks)
            
            self._add_correlation_edges(correlation_matrix, stocks)
            
            if sector_data is not None:
                self._add_sector_edges(sector_data, stocks)
            
            if market_cap_data is not None:
                self._add_market_cap_edges(market_cap_data, stocks)
            
            self.adjacency_matrix = self._build_adjacency_matrix(stocks)
            
            logger.info(f"Graph built with {len(stocks)} nodes and "
                       f"{self._get_edge_count()} edges")
            
            return self.graph
            
        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            raise
    
    def _build_correlation_matrix(
        self, 
        stock_data: pd.DataFrame, 
        stocks: List[str]
    ) -> np.ndarray:
        """Build correlation matrix from stock data."""
        logger.info("Computing stock correlations...")
        
        try:
            available_stocks = [s for s in stocks if s in stock_data.columns]
            
            if len(available_stocks) < 2:
                logger.warning("Insufficient stocks for correlation calculation")
                return np.eye(len(stocks))
            
            returns_data = stock_data[available_stocks].pct_change().dropna()
            
            correlation_matrix = returns_data.corr().values
            
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Correlation matrix calculation failed: {e}")
            return np.eye(len(stocks))
    
    def _add_correlation_edges(
        self, 
        correlation_matrix: np.ndarray, 
        stocks: List[str]
    ) -> None:
        """Add edges based on correlation patterns."""
        logger.info("Adding correlation-based edges...")
        
        try:
            n_stocks = len(stocks)
            
            for i in range(n_stocks):
                for j in range(i + 1, n_stocks):
                    if i < correlation_matrix.shape[0] and j < correlation_matrix.shape[1]:
                        correlation = abs(correlation_matrix[i, j])
                        
                        if correlation >= self.correlation_threshold:
                            weight = correlation * self.correlation_weight
                            
                            if NETWORKX_AVAILABLE:
                                self.graph.add_edge(
                                    stocks[i], 
                                    stocks[j], 
                                    weight=weight,
                                    edge_type='correlation',
                                    correlation=correlation_matrix[i, j]
                                )
                            else:
                                self.graph["edges"].append({
                                    "source": stocks[i],
                                    "target": stocks[j],
                                    "weight": weight,
                                    "edge_type": "correlation",
                                    "correlation": correlation_matrix[i, j]
                                })
            
        except Exception as e:
            logger.error(f"Correlation edge addition failed: {e}")
    
    def _add_sector_edges(self, sector_data: pd.DataFrame, stocks: List[str]) -> None:
        """Add edges based on sector relationships."""
        logger.info("Adding sector-based edges...")
        
        try:
            sector_groups = {}
            
            for stock in stocks:
                if stock in sector_data.index:
                    sector = sector_data.loc[stock, 'sector'] if 'sector' in sector_data.columns else None
                    if sector:
                        if sector not in sector_groups:
                            sector_groups[sector] = []
                        sector_groups[sector].append(stock)
            
            for sector, sector_stocks in sector_groups.items():
                for i, stock1 in enumerate(sector_stocks):
                    for stock2 in sector_stocks[i + 1:]:
                        weight = self.sector_weight
                        
                        if NETWORKX_AVAILABLE and hasattr(self.graph, 'add_edge'):
                            self.graph.add_edge(
                                stock1, 
                                stock2, 
                                weight=weight,
                                edge_type='sector',
                                sector=sector
                            )
                        elif isinstance(self.graph, dict) and "edges" in self.graph:
                            self.graph["edges"].append({
                                "source": stock1,
                                "target": stock2,
                                "weight": weight,
                                "edge_type": "sector",
                                "sector": sector
                            })
            
        except Exception as e:
            logger.error(f"Sector edge addition failed: {e}")
    
    def _add_market_cap_edges(
        self, 
        market_cap_data: pd.DataFrame, 
        stocks: List[str]
    ) -> None:
        """Add edges based on market cap similarities."""
        logger.info("Adding market cap similarity edges...")
        
        try:
            market_caps = {}
            for stock in stocks:
                if stock in market_cap_data.index:
                    cap = market_cap_data.loc[stock, 'market_cap'] if 'market_cap' in market_cap_data.columns else None
                    if cap and not pd.isna(cap):
                        try:
                            market_caps[stock] = float(cap)
                        except (ValueError, TypeError):
                            continue
            
            stock_list = list(market_caps.keys())
            for i, stock1 in enumerate(stock_list):
                for stock2 in stock_list[i + 1:]:
                    cap1, cap2 = market_caps[stock1], market_caps[stock2]
                    
                    ratio = max(cap1, cap2) / min(cap1, cap2)
                    if ratio <= 3.0:  # Similar market caps
                        similarity = 1.0 / np.log(ratio + 1)
                        weight = similarity * self.market_cap_weight
                        
                        if NETWORKX_AVAILABLE and hasattr(self.graph, 'add_edge'):
                            self.graph.add_edge(
                                stock1, 
                                stock2, 
                                weight=weight,
                                edge_type='market_cap',
                                cap_ratio=ratio
                            )
                        elif isinstance(self.graph, dict) and "edges" in self.graph:
                            self.graph["edges"].append({
                                "source": stock1,
                                "target": stock2,
                                "weight": weight,
                                "edge_type": "market_cap",
                                "cap_ratio": ratio
                            })
            
        except Exception as e:
            logger.error(f"Market cap edge addition failed: {e}")
    
    def _build_adjacency_matrix(self, stocks: List[str]) -> np.ndarray:
        """Build adjacency matrix from graph."""
        n_stocks = len(stocks)
        adjacency = np.zeros((n_stocks, n_stocks))
        
        try:
            stock_to_idx = {stock: i for i, stock in enumerate(stocks)}
            
            if NETWORKX_AVAILABLE and isinstance(self.graph, nx.Graph):
                for edge in self.graph.edges(data=True):
                    i = stock_to_idx[edge[0]]
                    j = stock_to_idx[edge[1]]
                    weight = edge[2].get('weight', 1.0)
                    adjacency[i, j] = weight
                    adjacency[j, i] = weight
            else:
                for edge in self.graph.get("edges", []):
                    if edge["source"] in stock_to_idx and edge["target"] in stock_to_idx:
                        i = stock_to_idx[edge["source"]]
                        j = stock_to_idx[edge["target"]]
                        weight = edge.get("weight", 1.0)
                        adjacency[i, j] = weight
                        adjacency[j, i] = weight
            
            return adjacency
            
        except Exception as e:
            logger.error(f"Adjacency matrix building failed: {e}")
            return adjacency
    
    def _get_edge_count(self) -> int:
        """Get number of edges in graph."""
        if NETWORKX_AVAILABLE and isinstance(self.graph, nx.Graph):
            return self.graph.number_of_edges()
        else:
            return len(self.graph.get("edges", []))
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get adjacency matrix."""
        return self.adjacency_matrix
    
    def get_node_features(self, stocks: List[str]) -> Dict[str, np.ndarray]:
        """Get node feature matrix."""
        n_stocks = len(stocks)
        
        features = {
            "degree": np.zeros(n_stocks),
            "clustering": np.zeros(n_stocks),
            "centrality": np.zeros(n_stocks)
        }
        
        try:
            if NETWORKX_AVAILABLE and isinstance(self.graph, nx.Graph):
                degree_dict = dict(self.graph.degree())
                clustering_dict = nx.clustering(self.graph)
                centrality_dict = nx.degree_centrality(self.graph)
                
                for i, stock in enumerate(stocks):
                    features["degree"][i] = degree_dict.get(stock, 0)
                    features["clustering"][i] = clustering_dict.get(stock, 0)
                    features["centrality"][i] = centrality_dict.get(stock, 0)
            
            return features
            
        except Exception as e:
            logger.error(f"Node feature extraction failed: {e}")
            return features
    
    def save_graph(self, output_path: Union[str, Path]) -> None:
        """Save graph to file."""
        output_path = Path(output_path)
        
        try:
            if NETWORKX_AVAILABLE and isinstance(self.graph, nx.Graph):
                if output_path.suffix == '.gml':
                    nx.write_gml(self.graph, output_path)
                elif output_path.suffix == '.graphml':
                    nx.write_graphml(self.graph, output_path)
                else:
                    import pickle
                    with open(output_path, 'wb') as f:
                        pickle.dump(self.graph, f)
            else:
                import json
                with open(output_path, 'w') as f:
                    json.dump(self.graph, f, indent=2)
            
            logger.info(f"Graph saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")
            raise
    
    def summary(self) -> Dict[str, Any]:
        """Get graph summary."""
        return {
            "graph_type": "FinancialGraph",
            "node_count": len(self.graph.get("nodes", [])) if not NETWORKX_AVAILABLE else self.graph.number_of_nodes(),
            "edge_count": self._get_edge_count(),
            "correlation_threshold": self.correlation_threshold,
            "sector_weight": self.sector_weight,
            "correlation_weight": self.correlation_weight,
            "market_cap_weight": self.market_cap_weight,
            "networkx_available": NETWORKX_AVAILABLE
        }


def create_financial_graph_builder(**kwargs) -> FinancialGraphBuilder:
    """Factory function to create financial graph builder."""
    return FinancialGraphBuilder(**kwargs)
