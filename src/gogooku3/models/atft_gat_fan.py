"""ATFT-GAT-FAN model implementation for financial time series prediction."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class ATFTGATFANModel(nn.Module):
    """
    ATFT-GAT-FAN: Attention-based Temporal Financial Transformer with 
    Graph Attention Network and Feature Attention Network.
    
    A hybrid model combining:
    - Temporal attention for time series patterns
    - Graph attention for stock relationships  
    - Feature attention for multi-modal financial data
    """
    
    def __init__(
        self,
        input_dim: int = 145,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        num_stocks: int = 2000,
        output_dim: int = 1,
        **kwargs
    ):
        """
        Initialize ATFT-GAT-FAN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            num_stocks: Number of stocks in universe
            output_dim: Output dimension (1 for regression)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_stocks = num_stocks
        self.output_dim = output_dim
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        logger.info(f"Initialized ATFT-GAT-FAN model with input_dim={input_dim}, "
                   f"hidden_dim={hidden_dim}, num_heads={num_heads}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        graph_adj: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of ATFT-GAT-FAN model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            graph_adj: Graph adjacency matrix (optional)
            mask: Attention mask (optional)
            
        Returns:
            Output predictions of shape (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        for layer in self.temporal_layers:
            x = layer(x, src_key_padding_mask=mask)
        
        if graph_adj is not None:
            x_graph, _ = self.graph_attention(x, x, x, key_padding_mask=mask)
            x = x + x_graph  # Residual connection
        
        attention_weights = self.feature_attention(x)
        x = x * attention_weights  # Apply feature attention
        
        x = self.layer_norm(x)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x_masked = x.masked_fill(mask_expanded, 0)
            x_pooled = x_masked.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).float()
        else:
            x_pooled = x.mean(dim=1)  # (batch_size, hidden_dim)
        
        output = self.output_layers(x_pooled)  # (batch_size, output_dim)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of attention weights
        """
        with torch.no_grad():
            x = self.input_projection(x)
            
            feature_weights = self.feature_attention(x)
            
            return {
                "feature_attention": feature_weights,
                "input_shape": x.shape
            }
    
    def configure_optimizers(self, learning_rate: float = 1e-4) -> torch.optim.Optimizer:
        """Configure optimizer for training."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device
    
    def summary(self) -> Dict[str, Any]:
        """Get model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_name": "ATFT-GAT-FAN",
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024 * 1024)  # Assuming float32
        }


def create_atft_gat_fan_model(**kwargs) -> ATFTGATFANModel:
    """Factory function to create ATFT-GAT-FAN model."""
    return ATFTGATFANModel(**kwargs)
