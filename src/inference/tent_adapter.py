"""
TENT (Test-time ENTropy minimization) Adapter

Implements test-time adaptation for financial ML models by:
1. Updating BatchNorm statistics during inference
2. Minimizing prediction entropy for better calibration
3. Adapting to distribution shifts in real-time

Based on "Tent: Fully Test-Time Adaptation by Entropy Minimization" (Wang et al., 2020)
Adapted for financial quantile predictions and multi-horizon models.

Key Features:
- Works with quantile predictions (converts to probabilities)
- Multi-horizon support
- Configurable adaptation steps and learning rate
- BatchNorm-only updates (other parameters frozen)
- Entropy-based confidence estimation
"""

import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD

logger = logging.getLogger(__name__)


@dataclass
class TENTConfig:
    """Configuration for TENT adaptation"""

    # Adaptation parameters
    steps: int = 3                    # Number of adaptation steps per batch
    lr: float = 1e-4                  # Learning rate for adaptation
    optimizer: str = "adam"           # Optimizer type: "adam" or "sgd"

    # BatchNorm settings
    update_bn_stats: bool = True      # Update BN running statistics
    bn_momentum: float = 0.1          # BN momentum for adaptation

    # Entropy minimization
    entropy_weight: float = 1.0       # Weight for entropy loss
    confidence_threshold: float = 0.9  # Skip adaptation if confidence > threshold

    # Multi-horizon settings
    horizon_weights: Dict[str, float] = None  # Weights per horizon for multi-horizon models
    primary_horizon: str = "horizon_1d"       # Primary horizon for single-horizon mode

    # Quantile handling
    quantile_to_prob_method: str = "softmax"  # "softmax" or "normalize"
    temperature: float = 1.0                  # Temperature for probability conversion

    # Stability and safety
    min_batch_size: int = 4           # Minimum batch size for adaptation
    max_grad_norm: float = 1.0        # Gradient clipping

    # Logging and debugging
    log_adaptation: bool = False      # Log adaptation statistics
    save_adaptation_history: bool = False  # Save detailed adaptation history


class TENTAdapter:
    """
    Test-time Entropy minimization adapter for financial ML models

    Adapts pre-trained models to distribution shifts during inference by:
    1. Enabling gradient computation for BatchNorm parameters only
    2. Minimizing entropy of predictions to improve calibration
    3. Updating BatchNorm running statistics
    """

    def __init__(self, model: nn.Module, config: TENTConfig):
        self.original_model = model
        self.config = config

        # Create adapted model (copy)
        self.adapted_model = self._prepare_adapted_model(model)

        # Setup optimizer for BN parameters only
        self.optimizer = self._setup_optimizer()

        # Statistics tracking
        self.adaptation_history: List[Dict[str, float]] = []
        self.total_adaptations = 0

        # Multi-horizon weights
        if config.horizon_weights is None:
            self.horizon_weights = {
                'horizon_1d': 0.4,
                'horizon_5d': 0.3,
                'horizon_10d': 0.2,
                'horizon_20d': 0.1
            }
        else:
            self.horizon_weights = config.horizon_weights

        logger.info(f"TENT adapter initialized with {self._count_adaptable_params()} adaptable parameters")

    def _prepare_adapted_model(self, model: nn.Module) -> nn.Module:
        """Prepare model for adaptation by freezing non-BN parameters"""

        # Deep copy the model
        adapted_model = copy.deepcopy(model)

        # Set to train mode for BN updates
        adapted_model.train()

        # Freeze all parameters except BatchNorm
        for name, param in adapted_model.named_parameters():
            if 'norm' in name.lower() or 'bn' in name.lower():
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Configure BatchNorm layers for adaptation
        for module in adapted_model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                if hasattr(module, 'momentum'):
                    module.momentum = self.config.bn_momentum
                # Keep track_running_stats enabled
                if hasattr(module, 'track_running_stats'):
                    module.track_running_stats = True

        return adapted_model

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer for adaptable parameters only"""

        # Collect adaptable parameters
        adaptable_params = []
        for name, param in self.adapted_model.named_parameters():
            if param.requires_grad:
                adaptable_params.append(param)

        if len(adaptable_params) == 0:
            logger.warning("No adaptable parameters found!")
            return None

        if self.config.optimizer.lower() == "adam":
            return Adam(adaptable_params, lr=self.config.lr)
        elif self.config.optimizer.lower() == "sgd":
            return SGD(adaptable_params, lr=self.config.lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _count_adaptable_params(self) -> int:
        """Count number of adaptable parameters"""
        return sum(p.numel() for p in self.adapted_model.parameters() if p.requires_grad)

    def adapt_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Adapt model to a batch and return adapted predictions

        Args:
            batch: Input batch dictionary

        Returns:
            Dictionary containing adapted predictions and adaptation stats
        """

        # Check minimum batch size
        if self._get_batch_size(batch) < self.config.min_batch_size:
            logger.debug("Batch size too small for adaptation, using original model")
            with torch.no_grad():
                return self.original_model(batch)

        # Enable gradient computation
        self.adapted_model.train()

        # Adaptation loop
        adaptation_stats = []

        for step in range(self.config.steps):
            # Forward pass
            outputs = self.adapted_model(batch)

            # Compute entropy loss
            entropy_loss, entropy_stats = self._compute_entropy_loss(outputs)

            # Check confidence threshold
            avg_confidence = entropy_stats.get('avg_confidence', 0.0)
            if avg_confidence > self.config.confidence_threshold:
                logger.debug(f"High confidence ({avg_confidence:.3f}), skipping remaining steps")
                break

            # Backward pass
            if self.optimizer is not None and entropy_loss is not None:
                self.optimizer.zero_grad()
                entropy_loss.backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.adapted_model.parameters() if p.requires_grad],
                        self.config.max_grad_norm
                    )

                self.optimizer.step()

            # Record stats
            step_stats = {
                'step': step,
                'entropy_loss': entropy_loss.item() if entropy_loss is not None else 0.0,
                **entropy_stats
            }
            adaptation_stats.append(step_stats)

            if self.config.log_adaptation:
                logger.debug(f"TENT step {step}: entropy_loss={step_stats['entropy_loss']:.4f}, "
                           f"confidence={step_stats.get('avg_confidence', 0):.3f}")

        # Final forward pass for prediction
        self.adapted_model.eval()
        with torch.no_grad():
            final_outputs = self.adapted_model(batch)

        # Add adaptation statistics
        final_outputs['tent_stats'] = {
            'adapted': True,
            'adaptation_steps': len(adaptation_stats),
            'final_entropy_loss': adaptation_stats[-1]['entropy_loss'] if adaptation_stats else 0.0,
            'final_confidence': adaptation_stats[-1].get('avg_confidence', 0.0) if adaptation_stats else 0.0,
            'adaptable_params': self._count_adaptable_params()
        }

        # Store history if requested
        if self.config.save_adaptation_history:
            self.adaptation_history.extend(adaptation_stats)

        self.total_adaptations += 1

        return final_outputs

    def _get_batch_size(self, batch: Dict[str, torch.Tensor]) -> int:
        """Get batch size from batch dictionary"""
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and tensor.dim() > 0:
                return tensor.shape[0]
        return 0

    def _compute_entropy_loss(self, outputs: Dict[str, Any]) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """
        Compute entropy loss for adaptation

        Args:
            outputs: Model outputs

        Returns:
            Tuple of (entropy_loss, statistics_dict)
        """

        try:
            predictions = outputs.get('predictions', {})

            if not predictions:
                return None, {'avg_confidence': 0.0, 'avg_entropy': 0.0}

            total_entropy_loss = 0.0
            total_weight = 0.0
            all_confidences = []
            all_entropies = []

            # Handle multi-horizon predictions
            if isinstance(predictions, dict):
                for horizon_key, pred in predictions.items():
                    if horizon_key in self.horizon_weights:
                        # Convert quantile predictions to probabilities
                        probs = self._quantiles_to_probabilities(pred)

                        # Compute entropy
                        entropy = self._compute_entropy(probs)

                        # Weighted entropy loss
                        weight = self.horizon_weights[horizon_key]
                        total_entropy_loss += weight * entropy.mean()
                        total_weight += weight

                        # Statistics
                        confidence = self._compute_confidence(probs)
                        all_confidences.extend(confidence.cpu().numpy())
                        all_entropies.extend(entropy.cpu().numpy())

            else:
                # Single prediction
                probs = self._quantiles_to_probabilities(predictions)
                entropy = self._compute_entropy(probs)
                total_entropy_loss = entropy.mean()
                total_weight = 1.0

                confidence = self._compute_confidence(probs)
                all_confidences.extend(confidence.cpu().numpy())
                all_entropies.extend(entropy.cpu().numpy())

            # Normalize by total weight
            if total_weight > 0:
                entropy_loss = total_entropy_loss * self.config.entropy_weight
            else:
                entropy_loss = None

            # Compute statistics
            stats = {
                'avg_confidence': float(sum(all_confidences) / len(all_confidences)) if all_confidences else 0.0,
                'avg_entropy': float(sum(all_entropies) / len(all_entropies)) if all_entropies else 0.0,
                'num_predictions': len(all_confidences)
            }

            return entropy_loss, stats

        except Exception as e:
            logger.warning(f"Error computing entropy loss: {e}")
            return None, {'avg_confidence': 0.0, 'avg_entropy': 0.0}

    def _quantiles_to_probabilities(self, quantiles: torch.Tensor) -> torch.Tensor:
        """
        Convert quantile predictions to probability distribution

        Args:
            quantiles: Tensor of shape [batch, n_quantiles]

        Returns:
            Probabilities of shape [batch, n_quantiles]
        """

        if self.config.quantile_to_prob_method == "softmax":
            # Apply temperature scaling and softmax
            logits = quantiles / self.config.temperature
            probs = F.softmax(logits, dim=-1)

        elif self.config.quantile_to_prob_method == "normalize":
            # Simple normalization (ensure positive and sum to 1)
            shifted = quantiles - quantiles.min(dim=-1, keepdim=True)[0]
            probs = F.softmax(shifted, dim=-1)

        else:
            raise ValueError(f"Unknown quantile_to_prob_method: {self.config.quantile_to_prob_method}")

        return probs

    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution"""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-8
        log_probs = torch.log(probs + epsilon)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy

    def _compute_confidence(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute confidence as max probability"""
        confidence = probs.max(dim=-1)[0]
        return confidence

    def reset_adaptation(self):
        """Reset adapted model to original state"""
        self.adapted_model.load_state_dict(self.original_model.state_dict())
        if self.optimizer is not None:
            # Reset optimizer state
            self.optimizer.state = {}
        logger.info("TENT adapter reset to original model state")

    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of adaptation statistics"""
        if not self.adaptation_history:
            return {
                'total_adaptations': self.total_adaptations,
                'avg_entropy_improvement': 0.0,
                'avg_confidence': 0.0,
                'adaptable_params': self._count_adaptable_params()
            }

        # Compute summary statistics
        first_entropy = [stats['entropy_loss'] for stats in self.adaptation_history if stats['step'] == 0]
        last_entropy = [stats['entropy_loss'] for stats in self.adaptation_history
                       if stats['step'] == max(s['step'] for s in self.adaptation_history)]

        avg_entropy_improvement = 0.0
        if first_entropy and last_entropy:
            avg_first = sum(first_entropy) / len(first_entropy)
            avg_last = sum(last_entropy) / len(last_entropy)
            avg_entropy_improvement = avg_first - avg_last

        avg_confidence = sum(stats.get('avg_confidence', 0.0) for stats in self.adaptation_history) / len(self.adaptation_history)

        return {
            'total_adaptations': self.total_adaptations,
            'total_adaptation_steps': len(self.adaptation_history),
            'avg_entropy_improvement': avg_entropy_improvement,
            'avg_confidence': avg_confidence,
            'adaptable_params': self._count_adaptable_params(),
            'adaptation_history_length': len(self.adaptation_history)
        }


def create_tent_adapter(model: nn.Module,
                       steps: int = 3,
                       lr: float = 1e-4,
                       **kwargs) -> TENTAdapter:
    """
    Convenience function to create TENT adapter with common settings

    Args:
        model: Model to adapt
        steps: Number of adaptation steps
        lr: Learning rate for adaptation
        **kwargs: Additional configuration parameters

    Returns:
        Configured TENTAdapter
    """

    config = TENTConfig(
        steps=steps,
        lr=lr,
        **kwargs
    )

    return TENTAdapter(model, config)