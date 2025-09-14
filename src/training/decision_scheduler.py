"""
Decision Layer Parameter Scheduler

Implements dynamic parameter adjustment for Decision Layer during training:
- detach_signal: Stabilize early training, then enable end-to-end learning
- sharpe_weight: Gradually increase emphasis on portfolio performance
- pos_l2: Adjust position regularization as model learns

Scheduling Strategy:
- Warmup Phase (0-10 epochs): Conservative parameters for stability
- Intermediate Phase (11-30 epochs): Transition to full learning
- Final Phase (31+ epochs): Optimal performance parameters
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class DecisionScheduleConfig:
    """Configuration for Decision Layer parameter scheduling"""

    # Phase boundaries
    warmup_epochs: int = 10
    intermediate_epochs: int = 30

    # Warmup phase parameters (epochs 0-warmup_epochs)
    warmup_detach_signal: bool = True
    warmup_sharpe_weight: float = 0.05
    warmup_pos_l2: float = 1e-3
    warmup_alpha: float = 1.5

    # Intermediate phase parameters (epochs warmup_epochs+1 to intermediate_epochs)
    intermediate_detach_signal: bool = False
    intermediate_sharpe_weight: float = 0.1
    intermediate_pos_l2: float = 8e-4
    intermediate_alpha: float = 2.0

    # Final phase parameters (epochs intermediate_epochs+1 and beyond)
    final_detach_signal: bool = False
    final_sharpe_weight: float = 0.15
    final_pos_l2: float = 5e-4
    final_alpha: float = 2.5

    # Transition smoothness
    use_smooth_transitions: bool = True
    transition_window: int = 3  # Number of epochs for smooth transition

    # Logging
    log_parameter_changes: bool = True


class DecisionScheduler:
    """
    Dynamic scheduler for Decision Layer parameters

    Adjusts Decision Layer parameters based on training progress to optimize:
    1. Training stability (early phases)
    2. Learning effectiveness (intermediate phases)
    3. Final performance (late phases)
    """

    def __init__(self,
                 config: DecisionScheduleConfig,
                 decision_layer: Optional[nn.Module] = None):
        self.config = config
        self.decision_layer = decision_layer
        self.current_epoch = 0
        self.parameter_history: List[Dict[str, Any]] = []

        # Track parameter changes
        self.last_parameters = {}

        logger.info(f"DecisionScheduler initialized with config: {config}")

    def step(self, epoch: int, decision_layer: Optional[nn.Module] = None) -> Dict[str, Any]:
        """
        Update Decision Layer parameters based on current epoch

        Args:
            epoch: Current training epoch
            decision_layer: Decision layer module to update (optional if set in __init__)

        Returns:
            Dictionary of current parameters
        """
        self.current_epoch = epoch
        target_layer = decision_layer or self.decision_layer

        # Determine current phase
        phase = self._get_current_phase(epoch)

        # Calculate target parameters
        target_params = self._calculate_target_parameters(epoch, phase)

        # Apply parameters to decision layer
        if target_layer is not None:
            self._apply_parameters(target_layer, target_params)

        # Log changes
        if self.config.log_parameter_changes and self._parameters_changed(target_params):
            self._log_parameter_change(epoch, phase, target_params)

        # Store history
        self.parameter_history.append({
            'epoch': epoch,
            'phase': phase,
            **target_params
        })

        self.last_parameters = target_params.copy()
        return target_params

    def _get_current_phase(self, epoch: int) -> str:
        """Determine current training phase"""
        if epoch <= self.config.warmup_epochs:
            return "warmup"
        elif epoch <= self.config.intermediate_epochs:
            return "intermediate"
        else:
            return "final"

    def _calculate_target_parameters(self, epoch: int, phase: str) -> Dict[str, Any]:
        """Calculate target parameters for current epoch"""

        if phase == "warmup":
            return {
                'detach_signal': self.config.warmup_detach_signal,
                'sharpe_weight': self.config.warmup_sharpe_weight,
                'pos_l2': self.config.warmup_pos_l2,
                'alpha': self.config.warmup_alpha
            }

        elif phase == "intermediate":
            if self.config.use_smooth_transitions:
                # Smooth transition from warmup to intermediate
                transition_progress = (epoch - self.config.warmup_epochs) / (
                    self.config.intermediate_epochs - self.config.warmup_epochs
                )
                transition_progress = min(1.0, max(0.0, transition_progress))

                return {
                    'detach_signal': self.config.intermediate_detach_signal,  # Boolean, no interpolation
                    'sharpe_weight': self._interpolate(
                        self.config.warmup_sharpe_weight,
                        self.config.intermediate_sharpe_weight,
                        transition_progress
                    ),
                    'pos_l2': self._interpolate(
                        self.config.warmup_pos_l2,
                        self.config.intermediate_pos_l2,
                        transition_progress
                    ),
                    'alpha': self._interpolate(
                        self.config.warmup_alpha,
                        self.config.intermediate_alpha,
                        transition_progress
                    )
                }
            else:
                return {
                    'detach_signal': self.config.intermediate_detach_signal,
                    'sharpe_weight': self.config.intermediate_sharpe_weight,
                    'pos_l2': self.config.intermediate_pos_l2,
                    'alpha': self.config.intermediate_alpha
                }

        else:  # final phase
            if self.config.use_smooth_transitions and epoch <= self.config.intermediate_epochs + self.config.transition_window:
                # Smooth transition from intermediate to final
                transition_progress = (epoch - self.config.intermediate_epochs) / self.config.transition_window
                transition_progress = min(1.0, max(0.0, transition_progress))

                return {
                    'detach_signal': self.config.final_detach_signal,
                    'sharpe_weight': self._interpolate(
                        self.config.intermediate_sharpe_weight,
                        self.config.final_sharpe_weight,
                        transition_progress
                    ),
                    'pos_l2': self._interpolate(
                        self.config.intermediate_pos_l2,
                        self.config.final_pos_l2,
                        transition_progress
                    ),
                    'alpha': self._interpolate(
                        self.config.intermediate_alpha,
                        self.config.final_alpha,
                        transition_progress
                    )
                }
            else:
                return {
                    'detach_signal': self.config.final_detach_signal,
                    'sharpe_weight': self.config.final_sharpe_weight,
                    'pos_l2': self.config.final_pos_l2,
                    'alpha': self.config.final_alpha
                }

    def _interpolate(self, start: float, end: float, progress: float) -> float:
        """Linear interpolation between start and end values"""
        return start + (end - start) * progress

    def _apply_parameters(self, decision_layer: nn.Module, params: Dict[str, Any]):
        """Apply parameters to decision layer"""
        if hasattr(decision_layer, 'cfg'):
            # Update decision layer configuration
            for param_name, param_value in params.items():
                if hasattr(decision_layer.cfg, param_name):
                    setattr(decision_layer.cfg, param_name, param_value)

    def _parameters_changed(self, new_params: Dict[str, Any]) -> bool:
        """Check if parameters changed from last update"""
        if not self.last_parameters:
            return True

        for key, value in new_params.items():
            if key not in self.last_parameters or self.last_parameters[key] != value:
                return True
        return False

    def _log_parameter_change(self, epoch: int, phase: str, params: Dict[str, Any]):
        """Log parameter changes"""
        logger.info(f"📊 Decision Layer Schedule Update - Epoch {epoch} ({phase} phase)")
        for param_name, param_value in params.items():
            old_value = self.last_parameters.get(param_name, "N/A")
            if old_value != param_value:
                logger.info(f"   {param_name}: {old_value} → {param_value}")

    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.last_parameters.copy()

    def get_parameter_history(self) -> List[Dict[str, Any]]:
        """Get full parameter history"""
        return self.parameter_history.copy()

    def save_schedule_summary(self, filepath: str):
        """Save scheduling summary to file"""
        import json

        summary = {
            'config': {
                'warmup_epochs': self.config.warmup_epochs,
                'intermediate_epochs': self.config.intermediate_epochs,
                'use_smooth_transitions': self.config.use_smooth_transitions
            },
            'parameter_history': self.parameter_history,
            'final_parameters': self.last_parameters
        }

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Decision scheduler summary saved to: {filepath}")


def create_decision_scheduler_from_config(config: DictConfig,
                                        decision_layer: Optional[nn.Module] = None) -> DecisionScheduler:
    """Create DecisionScheduler from Hydra config"""

    # Extract scheduler config from main config
    sched_cfg = config.get('train', {}).get('loss', {}).get('auxiliary', {}).get('decision_layer_schedule', {})

    schedule_config = DecisionScheduleConfig(
        warmup_epochs=sched_cfg.get('warmup_epochs', 10),
        intermediate_epochs=sched_cfg.get('intermediate_epochs', 30),

        warmup_detach_signal=sched_cfg.get('warmup_detach_signal', True),
        warmup_sharpe_weight=sched_cfg.get('warmup_sharpe_weight', 0.05),
        warmup_pos_l2=sched_cfg.get('warmup_pos_l2', 1e-3),
        warmup_alpha=sched_cfg.get('warmup_alpha', 1.5),

        intermediate_detach_signal=sched_cfg.get('intermediate_detach_signal', False),
        intermediate_sharpe_weight=sched_cfg.get('intermediate_sharpe_weight', 0.1),
        intermediate_pos_l2=sched_cfg.get('intermediate_pos_l2', 8e-4),
        intermediate_alpha=sched_cfg.get('intermediate_alpha', 2.0),

        final_detach_signal=sched_cfg.get('final_detach_signal', False),
        final_sharpe_weight=sched_cfg.get('final_sharpe_weight', 0.15),
        final_pos_l2=sched_cfg.get('final_pos_l2', 5e-4),
        final_alpha=sched_cfg.get('final_alpha', 2.5),

        use_smooth_transitions=sched_cfg.get('use_smooth_transitions', True),
        transition_window=sched_cfg.get('transition_window', 3),
        log_parameter_changes=sched_cfg.get('log_parameter_changes', True)
    )

    return DecisionScheduler(schedule_config, decision_layer)


# PyTorch Lightning Callback version
class DecisionSchedulerCallback:
    """PyTorch Lightning callback for Decision Layer scheduling"""

    def __init__(self, scheduler: DecisionScheduler):
        self.scheduler = scheduler

    def on_epoch_start(self, trainer, pl_module):
        """Called at the start of each epoch"""
        current_epoch = trainer.current_epoch

        # Update decision layer parameters
        if hasattr(pl_module, 'decision_layer') and pl_module.decision_layer is not None:
            params = self.scheduler.step(current_epoch, pl_module.decision_layer)

            # Also log parameters to Lightning logger
            if hasattr(pl_module, 'log'):
                for param_name, param_value in params.items():
                    pl_module.log(f'decision_schedule/{param_name}',
                                param_value, on_epoch=True, prog_bar=False)

    def on_train_end(self, trainer, pl_module):
        """Called at the end of training"""
        # Save scheduler summary
        log_dir = trainer.log_dir or "./logs"
        summary_path = f"{log_dir}/decision_scheduler_summary.json"
        self.scheduler.save_schedule_summary(summary_path)