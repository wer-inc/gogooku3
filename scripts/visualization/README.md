# Deep Learning Visualization Tools 📊

## Overview

This directory contains comprehensive visualization and monitoring tools for ATFT-GAT-FAN deep learning training.

## Tools Available

### 1. `visualize_training.py` - Post-Training Analysis
Comprehensive visualization of training results including:
- **Training Curves**: Loss, IC, RankIC, Sharpe Ratio over epochs
- **Prediction Analysis**: Scatter plots, distributions, residuals
- **Portfolio Performance**: Cumulative returns, drawdown, rolling Sharpe
- **Attention Weights**: GAT attention heatmaps
- **Interactive Dashboard**: Comprehensive HTML dashboard

### 2. `monitor_training.py` - Real-Time Monitoring
Live monitoring during training with:
- **Real-time Metrics**: Live tracking of loss, IC, RankIC
- **System Resources**: CPU, Memory, GPU utilization
- **Alert System**: Anomaly detection (NaN, divergence, OOM)
- **Rich Terminal UI**: Beautiful console display
- **Integration**: TensorBoard and W&B support

### 3. `compare_experiments.py` - Experiment Comparison
Compare multiple training runs:
- **Side-by-side Metrics**: Compare different configurations
- **Statistical Tests**: Significance testing between runs
- **Hyperparameter Analysis**: Impact of different settings
- **Best Model Selection**: Automatic best run identification

## Quick Start

### Post-Training Visualization
```bash
# Generate all visualizations for the last run
python scripts/visualization/visualize_training.py --run-dir runs/last --output-dir viz_output

# Launch interactive dashboard
python scripts/visualization/visualize_training.py --run-dir runs/last --dashboard
```

### Real-Time Monitoring
```bash
# Basic monitoring
python scripts/visualization/monitor_training.py --run-dir runs/current

# With TensorBoard
python scripts/visualization/monitor_training.py --run-dir runs/current --tensorboard

# With Weights & Biases
python scripts/visualization/monitor_training.py --run-dir runs/current --wandb
```

### Compare Experiments
```bash
# Compare multiple runs
python scripts/visualization/compare_experiments.py --run-dirs runs/exp1 runs/exp2 runs/exp3

# Generate comparison report
python scripts/visualization/compare_experiments.py --run-dirs runs/* --output report.html
```

## TensorBoard Integration

The training pipeline already has TensorBoard enabled. To launch:

```bash
# Launch TensorBoard
tensorboard --logdir runs/

# Or use the monitor script
python scripts/visualization/monitor_training.py --tensorboard
```

Access at: http://localhost:6006

## Weights & Biases Integration

W&B is configured in the training pipeline. To enable:

1. Install W&B:
```bash
pip install wandb
```

2. Login:
```bash
wandb login
```

3. Training will automatically log to W&B when `enable_wandb: true` in config

4. Or use monitor script:
```bash
python scripts/visualization/monitor_training.py --wandb --wandb-project "ATFT-GAT-FAN"
```

## Visualization Features

### Training Metrics
- Loss curves (train/validation)
- Information Coefficient (IC) by horizon
- Rank IC trends
- Sharpe Ratio evolution
- Learning rate schedules

### Prediction Quality
- Predicted vs Actual scatter plots
- Prediction distributions
- Residual analysis
- Quantile performance
- Time-series prediction plots

### Portfolio Analysis
- Cumulative returns
- Maximum drawdown
- Rolling Sharpe Ratio
- Win rate statistics
- Sector/industry performance

### Model Internals
- Attention weight heatmaps
- Feature importance
- Gradient flow visualization
- Activation distributions
- Weight histograms

### System Monitoring
- GPU memory usage
- Training speed (samples/sec)
- Gradient norms
- Parameter statistics
- Checkpoint sizes

## Output Formats

All visualizations support multiple output formats:
- **HTML**: Interactive Plotly charts
- **PNG/SVG**: Static images for reports
- **JSON**: Raw data export
- **CSV**: Tabular data export

## Configuration

Visualization settings can be configured via:
- Command-line arguments
- Configuration file (`viz_config.yaml`)
- Environment variables

### Example Configuration
```yaml
# viz_config.yaml
visualization:
  theme: "plotly_dark"
  save_format: ["html", "png"]
  figure_size: [1200, 800]

monitoring:
  refresh_rate: 1.0
  alert_thresholds:
    memory_percent: 90
    gpu_memory_percent: 95
    loss_divergence: 2.0

dashboard:
  port: 8050
  debug: false
  auto_reload: true
```

## Performance Tips

1. **Large Datasets**: Use sampling for faster visualization
2. **Memory Management**: Clear cache between large plots
3. **Remote Access**: Use SSH tunneling for remote dashboards
4. **Batch Processing**: Generate all plots in one run

## Troubleshooting

### Common Issues

**TensorBoard not showing data**:
- Check log directory exists
- Ensure training has `enable_tensorboard: true`
- Verify port 6006 is not in use

**Dashboard not loading**:
- Install required packages: `pip install dash plotly`
- Check port 8050 availability
- Verify HTML files were generated

**Out of Memory during visualization**:
- Reduce data sampling
- Process visualizations sequentially
- Clear matplotlib cache

## Advanced Features

### Custom Visualizations
Create custom visualizations by extending base classes:

```python
from scripts.visualization.base import BaseVisualizer

class CustomVisualizer(BaseVisualizer):
    def plot_custom_metric(self):
        # Your visualization code
        pass
```

### Automated Reports
Generate automated training reports:

```bash
# Generate comprehensive PDF report
python scripts/visualization/generate_report.py --run-dir runs/last --format pdf
```

### Integration with MLflow
Track experiments with MLflow:

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.log_metrics(metrics)
```

## Dependencies

Required packages:
```bash
pip install plotly dash pandas numpy matplotlib seaborn rich tensorboard wandb
```

Optional packages:
```bash
pip install mlflow streamlit gradio
```

## Contributing

To add new visualizations:
1. Create new plot function in appropriate module
2. Add to dashboard if interactive
3. Update documentation
4. Add tests if applicable

## Support

For issues or questions:
- Check logs in `runs/*/logs/`
- Review training configuration
- Verify data format compatibility