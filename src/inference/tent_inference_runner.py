"""
TENT Inference Runner

Implements complete TENT-based inference pipeline:
1. Load pre-trained model
2. Setup TENT adapter
3. Process batches with test-time adaptation
4. Save adapted predictions with confidence scores

Usage:
    python -m src.inference.tent_inference_runner \
        --model-path models/best_model.pth \
        --input-path data/test.parquet \
        --output-path predictions.parquet \
        --tent-steps 3 \
        --tent-lr 1e-4
"""

import argparse
import logging
from pathlib import Path
import time
from typing import Dict, Any, List, Optional, Union
import sys

import torch
import torch.nn as nn
import polars as pl
import numpy as np
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: Union[str, Path]) -> nn.Module:
    """Load pre-trained model from checkpoint"""
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Extract model architecture information
        if 'hyper_parameters' in checkpoint:
            config = checkpoint['hyper_parameters']
        elif 'config' in checkpoint:
            config = checkpoint['config']
        else:
            raise ValueError("Model config not found in checkpoint")

        # Reconstruct model (this is simplified - in practice, need proper config handling)
        from ...atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN

        model = ATFT_GAT_FAN(config)

        # Load state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            # Remove 'model.' prefix if present (from Lightning)
            state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()

        logger.info("✅ Model loaded successfully")
        return model

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def load_data(input_path: Union[str, Path]) -> pl.DataFrame:
    """Load input data for inference"""
    try:
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        logger.info(f"Loading data from: {input_path}")

        if input_path.suffix == '.parquet':
            df = pl.read_parquet(input_path)
        elif input_path.suffix == '.csv':
            df = pl.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")

        logger.info(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df

    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        raise


def prepare_batch(df: pl.DataFrame, start_idx: int, batch_size: int) -> Dict[str, torch.Tensor]:
    """Prepare batch for model inference"""
    end_idx = min(start_idx + batch_size, len(df))
    batch_df = df[start_idx:end_idx]

    # Convert to tensors (simplified - in practice need proper feature engineering)
    batch = {}

    # Assume 'dynamic_features' column exists or create from feature columns
    if 'dynamic_features' in batch_df.columns:
        features = torch.tensor(batch_df['dynamic_features'].to_list(), dtype=torch.float32)
    else:
        # Extract numeric columns as features
        numeric_cols = [col for col in batch_df.columns if batch_df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found for features")

        feature_matrix = batch_df.select(numeric_cols).to_numpy().astype(np.float32)
        # Reshape to [batch, seq_len, features] - assume seq_len=1 for simplicity
        features = torch.tensor(feature_matrix).unsqueeze(1)

    batch['dynamic_features'] = features

    # Add regime features if available
    if 'regime_features' in batch_df.columns:
        regime_features = torch.tensor(batch_df['regime_features'].to_list(), dtype=torch.float32)
        batch['regime_features'] = regime_features

    return batch


def save_predictions(predictions: List[Dict[str, Any]], output_path: Union[str, Path]):
    """Save predictions to file"""
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert predictions to DataFrame
        rows = []
        for i, pred in enumerate(predictions):
            row = {'batch_idx': i}

            # Extract predictions
            if 'predictions' in pred:
                preds = pred['predictions']
                if isinstance(preds, dict):
                    for horizon, horizon_pred in preds.items():
                        if isinstance(horizon_pred, torch.Tensor):
                            # Save quantiles as separate columns
                            horizon_pred_np = horizon_pred.cpu().numpy()
                            for q_idx in range(horizon_pred_np.shape[1]):
                                row[f'{horizon}_q{q_idx}'] = horizon_pred_np[:, q_idx].tolist()

            # Extract TENT statistics
            if 'tent_stats' in pred:
                tent_stats = pred['tent_stats']
                for key, value in tent_stats.items():
                    row[f'tent_{key}'] = value

            # Extract gate analysis if available
            if 'gate_analysis' in pred:
                gate_analysis = pred['gate_analysis']
                if 'gate_probs' in gate_analysis:
                    gate_probs = gate_analysis['gate_probs'].cpu().numpy()
                    for expert_idx in range(gate_probs.shape[1]):
                        row[f'expert_{expert_idx}_prob'] = gate_probs[:, expert_idx].tolist()

            rows.append(row)

        # Create DataFrame and save
        df_predictions = pl.DataFrame(rows)

        if output_path.suffix == '.parquet':
            df_predictions.write_parquet(output_path)
        elif output_path.suffix == '.csv':
            df_predictions.write_csv(output_path)
        else:
            # Default to parquet
            output_path = output_path.with_suffix('.parquet')
            df_predictions.write_parquet(output_path)

        logger.info(f"✅ Predictions saved to: {output_path}")

    except Exception as e:
        logger.error(f"❌ Failed to save predictions: {e}")
        raise


def run_tent_inference(
    model_path: Union[str, Path],
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    tent_steps: int = 3,
    tent_lr: float = 1e-4,
    batch_size: int = 32,
    **tent_kwargs
) -> Dict[str, Any]:
    """
    Run TENT inference on input data

    Args:
        model_path: Path to pre-trained model
        input_path: Path to input data
        output_path: Path to save predictions
        tent_steps: Number of TENT adaptation steps
        tent_lr: Learning rate for TENT adaptation
        batch_size: Batch size for processing
        **tent_kwargs: Additional TENT configuration

    Returns:
        Dictionary with inference results and statistics
    """

    start_time = time.time()
    results = {
        'success': False,
        'error': None,
        'batches_processed': 0,
        'total_samples': 0,
        'avg_entropy_improvement': 0.0,
        'avg_confidence': 0.0,
        'processing_time': 0.0
    }

    try:
        # Load model and data
        model = load_model(model_path)
        data = load_data(input_path)

        # Setup TENT adapter
        from .tent_adapter import create_tent_adapter

        tent_adapter = create_tent_adapter(
            model=model,
            steps=tent_steps,
            lr=tent_lr,
            log_adaptation=True,
            save_adaptation_history=True,
            **tent_kwargs
        )

        logger.info(f"🧠 TENT adapter ready: {tent_steps} steps, lr={tent_lr}")

        # Process data in batches
        predictions = []
        num_batches = (len(data) + batch_size - 1) // batch_size

        with tqdm(total=num_batches, desc="TENT Inference") as pbar:
            for batch_idx in range(0, len(data), batch_size):
                # Prepare batch
                batch = prepare_batch(data, batch_idx, batch_size)

                # Run TENT adaptation and inference
                with torch.no_grad():
                    adapted_output = tent_adapter.adapt_batch(batch)

                predictions.append(adapted_output)
                results['batches_processed'] += 1
                results['total_samples'] += len(batch['dynamic_features'])

                pbar.update(1)
                pbar.set_postfix({
                    'entropy': adapted_output.get('tent_stats', {}).get('final_entropy_loss', 0.0),
                    'confidence': adapted_output.get('tent_stats', {}).get('final_confidence', 0.0)
                })

        # Save predictions
        save_predictions(predictions, output_path)

        # Compute final statistics
        adaptation_summary = tent_adapter.get_adaptation_summary()
        results.update({
            'success': True,
            'avg_entropy_improvement': adaptation_summary.get('avg_entropy_improvement', 0.0),
            'avg_confidence': adaptation_summary.get('avg_confidence', 0.0),
            'total_adaptation_steps': adaptation_summary.get('total_adaptation_steps', 0),
            'adaptable_params': adaptation_summary.get('adaptable_params', 0),
            'processing_time': time.time() - start_time
        })

        logger.info("🎉 TENT inference completed successfully!")
        logger.info(f"   Processed: {results['batches_processed']} batches, {results['total_samples']} samples")
        logger.info(f"   Entropy improvement: {results['avg_entropy_improvement']:.4f}")
        logger.info(f"   Average confidence: {results['avg_confidence']:.3f}")
        logger.info(f"   Processing time: {results['processing_time']:.2f}s")

    except Exception as e:
        logger.error(f"❌ TENT inference failed: {e}")
        results['error'] = str(e)

    return results


def main():
    """Command-line interface for TENT inference"""
    parser = argparse.ArgumentParser(description="TENT Test-time Adaptation Inference")

    parser.add_argument("--model-path", required=True, help="Path to pre-trained model")
    parser.add_argument("--input-path", required=True, help="Path to input data")
    parser.add_argument("--output-path", required=True, help="Path to save predictions")

    # TENT parameters
    parser.add_argument("--tent-steps", type=int, default=3, help="TENT adaptation steps per batch")
    parser.add_argument("--tent-lr", type=float, default=1e-4, help="TENT learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size")

    # Additional TENT configuration
    parser.add_argument("--confidence-threshold", type=float, default=0.9,
                       help="Skip adaptation if confidence > threshold")
    parser.add_argument("--entropy-weight", type=float, default=1.0,
                       help="Weight for entropy loss")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Temperature for probability conversion")

    # Logging
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Run TENT inference
    result = run_tent_inference(
        model_path=args.model_path,
        input_path=args.input_path,
        output_path=args.output_path,
        tent_steps=args.tent_steps,
        tent_lr=args.tent_lr,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        entropy_weight=args.entropy_weight,
        temperature=args.temperature
    )

    if result['success']:
        print("✅ TENT inference completed successfully")
        return 0
    else:
        print(f"❌ TENT inference failed: {result['error']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())