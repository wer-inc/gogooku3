#!/usr/bin/env python3
"""
日次予測生成スクリプト（production ready）
単一モデルから Date, Code, predicted_return 形式で出力
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import polars as pl
import torch
from typing import Dict, List, Tuple, Optional

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_model(model_path: str, device: str = 'cuda') -> torch.nn.Module:
    """学習済みモデルを読み込み（簡略版）"""
    logger.info(f"Loading model from {model_path}")

    from src.models.architectures.atft_gat_fan import ATFT_GAT_FAN
    from omegaconf import OmegaConf

    # チェックポイント読み込み
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # 設定復元
    cfg = None
    if isinstance(ckpt, dict):
        for k in ("config", "final_config", "cfg"):
            if k in ckpt and ckpt[k] is not None:
                cfg = OmegaConf.create(ckpt[k])
                break

    if cfg is None:
        # フォールバック設定
        cfg = OmegaConf.create({
            "data": {"features": {"input_dim": 300}, "time_series": {"prediction_horizons": [1, 2, 3, 5, 10]}},
            "model": {"hidden_size": 64}
        })

    # モデル初期化
    model = ATFT_GAT_FAN(cfg)
    
    # 期待入力次元保持
    try:
        exp_dim = int(cfg.data.features.input_dim)
        if exp_dim > 0:
            setattr(model, "_expected_input_dim", exp_dim)
    except Exception:
        pass

    # 重みロード（非厳密）
    try:
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            logger.warning(f"Non-strict load: missing={len(missing)} unexpected={len(unexpected)}")
    except Exception as e:
        logger.error(f"Failed to load state_dict: {e}")
        raise

    model.to(device)
    model.eval()
    return model

def infer_batch_simple(model: torch.nn.Module, batch_data: pl.DataFrame, device: str = 'cuda') -> np.ndarray:
    """シンプルバッチ推論（ノーグラフモード）"""
    # 数値特徴量のみ
    numeric_types = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }
    candidate_cols = [c for c in batch_data.columns if c not in ['Code', 'Date', 'target', 'returns_1d']]
    feature_cols = [c for c in candidate_cols if batch_data[c].dtype in numeric_types]
    
    features = batch_data.select(feature_cols).to_numpy().astype(np.float32)
    features_tensor = torch.from_numpy(features).to(device)
    
    # 時系列形式に変換（簡易版：単一時点を複製）
    seq_len = 20
    features_3d = features_tensor.unsqueeze(1).expand(-1, seq_len, -1)

    with torch.no_grad():
        # 次元調整（パディング/切詰め）
        target_dim = 300  # デフォルト
        try:
            td = int(getattr(model, '_expected_input_dim', 0))
            if td > 0:
                target_dim = td
        except Exception:
            pass
        
        cur_dim = features_3d.shape[-1]
        if cur_dim > target_dim:
            features_3d = features_3d[:, :, :target_dim]
        elif cur_dim < target_dim:
            pad = torch.zeros(features_3d.size(0), features_3d.size(1), target_dim - cur_dim, device=features_3d.device)
            features_3d = torch.cat([features_3d, pad], dim=-1)
        
        # ノーグラフ推論（GATエラー回避）
        try:
            predictions = model(features_3d)  # グラフ引数なし
        except Exception as e:
            logger.warning(f"Direct inference failed: {e}. Retrying with dummy graph.")
            # ダミーグラフで再試行
            batch_size = features_3d.size(0)
            edge_index = torch.tensor([[0, 1], [1, 0]], device=device).long()
            edge_attr = torch.zeros((2, 1), device=device)
            predictions = model(features_3d, edge_index, edge_attr)
        
        # 予測値抽出（堅牢化済みロジック）
        def extract_predictions(output):
            if isinstance(output, torch.Tensor):
                return output.detach().cpu()
            
            if isinstance(output, dict):
                B = features_3d.size(0)
                # 優先キー順
                for key in ['point_horizon_1', 'horizon_1', 'pred_1', 'output_1']:
                    if key in output and isinstance(output[key], torch.Tensor):
                        tensor = output[key]
                        if tensor.shape[0] == B:
                            return tensor.detach().cpu()
                
                # バッチサイズ一致する適切なテンソルを探す
                for k, v in output.items():
                    if isinstance(v, torch.Tensor) and v.shape[:1] == (B,) and v.numel() >= B:
                        vv = v
                        while vv.dim() > 1:
                            vv = vv.squeeze(-1)
                        return vv.detach().cpu()
                
                raise RuntimeError(f"Could not extract batch predictions from keys: {list(output.keys())}")
            
            raise RuntimeError(f"Unsupported output type: {type(output)}")
        
        pred_tensor = extract_predictions(predictions)
        pred_values = pred_tensor.numpy()
    
    # バッチサイズ調整
    actual_size = min(len(batch_data), len(pred_values))
    return pred_values[:actual_size].flatten()

def generate_predictions(model_path: str, data_path: str, output_path: str, 
                        max_dates: int = None, device: str = 'cuda'):
    """日次予測生成メイン処理"""
    logger.info(f"🚀 Generating predictions: {model_path}")
    
    # データ読み込み
    logger.info("📊 Loading dataset...")
    df = pl.read_parquet(data_path)
    logger.info(f"Dataset shape: {df.shape}")
    
    # モデル読み込み
    model = load_model(model_path, device)
    
    # 最新日付取得（production用）
    if max_dates:
        test_dates = df['Date'].unique().sort()
        if isinstance(test_dates, pl.Series):
            test_dates = test_dates.to_list()
        test_dates = test_dates[-max_dates:]  # 最新N日間
    else:
        test_dates = df['Date'].unique().sort().to_list()
    
    logger.info(f"Processing {len(test_dates)} dates...")
    
    results = []
    
    for i, date in enumerate(test_dates):
        if i % 10 == 0:
            logger.info(f"Processing date {i+1}/{len(test_dates)}: {date}")
        
        date_data = df.filter(pl.col('Date') == date)
        
        if len(date_data) == 0:
            continue
        
        # 推論実行
        try:
            predictions = infer_batch_simple(model, date_data, device)
            codes = date_data['Code'].to_numpy()
            
            # 結果保存
            for code, pred in zip(codes, predictions):
                results.append({
                    'Date': date,
                    'Code': code,
                    'predicted_return': float(pred)
                })
        
        except Exception as e:
            logger.warning(f"Failed to process date {date}: {e}")
            continue
    
    # 結果をDataFrameに変換して保存
    result_df = pd.DataFrame(results)
    
    # 出力ディレクトリ作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Parquet形式で保存
    result_df.to_parquet(output_path, index=False)
    
    logger.info(f"✅ Predictions saved: {output_path}")
    logger.info(f"📈 Generated {len(result_df)} predictions for {len(result_df['Date'].unique())} dates")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="日次予測生成")
    parser.add_argument("--model-path", type=str, required=True, help="モデルチェックポイントパス")
    parser.add_argument("--data-path", type=str, required=True, help="データセットパス")
    parser.add_argument("--output", type=str, required=True, help="予測出力パス")
    parser.add_argument("--max-dates", type=int, default=10, help="処理する最大日数（最新から）")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    generate_predictions(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output,
        max_dates=args.max_dates,
        device=args.device
    )

if __name__ == "__main__":
    main()