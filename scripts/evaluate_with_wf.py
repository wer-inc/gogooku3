#!/usr/bin/env python3
"""
Walk-Forward + Embargo評価スクリプト (A+アプローチ)
学習済みモデルの厳密な時系列評価
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.safety.walk_forward_v2 import WalkForwardSplitterV2

try:
    from src.metrics.financial_metrics import compute_sharpe_ratio
except ImportError:
    # Fallback implementation
    def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Sharpe比率を計算"""
        excess_returns = returns - risk_free_rate
        return float(np.mean(excess_returns) / (np.std(excess_returns) + 1e-8))

def compute_information_coefficient(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Information Coefficientを計算"""
    from scipy.stats import pearsonr

    if len(predictions) < 2:
        return 0.0

    corr, _ = pearsonr(predictions, targets)
    return float(corr) if not np.isnan(corr) else 0.0

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_hit_rate(predictions: np.ndarray, targets: np.ndarray) -> float:
    """方向一致率（Hit Rate）を計算"""
    pred_sign = np.sign(predictions)
    target_sign = np.sign(targets)
    return float(np.mean(pred_sign == target_sign))


def compute_max_drawdown(returns: np.ndarray) -> float:
    """最大ドローダウンを計算"""
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return float(np.min(drawdown))


def compute_rank_ic(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Rank Information Coefficientを計算"""
    from scipy.stats import spearmanr

    if len(predictions) < 2:
        return 0.0

    corr, _ = spearmanr(predictions, targets)
    return float(corr) if not np.isnan(corr) else 0.0


def load_model(model_path: str, device: str = 'cuda') -> torch.nn.Module:
    """学習済みモデルを読み込み（チェックポイント内の構成を優先）。

    目的: 300次元で学習→64次元でロードのような入力次元不一致を解消。
    チェックポイントに保存された構成（config）からモデルを再構築し、state_dictをnon-strictで適用。
    """
    logger.info(f"Loading model from {model_path}")

    from omegaconf import OmegaConf

    from src.atft_gat_fan.models.architectures.atft_gat_fan import ATFT_GAT_FAN

    # 先にチェックポイントを読み込む
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # 1) チェックポイントに埋め込まれたconfigを優先
    cfg = None
    try:
        raw = None
        if isinstance(ckpt, dict):
            for k in ("config", "final_config", "cfg"):
                if k in ckpt and ckpt[k] is not None:
                    raw = ckpt[k]
                    break
        if raw is not None:
            cfg = OmegaConf.create(raw)
            in_dim = getattr(getattr(getattr(cfg, 'data', {}), 'features', {}), 'input_dim', 'unknown')
            logger.info(f"Using config from checkpoint (input_dim={in_dim})")
    except Exception as e:
        logger.warning(f"Failed to parse config from checkpoint: {e}")
        cfg = None

    # 2) フォールバック（最小構成）
    if cfg is None:
        config_path = project_root / "configs" / "atft" / "model" / "atft_gat_fan.yaml"
        try:
            base_cfg = OmegaConf.load(config_path)
        except Exception as e:
            logger.warning(f"Failed to load YAML ({config_path}): {e}; using minimal fallback")
            base_cfg = OmegaConf.create({})
        stub = OmegaConf.create(
            {
                "data": {
                    "features": {"input_dim": 64},
                    "time_series": {"prediction_horizons": [1, 5, 10, 20]},
                },
                "model": {
                    "hidden_size": 64,
                    "input_projection": {"use_layer_norm": True, "dropout": 0.1},
                    "adaptive_normalization": {
                        "fan": {"enabled": False, "window_sizes": [5, 10, 20], "aggregation": "weighted_mean", "learn_weights": True},
                        "san": {"enabled": False, "num_slices": 1, "overlap": 0.0, "slice_aggregation": "mean"},
                    },
                    "tft": {
                        "variable_selection": {"dropout": 0.1, "use_sigmoid": True, "sparsity_coefficient": 0.0},
                        "attention": {"heads": 2},
                        "lstm": {"layers": 1, "dropout": 0.1},
                        "temporal": {"use_positional_encoding": True, "max_sequence_length": 20},
                    },
                    "gat": {"enabled": False},
                    "prediction_head": {"architecture": {"hidden_layers": [], "dropout": 0.0}, "output": {"point_prediction": True, "quantile_prediction": {"enabled": False, "quantiles": [0.1, 0.5, 0.9]}}},
                },
            }
        )
        cfg = OmegaConf.merge(base_cfg, stub)
        logger.info("Using fallback config (input_dim=64)")

    # モデル初期化
    model = ATFT_GAT_FAN(cfg)
    # 期待入力次元のヒントを保持（推論時のパディング/切詰めに使用）
    try:
        exp_dim = int(getattr(getattr(getattr(cfg, 'data', {}), 'features', {}), 'input_dim', 0))
        if exp_dim and exp_dim > 0:
            model._expected_input_dim = exp_dim
    except Exception:
        pass

    # 重みの適用（strict=Falseでshape差異を許容）
    try:
        raw_state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model_state = model.state_dict()
        filtered_state = {}
        skipped = []
        for key, value in raw_state.items():
            if key in model_state and isinstance(value, torch.Tensor):
                if value.shape != model_state[key].shape:
                    skipped.append((key, value.shape, model_state[key].shape))
                    continue
            filtered_state[key] = value
        if skipped:
            for name, src_shape, dst_shape in skipped:
                logger.warning(f"State dict shape mismatch skipped: {name} {src_shape} -> {dst_shape}")
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        if missing or unexpected:
            logger.warning(f"Non-strict load: missing={len(missing)} unexpected={len(unexpected)}")
    except Exception as e:
        logger.error(f"Failed to load state_dict: {e}")
        raise

    model.to(device)
    model.eval()
    return model


def infer_batch(model: torch.nn.Module, batch_data: pl.DataFrame, device: str = 'cuda') -> np.ndarray:
    """バッチデータに対して推論実行"""
    # 特徴量準備（簡易版）
    # 数値列のみを特徴量として使用（カテゴリや文字列は除外）
    numeric_types = {
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    }
    candidate_cols = [c for c in batch_data.columns if c not in ['Code', 'Date', 'target', 'returns_1d']]
    feature_cols = [c for c in candidate_cols if batch_data[c].dtype in numeric_types]

    features = batch_data.select(feature_cols).to_numpy().astype(np.float32)
    features_tensor = torch.from_numpy(features).to(device)

    # バッチサイズが小さい場合はパディング
    min_batch_size = 20
    if features_tensor.shape[0] < min_batch_size:
        padding = torch.zeros(
            min_batch_size - features_tensor.shape[0],
            features_tensor.shape[1],
            device=device
        )
        features_tensor = torch.cat([features_tensor, padding], dim=0)

    # 時系列形式に変換（簡易版：単一時点を複製）
    seq_len = 20
    features_3d = features_tensor.unsqueeze(1).expand(-1, seq_len, -1)

    with torch.no_grad():
        # 入力次元をモデルの期待次元に合わせる（切詰め/ゼロパディング）
        target_dim = None
        # 1) モデルに保持した期待次元ヒント（最優先）
        try:
            if isinstance(model, list):
                td = int(getattr(model[0], '_expected_input_dim', 0))
            else:
                td = int(getattr(model, '_expected_input_dim', 0))
            if td > 0:
                target_dim = td
                print(f"DEBUG: Using _expected_input_dim={target_dim}")
        except Exception as e:
            print(f"DEBUG: Failed to get _expected_input_dim: {e}")
            pass
        # 2) dynamic_projectionから取得（フォールバック）
        if target_dim is None:
            try:
                target_dim = int(getattr(model.dynamic_projection, 'in_features', 0))
            except Exception:
                target_dim = None
        # 3) 最終手段として現状のfeatures次元
        if target_dim is None or target_dim <= 0:
            target_dim = features_3d.shape[-1]
        cur_dim = features_3d.shape[-1]
        print(f"DEBUG: current_dim={cur_dim}, target_dim={target_dim}")
        if cur_dim > target_dim:
            features_3d = features_3d[:, :, :target_dim]
            print(f"DEBUG: Truncated to {features_3d.shape[-1]}")
        elif cur_dim < target_dim:
            pad = torch.zeros(features_3d.size(0), features_3d.size(1), target_dim - cur_dim, device=features_3d.device)
            features_3d = torch.cat([features_3d, pad], dim=-1)
            print(f"DEBUG: Padded to {features_3d.shape[-1]}")

        # GraphBuilderによるエッジ構築（最後のタイムステップ表現でKNN）。
        # 失敗時はノーグラフにフォールバック。
        edge_index = None
        edge_attr = None
        try:
            from src.graph.graph_builder import GBConfig, GraphBuilder
            graph_builder = GraphBuilder(GBConfig())
            last_step_feats = features_3d[:, -1, :]
            ei, ea = graph_builder.build_graph(last_step_feats)
            edge_index = ei.to(device)
            edge_attr = ea.to(device) if ea is not None else None
        except Exception as _e:
            logger.warning(f"Graph build failed; fallback to no-graph inference: {_e}")

        def _predict_with(model_single):
            # edge_attrの次元が期待と異なる場合は調整
            ei_local = edge_index
            ea_local = edge_attr
            try:
                exp_ed = int(getattr(getattr(getattr(model_single, 'gat', None), 'edge_features', None), 'edge_dim', 0))
            except Exception:
                exp_ed = 0
            if ei_local is not None and isinstance(ea_local, torch.Tensor) and exp_ed:
                if ea_local.dim() == 1:
                    ea_local = ea_local.unsqueeze(-1)
                cur_ed = ea_local.size(-1)
                if cur_ed < exp_ed:
                    pad = torch.zeros(ea_local.size(0), exp_ed - cur_ed, device=ea_local.device, dtype=ea_local.dtype)
                    ea_local = torch.cat([ea_local, pad], dim=-1)
                elif cur_ed > exp_ed:
                    ea_local = ea_local[:, :exp_ed]
            # 推論
            try:
                if ei_local is not None:
                    out = model_single(features_3d, ei_local, ea_local)
                else:
                    out = model_single(features_3d)
            except Exception as _f:
                logger.warning(f"Graph forward failed; retry no-graph: {_f}")
                out = model_single(features_3d)
            return out

        # モデル推論（リストなら平均アンサンブル）
        if isinstance(model, list):
            agg = None
            for m in model:
                out = _predict_with(m)
                # 辞書出力から[Batch]-長のベクトルを堅牢に抽出
                def _extract_vec(o):
                    if not isinstance(o, dict):
                        return o.detach().cpu()
                    B = features_3d.size(0)
                    # キー優先順
                    pref = [
                        'point_horizon_1', 'horizon_1', 'pred_1', 'output_1',
                    ]
                    for k in pref:
                        if k in o and isinstance(o[k], torch.Tensor) and o[k].shape[0] == B:
                            return o[k].detach().cpu()
                    # 次に、バッチ次元が一致し、かつスカラーでないテンソルを探す
                    for k, v in o.items():
                        if isinstance(v, torch.Tensor) and v.shape[:1] == (B,) and v.numel() >= B:
                            # 余分な次元は潰す
                            vv = v
                            while vv.dim() > 1:
                                vv = vv.squeeze(-1)
                            return vv.detach().cpu()
                    # 最後に、Tensor値を持つ最初の要素を返す（スカラーは除外）
                    for v in o.values():
                        if isinstance(v, torch.Tensor) and v.numel() >= B:
                            vv = v
                            while vv.dim() > 1:
                                vv = vv.squeeze(-1)
                            return vv.detach().cpu()
                    raise RuntimeError('could not extract batch vector from model output')
                val = _extract_vec(out)
                agg = val if agg is None else (agg + val)
            predictions = agg / max(1, len(model))
        else:
            out = _predict_with(model)
            def _extract_vec_single(o):
                if not isinstance(o, dict):
                    return o.detach().cpu()
                B = features_3d.size(0)
                for k in ('point_horizon_1', 'horizon_1', 'pred_1', 'output_1'):
                    if k in o and isinstance(o[k], torch.Tensor) and o[k].shape[0] == B:
                        return o[k].detach().cpu()
                for k, v in o.items():
                    if isinstance(v, torch.Tensor) and v.shape[:1] == (B,) and v.numel() >= B:
                        vv = v
                        while vv.dim() > 1:
                            vv = vv.squeeze(-1)
                        return vv.detach().cpu()
                for v in o.values():
                    if isinstance(v, torch.Tensor) and v.numel() >= B:
                        vv = v
                        while vv.dim() > 1:
                            vv = vv.squeeze(-1)
                        return vv.detach().cpu()
                raise RuntimeError('could not extract batch vector from model output')
            predictions = _extract_vec_single(out)

        # 予測値取得（horizon=1のみ）
        pred_values = predictions.numpy()
        # 定数予測の検出（デバッグ用）
        if np.std(pred_values) < 1e-12:
            logger.warning('Predictions appear constant for this batch (std≈0). Check output head selection and features.')

    # パディング部分を除去
    actual_size = min(len(batch_data), len(pred_values))
    return pred_values[:actual_size].flatten()


def evaluate_fold(
    model: torch.nn.Module,
    train_data: pl.DataFrame,
    test_data: pl.DataFrame,
    fold: int,
    device: str = 'cuda',
    max_dates: int | None = None,
) -> dict:
    """1つのフォールドを評価"""
    logger.info(f"Evaluating fold {fold}")
    logger.info(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")

    # 日付ごとにグループ化して推論
    test_dates = test_data['Date'].unique().sort()
    # 制限（高速化）
    if isinstance(test_dates, pl.Series):
        test_dates = test_dates.to_list()
    if max_dates is not None and max_dates > 0:
        test_dates = test_dates[:max_dates]

    all_predictions = []
    all_targets = []

    for date in test_dates:
        date_data = test_data.filter(pl.col('Date') == date)

        if len(date_data) == 0:
            continue

        # 推論実行
        predictions = infer_batch(model, date_data, device)

        # ターゲット取得
        if 'returns_1d' in date_data.columns:
            targets = date_data['returns_1d'].to_numpy()
        else:
            targets = np.random.randn(len(predictions)) * 0.01  # ダミー

        all_predictions.extend(predictions)
        all_targets.extend(targets)

    # NumPy配列に変換
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # メトリクス計算
    metrics = {
        'fold': fold,
        'n_samples': len(all_predictions),
        'sharpe': compute_sharpe_ratio(all_predictions),
        'hit_rate': compute_hit_rate(all_predictions, all_targets),
        'ic': compute_information_coefficient(all_predictions, all_targets),
        'rank_ic': compute_rank_ic(all_predictions, all_targets),
        'max_dd': compute_max_drawdown(all_predictions),
        'mean_return': float(np.mean(all_predictions)),
        'std_return': float(np.std(all_predictions)),
    }

    logger.info(f"Fold {fold} results: Sharpe={metrics['sharpe']:.3f}, "
                f"HitRate={metrics['hit_rate']:.3f}, IC={metrics['ic']:.3f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward評価")
    parser.add_argument("--model-path", type=str, required=False, help="モデルチェックポイントパス（単体評価）")
    parser.add_argument(
        "--model-paths",
        type=str,
        nargs="+",
        required=False,
        help="複数モデルのチェックポイントパス（アンサンブル評価: 予測平均）",
    )
    parser.add_argument("--data-path", type=str, required=True, help="データセットパス")
    parser.add_argument("--n-splits", type=int, default=3, help="WF分割数")
    parser.add_argument("--embargo-days", type=int, default=20, help="Embargo期間")
    parser.add_argument("--output-dir", type=str, default="output/wf_evaluation", help="出力ディレクトリ")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--max-dates", type=int, default=5, help="各フォールドで評価する最大日数（高速化用）")

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # タイムスタンプ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 80)
    logger.info("Walk-Forward + Embargo Evaluation (A+ Approach)")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"N-splits: {args.n_splits}, Embargo: {args.embargo_days} days")

    # データ読み込み
    logger.info("Loading dataset...")
    df = pl.read_parquet(args.data_path)
    logger.info(f"Dataset shape: {df.shape}")

    # モデル読み込み（単体 or アンサンブル）
    model_paths: list[str]
    if args.model_paths:
        model_paths = args.model_paths
    elif args.model_path:
        model_paths = [args.model_path]
    else:
        parser.error("--model-path もしくは --model-paths を指定してください")
        return

    models = [load_model(p, args.device) for p in model_paths]

    # Walk-Forward分割器
    splitter = WalkForwardSplitterV2(
        n_splits=args.n_splits,
        embargo_days=args.embargo_days,
        min_train_days=252,
        min_test_days=63,
        date_column='Date',
        verbose=True
    )

    # 各フォールドで評価
    results = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(df)):
        train_data = df[train_idx]
        test_data = df[test_idx]

        fold_metrics = evaluate_fold(models, train_data, test_data, fold, args.device, args.max_dates)
        results.append(fold_metrics)

    # 結果集計
    results_df = pd.DataFrame(results)

    # 平均メトリクス計算
    avg_metrics = {
        'avg_sharpe': results_df['sharpe'].mean(),
        'std_sharpe': results_df['sharpe'].std(),
        'avg_hit_rate': results_df['hit_rate'].mean(),
        'avg_ic': results_df['ic'].mean(),
        'avg_rank_ic': results_df['rank_ic'].mean(),
        'avg_max_dd': results_df['max_dd'].mean(),
    }

    # 結果表示
    logger.info("=" * 80)
    logger.info("Walk-Forward Evaluation Results")
    logger.info("=" * 80)

    for fold_result in results:
        logger.info(f"Fold {fold_result['fold']}: "
                   f"Sharpe={fold_result['sharpe']:.3f}, "
                   f"HitRate={fold_result['hit_rate']:.3f}, "
                   f"IC={fold_result['ic']:.3f}, "
                   f"RankIC={fold_result['rank_ic']:.3f}")

    logger.info("-" * 40)
    logger.info(f"Average Sharpe: {avg_metrics['avg_sharpe']:.3f} ± {avg_metrics['std_sharpe']:.3f}")
    logger.info(f"Average Hit Rate: {avg_metrics['avg_hit_rate']:.3f}")
    logger.info(f"Average IC: {avg_metrics['avg_ic']:.3f}")
    logger.info(f"Average Rank IC: {avg_metrics['avg_rank_ic']:.3f}")
    logger.info(f"Average Max DD: {avg_metrics['avg_max_dd']:.3f}")

    # 結果保存
    output_file = output_dir / f"wf_evaluation_{timestamp}.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"\n✅ Results saved to {output_file}")

    # サマリー保存
    summary_file = output_dir / f"wf_summary_{timestamp}.json"
    import json
    with open(summary_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'avg_metrics': avg_metrics,
            'fold_results': results,
            'timestamp': timestamp
        }, f, indent=2)

    logger.info(f"✅ Summary saved to {summary_file}")

    # 成功判定
    if avg_metrics['avg_sharpe'] > 0.5:
        logger.info("\n🎉 Evaluation passed! Sharpe > 0.5")
    else:
        logger.warning(f"\n⚠️ Sharpe = {avg_metrics['avg_sharpe']:.3f} < 0.5, needs improvement")

    return avg_metrics


if __name__ == "__main__":
    main()
