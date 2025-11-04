"""
RFI-5/6 Metrics Logging Helpers

P0-3/4/6/7で必要な診断メトリクスを一箇所で計算・フォーマット。
学習ループから1行で呼び出し可能。

Required for:
- RFI-5: Graph health (deg_avg, isolates, edge_attr stats)
- RFI-6: Loss metrics (Sharpe_EMA, RankIC, CRPS, WQL, quantile_crossing_rate)
- P0-3: Gate statistics (gat_gate_mean, gat_gate_std)
"""

import torch


def graph_stats_from_batch(batch: dict) -> dict:
    """
    グラフ統計を抽出（RFI-5）

    Args:
        batch: DataLoaderからのバッチ（edge_index, edge_attr含む）

    Returns:
        dict: N, E, deg_avg, isolates, corr_mean/std/min/max
    """
    # Batch内のノード数
    x_dyn = batch.get("dynamic_features", batch.get("x_dyn", None))
    if x_dyn is None:
        x_dyn = batch.get("features", None)

    N = int(x_dyn.shape[0]) if x_dyn is not None else 0

    ei = batch.get("edge_index", None)
    ea = batch.get("edge_attr", None)

    if ei is None or ea is None or ei.numel() == 0:
        return dict(
            N=N, E=0, deg_avg=0.0, isolates=1.0,
            corr_min=None, corr_mean=None, corr_std=None, corr_max=None
        )

    E = int(ea.shape[0])

    # 次数計算
    deg = torch.bincount(ei[0], minlength=N).float()
    deg_avg = float(deg.mean().item())

    # 孤立ノード率
    isolates = float((deg == 0).sum().item()) / max(1, N)

    # Edge attribute統計（0列目を相関強度とみなす）
    c0 = ea[:, 0]

    stats = dict(
        N=N, E=E,
        deg_avg=deg_avg,
        isolates=isolates,
        corr_min=float(c0.min().item()),
        corr_mean=float(c0.mean().item()),
        corr_std=float(c0.std().item()),
        corr_max=float(c0.max().item())
    )

    return stats


def rank_ic(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Spearman Rank IC（簡易版）

    同日バッチ想定で、時間軸平均後の断面Rank相関を計算。

    Args:
        pred: 予測値 [B, H] or [B, H, 1]
        true: 真値 [B, H] or [B, H, 1]

    Returns:
        float: Rank IC (-1.0 to 1.0)
    """
    # 時間軸平均
    if pred.dim() == 3:
        pred = pred.squeeze(-1)
    if true.dim() == 3:
        true = true.squeeze(-1)

    p = pred.mean(dim=1).detach().cpu().numpy()
    t = true.mean(dim=1).detach().cpu().numpy()

    if len(p) < 3:
        return float("nan")

    # Rank変換
    rp = p.argsort().argsort()
    rt = t.argsort().argsort()

    # 標準化
    rp = (rp - rp.mean()) / (rp.std() + 1e-12)
    rt = (rt - rt.mean()) / (rt.std() + 1e-12)

    # 相関
    ic = float((rp * rt).mean())

    return ic


def weighted_quantile_loss(
    yhat_q: torch.Tensor,
    y: torch.Tensor,
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
) -> float:
    """
    Weighted Quantile Loss (WQL)

    Args:
        yhat_q: 分位点予測 [B, H, Q]
        y: 真値 [B, H]
        quantiles: 分位点リスト

    Returns:
        float: WQL (lower is better)
    """
    B, H, Q = yhat_q.shape
    Y = y.unsqueeze(-1).expand(-1, -1, Q)
    E = Y - yhat_q

    loss = 0.0
    for i, q in enumerate(quantiles):
        pinball = torch.maximum(q * E[:, :, i], (q - 1) * E[:, :, i])
        loss += pinball.mean()

    return float(loss.item() / len(quantiles))


def crps_from_quantiles(
    yhat_q: torch.Tensor,
    y: torch.Tensor,
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
) -> float:
    """
    CRPS近似（Pinballの離散積分）

    Args:
        yhat_q: 分位点予測 [B, H, Q]
        y: 真値 [B, H]
        quantiles: 分位点リスト

    Returns:
        float: CRPS (lower is better)
    """
    B, H, Q = yhat_q.shape
    Y = y.unsqueeze(-1).expand(-1, -1, Q)
    E = Y - yhat_q

    pin = []
    for i, q in enumerate(quantiles):
        pinball = torch.maximum(q * E[:, :, i], (q - 1) * E[:, :, i])
        pin.append(pinball)

    # 均等ウェイト近似
    crps = torch.stack(pin, dim=-1).mean()

    return float(crps.item())


def quantile_crossing_rate(yhat_q: torch.Tensor) -> float:
    """
    分位点交差率（Quantile Crossing Rate）

    y(q_i) <= y(q_{i+1}) を満たさない割合。

    Args:
        yhat_q: 分位点予測 [B, H, Q]

    Returns:
        float: 交差率 (0.0-1.0, lower is better, target < 0.05)
    """
    # 隣接分位点の差
    diffs = yhat_q[:, :, 1:] - yhat_q[:, :, :-1]

    # 違反（負の差）の割合
    violations = (diffs < 0).float().mean().item()

    return float(violations)


def gradient_ratio(model) -> float:
    """
    Base vs GAT/Fuse gradient ratio

    Base系とGAT/Fuse系の勾配ノルム比。
    P0-3が正しく動作していれば 0.5-2.0 の範囲に収まる。

    Args:
        model: ATFT_GAT_FAN model

    Returns:
        float: ||∂L/∂base|| / ||∂L/∂gat|| (target: 0.5-2.0)
    """
    grad_base = 1e-12
    grad_gat = 1e-12

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad_norm = param.grad.detach().float().norm().item()

        if ("gat" in name) or ("fuse" in name):
            grad_gat += grad_norm
        else:
            grad_base += grad_norm

    ratio = float(grad_base / grad_gat)

    return ratio


def gate_statistics(model) -> tuple[float | None, float | None]:
    """
    GAT Gate統計（P0-3特有）

    Args:
        model: ATFT_GAT_FAN model

    Returns:
        (gate_mean, gate_std): None if no gate available
    """
    if not hasattr(model, "_last_gate") or model._last_gate is None:
        return None, None

    gate = model._last_gate
    gate_mean = float(gate.mean().item())
    gate_std = float(gate.std().item())

    return gate_mean, gate_std


def log_rfi_56_metrics(
    logger,
    model,
    batch: dict,
    y_point: torch.Tensor,
    y_q: torch.Tensor,
    y_true: torch.Tensor,
    epoch: int = 0
) -> dict:
    """
    RFI-5/6メトリクスを一括計算・ログ出力

    Args:
        logger: Python logger
        model: ATFT_GAT_FAN model
        batch: DataLoader batch
        y_point: Point forecast [B, H]
        y_q: Quantile forecast [B, H, Q]
        y_true: Ground truth [B, H]
        epoch: Current epoch number

    Returns:
        dict: All computed metrics
    """
    # Graph statistics (RFI-5)
    gs = graph_stats_from_batch(batch)

    # Gate statistics (P0-3)
    gate_mean, gate_std = gate_statistics(model)

    # Loss metrics (RFI-6)
    ic = rank_ic(y_point, y_true)
    wql = weighted_quantile_loss(y_q, y_true)
    crps = crps_from_quantiles(y_q, y_true)
    xrate = quantile_crossing_rate(y_q)

    # Gradient ratio (P0-3 diagnostic)
    gr = gradient_ratio(model)

    # Consolidated metrics
    metrics = {
        "epoch": epoch,
        "gat_gate_mean": gate_mean,
        "gat_gate_std": gate_std,
        "deg_avg": gs["deg_avg"],
        "isolates": gs["isolates"],
        "corr_mean": gs["corr_mean"],
        "corr_std": gs["corr_std"],
        "rank_ic": ic,
        "wql": wql,
        "crps": crps,
        "quantile_crossing_rate": xrate,
        "grad_ratio": gr,
        "N": gs["N"],
        "E": gs["E"],
    }

    # Format log message
    log_msg = (
        f"RFI56 | epoch={epoch} "
        f"gat_gate_mean={gate_mean:.4f} gat_gate_std={gate_std:.4f} "
        f"deg_avg={gs['deg_avg']:.2f} isolates={gs['isolates']:.3f} "
        f"corr_mean={gs['corr_mean']:.3f} corr_std={gs['corr_std']:.3f} "
        f"RankIC={ic:.4f} WQL={wql:.6f} CRPS={crps:.6f} qx_rate={xrate:.4f} "
        f"grad_ratio={gr:.3f}"
    )

    logger.info(log_msg)

    return metrics
