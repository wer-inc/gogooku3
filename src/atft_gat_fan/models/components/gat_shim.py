"""
P0-3: PyG-free Graph Convolution Fallback (安全シム)

torch_geometric未インストール or segfault環境での代替実装。
依存ゼロで近傍平均コンボリューションを実現。

性能:
- 速度: PyG実装の60-80%程度
- 精度: GATv2Convには劣るが学習・評価は可能
- 安定性: CPU/GPU両対応、依存なし

用途:
- RFI-5/6データ採取（ゲート統計、グラフ統計、Sharpe/RankIC）
- 環境整備完了までの暫定運用
- PyG環境問題の回避
"""
import torch
import torch.nn as nn


class GraphConvShim(nn.Module):
    """
    依存ゼロの近傍平均コンボリューション（GATv2Convの安全代替）。

    動作:
    - 各ノードiについて、近傍j→iからのメッセージを集約
    - edge_attrを線形ゲインとしてメッセージに掛ける（列標準化済みを想定）
    - 次数で正規化（平均）
    - 自己ループとメッセージパッシングを線形結合

    制約:
    - Attentionメカニズムなし（GATの主要機能を欠く）
    - マルチヘッドなし（単一表現）
    - 速度/精度はPyG実装に劣る

    Args:
        in_dim: 入力次元
        hidden_dim: 出力次元
        edge_dim: エッジ特徴次元（0の場合は無視）
        dropout: ドロップアウト率
    """
    def __init__(self, in_dim: int, hidden_dim: int, edge_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim

        # 自己ループ変換
        self.lin_self = nn.Linear(in_dim, hidden_dim, bias=False)

        # 近傍メッセージ変換
        self.lin_nei = nn.Linear(in_dim, hidden_dim, bias=False)

        # エッジ特徴ゲート（オプション）
        self.lin_edge = nn.Linear(edge_dim, 1, bias=False) if edge_dim > 0 else None

        # 正規化・正則化
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z: ノード特徴 [N, in_dim]
            edge_index: エッジインデックス [2, E] (src, dst)
            edge_attr: エッジ特徴 [E, edge_dim] (オプション)

        Returns:
            h: 更新後のノード特徴 [N, hidden_dim]
        """
        N, _ = z.size()

        # エッジインデックスの展開
        if edge_index.numel() == 0:
            # グラフが空の場合、自己ループのみ
            h = self.lin_self(z)
            return self.norm(self.drop(torch.relu(h)))

        src, dst = edge_index[0], edge_index[1]

        # メッセージの構築: z[src] -> [E, in_dim]
        msg = z[src]

        # エッジ特徴によるゲーティング（オプション）
        if self.lin_edge is not None and edge_attr is not None and edge_attr.numel() > 0:
            # edge_attr: [E, edge_dim] -> w: [E, 1]
            w = torch.sigmoid(self.lin_edge(edge_attr)).clamp_min(1e-6)
            msg = msg * w  # [E, in_dim] * [E, 1] -> [E, in_dim]

        # メッセージの集約: sum_{j->i} msg_j
        agg = torch.zeros(N, self.in_dim, device=z.device, dtype=z.dtype)
        agg = agg.index_add_(0, dst, msg)

        # 次数計算（正規化用）
        deg = torch.zeros(N, 1, device=z.device, dtype=z.dtype)
        deg = deg.index_add_(0, dst, torch.ones_like(dst, dtype=z.dtype).unsqueeze(-1))
        deg = deg.clamp_min(1.0)  # 孤立ノード対策

        # 自己ループ + 正規化済み近傍メッセージ
        h = self.lin_self(z) + self.lin_nei(agg / deg)

        # 正規化・活性化・ドロップアウト
        return self.norm(self.drop(torch.relu(h)))


class GATBlockShim(nn.Module):
    """
    2層GraphConvShimのスタック（GATBlockの代替）

    GATBlockの構造を模倣:
    - Layer 1: in_dim -> hidden_dim (ReLU + Dropout)
    - Layer 2: hidden_dim -> hidden_dim (LayerNorm)

    Args:
        in_dim: 入力次元
        hidden_dim: 出力次元
        edge_dim: エッジ特徴次元
        dropout: ドロップアウト率
    """
    def __init__(self, in_dim: int, hidden_dim: int, edge_dim: int = 3, dropout: float = 0.2):
        super().__init__()
        self.layer1 = GraphConvShim(in_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout)
        self.layer2 = GraphConvShim(hidden_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout)

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            z: ノード特徴 [N, in_dim]
            edge_index: エッジインデックス [2, E]
            edge_attr: エッジ特徴 [E, edge_dim]

        Returns:
            h: 更新後のノード特徴 [N, hidden_dim]
        """
        h = self.layer1(z, edge_index, edge_attr)
        h = self.layer2(h, edge_index, edge_attr)
        return h
