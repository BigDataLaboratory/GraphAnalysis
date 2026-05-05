import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """2-layer GCN encoder. Does not use edge features."""

    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 edge_dim: int = 0, dropout: float = 0.3):
        super().__init__()
        # edge_dim accepted for unified interface but unused
        self.conv1 = GCNConv(in_channels, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.conv2 = GCNConv(hidden, out_channels)
        self.dropout = dropout
        self.scale = out_channels ** 0.5

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x / self.scale
