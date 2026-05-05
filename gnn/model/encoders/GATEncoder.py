import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """2-layer GAT encoder. Does not use edge features."""

    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 edge_dim: int = 0, dropout: float = 0.3):
        super().__init__()
        # edge_dim accepted for unified interface but unused
        self.conv1 = GATConv(in_channels, hidden, heads=4, concat=False)
        self.ln1 = nn.LayerNorm(hidden)
        self.conv2 = GATConv(hidden, out_channels, heads=4, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x
