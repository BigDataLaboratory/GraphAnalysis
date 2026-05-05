import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv


class TransformerEncoder(nn.Module):
    """2-layer Transformer encoder. Uses edge features natively."""

    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 edge_dim: int = 0, dropout: float = 0.2):
        super().__init__()
        real_edge_dim = max(edge_dim, 1)
        self.conv1 = TransformerConv(in_channels, hidden, edge_dim=real_edge_dim,
                                     heads=4, concat=False, beta=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.conv2 = TransformerConv(hidden, out_channels, edge_dim=real_edge_dim,
                                     heads=4, concat=False, beta=True)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.ln1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x
