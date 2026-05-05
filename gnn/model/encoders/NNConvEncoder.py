import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, BatchNorm


class NNConvEncoder(nn.Module):
    """2-layer NNConv encoder. Uses edge features natively via learned edge networks."""

    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 edge_dim: int = 0, dropout: float = 0.2):
        super().__init__()
        real_edge_dim = max(edge_dim, 1)
        self.nn1 = nn.Sequential(
            nn.Linear(real_edge_dim, 64), nn.ReLU(),
            nn.Linear(64, in_channels * hidden), nn.Tanh(),
        )
        self.conv1 = NNConv(in_channels, hidden, self.nn1, aggr='sum')
        self.bn1 = BatchNorm(hidden)

        self.nn2 = nn.Sequential(
            nn.Linear(real_edge_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden * out_channels), nn.Tanh(),
        )
        self.conv2 = NNConv(hidden, out_channels, self.nn2, aggr='sum')
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return x
