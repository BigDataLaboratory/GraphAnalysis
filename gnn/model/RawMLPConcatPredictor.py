import torch
import torch.nn as nn


class RawMLPConcatPredictor(nn.Module):
    """
    Simple MLP baseline that concatenates raw source and destination node features
    and passes them through an MLP. No GNN message passing is performed.
    """
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward_logits(self, data, edge_index):
        x = data.x
        s, d = edge_index[0], edge_index[1]
        h = torch.cat([x[s], x[d]], dim=1)
        return self.net(h).view(-1)
