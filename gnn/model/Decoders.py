import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """Two-layer MLP decoder: [z_u || z_v] -> Linear -> ReLU -> Dropout -> Linear -> scalar."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        x = torch.cat([z[row], z[col]], dim=-1)
        return self.net(x).squeeze(-1)


class DotProductDecoder(nn.Module):
    """Simple dot-product decoder: score(u,v) = z_u . z_v (no learnable params)."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        return (z[row] * z[col]).sum(dim=-1)
