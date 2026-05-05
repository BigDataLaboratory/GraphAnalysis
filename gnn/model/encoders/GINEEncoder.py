import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, BatchNorm


def _make_mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    """3-layer MLP used inside each GINEConv message-passing step."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class GINEEncoder(nn.Module):
    """
    2-layer GINE encoder with residual projection and L2-normalized output.

    Architecture improvements over the vanilla version:
    - Two GINEConv layers instead of one (deeper message passing).
    - Residual connection between layers: helps gradient flow and prevents
      over-smoothing on deep graphs.
    - L2 output normalization (normalize=True by default): projects the final
      embeddings onto the unit hypersphere. This is the correct way to apply
      "normalization at the output" for metric-learning / link-prediction tasks.
      It is equivalent in spirit to a softmax on cosine similarities, as it makes
      dot-product(u, v) == cosine_similarity(u, v) — crucial for the MLP decoder.

    Note on 'softmax at the output': applying nn.Softmax to node embeddings
    directly would collapse them into a probability simplex (all dims sum to 1),
    destroying the distance geometry needed for link prediction.
    L2-normalization is the right substitute for embedding models.
    """

    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 edge_dim: int = 0, dropout: float = 0.2, normalize: bool = True):
        """
        Args:
            in_channels:  Input node feature dimension.
            hidden:       Hidden dimension for MLP and intermediate layers.
            out_channels: Output embedding dimension.
            edge_dim:     Edge feature dimension (0 → treated as 1 internally).
            dropout:      Dropout probability applied between layers.
            normalize:    If True, L2-normalizes the output embeddings.
                          Recommended for link prediction (cosine-like geometry).
        """
        super().__init__()
        real_edge_dim = max(edge_dim, 1)
        self.normalize = normalize
        self.dropout = dropout

        # --- Layer 1: in_channels → hidden ---
        self.conv1 = GINEConv(
            _make_mlp(in_channels, hidden, hidden),
            edge_dim=real_edge_dim,
        )
        self.bn1 = BatchNorm(hidden)

        # --- Layer 2: hidden → out_channels ---
        self.conv2 = GINEConv(
            _make_mlp(hidden, hidden, out_channels),
            edge_dim=real_edge_dim,
        )
        self.bn2 = BatchNorm(out_channels)

        # Residual projection: align dim(layer1 output) with dim(layer2 output)
        # so we can add them. Only needed when hidden != out_channels.
        if hidden != out_channels:
            self.res_proj = nn.Linear(hidden, out_channels, bias=False)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x, edge_index, edge_attr=None):
        # Layer 1
        h = self.conv1(x, edge_index, edge_attr)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2 with residual
        res = self.res_proj(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = self.bn2(h)
        h = h + res                                # residual skip
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # L2 normalization: maps to unit sphere.
        # Makes dot(u,v) == cosine(u,v), which benefits both MLP and dot-product decoders.
        if self.normalize:
            h = F.normalize(h, p=2, dim=-1)

        return h
