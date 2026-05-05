import torch
from torch import nn
from torch_geometric.data import Data

from .encoders.LateFuseEncoder import LateFuseEncoder


class LinkPredictor(nn.Module):
    """
    Wraps a GNN encoder and a decoder for link prediction.
    Handles standard graphs as well as LateFuse logic by passing extra kwargs.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._is_late_fuse = isinstance(encoder, LateFuseEncoder)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, data: Data) -> torch.Tensor:
        """Run the GNN encoder on a Data object and return node embeddings."""
        data = data.to(self.device)
        
        if self._is_late_fuse:
            # Pass all per-channel tensors directly
            return self.encoder(
                data.x, 
                edge_index=data.edge_index, 
                edge_attr=getattr(data, 'edge_attr', None),
                edge_index_retweet=getattr(data, 'edge_index_retweet', None),
                edge_attr_retweet=getattr(data, 'edge_attr_retweet', None),
                edge_index_reply=getattr(data, 'edge_index_reply', None),
                edge_attr_reply=getattr(data, 'edge_attr_reply', None),
                edge_index_mention=getattr(data, 'edge_index_mention', None),
                edge_attr_mention=getattr(data, 'edge_attr_mention', None),
            )
            
        # Standard encoding (edge_attr is always passed, the encoder will ignore it if it doesn't need it)
        return self.encoder(data.x, data.edge_index, getattr(data, 'edge_attr', None))

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        """Score candidate edges from node embeddings."""
        edge_label_index = edge_label_index.to(z.device)
        return self.decoder(z, edge_label_index)

    def forward(self, data: Data, edge_label_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(data)
        return self.decode(z, edge_label_index)
