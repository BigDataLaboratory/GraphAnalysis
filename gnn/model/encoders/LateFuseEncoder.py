import torch
import torch.nn as nn


class LateFuseEncoder(nn.Module):
    """
    Late-fusion encoder: applies a base encoder separately to 3 sub-graphs
    (retweet, reply, mention), then fuses the resulting embeddings per node.
    
    Each channel can have a different edge_dim.
    """

    def __init__(self, encoder_class, in_ch: int, hidden: int, embed_dim: int,
                 edge_dim_retweet: int = 3, edge_dim_reply: int = 3, edge_dim_mention: int = 3,
                 dropout: float = 0.5, fusion: str = 'mean'):
        super().__init__()
        
        # Instantiate 3 independent sub-encoders
        self.enc_retweet = encoder_class(
            in_channels=in_ch, hidden=hidden, out_channels=embed_dim, 
            edge_dim=edge_dim_retweet, dropout=dropout)
            
        self.enc_reply = encoder_class(
            in_channels=in_ch, hidden=hidden, out_channels=embed_dim, 
            edge_dim=edge_dim_reply, dropout=dropout)
            
        self.enc_mention = encoder_class(
            in_channels=in_ch, hidden=hidden, out_channels=embed_dim, 
            edge_dim=edge_dim_mention, dropout=dropout)
            
        self.fusion = fusion
        self.embed_dim = embed_dim

    @property
    def out_dim(self):
        """Output dimension after fusion."""
        return 3 * self.embed_dim if self.fusion == 'concat' else self.embed_dim

    def forward(self, x, edge_index=None, edge_attr=None, **kwargs):
        """
        Forward pass.
        Expects per-channel edge data in kwargs:
          - edge_index_retweet, edge_attr_retweet
          - edge_index_reply,   edge_attr_reply
          - edge_index_mention, edge_attr_mention
        
        The standard `edge_index` and `edge_attr` are ignored because 
        this encoder explicitly relies on the sub-graphs.
        """
        # Encode Retweet channel
        ei_rt = kwargs.get('edge_index_retweet')
        ea_rt = kwargs.get('edge_attr_retweet')
        if ei_rt is None:
            ei_rt = torch.empty((2, 0), dtype=torch.long, device=x.device)
        z_rt = self.enc_retweet(x, ei_rt, ea_rt)

        # Encode Reply channel
        ei_re = kwargs.get('edge_index_reply')
        ea_re = kwargs.get('edge_attr_reply')
        if ei_re is None:
            ei_re = torch.empty((2, 0), dtype=torch.long, device=x.device)
        z_re = self.enc_reply(x, ei_re, ea_re)

        # Encode Mention channel
        ei_mn = kwargs.get('edge_index_mention')
        ea_mn = kwargs.get('edge_attr_mention')
        if ei_mn is None:
            ei_mn = torch.empty((2, 0), dtype=torch.long, device=x.device)
        z_mn = self.enc_mention(x, ei_mn, ea_mn)

        # Fuse
        if self.fusion == 'mean':
            return torch.stack([z_rt, z_re, z_mn]).mean(dim=0)
        elif self.fusion == 'sum':
            return torch.stack([z_rt, z_re, z_mn]).sum(dim=0)
        else:  # concat
            return torch.cat([z_rt, z_re, z_mn], dim=-1)
