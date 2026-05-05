from .encoders import (
    GCNEncoder, GATEncoder, GATv2Encoder, GINEEncoder,
    SAGEEncoder, TransformerEncoder, NNConvEncoder, LateFuseEncoder
)
from .Decoders import MLPDecoder, DotProductDecoder
from .LinkPredictor import LinkPredictor


class ModelFactory:
    """Factory for creating encoders, decoders, and full link predictors."""

    @staticmethod
    def _resolve_encoder_class(name: str):
        name = name.lower()
        if name == 'gcn': return GCNEncoder
        if name == 'gat': return GATEncoder
        if name == 'gatv2': return GATv2Encoder
        if name == 'gine': return GINEEncoder
        if name == 'sage': return SAGEEncoder
        if name == 'transformer': return TransformerEncoder
        if name == 'nnconv': return NNConvEncoder
        raise ValueError(f"Unknown encoder base name: {name}")

    @staticmethod
    def create_encoder(name: str, in_channels: int, hidden: int, embed_dim: int,
                       edge_dim: int = 0, dropout: float = 0.3):
        cls = ModelFactory._resolve_encoder_class(name)
        return cls(
            in_channels=in_channels,
            hidden=hidden,
            out_channels=embed_dim,
            edge_dim=edge_dim,
            dropout=dropout
        )

    @staticmethod
    def create_decoder(name: str, embed_dim: int, hidden_dim: int = 64, dropout: float = 0.3):
        name = name.lower()
        if name == 'mlp':
            return MLPDecoder(embed_dim, hidden_dim, dropout)
        elif name == 'dot':
            return DotProductDecoder()
        raise ValueError(f"Unknown decoder name: {name}")

    @staticmethod
    def create_predictor(encoder_name: str, decoder_name: str,
                         in_channels: int, hidden: int, embed_dim: int,
                         edge_dim: int = 0, dropout: float = 0.3,
                         decoder_hidden: int = 64, decoder_dropout: float = 0.3,
                         fusion: str = 'mean',
                         edge_dim_retweet: int = 3, edge_dim_reply: int = 3, edge_dim_mention: int = 3,
                         late_fuse_base_encoder: str = None) -> LinkPredictor:
        """
        Create a LinkPredictor.
        If encoder_name == 'late-fuse', it will instantiate a LateFuseEncoder using
        the late_fuse_base_encoder class.
        """
        encoder_name = encoder_name.lower()
        
        if encoder_name == 'late-fuse':
            if late_fuse_base_encoder is None:
                raise ValueError("late_fuse_base_encoder must be provided when using late-fuse graph_type")
                
            base_cls = ModelFactory._resolve_encoder_class(late_fuse_base_encoder)
            encoder = LateFuseEncoder(
                encoder_class=base_cls,
                in_ch=in_channels,
                hidden=hidden,
                embed_dim=embed_dim,
                edge_dim_retweet=edge_dim_retweet,
                edge_dim_reply=edge_dim_reply,
                edge_dim_mention=edge_dim_mention,
                dropout=dropout,
                fusion=fusion
            )
            dec_dim = encoder.out_dim
        else:
            encoder = ModelFactory.create_encoder(
                name=encoder_name,
                in_channels=in_channels,
                hidden=hidden,
                embed_dim=embed_dim,
                edge_dim=edge_dim,
                dropout=dropout
            )
            dec_dim = embed_dim

        decoder = ModelFactory.create_decoder(
            name=decoder_name,
            embed_dim=dec_dim,
            hidden_dim=decoder_hidden,
            dropout=decoder_dropout
        )
        
        return LinkPredictor(encoder, decoder)
