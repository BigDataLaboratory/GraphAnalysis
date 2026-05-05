import torch
from .BaselineScorer import BaselineScorer


class MLPBaselineScorer(BaselineScorer):
    """
    Wraps a trained MLP (e.g. RawMLPConcatPredictor) to produce probabilities.
    """

    def __init__(self, model):
        self.model = model

    def score(self, data, edge_index):
        self.model.eval()
        with torch.no_grad():
            logits = self.model.forward_logits(data, edge_index)
            return torch.sigmoid(logits)
