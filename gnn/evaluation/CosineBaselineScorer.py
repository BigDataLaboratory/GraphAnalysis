import torch.nn.functional as F
from .BaselineScorer import BaselineScorer


class CosineBaselineScorer(BaselineScorer):
    """
    Stateless cosine similarity scorer.
    Score = 0.5 * (cosine(src, dst) + 1), mapped to [0, 1].
    """

    def score(self, data, edge_index):
        x = data.x.float()
        s, d = edge_index[0], edge_index[1]
        cos = F.cosine_similarity(x[s], x[d], dim=1)
        return 0.5 * (cos + 1.0)
