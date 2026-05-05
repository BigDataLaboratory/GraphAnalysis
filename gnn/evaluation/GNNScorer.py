import torch
from .BaselineScorer import BaselineScorer


class GNNScorer(BaselineScorer):
    """
    Uses a trained LinkPredictor (encoder + decoder) to score edges.
    """

    def __init__(self, model):
        self.model = model

    def score(self, data, edge_index):
        self.model.eval()
        with torch.no_grad():
            try:
                z = self.model.encode(data)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Fallback to CPU for encoding large graphs
                    device = next(self.model.parameters()).device
                    self.model.to('cpu')
                    data = data.to('cpu')
                    z = self.model.encode(data)
                    self.model.to(device)
                    data = data.to(device)
                    z = z.to(device)
                else:
                    raise e
                    
            # Batch decoding to prevent OOM
            batch_size = 100000
            all_scores = []
            for i in range(0, edge_index.size(1), batch_size):
                batch_ei = edge_index[:, i:i+batch_size].to(z.device)
                logits = self.model.decode(z, batch_ei).view(-1)
                all_scores.append(torch.sigmoid(logits))
                
            return torch.cat(all_scores)
