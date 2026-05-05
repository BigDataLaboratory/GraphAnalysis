import torch
from torch_geometric.utils import negative_sampling


class GraphUtils:

    @staticmethod
    def get_pos_edges_from_split(data):
        """Extract positive edge indices from a split Data object."""
        if not hasattr(data, 'edge_label_index') or not hasattr(data, 'edge_label'):
            return None
        mask = data.edge_label > 0
        return data.edge_label_index[:, mask] if mask.sum() > 0 else None

    @staticmethod
    def get_neg_edges_from_split(data):
        """Extract negative edge indices from a split Data object."""
        if not hasattr(data, 'edge_label_index') or not hasattr(data, 'edge_label'):
            return None
        mask = data.edge_label == 0
        return data.edge_label_index[:, mask] if mask.sum() > 0 else None

    @staticmethod
    def sample_negatives_excluding_positives(full_pos, num_nodes, target_neg, device=None):
        """Sample negative edges while excluding known positives."""
        if target_neg <= 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        req = int(target_neg * 1.3) + 10
        neg = negative_sampling(full_pos, num_nodes=num_nodes, num_neg_samples=req)
        if device:
            neg = neg.to(device)
        if neg.numel() > 0:
            mask = neg[0] != neg[1]
            neg = neg[:, mask]
        if neg.size(1) > target_neg:
            neg = neg[:, :target_neg]
        return neg
