import torch
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, auc
)
from gnn.utils.GraphUtils import GraphUtils


class Evaluator:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def metrics_from_scores(self, y_true, y_score, thr):
        """Convert scores to binary predictions and compute metrics."""
        y_pred = (y_score >= thr).astype(int)
        return (
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0),
        )

    def _collect_scores(self, data, neg_ratio, scorer, full_pos_edges=None):
        """
        Score edges with the given scorer, and return concatenated labels and scores.
        data is a PyG Data object (from RandomLinkSplit).
        
        Args:
            full_pos_edges: Complete edge_index from the original (pre-split) graph.
                           Used to exclude ALL known positives when sampling negatives.
                           Falls back to data.edge_index if not provided.
        """
        data = data.to(self.device)
        pos_ei = GraphUtils.get_pos_edges_from_split(data)
        if pos_ei is None:
            return None, None
            
        pos_ei = pos_ei.to(self.device)
        num_nodes = data.num_nodes
        
        target_neg = int(pos_ei.size(1) * neg_ratio)
        if target_neg == 0:
            target_neg = 1
        
        # Use full_pos_edges to exclude ALL known positives (not just the split's MP edges)
        exclusion_edges = full_pos_edges.to(self.device) if full_pos_edges is not None else data.edge_index.to(self.device)
        neg_ei = GraphUtils.sample_negatives_excluding_positives(
            exclusion_edges, num_nodes, target_neg, self.device
        )
        
        if neg_ei.size(1) == 0:
            return None, None

        edge_index = torch.cat([pos_ei, neg_ei], dim=1)
        scores = scorer.score(data, edge_index).view(-1)
        
        y = torch.cat([
            torch.ones(pos_ei.size(1), device=self.device),
            torch.zeros(neg_ei.size(1), device=self.device),
        ])

        return y.cpu().numpy().astype(int), scores.cpu().numpy().astype(float)

    def find_best_threshold(self, val_data, neg_ratio, scorer, full_pos_edges=None):
        """Find the threshold maximizing F1 on the validation set."""
        y, p = self._collect_scores(val_data, neg_ratio, scorer, full_pos_edges=full_pos_edges)
        if y is None:
            return 0.5

        thrs = np.unique(p)
        best_t, best_f = 0.5, -1.0
        step = max(1, len(thrs) // 500)
        
        for t in thrs[::step]:
            _, _, _, f = self.metrics_from_scores(y, p, t)
            if f > best_f:
                best_f, best_t = f, float(t)
                
        return best_t

    def evaluate_metrics_master(self, name, data, neg_ratio, scorer, threshold=0.5, full_pos_edges=None):
        """
        Evaluate link prediction performance with PR-AUC support.
        """
        y, p = self._collect_scores(data, neg_ratio, scorer, full_pos_edges=full_pos_edges)
        if y is None:
            print(f"[{name}] No edges evaluated.")
            return

        acc, pr, re, f1 = self.metrics_from_scores(y, p, threshold)
        
        try:
            auc_roc = roc_auc_score(y, p)
            ap = average_precision_score(y, p)
            
            # PR-AUC
            prec_curve, rec_curve, _ = precision_recall_curve(y, p)
            pr_auc = auc(rec_curve, prec_curve)
        except ValueError:
            # Occurs if only one class is present
            auc_roc = 0.0
            ap = 0.0
            pr_auc = 0.0
            
        # Format matching the benchmark table
        res = {
            'Scorer': name,
            'Neg Ratio': f"1:{neg_ratio}",
            'F1': f1,
            'P': pr,
            'R': re,
            'Acc': acc,
            'AUC': auc_roc,
            'AP': ap,
            'PR-AUC': pr_auc
        }
        
        print(f"[{name} | 1:{neg_ratio} | Thr={threshold:.4f}] "
              f"F1={f1:.4f} P={pr:.4f} R={re:.4f} Acc={acc:.4f} "
              f"AUC={auc_roc:.4f} AP={ap:.4f} PR-AUC={pr_auc:.4f}")
              
        return res
