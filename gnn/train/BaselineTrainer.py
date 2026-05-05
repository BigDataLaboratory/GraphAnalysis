import torch
import torch.nn.functional as F
from gnn.utils.GraphUtils import GraphUtils


class BaselineTrainer:
    """
    Trains a baseline model (e.g., RawMLPConcatPredictor) on link prediction.
    The model must implement forward_logits(data, edge_index).
    """

    def __init__(self, model, lr=1e-3, pos_weight=1.0):
        self.model = model
        self.lr = lr
        self.pos_weight = pos_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, train_data, val_data, epochs=20, neg_ratio=1.0):
        """
        Train the baseline model on a single full graph split.
        Note: The baseline model (MLP) does not require neighbor sampling since it 
        only concatenates source and target features directly.
        """
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], device=self.device)
        )

        train_data = train_data.to(self.device)
        val_data = val_data.to(self.device)
        
        pos = GraphUtils.get_pos_edges_from_split(train_data)
        if pos is None:
            print("WARNING: No positive edges found in training data.")
            return self.model
            
        num_pos = pos.size(1)

        best_val_auc = 0.0
        best_state = None

        for ep in range(1, epochs + 1):
            self.model.train()
            opt.zero_grad()

            # Dynamic negative sampling for each epoch
            target_neg = int(neg_ratio * num_pos)
            neg = GraphUtils.sample_negatives_excluding_positives(
                train_data.edge_index, train_data.num_nodes, target_neg, self.device)
                
            if neg.size(1) == 0:
                continue

            # Combine positive and negative edges for supervision
            num_neg_sampled = neg.size(1)
            edge_index = torch.cat([pos, neg], dim=1)
            y = torch.cat([
                torch.ones(num_pos, device=self.device),
                torch.zeros(num_neg_sampled, device=self.device),
            ])

            logits = self.model.forward_logits(train_data, edge_index)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            # Validation step
            val_auc = self._evaluate_auc(val_data)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if ep % 10 == 0 or ep == 1:
                print(f"  [{self.model.__class__.__name__}] Epoch {ep:3d} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

        if best_state:
            self.model.load_state_dict(best_state)
            
        return self.model
        
    @torch.no_grad()
    def _evaluate_auc(self, data):
        """Compute AUC on a validation split."""
        from sklearn.metrics import roc_auc_score
        self.model.eval()
        
        pos = GraphUtils.get_pos_edges_from_split(data)
        neg = GraphUtils.get_neg_edges_from_split(data)
        
        if pos is None or neg is None:
            return 0.0
            
        edge_index = torch.cat([pos, neg], dim=1)
        y_true = torch.cat([
            torch.ones(pos.size(1)),
            torch.zeros(neg.size(1))
        ]).cpu().numpy()
        
        logits = self.model.forward_logits(data, edge_index)
        scores = torch.sigmoid(logits).cpu().numpy()
        
        return roc_auc_score(y_true, scores)
