from pathlib import Path

import torch
import torch.nn.functional as F

from gnn.model.ModelFactory import ModelFactory
from gnn.utils.GraphUtils import GraphUtils


class GNNTraining:
    def __init__(self, output_dir: str,
                 encoder: str, decoder: str,
                 in_channels: int, hidden_dim: int, embed_dim: int, edge_dim: int,
                 learning_rate: float, num_epochs: int, weight_decay: float,
                 dropout: float, neg_ratio: float, 
                 decoder_hidden: int, decoder_dropout: float,
                 scheduler: bool, grad_clip: float,
                 fusion: str,
                 late_fuse_base_encoder: str = None,
                 edge_dim_retweet: int = 3, edge_dim_reply: int = 3, edge_dim_mention: int = 3,
                 early_stopping_patience: int = 10):
        self.output_dir = Path(output_dir)
        self.encoder = encoder
        self.decoder = decoder
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.edge_dim = edge_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.neg_ratio = neg_ratio
        self.decoder_hidden = decoder_hidden
        self.decoder_dropout = decoder_dropout
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.fusion = fusion
        self.late_fuse_base_encoder = late_fuse_base_encoder
        self.edge_dim_retweet = edge_dim_retweet
        self.edge_dim_reply = edge_dim_reply
        self.edge_dim_mention = edge_dim_mention
        self.early_stopping_patience = early_stopping_patience

    def _link_pred_loss(self, z, pos_edge_index, decoder, neg_ratio=1.0):
        """BCE loss with dynamic negative sampling for the batch."""
        device = z.device
        num_pos = pos_edge_index.size(1)
        
        # Sample negatives on the fly for this batch
        # We sample randomly from the current batch's nodes
        target_neg = int(num_pos * neg_ratio)
        if target_neg == 0:
            target_neg = 1
            
        # Fast random negative sampling within the batch embeddings
        num_nodes_in_batch = z.size(0)
        neg_src = torch.randint(0, num_nodes_in_batch, (target_neg,), device=device)
        neg_dst = torch.randint(0, num_nodes_in_batch, (target_neg,), device=device)
        neg_edge_index = torch.stack([neg_src, neg_dst], dim=0)

        scores = torch.cat([decoder(z, pos_edge_index), decoder(z, neg_edge_index)])
        labels = torch.cat([
            torch.ones(num_pos, device=device),
            torch.zeros(target_neg, device=device),
        ])
        return F.binary_cross_entropy_with_logits(scores, labels)

    def train(self, train_loader, val_data):
        """
        Mini-batch training loop using LinkNeighborLoader.
        Returns the best (model) by validation AUC.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ModelFactory.create_predictor(
            encoder_name=self.encoder,
            decoder_name=self.decoder,
            in_channels=self.in_channels,
            hidden=self.hidden_dim,
            embed_dim=self.embed_dim,
            edge_dim=self.edge_dim,
            dropout=self.dropout,
            decoder_hidden=self.decoder_hidden,
            decoder_dropout=self.decoder_dropout,
            fusion=self.fusion,
            late_fuse_base_encoder=self.late_fuse_base_encoder,
            edge_dim_retweet=self.edge_dim_retweet,
            edge_dim_reply=self.edge_dim_reply,
            edge_dim_mention=self.edge_dim_mention,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        lr_scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
            if self.scheduler else None
        )

        best_val_auc = 0.0
        best_state = None
        no_improve = 0
        
        # Determine a safe filename based on whether it's late-fuse or standard
        enc_name_for_file = self.late_fuse_base_encoder if self.encoder == 'late-fuse' else self.encoder
        model_path = self.output_dir / f"best_model_{self.encoder}_{enc_name_for_file}.pt"

        # Early check if we are doing late fuse to map global to local indices
        is_late_fuse = getattr(model, '_is_late_fuse', False)

        for epoch in range(1, self.num_epochs + 1):
            model.train()
            total_loss = 0
            used_batches = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # --- Map LateFuse edges to local batch indices if needed ---
                if is_late_fuse:
                    # LinkNeighborLoader remaps batch.edge_index to local indices
                    # 0..batch.num_nodes-1, but custom attributes (edge_index_retweet
                    # etc.) keep their global IDs. We remap them here using a
                    # vectorized lookup table — O(E) instead of O(E*N).
                    num_global_nodes = batch.n_id.max().item() + 1
                    lookup = torch.full((num_global_nodes,), -1, dtype=torch.long, device=device)
                    lookup[batch.n_id] = torch.arange(batch.n_id.size(0), device=device)
                    
                    for etype in ['retweet', 'reply', 'mention']:
                        global_ei = getattr(batch, f'edge_index_{etype}')
                        if global_ei.size(1) == 0:
                            continue
                        
                        # Vectorized: remap and filter in one pass
                        local_src = lookup[global_ei[0]]
                        local_dst = lookup[global_ei[1]]
                        mask = (local_src >= 0) & (local_dst >= 0)
                        
                        setattr(batch, f'edge_index_{etype}',
                                torch.stack([local_src[mask], local_dst[mask]]))
                        
                        ea = getattr(batch, f'edge_attr_{etype}')
                        if ea is not None:
                            setattr(batch, f'edge_attr_{etype}', ea[mask])
                
                # Encode the sampled subgraph
                z = model.encode(batch)

                # Supervision edges in this batch (LinkNeighborLoader provides edge_label_index)
                pos_mask = batch.edge_label == 1
                pos_ei = batch.edge_label_index[:, pos_mask]
                
                if pos_ei.size(1) == 0:
                    continue

                loss = self._link_pred_loss(z, pos_ei, model.decoder, neg_ratio=self.neg_ratio)
                loss.backward()
                
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                    
                optimizer.step()
                total_loss += loss.item()
                used_batches += 1

            avg_loss = total_loss / max(used_batches, 1)

            # Full-graph Validation
            val_auc = self._evaluate_auc(model, val_data, device)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(model.state_dict(), model_path)
                no_improve = 0
            else:
                no_improve += 1

            if lr_scheduler:
                lr_scheduler.step(val_auc)

            if epoch % 10 == 0 or epoch == 1:
                lr_str = f" | LR: {optimizer.param_groups[0]['lr']:.6f}" if lr_scheduler else ""
                print(f"  Epoch {epoch:4d} | Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}{lr_str}")

            if self.early_stopping_patience and no_improve >= self.early_stopping_patience:
                print(f"  Early stopping triggered at epoch {epoch} (no improvement for {no_improve} epochs).")
                break

        if best_state:
            model.load_state_dict(best_state)
            
        return model

    @torch.no_grad()
    def _evaluate_auc(self, model, data, device):
        """Compute AUC over the full validation graph."""
        from sklearn.metrics import roc_auc_score
        model.eval()
        data = data.to(device)
        
        # Note: for very large graphs, full-graph encode might OOM. 
        # But for 16M nodes, if it OOMs we might need a NeighborLoader for validation too.
        # Assuming inference fits in memory (or using CPU if needed).
        # We can move data to cpu for encoding if GPU is full, but LinkPredictor expects device
        
        try:
            z = model.encode(data)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("  [Warning] OOM during full validation encode. Falling back to CPU for validation.")
                model.to('cpu')
                data = data.to('cpu')
                z = model.encode(data)
                model.to(device)  # Move back to GPU
            else:
                raise e

        pos = GraphUtils.get_pos_edges_from_split(data)
        neg = GraphUtils.get_neg_edges_from_split(data)
        
        if pos is None or neg is None:
            return 0.0
            
        pos = pos.to(z.device)
        neg = neg.to(z.device)

        # Batch decoding if edges are too many
        all_scores = []
        batch_size = 100000
        edge_index = torch.cat([pos, neg], dim=1)
        
        for i in range(0, edge_index.size(1), batch_size):
            batch_ei = edge_index[:, i:i+batch_size]
            scores = model.decode(z, batch_ei)
            all_scores.append(scores.cpu())
            
        scores = torch.sigmoid(torch.cat(all_scores)).numpy()
        labels = torch.cat([
            torch.ones(pos.size(1)), torch.zeros(neg.size(1))
        ]).numpy()
        
        return roc_auc_score(labels, scores)
