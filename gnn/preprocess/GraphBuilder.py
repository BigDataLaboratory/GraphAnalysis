"""
GraphBuilder — Convert extracted CSV files to PyG Data objects.

Reads the extraction output (user_features, edges_retweet, edges_reply, edges_mention)
and builds PyTorch Geometric Data objects for link prediction.

Supports 4 graph types: mention, retweet, reply, late_fuse.
All nodes are always present regardless of edge type.
"""

import math
import os
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Utils.Utils import Utils

def _parse_uid(val):
    try:
        return int(val)
    except ValueError:
        return Utils.from_node_id(val)

class GraphBuilder:
    """Build PyG Data objects from extracted CSV files."""

    # Edge file names by type
    EDGE_FILES = {
        'retweet': 'edges_retweet',
        'reply': 'edges_reply',
        'mention': 'edges_mention',
    }

    def __init__(self, data_dir: str, graph_type: str):
        """
        Args:
            data_dir: Path to the extraction output directory (e.g. 'resources/<run_id>')
            graph_type: One of 'mention', 'retweet', 'reply', 'late_fuse'
        """
        self.data_dir = Path(data_dir)
        self.graph_type = graph_type

    def build_and_save(self) -> str:
        """Build and save the graph as a .pt file. Returns the path to the saved file."""
        output_dir = self.data_dir / 'graphs'
        output_dir.mkdir(parents=True, exist_ok=True)
        pt_path = output_dir / f'graph_{self.graph_type}.pt'

        if pt_path.exists():
            print(f"  [GraphBuilder] Graph already exists: {pt_path}")
            return str(pt_path)

        print(f"  [GraphBuilder] Building graph: {self.graph_type} from {self.data_dir}")

        # Load node features
        node_features, uid_to_idx = self._load_user_features()
        num_nodes = len(uid_to_idx)

        if self.graph_type == 'late_fuse':
            data = self._build_late_fuse(node_features, uid_to_idx, num_nodes)
        else:
            data = self._build_single_type(node_features, uid_to_idx, num_nodes, self.graph_type)

        torch.save(data, pt_path)
        print(f"  [GraphBuilder] Saved: {pt_path} | Nodes: {data.num_nodes} | "
              f"Edges: {data.edge_index.size(1)}")
        return str(pt_path)

    # ----------------------------------------------------------------
    # Node features
    # ----------------------------------------------------------------
    def _load_user_features(self):
        """Load user features CSV and return (feature_tensor, uid_to_idx mapping)."""
        feat_path = self.data_dir / 'user_features'
        if not feat_path.exists():
            raise FileNotFoundError(f"User features not found: {feat_path}")

        uid_to_idx = {}
        features = []

        with open(feat_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                # Format: user_id, total, retweets, replies, original, likes,
                #         followers, following, verified, account_age_days,
                #         n_unique_hashtags, n_unique_mentions, screen_name
                if len(parts) < 13:
                    continue

                uid = _parse_uid(parts[0])
                if uid not in uid_to_idx:
                    uid_to_idx[uid] = len(uid_to_idx)

                # 11 numeric features (exclude user_id and screen_name)
                raw = [float(parts[i]) for i in range(1, 12)]
                features.append(self._build_feature_vector(raw))

        node_features = torch.tensor(features, dtype=torch.float)
        return node_features, uid_to_idx

    @staticmethod
    def _build_feature_vector(raw_values: list) -> list:
        """Convert raw user stats to feature vector.

        To disable normalization, comment out the normalize block below
        and uncomment the raw passthrough.
        """
        # raw_values = [total, retweets, replies, original, likes,
        #               followers, following, verified, account_age_days,
        #               n_unique_hashtags, n_unique_mentions]

        # --- RAW PASSTHROUGH (uncomment to disable normalization) ---
        # return [float(v) for v in raw_values]

        # --- NORMALIZED (comment out to use raw values) ---
        # Each divisor controls the log1p scaling for that feature.
        # Higher divisor → more compression. 1 means binary passthrough.
        NORMS = [
            10,   # total tweets:        log1p(x)/10
            10,   # retweets:            log1p(x)/10
            10,   # replies:             log1p(x)/10
            10,   # original tweets:     log1p(x)/10
            15,   # likes:               log1p(x)/15  (wider range)
            15,   # followers:           log1p(x)/15  (wider range)
            12,   # following:           log1p(x)/12
            1,    # verified:            binary passthrough
            10,   # account_age_days:    log1p(x)/10
            8,    # n_unique_hashtags:   log1p(x)/8
            8,    # n_unique_mentions:   log1p(x)/8
        ]
        result = []
        for v, div in zip(raw_values, NORMS):
            if div == 1:  # binary feature (verified)
                result.append(float(v))
            else:
                result.append(math.log1p(max(float(v), 0)) / div)
        return result

    # ----------------------------------------------------------------
    # Edge loading
    # ----------------------------------------------------------------
    def _load_edges(self, edge_type: str, uid_to_idx: dict):
        """Load an edge CSV and return (edge_index [2, E], edge_attr [E, D]).

        Edge file format: src_uid, dst_uid, weight[, extra_features...]
        All columns after src and dst are treated as edge features.
        """
        edge_file = self.data_dir / self.EDGE_FILES[edge_type]
        if not edge_file.exists():
            print(f"    WARNING: Edge file not found: {edge_file}")
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 1), dtype=torch.float),
            )

        src_list, dst_list = [], []
        feat_list = []

        with open(edge_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) < 3:
                    continue

                src_uid = _parse_uid(parts[0])
                dst_uid = _parse_uid(parts[1])

                # Skip edges with unknown nodes
                if src_uid not in uid_to_idx or dst_uid not in uid_to_idx:
                    continue

                src_list.append(uid_to_idx[src_uid])
                dst_list.append(uid_to_idx[dst_uid])

                # All columns from index 2 onward are edge features
                raw_feats = [float(p) for p in parts[2:]]
                feat_list.append(raw_feats)

        if not src_list:
            return (
                torch.empty((2, 0), dtype=torch.long),
                torch.empty((0, 1), dtype=torch.float),
            )

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

        # Build edge_attr — normalize weight with log1p
        feat_tensor = torch.tensor(feat_list, dtype=torch.float)
        # Normalize the first column (weight) with log1p / 5
        if feat_tensor.size(1) > 0:
            feat_tensor[:, 0] = torch.log1p(feat_tensor[:, 0]) / 5.0

        return edge_index, feat_tensor

    # ----------------------------------------------------------------
    # Structural features (lightweight, scalable to 16M nodes)
    # ----------------------------------------------------------------
    @staticmethod
    def _compute_structural_features(edge_index, num_nodes):
        """Compute lightweight structural features from graph topology.

        Only O(E) features are computed here (degree-based).
        Heavy features like PageRank should be pre-computed separately.
        """
        in_deg = torch.zeros(num_nodes, dtype=torch.float)
        out_deg = torch.zeros(num_nodes, dtype=torch.float)

        if edge_index.size(1) > 0:
            in_deg.index_add_(0, edge_index[1], torch.ones(edge_index.size(1)))
            out_deg.index_add_(0, edge_index[0], torch.ones(edge_index.size(1)))

        deg_ratio = out_deg / (in_deg + 1.0)  # avoid div by zero

        return torch.stack([
            torch.log1p(in_deg) / 10.0,
            torch.log1p(out_deg) / 10.0,
            deg_ratio.clamp(max=5.0) / 5.0,  # cap and normalize
        ], dim=1)

    # ----------------------------------------------------------------
    # Graph builders
    # ----------------------------------------------------------------
    def _build_single_type(self, node_features, uid_to_idx, num_nodes, edge_type):
        """Build a single-type graph (mention, retweet, or reply)."""
        edge_index, edge_attr = self._load_edges(edge_type, uid_to_idx)

        # Compute structural features and append to node features
        struct_feats = self._compute_structural_features(edge_index, num_nodes)
        x = torch.cat([node_features, struct_feats], dim=1)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
        )
        return data

    def _build_late_fuse(self, node_features, uid_to_idx, num_nodes):
        """Build a late-fuse graph with all 3 edge types."""
        all_edges = []
        edge_data = {}

        for etype in ['retweet', 'reply', 'mention']:
            ei, ea = self._load_edges(etype, uid_to_idx)
            edge_data[etype] = (ei, ea)
            if ei.size(1) > 0:
                all_edges.append(ei)

        # Union edge_index for RandomLinkSplit
        if all_edges:
            all_edge_index = torch.cat(all_edges, dim=1)
        else:
            all_edge_index = torch.empty((2, 0), dtype=torch.long)

        # Structural features from the union graph
        struct_feats = self._compute_structural_features(all_edge_index, num_nodes)
        x = torch.cat([node_features, struct_feats], dim=1)

        data = Data(
            x=x,
            edge_index=all_edge_index,
            edge_attr=None,  # not used for MP in late-fuse (each channel has its own)
            num_nodes=num_nodes,
        )

        # Store per-channel edges
        for etype in ['retweet', 'reply', 'mention']:
            ei, ea = edge_data[etype]
            setattr(data, f'edge_index_{etype}', ei)
            setattr(data, f'edge_attr_{etype}', ea)
            setattr(data, f'edge_dim_{etype}', ea.size(1) if ea.size(1) > 0 else 1)

        return data
