#!/usr/bin/env python3
"""
compute_hashtag_entropy_from_graph.py

Purpose:
- Given a Leiden CSV (columns: id, name, type, 0.1, 0.2, ..., 1.0) and a pickled graph,
  build user->hashtags usage from the graph edges and compute per-community hashtag
  entropy (and dominant share) for each resolution.

Usage:
    python compute_hashtag_entropy_from_graph.py \
        --leiden_csv retweet_plus_hashtag_leiden.csv \
        --graph_pickle retweet_plus_hashtag_graph.pkl \
        --outdir results_rt_rh

Notes:
- The script tries to be flexible about the pickled graph format:
  it first tries networkx.read_gpickle, then pickle.load and inspects the object.
  It supports NetworkX Graph / DiGraph / MultiGraph and igraph Graph (if python-igraph is installed).
- The Leiden CSV is used to map node 'name' -> 'type' (user/hashtag) and to get community labels.
- The graph is expected to contain the same node names in some attribute or as node keys. The script
  will attempt to match by node name string. If your node ids differ from the 'name' field, adapt the mapping section.

Outputs (in --outdir):
- entropy_per_community_per_resolution.csv
- mean_entropy_by_resolution.csv
- plot entropy mean vs resolution: entropy_mean_plot.png
"""

import argparse
import os
import glob
import sys
import math
import pickle
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import json

# Try import igraph

try:
    import igraph as ig
except Exception:
    ig = None

# --- Utilities ---

def shannon_entropy_from_counts(counts, base=math.e):
    """
    counts: iterable of integer counts
    returns entropy (natural log base by default). To get bits use base=2.
    """
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p, base)
    return ent

# --- Graph loading / user-hashtag extraction ---

def read_pickle(fn):
    print(f"Reading file: {fn} (PID: {os.getpid()})")
    return ig.Graph.Read_Pickle(fn)

def merge_graphs(graph_list):
    """Sequentially merges a list of igraph.Graph objects using name/type identity."""
    print(f"Merging {len(graph_list)} graphs (PID: {os.getpid()})")
    names, types = [], []
    name2idx = {}
    for num_g, g in enumerate(graph_list):
        print(f"Processing graph {num_g} / {len(graph_list)} with {len(g.vs)} vertices and {len(g.es)} edges (PID: {os.getpid()})")
        for v in g.vs:
            n = v["name"]
            if n not in name2idx:
                name2idx[n] = len(names)
                names.append(n)
                types.append(v["type"])
            else:
                assert v["type"] == types[name2idx[n]]

    master = ig.Graph(directed=True)
    master.add_vertices(len(names))
    master.vs["name"] = names
    master.vs["type"] = types

    sources, targets, types_edge, weights = [], [], [], []
    for num_g, g in enumerate(graph_list):
        print(f"Graph {num_g}/{len(graph_list)}, adding edges from graph with {len(g.vs)} vertices and {len(g.es)} edges (PID: {os.getpid()})")
        src_ids = [name2idx[n] for n in g.vs["name"]]
        for e in g.es:
            sources.append(src_ids[e.source])
            targets.append(src_ids[e.target])
        types_edge.extend(g.es["type"])
        weights.extend(g.es["weight"])

    master.add_edges(list(zip(sources, targets)))
    master.es["type"] = types_edge
    master.es["weight"] = weights
    print(f"Finished merging graphs (PID: {os.getpid()})")
    return master

def chunkify(lst, n):
    size = math.ceil(len(lst) / n)
    return [lst[i:i + size] for i in range(0, len(lst), size)]

def try_load_graph(path):
    """
    Try several ways to load a pickled graph:
    - networkx.read_gpickle (if networkx available and file looks like gpickle)
    - pickle.load (and trust the returned object)
    - if igraph installed and file ends with .graphml or .gml attempt igraph read (not used for pickles)
    Returns loaded object (could be networkx.Graph, igraph.Graph, or anything).
    """
    try:
        snapshot_dir = path
        files = sorted(glob.glob(os.path.join(snapshot_dir, "*.pkl")))

        with ProcessPoolExecutor() as executor:
            graphs = list(executor.map(read_pickle, files))

        chunks = chunkify(graphs, 15)  # Use 4 processes for merge

        with ProcessPoolExecutor() as executor:
            partial_merged = list(executor.map(merge_graphs, chunks))

        print(f"Final merge {len(partial_merged)} partial graphs (PID: {os.getpid()})")
        full_g = merge_graphs(partial_merged)

        count_u = sum(1 for v in full_g.vs if v["type"] == "u")
        print("Number of 'u' nodes:", count_u)

        count_h = sum(1 for v in full_g.vs if v["type"] == "h")
        print("Number of 'h' nodes:", count_h)
        return full_g
    except Exception as e:
            print("pickle.load failed:", e)
        

def build_user_hashtag_mapping_from_graph(graph, name_to_type, name_nodekey_hint=None):
    """
    Build mapping user -> list of hashtags (usage events) from the graph.
    Parameters:
      - graph: networkx.Graph/DiGraph/MultiGraph or igraph.Graph
      - name_to_type: dict mapping node name string -> 'user' or 'hashtag' (from Leiden CSV)
      - name_nodekey_hint: optional function(node) -> node_name_string if graph uses node attributes
    Returns:
      - user_hashtags: dict user_name -> list of hashtag names (may contain duplicates if multiple edges)
    Notes:
      - The function tries multiple strategies to map graph nodes to the 'name' used in the Leiden CSV:
        * if graph is networkx and node key is a string matching name, use it directly
        * else if node attribute 'name' exists, use node['name']
        * else if user supplied a hint function, use it
      - For MultiGraphs, duplicates count as multiple usages (useful).
    """
    user_hashtags = defaultdict(list)

    # igraph helper
    def get_node_name_igraph(idx, g):
        # try vertex attribute 'name' or 'label'
        if 'name' in g.vs.attribute_names():
            vname = str(g.vs[idx]['name'])
            if vname in name_to_type:
                return vname
        if 'label' in g.vs.attribute_names():
            vname = str(g.vs[idx]['label'])
            if vname in name_to_type:
                return vname
        # fallback to str(idx)
        if str(idx) in name_to_type:
            return str(idx)
        return None

    # igraph case
    if ig is not None:
        # iterate edges
        for e in graph.es:
            u_idx = e.tuple[0]
            v_idx = e.tuple[1]
            nu = get_node_name_igraph(u_idx, graph)
            nv = get_node_name_igraph(v_idx, graph)
            if nu is None or nv is None:
                continue
            tu = name_to_type.get(nu, '').lower()
            tv = name_to_type.get(nv, '').lower()
            if tu == 'u' and tv != 'u':
                user_hashtags[nu].append(nv)
            elif tv == 'u' and tu != 'u':
                user_hashtags[nv].append(nu)
        return dict(user_hashtags)

    raise RuntimeError("Graph type not recognized or missing graph library (networkx/igraph).")

# --- Main analysis ---

def process_community(res, comm, users_list, df, user_hashtags, type_col, name_col, normalize=True):
    from collections import Counter

    hashtags_flat = []
    for u in users_list:
        hashtags_flat.extend(user_hashtags.get(u, []))

    n_users = len(users_list)
    n_hashtag_nodes_in_comm = len([
        n for n in df[df[res].astype(str) == str(comm)].itertuples()
        if getattr(n, type_col) != 'u'
    ])
    n_hashtag_usages = len(hashtags_flat)
    unique_hashtags = set(hashtags_flat)

    if n_hashtag_usages == 0:
        entropy = None
        dominant_share = None
    else:
        counts = Counter(hashtags_flat)
        entropy = shannon_entropy_from_counts(counts.values(), base=2)
        if normalize and len(unique_hashtags) > 1:
            entropy = entropy / math.log2(len(unique_hashtags))
        dominant_share = max(counts.values()) / n_hashtag_usages

    return {
        'resolution': res,
        'community': str(comm),
        'n_users': n_users,
        'n_hashtag_nodes_in_comm': n_hashtag_nodes_in_comm,
        'n_hashtag_usages_by_users': n_hashtag_usages,
        'n_unique_hashtags_used': len(unique_hashtags),
        'entropy_bits_normalized' if normalize else 'entropy_bits': entropy,
        'dominant_share': dominant_share
    }


def analyze_entropy_from_graph(leiden_csv, graph_pickle, outdir,
                               resolutions=None, normalize=True, name_col='name', type_col='type'):
    """
    Parallel version: communities are processed in parallel across resolutions.
    """
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(leiden_csv, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # name->type map
    name_to_type = {}
    for _, row in df[[name_col, type_col]].iterrows():
        name_to_type[str(row[name_col])] = str(row[type_col]).strip().lower()

    # detect resolution columns
    if resolutions is None:
        resolutions = [c for c in df.columns if c.replace('.', '', 1).isdigit()]
        resolutions = sorted(resolutions, key=lambda x: float(x))

    print("Resolutions detected:", resolutions)

    # load graph
    G = try_load_graph(graph_pickle)

    # build user->hashtags mapping
    user_hashtags = build_user_hashtag_mapping_from_graph(G, name_to_type)
    print(f"Found {len(user_hashtags)} users with at least one hashtag usage in graph.")

    # ensure all users exist in mapping
    users_in_csv = [n for n, t in name_to_type.items() if t == 'u']
    for u in users_in_csv:
        user_hashtags.setdefault(u, [])

    # --- parallel processing ---
    rows = []
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor() as executor:
        futures = []
        for res in resolutions:
            users = df[df[type_col].str.lower() == 'u'][[name_col, res]].dropna()
            user_to_comm = dict(zip(users[name_col].astype(str), users[res].astype(str)))

            comm_to_users = defaultdict(list)
            for u, comm in user_to_comm.items():
                comm_to_users[comm].append(u)

            for comm, users_list in comm_to_users.items():
                futures.append(
                    executor.submit(process_community, res, comm, users_list,
                                    df, user_hashtags, type_col, name_col, normalize)
                )

        for f in as_completed(futures):
            rows.append(f.result())

    # --- collect results ---
    out_df = pd.DataFrame(rows)
    out_csv = os.path.join(outdir, 'entropy_per_community_per_resolution.csv')
    out_df.to_csv(out_csv, index=False)
    print("Wrote", out_csv)

    ent_col = 'entropy_bits_normalized' if normalize else 'entropy_bits'
    summary = out_df.dropna(subset=[ent_col]).groupby('resolution')[ent_col] \
                    .agg(['mean', 'median', 'count']).reset_index()
    summary_csv = os.path.join(outdir, 'mean_entropy_by_resolution.csv')
    summary.to_csv(summary_csv, index=False)
    print("Wrote", summary_csv)

    try:
        plt.figure(figsize=(8,5))
        xs = [float(x) for x in summary['resolution'].tolist()]
        ys = summary['mean'].tolist()
        plt.plot(xs, ys, marker='o')
        plt.xlabel('resolution (gamma)')
        plt.ylabel('mean normalized entropy (bits / log2(unique_hashtags))' if normalize else 'mean entropy (bits)')
        plt.title('Mean community hashtag entropy across resolutions')
        plt.grid(alpha=0.2)
        plot_path = os.path.join(outdir, 'entropy_mean_plot.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print("Wrote plot:", plot_path)
    except Exception as e:
        print("Plotting failed:", e)

    return out_df, summary


# --- CLI ---

def parse_args():
    p = argparse.ArgumentParser(description="Compute hashtag entropy per community using graph usage edges.")
    p.add_argument('--leiden_csv', required=True, help='Leiden CSV with id,name,type, and resolution columns')
    p.add_argument('--graph_pickle', required=True, help='Path to pickled graph (NetworkX pickle or similar)')
    p.add_argument('--outdir', required=True, help='Output directory')
    p.add_argument('--resolutions', nargs='*', help='Optional list of resolution columns (e.g. 0.1 0.2)')
    p.add_argument('--no_normalize', action='store_true', help='Do not normalize entropy by log2(#unique_hashtags)')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    res = args.resolutions if args.resolutions else None
    normalize = True
    analyze_entropy_from_graph(args.leiden_csv, args.graph_pickle, args.outdir, resolutions=res, normalize=normalize)
