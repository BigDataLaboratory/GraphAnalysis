import math
import glob
import os
import json
import pickle
import igraph as ig
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict

snapshot_dir = "/ipazianas/pasquini/twitter_graph_dump/feb-aug_2022"
files = sorted(glob.glob(os.path.join(snapshot_dir, "*.pkl")))

def read_pickle(fn):
    print(f"Reading file: {fn} (PID: {os.getpid()})")
    return ig.Graph.Read_Pickle(fn)

with ProcessPoolExecutor() as executor:
    graphs = list(executor.map(read_pickle, files))

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

chunks = chunkify(graphs, 30)  # Use 4 processes for merge

with ProcessPoolExecutor() as executor:
    partial_merged = list(executor.map(merge_graphs, chunks))

print(f"Final merge {len(partial_merged)} partial graphs (PID: {os.getpid()})")
full_g = merge_graphs(partial_merged)