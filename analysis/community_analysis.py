import pandas as pd
import igraph as ig
import numpy as np
import glob
import os
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from sklearn.metrics import (
    normalized_mutual_info_score,
    mutual_info_score,
    adjusted_rand_score,
    silhouette_score
)

import math

from sklearn.metrics.cluster import entropy
from scipy.stats import entropy

def comm_conductance(graph, vertex_ids):
    """
    vertex_ids: list[int]  (indices, not names)
    returns: float   (0 = perfectly separated, 1 = fully mixed)
    """
    return graph.conductance(vertex_ids)   # uses edge weights if 'weight' attr exists

def variation_of_information(labels_true, labels_pred):
    """Compute VI between two labelings."""
    H_true = entropy(np.bincount(labels_true))
    H_pred = entropy(np.bincount(labels_pred))
    I = mutual_info_score(labels_true, labels_pred)
    return H_true + H_pred - 2 * I


snapshot_dir = "/ipazianas/pasquini/twitter_graph_dump/feb-aug_2022"
files = sorted(glob.glob(os.path.join(snapshot_dir, "*.pkl")))

def read_pickle(fn):
    print(f"Reading file: {fn} (PID: {os.getpid()})")
    return ig.Graph.Read_Pickle(fn)

if True:
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

if True:
    chunks = chunkify(graphs, 30)  # Use 4 processes for merge

    with ProcessPoolExecutor() as executor:
        partial_merged = list(executor.map(merge_graphs, chunks))

    print(f"Final merge {len(partial_merged)} partial graphs (PID: {os.getpid()})")
    full_g = merge_graphs(partial_merged)

    count_u = sum(1 for v in full_g.vs if v["type"] == "u")
    print("Number of 'u' nodes:", count_u)

    count_h = sum(1 for v in full_g.vs if v["type"] == "h")
    print("Number of 'h' nodes:", count_h)

if True:
    df = pd.read_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/6273205e640511f08e0b08f1eaf4fe18/nodes_with_communities.csv", 
                    sep=",", 
                    header=0, low_memory=False)

    res_cols = sorted([c for c in df.columns if c.replace('.', '', 1).isdigit()],
                    key=float)          # ['0.1','0.2', … '1.0']

    labels_by_res = {r: df[str(r)].to_numpy() for r in res_cols}

    # number of communities at each resolution
    n_clusters = [pd.Series(labels_by_res[r]).nunique() for r in res_cols]

    print(f"Number of communities at each resolution: {n_clusters}")

basic_analysis = True
if basic_analysis:
    # ---- Basic input ----
    edges = np.asarray(full_g.get_edgelist())
    e_src, e_tgt = edges[:, 0], edges[:, 1]
    e_w = np.asarray(full_g.es["weight"], dtype=float)
    w_tot = e_w.sum()

    # Partition similarity between *adjacent* resolutions (NMI & ARI)
    nmi_adj, ari_adj, vi_adj = [], [], []
    coverage_rows = []
    size_rows = []

    for col in res_cols:
        print(f"Computing metrics for resolution = {col}")
        r = float(col)
        labels = df[col].to_numpy()

        # Coverage
        same_comm = labels[e_src] == labels[e_tgt]
        coverage = e_w[same_comm].sum() / w_tot
        coverage_rows.append({"resolution": r, "coverage": coverage})

        # community size stats
        comm_sizes = Counter(labels).values()
        comm_sizes = np.fromiter(comm_sizes, dtype=int)
        size_rows.append({
            "resolution": r,
            "count": comm_sizes.size,
            "mean": np.mean(comm_sizes),
            "median": np.median(comm_sizes),
            "max": np.max(comm_sizes),
        })
        
    for r1, r2 in zip(res_cols[:-1], res_cols[1:]):
        nmi_adj.append(normalized_mutual_info_score(labels_by_res[r1], labels_by_res[r2]))
        ari_adj.append(adjusted_rand_score(labels_by_res[r1], labels_by_res[r2]))
        vi = variation_of_information(labels_by_res[r1], labels_by_res[r2])
        vi_adj.append(vi)

    metrics_df = pd.DataFrame({
    "res_lo": res_cols[:-1],
    "res_hi": res_cols[1:],
    "nmi": nmi_adj,
    "ari": ari_adj,
    "vi": vi_adj
    })
    if False:
        metrics_df.to_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/6273205e640511f08e0b08f1eaf4fe18/hierarchical/bf5b3056674b11f0838c08f1eaf4fe18/metrics_nmi_ari_vi.csv", index=False)
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/6273205e640511f08e0b08f1eaf4fe18/coverage_by_resolution.csv", index=False)
    size_df = pd.DataFrame(size_rows)
    size_df.to_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/6273205e640511f08e0b08f1eaf4fe18/sizes_by_resolution.csv", index=False)


read_graphs = False
if read_graphs:
    full_g = ig.Graph(directed=True)
    temporal_graph = False
    for i, fn in enumerate(files):
        print(f"Reading pickle #{i} / {len(files)}, file: {fn}")
        sg = ig.Graph.Read_Pickle(fn)
        if full_g.vcount() == 0 and full_g.ecount() == 0:
            full_g = sg
        else:
            graphs = [full_g, sg]
            names, types = [], []
            name2idx     = {}

            for g in graphs:
                for v in g.vs:
                    n = v["name"]
                    if n not in name2idx:               # first time we see this name
                        name2idx[n] = len(names)
                        names.append(n)
                        types.append(v["type"])
                    else:
                        assert v["type"] == types[name2idx[n]], f"vertex {n!r} carries inconsistent type"
            
            master   = ig.Graph(directed=True)
            master.add_vertices(len(names))
            master.vs["name"] = names
            master.vs["type"] = types

            sources, targets = [], []
            types_edge,  weights  = [], []

            times = [] if temporal_graph else None

            for g in graphs:
                # translate local vertex IDs to master IDs *vectorised*
                src_ids = [name2idx[n] for n in g.vs["name"]]          # list is OK once
                for e in g.es:
                    sources.append(src_ids[e.source])
                    targets.append(src_ids[e.target])
                types_edge.extend(g.es["type"])
                weights.extend(g.es["weight"])
                if temporal_graph:
                    times.extend(g.es["time"])
            master.add_edges(list(zip(sources, targets)))
            master.es["type"]   = types_edge
            master.es["weight"] = weights
            if temporal_graph:
                master.es["time"] = times
            full_g = master


def collapse_small(labels, min_size=10, dummy=-1):
    labels = np.asarray(labels)
    counts = Counter(labels)
    keep = set()
    for k, v in counts.items():
        if v >= min_size:
            keep.add(k)
    #keep = {k for k, v in counts.items() if v >= min_size}
    return np.array([l if l in keep else dummy for l in labels])

collapse_dataset = False

if collapse_dataset:
    # ---- Basic input ----
    edges = np.asarray(full_g.get_edgelist())
    e_src, e_tgt = edges[:, 0], edges[:, 1]
    e_w = np.asarray(full_g.es["weight"], dtype=float)
    w_tot = e_w.sum()

    min_sizes = [10, 20, 30, 40, 50]
    dummy_id = -1


    for min_size in min_sizes:
        # ---- Outputs ----
        coverage_rows = []
        size_rows = []
        nmi_vals = []
        ari_vals = []
        vi_vals = []
        collapsed = {}
        for col in res_cols:
            print(f"Computing metrics for resolution = {col}")
            r = float(col)
            labels = df[col].to_numpy()
            print(f"min_size = {min_size}, number of communities before collapsing: {len(set(labels))}")
            labels_c = collapse_small(labels, min_size=min_size, dummy=dummy_id)
            print(f"number of communities after collapsing: {len(set(labels_c))}")
            collapsed[col] = labels_c

            # Coverage
            same_comm = labels_c[e_src] == labels_c[e_tgt]
            coverage = e_w[same_comm].sum() / w_tot

            # Kept communities
            kept = labels_c != dummy_id
            num_comms = len(set(labels_c[kept]))

            # Community size stats
            comm_sizes = Counter(labels_c[kept]).values()
            comm_sizes = np.fromiter(comm_sizes, dtype=int)
            size_rows.append({
                "resolution": r,
                "count": comm_sizes.size,
                "mean": np.mean(comm_sizes),
                "median": np.median(comm_sizes),
                "max": np.max(comm_sizes),
            })

            # Store coverage and count
            coverage_rows.append({
                "resolution": r,
                "coverage": coverage,
                "kept_comms": num_comms
            })

        # ---- ARI / NMI between adjacent resolutions ----
        for i in range(len(res_cols) - 1):
            c1, c2 = res_cols[i], res_cols[i + 1]
            labels1, labels2 = collapsed[c1], collapsed[c2]
            mask = (labels1 != dummy_id) & (labels2 != dummy_id)
            if mask.sum() == 0:
                nmi, ari = np.nan, np.nan
                vi = np.nan
            else:
                nmi = normalized_mutual_info_score(labels1[mask], labels2[mask])
                ari = adjusted_rand_score(labels1[mask], labels2[mask])
                vi = variation_of_information(labels1[mask], labels2[mask])
            nmi_vals.append({"res_lo": float(c1), "res_hi": float(c2), "nmi": nmi})
            ari_vals.append({"res_lo": float(c1), "res_hi": float(c2), "ari": ari})
            vi_vals.append({"res_lo": float(c1), "res_hi": float(c2), "vi": vi})

        # ---- Export ----
        coverage_df = pd.DataFrame(coverage_rows)
        size_df = pd.DataFrame(size_rows)
        nmi_df = pd.DataFrame(nmi_vals)
        ari_df = pd.DataFrame(ari_vals)
        vi_df = pd.DataFrame(vi_vals)

        coverage_df.to_csv(f"/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/021c7ef263cf11f0b20008f1eaf4fe18/{min_size}_collapsed_metrics_coverage.csv", index=False)
        size_df.to_csv(f"/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/021c7ef263cf11f0b20008f1eaf4fe18/{min_size}_collapsed_metrics_sizes.csv", index=False)
        nmi_df.to_csv(f"/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/021c7ef263cf11f0b20008f1eaf4fe18/{min_size}_collapsed_metrics_nmi.csv", index=False)
        ari_df.to_csv(f"/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/021c7ef263cf11f0b20008f1eaf4fe18/{min_size}_collapsed_metrics_ari.csv", index=False)
        vi_df.to_csv(f"/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/021c7ef263cf11f0b20008f1eaf4fe18/{min_size}_collapsed_metrics_vi.csv", index=False)


if False:
    edges = np.asarray(full_g.get_edgelist())          # shape (m, 2)
    e_src, e_tgt = edges[:, 0], edges[:, 1]
    e_w = np.asarray(full_g.es["weight"], dtype=float)
    w_tot = e_w.sum()

    coverage_rows = []
    size_dist_rows = []         #  one row per community
    summary_rows = []           #  one row per resolution

    for col in res_cols:
        print(f"doing coverage distribution for {col}")
        r = float(col)
        labels = df[col].to_numpy()

        # ---- 3a. Coverage ----------------------------------------
        same_comm = labels[e_src] == labels[e_tgt]
        cov = (e_w[same_comm].sum() / w_tot)           # fraction of edge‑weight inside

        coverage_rows.append({"resolution": r, "coverage": cov})

        # ---- 3b. Community‑size distribution ---------------------
        sizes = pd.Series(labels).value_counts()

        for cid, sz in sizes.items():
            size_dist_rows.append({"resolution": r, "community_id": int(cid), "size": int(sz)})

        summary_rows.append({
            "resolution": r,
            "count": sizes.shape[0],            # number of communities
            "mean": sizes.mean(),               # avg community size
            "median": sizes.median(),           # median size
            "max": sizes.max()                  # size of largest community
        })

    coverage_df = pd.DataFrame(coverage_rows)           # 10 × 2
    sizes_df    = pd.DataFrame(size_dist_rows)          # ≈ (#res × #communities) rows
    summary_df = pd.DataFrame(summary_rows)

    # Save to disk
    coverage_df.to_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/6273205e640511f08e0b08f1eaf4fe18/hierarchical/bf5b3056674b11f0838c08f1eaf4fe18/coverage_by_resolution.csv", index=False)
    sizes_df.to_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/6273205e640511f08e0b08f1eaf4fe18/hierarchical/bf5b3056674b11f0838c08f1eaf4fe18/sizes_by_resolution.csv", index=False)
    summary_df.to_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/6273205e640511f08e0b08f1eaf4fe18/hierarchical/bf5b3056674b11f0838c08f1eaf4fe18/community_sizes_by_resolution.csv", index=False)
    print("Files written:",
        "coverage_by_resolution.csv",
        "community_sizes_by_resolution.csv", sep="\n")

if False:
    partition_stats = {}          # {resolution: {cid: φ, … , '_mean': …}}

    for col in res_cols:
        print(f"doing conductance for {col}")
        labels = df[col].to_numpy()
        cond_per_comm = {}

        for cid in np.unique(labels):
            # vertex indices belonging to this community
            vids = df.loc[labels == cid, "id"].astype(int).to_numpy()
            # OR use "name" with g.vs.find(name=…) mapping
            cond_per_comm[int(cid)] = conductance_manual(full_g, vids)

        # aggregate
        values = np.fromiter(cond_per_comm.values(), dtype=float)
        cond_per_comm["_mean"]   = values.mean()
        cond_per_comm["_median"] = np.median(values)
        cond_per_comm["_wmean"]  = (
            values * np.array([len(df[df[col] == cid]) for cid in cond_per_comm.keys()
                            if cid not in {"_mean","_median","_wmean"}], dtype=float)
        ).sum() / len(df)

        partition_stats[float(col)] = cond_per_comm

    # JSON or pickle
    with open("/ipazianas/pasquini/output_graph_analysis/communities_leiden/772845e4566b11f0bf7e08f1eaf4fe18/conductance_partition_stats.json", "w") as f:
        json.dump(partition_stats, f)

    # Flat CSV (one row per community)
    rows = [
        {"resolution": res, "community_id": cid, "conductance": phi}
        for res, d in partition_stats.items()
        for cid, phi in d.items()
    ]
    pd.DataFrame(rows).to_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/772845e4566b11f0bf7e08f1eaf4fe18/conductance_per_community.csv", index=False)


ei_index = False

if ei_index:
    original_comms = pd.read_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/6273205e640511f08e0b08f1eaf4fe18/communities_rp_0.6.csv",
                                 sep=",", header=0, low_memory=False)
    top500_comms = pd.read_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/6273205e640511f08e0b08f1eaf4fe18/hierarchical/bf5b3056674b11f0838c08f1eaf4fe18/nodes_with_communities.csv",
                               sep=",", header=0, low_memory=False)
    
    original_comms = original_comms[['id', 'name', 'type']]

    comms = original_comms.merge(top500_comms, on='name', how='left', suffixes=('', '_top500'))
    comms[['0.1', '0.2', '0.3','0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']] = comms[['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']].fillna(value=-1)
    df = comms.drop(columns=['id_top500', 'type_top500'])

    res_cols = sorted([c for c in df.columns if c.replace('.', '', 1).isdigit()],
                key=float)          # ['0.1','0.2', … '1.0']

    results = []
    name_to_vidx = {v["name"]: v.index for v in full_g.vs}  # map real-world ID (name) to vertex index
    min_size = 10
    dummy = -1
    name_col = "name"

    for col in res_cols:
        labels_raw = df[col].to_numpy()
        node_names = df[name_col].to_numpy()
        labels_c = collapse_small(labels_raw, min_size=min_size, dummy=dummy)

        # Map vertex index to community
        vertex_to_comm = {}
        for node_name, label in zip(node_names, labels_c):
            if node_name in name_to_vidx:
                vertex_to_comm[name_to_vidx[node_name]] = label
        

        # Track internal and external weights per community
        internal_w = defaultdict(float)
        external_w = defaultdict(float)
        comm_sizes = defaultdict(int)
        total_nodes = len(full_g.vs)

        for v in full_g.vs:
            print(f"Processing vertex {v.index} (total = {total_nodes}) with name {v['name']}")
            v_idx = v.index
            if v_idx not in vertex_to_comm:
                continue
            c_v = vertex_to_comm[v_idx]
            if c_v == dummy:
                continue
            comm_sizes[c_v] += 1
            for neighbor_idx, eid in zip(full_g.neighbors(v_idx, mode="ALL"), full_g.incident(v_idx, mode="ALL")):
                if neighbor_idx not in vertex_to_comm:
                    continue
                c_u = vertex_to_comm[neighbor_idx]
                w = full_g.es[eid]["weight"] if "weight" in full_g.es.attributes() else 1.0
                if c_v == c_u:
                    internal_w[c_v] += w
                else:
                    external_w[c_v] += w

        for comm in comm_sizes:
            iw = internal_w[comm]
            ew = external_w[comm]
            denom = iw + ew
            ei = (ew - iw) / denom if denom > 0 else 0
            results.append({
                "resolution": float(col),
                "community": comm,
                "size": comm_sizes[comm],
                "internal_weight": iw,
                "external_weight": ew,
                "ei_index": ei
            })

    final_result = pd.DataFrame(results)
    final_result.to_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/6273205e640511f08e0b08f1eaf4fe18/hierarchical/bf5b3056674b11f0838c08f1eaf4fe18/ei_index.csv", index=False)


link_hashtag = False
if link_hashtag:
    hashtag = pd.read_csv("/ipazianas/pasquini/output_graph_analysis/feb-aug_2022/2f019fee5d7411f0be9308f1eaf4fe18/hashtag",sep=",", header=None, dtype={0: "str", 1: "str", 2: "int8"})
    hashtag.columns = ["hashtag", "hash", "type"]
    hashtag = hashtag.groupby(["hashtag", "hash", "type"]).size().reset_index(name="count")
    hashtag = hashtag.drop(columns=["count"])

    communities_dir = "/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/6273205e640511f08e0b08f1eaf4fe18/hierarchical/bf5b3056674b11f0838c08f1eaf4fe18"
    files = sorted(glob.glob(os.path.join(communities_dir, "communities_rp_*.csv"))) 
    for i, fn in enumerate(files):
        print(f"Reading communities #{i}, file: {fn}")
        comm = pd.read_csv(fn, sep=",", header=0, dtype={0: "int32", 1: "str", 2: "str", 3: "int32"})
        merged = pd.merge(comm, hashtag, left_on="name", right_on="hash", how="left")
        cols_to_keep = [col for col in comm.columns if col in merged.columns] + ["type_x", "hashtag"]
        cols_to_keep = list(dict.fromkeys(cols_to_keep))
        merged = merged[cols_to_keep]
        merged.to_csv(fn.replace(".csv", "_with_hashtags.csv"), index=False)
        print(f"Saved merged communities with hashtags to {fn.replace('.csv', '_with_hashtags.csv')}")