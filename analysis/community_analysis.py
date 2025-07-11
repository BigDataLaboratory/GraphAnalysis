import pandas as pd
import igraph as ig
import numpy as np
import glob
import os
import json
import pickle
from collections import Counter
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score
)

def comm_conductance(graph, vertex_ids):
    """
    vertex_ids: list[int]  (indices, not names)
    returns: float   (0 = perfectly separated, 1 = fully mixed)
    """
    return graph.conductance(vertex_ids)   # uses edge weights if 'weight' attr exists

basic_analysis = True
if basic_analysis:
    snapshot_dir = "/ipazianas/pasquini/twitter_graph_dump/graph_ten_days"
    files = sorted(glob.glob(os.path.join(snapshot_dir, "*.pkl")))

    df = pd.read_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/56a60a425ca411f0a7a908f1eaf4fe18/nodes_with_communities.csv", 
                    sep=",", 
                    header=0, low_memory=False)

    res_cols = sorted([c for c in df.columns if c.replace('.', '', 1).isdigit()],
                    key=float)          # ['0.1','0.2', … '1.0']

    labels_by_res = {r: df[str(r)].to_numpy() for r in res_cols}

    # number of communities at each resolution
    n_clusters = [pd.Series(labels_by_res[r]).nunique() for r in res_cols]

    print(f"Number of communities at each resolution: {n_clusters}")

    # Partition similarity between *adjacent* resolutions (NMI & ARI)
    nmi_adj, ari_adj = [], []
    for r1, r2 in zip(res_cols[:-1], res_cols[1:]):
        nmi_adj.append(normalized_mutual_info_score(labels_by_res[r1], labels_by_res[r2]))
        ari_adj.append(adjusted_rand_score(labels_by_res[r1], labels_by_res[r2]))

    print("CPM - NMI between adjacent resolutions:", nmi_adj)
    print("CPM - ARI between adjacent resolutions:", ari_adj)

read_graphs = True
if read_graphs:
    full_g = ig.Graph(directed=True)
    temporal_graph = False
    for i, fn in enumerate(files):
        print(f"Reading pickle #{i}")
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

def conductance_manual(g: ig.Graph, vids, weight_attr="weight"):
    """
    Compute φ(S) for a set of vertices S = `vids`.

    φ(S) = cut(S, ~S) / min( vol(S), vol(~S) )
    where vol(X) = sum of (weighted) degrees of vertices in X.
    """
    vids = np.fromiter(vids, dtype=int)
    mask = np.zeros(g.vcount(), dtype=bool)
    mask[vids] = True                     # mask[i] == True  ⇔  i ∈ S

    # adjacency as two arrays of endpoints
    ends = np.asarray(g.get_edgelist())     # shape (m, 2)
    src, tgt = ends[:, 0], ends[:, 1]

    # optional edge weights
    if weight_attr in g.es.attribute_names():
        w = np.asarray(g.es[weight_attr], dtype=float)
    else:
        w = np.ones(len(g.es), dtype=float)

    src_in = mask[src]
    tgt_in = mask[tgt]

    # cut edges: exactly one endpoint in S
    cut_mask = src_in ^ tgt_in
    cut_w = w[cut_mask].sum()

    # volumes
    deg = g.strength(weights=weight_attr if weight_attr in g.es.attributes() else None)
    deg = np.asarray(deg, dtype=float)

    vol_S    = deg[mask].sum()
    vol_compl = deg[~mask].sum()

    denom = min(vol_S, vol_compl)
    return 0.0 if denom == 0 else cut_w / denom

def collapse_small(labels, min_size=10, dummy=-1):
    labels = np.asarray(labels)
    counts = Counter(labels)
    keep = {k for k, v in counts.items() if v >= min_size}
    return np.array([l if l in keep else dummy for l in labels])

collapse_dataset = True

if collapse_dataset:
    # ---- Basic input ----
    edges = np.asarray(full_g.get_edgelist())
    e_src, e_tgt = edges[:, 0], edges[:, 1]
    e_w = np.asarray(full_g.es["weight"], dtype=float)
    w_tot = e_w.sum()

    min_size = 20
    dummy_id = -1

    # ---- Outputs ----
    coverage_rows = []
    size_rows = []
    nmi_vals = []
    ari_vals = []

    collapsed = {}

    for col in res_cols:
        print(f"Computing metrics for resolution = {col}")
        r = float(col)
        labels = df[col].to_numpy()
        labels_c = collapse_small(labels, min_size=min_size, dummy=dummy_id)
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
        else:
            nmi = normalized_mutual_info_score(labels1[mask], labels2[mask])
            ari = adjusted_rand_score(labels1[mask], labels2[mask])
        nmi_vals.append({"res_lo": float(c1), "res_hi": float(c2), "nmi": nmi})
        ari_vals.append({"res_lo": float(c1), "res_hi": float(c2), "ari": ari})

    # ---- Export ----
    coverage_df = pd.DataFrame(coverage_rows)
    size_df = pd.DataFrame(size_rows)
    nmi_df = pd.DataFrame(nmi_vals)
    ari_df = pd.DataFrame(ari_vals)

    coverage_df.to_csv(f"/ipazianas/pasquini/output_graph_analysis/communities_leiden/56a60a425ca411f0a7a908f1eaf4fe18/{min_size}_collapsed_metrics_coverage.csv", index=False)
    size_df.to_csv(f"/ipazianas/pasquini/output_graph_analysis/communities_leiden/56a60a425ca411f0a7a908f1eaf4fe18/{min_size}_collapsed_metrics_sizes.csv", index=False)
    nmi_df.to_csv(f"/ipazianas/pasquini/output_graph_analysis/communities_leiden/56a60a425ca411f0a7a908f1eaf4fe18/{min_size}_collapsed_metrics_nmi.csv", index=False)
    ari_df.to_csv(f"/ipazianas/pasquini/output_graph_analysis/communities_leiden/56a60a425ca411f0a7a908f1eaf4fe18/{min_size}_collapsed_metrics_ari.csv", index=False)

if False:
    edges = np.asarray(full_g.get_edgelist())          # shape (m, 2)
    e_src, e_tgt = edges[:, 0], edges[:, 1]
    e_w = np.asarray(full_g.es["weight"], dtype=float)
    w_tot = e_w.sum()

    min_size = 10      # communities with < min_size nodes are collapsed
    dummy_id = -1      # label assigned to the “collapsed” bucket

    coverage_rows = []

    for col in res_cols:
        print(f"computing coverage for {col}   (min_size = {min_size})")
        r = float(col)
        labels = df[col].to_numpy()
        labels_c = collapse_small(labels, min_size=min_size, dummy=dummy_id)
        # ---- Coverage ----------------------------------------
        same_comm = labels_c[e_src] == labels_c[e_tgt]
        cov = e_w[same_comm].sum() / w_tot           # fraction of edge‑weight inside

        coverage_rows.append({"resolution": r,
                            "coverage":   cov,
                            "kept_comms": (labels_c != dummy_id).sum()})

    coverage_df = pd.DataFrame(coverage_rows)           # 10 × 2

    # Save to disk
    coverage_df.to_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/772845e4566b11f0bf7e08f1eaf4fe18/coverage_by_resolution_collapsed_lte_10.csv", index=False)


if False:
    edges = np.asarray(full_g.get_edgelist())          # shape (m, 2)
    e_src, e_tgt = edges[:, 0], edges[:, 1]
    e_w = np.asarray(full_g.es["weight"], dtype=float)
    w_tot = e_w.sum()

    coverage_rows = []
    size_dist_rows = []         #  one row per community

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

    coverage_df = pd.DataFrame(coverage_rows)           # 10 × 2
    sizes_df    = pd.DataFrame(size_dist_rows)          # ≈ (#res × #communities) rows

    # Save to disk
    coverage_df.to_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/54b065ee5c9911f0af8708f1eaf4fe18/10days_cpm_coverage_by_resolution.csv", index=False)
    sizes_df.to_csv("/ipazianas/pasquini/output_graph_analysis/communities_leiden/54b065ee5c9911f0af8708f1eaf4fe18/10days_cpm_community_sizes_by_resolution.csv", index=False)
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


link_hashtag = False
if link_hashtag:
    hashtag = pd.read_csv("/ipazianas/pasquini/output_graph_analysis/ten_days/e1ea085a5c0511f091ac08f1eaf4fe18/hashtag",sep=",", header=None, dtype={0: "str", 1: "str", 2: "int8"})
    hashtag.columns = ["hashtag", "hash", "type"]
    hashtag = hashtag.groupby(["hashtag", "hash", "type"]).size().reset_index(name="count")
    hashtag = hashtag.drop(columns=["count"])

    communities_dir = "/ipazianas/pasquini/output_graph_analysis/communities_leiden/56a60a425ca411f0a7a908f1eaf4fe18"
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