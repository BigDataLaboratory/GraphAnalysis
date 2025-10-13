#!/usr/bin/env python3
"""
analyze_communities.py

Applies a size threshold to communities before analysis to focus on substantive groups.

Usage:
    python analyze_communities.py \\
        --rt_csv retweet_leiden.csv \\
        --rth_csv retweet_plus_hashtag_leiden.csv \\
        --rp_csv reply_leiden.csv \\
        --rph_csv reply_plus_hashtag_leiden.csv \\
        --outdir results \\
        --min_comm_size 11

Assumptions:
- Each CSV has columns: id, name, type, 0.1, 0.2, ..., 1.0
- Community labels are integers (or convertible to ints/strings).

Outputs (in --outdir):
- Filtered metrics CSVs and plots based on communities larger than the threshold.
"""

import argparse
import os
import math
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, mutual_info_score
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

# ---------------------------
# Utilities: metrics & helpers
# ---------------------------

def variation_of_information(labels1, labels2):
    """Compute Variation of Information (VI) in bits."""
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    mi = mutual_info_score(labels1, labels2)
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum([p * math.log(p, 2) for p in probs if p > 0])
    h1 = entropy(labels1)
    h2 = entropy(labels2)
    vi = h1 + h2 - 2 * mi
    return vi

def gini_coefficient(x):
    """Compute Gini coefficient of array x (non-negative)."""
    x = np.array(x, dtype=float)
    if x.size == 0:
        return np.nan
    if np.all(x == 0):
        return 0.0
    x = x.flatten()
    if np.any(x < 0):
        x = x - x.min()
    x = x + 1e-12
    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)
    gini = (2.0 * np.sum((np.arange(1, n+1) * x_sorted))) / (n * cum[-1]) - (n + 1) / n
    return gini

# ---------------------------
# Partition helpers
# ---------------------------

def get_labels_for_resolution(df, res_col, node_type='u'):
    """Return (names_list, labels_list) for nodes of type node_type at a given resolution."""
    mask = (df['type'] == node_type)
    sub = df.loc[mask, ['name', res_col]].copy()
    labels = sub[res_col].fillna('-1').astype(str).tolist()
    names = sub['name'].tolist()
    return names, labels

def invert_partition(names, labels):
    """Given parallel lists of names and labels, return a dict: label -> set(names)."""
    comm = defaultdict(set)
    for n, l in zip(names, labels):
        comm[l].add(n)
    return comm

def filter_partition(names, labels, min_size):
    """
    Filters a partition to exclude communities smaller than min_size.

    Args:
        names (list): List of node names.
        labels (list): List of community labels corresponding to names.
        min_size (int): The minimum size for a community to be kept.

    Returns:
        tuple: A tuple containing:
            - filtered_labels (list): A new list of labels where users in small
              communities are reassigned to the label '-1'.
            - filtered_communities (dict): A dictionary of the communities that
              met the size threshold.
    """
    if min_size <= 1:
        return labels, invert_partition(names, labels)

    communities = invert_partition(names, labels)
    
    filtered_communities = {
        label: users for label, users in communities.items()
        if len(users) >= min_size
    }
    
    valid_labels = set(filtered_communities.keys())
    
    filtered_labels = [l if l in valid_labels else '-1' for l in labels]
    
    return filtered_labels, filtered_communities


def jaccard_best_match(commA, commB, topk=3):
    """For each community in commA, find best matches in commB by Jaccard similarity."""
    results = {}
    for a_label, a_nodes in commA.items():
        matches = []
        for b_label, b_nodes in commB.items():
            inter = len(a_nodes & b_nodes)
            union = len(a_nodes | b_nodes)
            j = inter / union if union > 0 else 0.0
            matches.append((b_label, j))
        matches.sort(key=lambda x: x[1], reverse=True)
        results[a_label] = matches[:topk]
    return results

# ---------------------------
# Main analysis procedure
# ---------------------------

def analyze_pair(df_base, df_aug, pair_name, resolutions, outdir, plot_prefix, min_comm_size):
    """Compare base graph vs augmented graph across resolutions after filtering small communities."""
    metrics = []
    stability_rows = []
    
    users_base = set(df_base.loc[df_base['type'] == 'u', 'name'])
    users_aug = set(df_aug.loc[df_aug['type'] == 'u', 'name'])
    common_users = sorted(list(users_base & users_aug))
    
    if not common_users:
        raise ValueError("No common users found between base and augmented graphs.")
    print(f"[{pair_name}] common users: {len(common_users)}")

    for i, res in enumerate(resolutions):
        print(f"[{pair_name}] Analyzing resolution {res} (min size: {min_comm_size})...")
        
        # Get original labels
        names_b, labels_b_orig = get_labels_for_resolution(df_base, res, node_type='u')
        map_b = dict(zip(names_b, labels_b_orig))
        names_a, labels_a_orig = get_labels_for_resolution(df_aug, res, node_type='u')
        map_a = dict(zip(names_a, labels_a_orig))

        # Align original labels to the set of common users
        labels_b_aligned = [map_b.get(u, '-1') for u in common_users]
        labels_a_aligned = [map_a.get(u, '-1') for u in common_users]

        # Filter partitions by community size
        labels_b_filtered, comms_b_filtered = filter_partition(common_users, labels_b_aligned, min_comm_size)
        labels_a_filtered, comms_a_filtered = filter_partition(common_users, labels_a_aligned, min_comm_size)

        # Compute metrics on filtered partitions
        nmi = normalized_mutual_info_score(labels_b_filtered, labels_a_filtered)
        ari = adjusted_rand_score(labels_b_filtered, labels_a_filtered)
        vi = variation_of_information(labels_b_filtered, labels_a_filtered)

        # Calculate size stats on the filtered communities
        sizes_b = [len(s) for s in comms_b_filtered.values()]
        sizes_a = [len(s) for s in comms_a_filtered.values()]
        gini_b = gini_coefficient(sizes_b)
        gini_a = gini_coefficient(sizes_a)

        metrics.append({
            'pair': pair_name,
            'resolution': res,
            'nmi': nmi,
            'ari': ari,
            'vi': vi,
            'n_comms_base': len(comms_b_filtered),
            'n_comms_aug': len(comms_a_filtered),
            'mean_size_base': np.mean(sizes_b) if sizes_b else 0,
            'mean_size_aug': np.mean(sizes_a) if sizes_a else 0,
            'gini_base': gini_b,
            'gini_aug': gini_a
        })

        # Stability analysis also uses filtered partitions
        if i > 0:
            prev = resolutions[i-1]
            names_b_prev, labels_b_prev_orig = get_labels_for_resolution(df_base, prev, node_type='u')
            map_b_prev = dict(zip(names_b_prev, labels_b_prev_orig))
            labels_b_prev_aligned = [map_b_prev.get(u, '-1') for u in common_users]
            labels_b_prev_filtered, _ = filter_partition(common_users, labels_b_prev_aligned, min_comm_size)
            
            nmi_base_prev = normalized_mutual_info_score(labels_b_prev_filtered, labels_b_filtered)

            names_a_prev, labels_a_prev_orig = get_labels_for_resolution(df_aug, prev, node_type='u')
            map_a_prev = dict(zip(names_a_prev, labels_a_prev_orig))
            labels_a_prev_aligned = [map_a_prev.get(u, '-1') for u in common_users]
            labels_a_prev_filtered, _ = filter_partition(common_users, labels_a_prev_aligned, min_comm_size)

            nmi_aug_prev = normalized_mutual_info_score(labels_a_prev_filtered, labels_a_filtered)
            
            stability_rows.append({
                'pair': pair_name, 'resolution': res,
                'nmi_base_prev': nmi_base_prev, 'nmi_aug_prev': nmi_aug_prev
            })

    metrics_df = pd.DataFrame(metrics)
    metrics_csv = os.path.join(outdir, f'metrics_{pair_name}_filtered.csv')
    metrics_df.to_csv(metrics_csv, index=False)

    if stability_rows:
        stability_df = pd.DataFrame(stability_rows)
        stability_csv = os.path.join(outdir, f'stability_{pair_name}_filtered.csv')
        stability_df.to_csv(stability_csv, index=False)
    
    # --- Plotting ---
    plt.figure(figsize=(8,5))
    plt.plot(metrics_df['resolution'].astype(float), metrics_df['nmi'], marker='o', label='NMI')
    plt.plot(metrics_df['resolution'].astype(float), metrics_df['ari'], marker='s', label='ARI')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.xlabel('resolution (gamma)')
    plt.ylabel('score')
    plt.title(f'Partition Similarity (filtered, min size {min_comm_size}): {pair_name}')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(outdir, f'{plot_prefix}_nmi_{pair_name}_filtered.png'), dpi=150)
    plt.close()

def compute_jaccard_for_res(dfA, dfB, res, out_csv, topk, min_comm_size):
    """Compute Jaccard best matches after filtering communities by size."""
    usersA = set(dfA.loc[dfA['type'] == 'u', 'name'])
    usersB = set(dfB.loc[dfB['type'] == 'u', 'name'])
    common = sorted(list(usersA & usersB))

    names_a, labels_a = get_labels_for_resolution(dfA, res, node_type='u')
    names_b, labels_b = get_labels_for_resolution(dfB, res, node_type='u')

    map_a = dict(zip(names_a, labels_a))
    map_b = dict(zip(names_b, labels_b))

    labels_a_aligned = [map_a.get(u, '-1') for u in common]
    labels_b_aligned = [map_b.get(u, '-1') for u in common]
    
    # Filter communities before matching
    _, commA = filter_partition(common, labels_a_aligned, min_comm_size)
    _, commB = filter_partition(common, labels_b_aligned, min_comm_size)

    if not commA or not commB:
        print(f"  Warning: No communities of min size {min_comm_size} found for Jaccard at res {res}. Skipping.")
        return pd.DataFrame()

    matches = jaccard_best_match(commA, commB, topk=topk)
    rows = []
    for a_label, matchlist in matches.items():
        a_size = len(commA.get(a_label, []))
        row = {'a_comm': a_label, 'a_size': a_size}
        for k, (b_label, j) in enumerate(matchlist, start=1):
            row[f'b_comm_{k}'] = b_label
            row[f'jaccard_{k}'] = j
            row[f'b_size_{k}'] = len(commB.get(b_label, [])) if b_label is not None else 0
        rows.append(row)
        
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    return df_out

# ---------------------------
# CLI & orchestrator
# ---------------------------

def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    print("Loading data...")
    df_R = pd.read_csv(args.rt_csv, dtype=str, keep_default_na=False)
    df_RH = pd.read_csv(args.rth_csv, dtype=str, keep_default_na=False)
    df_P = pd.read_csv(args.rp_csv, dtype=str, keep_default_na=False)
    df_PH = pd.read_csv(args.rph_csv, dtype=str, keep_default_na=False)

    resolutions = [f"{x:.1f}" for x in np.arange(0.1, 1.0 + 1e-9, 0.1)]
    for df, name in [(df_R, 'R'), (df_RH, 'R+H'), (df_P, 'P'), (df_PH, 'P+H')]:
        missing = [r for r in resolutions if r not in df.columns]
        if missing:
            raise ValueError(f"Missing resolution columns {missing} in dataframe {name}")
        df['type'] = df['type'].str.lower()

    print("\n--- Starting Analysis (Excluding communities with < " + str(args.min_comm_size) + " users) ---")

    print("\nAnalyzing R vs R+H ...")
    analyze_pair(df_R, df_RH, 'R_vs_RH', resolutions, args.outdir, 'plot', args.min_comm_size)

    print("\nAnalyzing P vs P+H ...")
    analyze_pair(df_P, df_PH, 'P_vs_PH', resolutions, args.outdir, 'plot', args.min_comm_size)

    res_list = args.jaccard_resolutions if args.jaccard_resolutions else resolutions
    for res in res_list:
        print(f"\nComputing Jaccard matches for resolution {res}...")
        out_jaccard_R = os.path.join(args.outdir, f'jaccard_R_vs_RH_res_{res}_filtered.csv')
        compute_jaccard_for_res(df_R, df_RH, res, out_jaccard_R, args.jaccard_topk, args.min_comm_size)
        
        out_jaccard_P = os.path.join(args.outdir, f'jaccard_P_vs_PH_res_{res}_filtered.csv')
        compute_jaccard_for_res(df_P, df_PH, res, out_jaccard_P, args.jaccard_topk, args.min_comm_size)

    print("\n✅ All analyses complete. Results saved to:", args.outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze Leiden partitions after filtering small communities.")
    parser.add_argument('--rt_csv', required=True, help='Retweet-only Leiden CSV')
    parser.add_argument('--rth_csv', required=True, help='Retweet+Hashtag Leiden CSV')
    parser.add_argument('--rp_csv', required=True, help='Reply-only Leiden CSV')
    parser.add_argument('--rph_csv', required=True, help='Reply+Hashtag Leiden CSV')
    parser.add_argument('--outdir', required=True, help='Output directory for CSVs and plots')
    parser.add_argument('--min_comm_size', type=int, default=11, help='Minimum size for a community to be included (e.g., 11 means >10 users).')
    parser.add_argument('--jaccard_resolutions', nargs='*', help='Resolutions (e.g. 0.3 0.5) to compute jaccard matching')
    parser.add_argument('--jaccard_topk', type=int, default=3, help='Top-k matches for jaccard output')
    args = parser.parse_args()
    main(args)
