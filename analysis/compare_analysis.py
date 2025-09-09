#!/usr/bin/env python3
"""
analyze_communities.py

Usage:
    python analyze_communities.py \
        --rt_csv retweet_leiden.csv \
        --rth_csv retweet_plus_hashtag_leiden.csv \
        --rp_csv reply_leiden.csv \
        --rph_csv reply_plus_hashtag_leiden.csv \
        --outdir results

Assumptions:
- Each CSV has columns: id, name, type, 0.1, 0.2, ..., 1.0
  where "type" is 'user' or 'hashtag' (or similar).
- Community labels are integers (or convertible to ints/strings).
- For augmented graphs (R+H, P+H) hashtags are present as nodes (type=='hashtag').

Outputs (in --outdir):
- metrics_R_vs_RH.csv, metrics_P_vs_PH.csv  (NMI, ARI, VI per resolution)
- entropy_RH.csv, entropy_PH.csv  (hashtag entropy per community per resolution)
- plots: nmi_plot_*.png, entropy_box_*.png, jaccard_*.png, stability_*.png
"""

import argparse
import os
import math
from collections import defaultdict, Counter
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, mutual_info_score
import matplotlib.pyplot as plt
from matplotlib import ticker

# ---------------------------
# Utilities: metrics & helpers
# ---------------------------

def variation_of_information(labels1, labels2):
    """Compute Variation of Information (VI) in bits."""
    # Ensure integer labels starting at 0 for bincount
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    mi = mutual_info_score(labels1, labels2)
    def entropy(labels):
        vals, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        # entropy base 2 (bits)
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
        # shift
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
    """
    Return (names_list, labels_list) for nodes of type node_type at resolution res_col.
    - df: DataFrame with columns name, type, and resolution columns.
    """
    mask = (df['type'] == node_type)
    sub = df.loc[mask, ['name', res_col]].copy()
    # Ensure consistent label types
    labels = sub[res_col].fillna(-1).astype(str).tolist()
    names = sub['name'].tolist()
    return names, labels

def invert_partition(names, labels):
    """
    Given parallel lists names, labels, return dict: label -> set(names)
    """
    comm = defaultdict(set)
    for n, l in zip(names, labels):
        comm[l].add(n)
    return comm

def jaccard_best_match(commA, commB, topk=3):
    """
    For each community in commA (dict label->set), find best matches in commB by Jaccard.
    Returns dict: a_label -> list of (b_label, jaccard) sorted desc up to topk
    """
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

def analyze_pair(df_base, df_aug, pair_name, resolutions, outdir, plot_prefix):
    """
    Compare base graph vs augmented graph across resolutions.
    - df_base: DataFrame for base graph (only nodes present in its run)
    - df_aug: DataFrame for augmented graph (contains hashtags nodes)
    - resolutions: list of resolution column names (strings)
    - pair_name: e.g., 'R_vs_RH'
    """
    metrics = []
    stability_rows = []
    # We'll align on 'user' nodes only for comparisons
    # get set of users present in both dataframes
    users_base = set(df_base.loc[df_base['type'] == 'u', 'name'])
    users_aug = set(df_aug.loc[df_aug['type'] == 'u', 'name'])
    common_users = sorted(list(users_base & users_aug))
    if not common_users:
        raise ValueError("No common users found between base and augmented graphs.")
    print(f"[{pair_name}] common users: {len(common_users)}")

    for i, res in enumerate(resolutions):
        print(f"[{pair_name}] Analyzing resolution {res} ...")
        # labels for base and augmented (users only)
        names_b, labels_b = get_labels_for_resolution(df_base, res, node_type='u')
        # Map name->label for quick alignment
        map_b = dict(zip(names_b, labels_b))
        names_a, labels_a = get_labels_for_resolution(df_aug, res, node_type='u')
        map_a = dict(zip(names_a, labels_a))

        # Build aligned label lists
        labels_b_aligned = [map_b.get(u, '-1') for u in common_users]
        labels_a_aligned = [map_a.get(u, '-1') for u in common_users]

        # Compute metrics
        try:
            nmi = normalized_mutual_info_score(labels_b_aligned, labels_a_aligned)
        except Exception:
            nmi = np.nan
        try:
            ari = adjusted_rand_score(labels_b_aligned, labels_a_aligned)
        except Exception:
            ari = np.nan
        try:
            vi = variation_of_information(labels_b_aligned, labels_a_aligned)
        except Exception:
            vi = np.nan

        # community size distributions (users only) for both graphs
        comm_b = invert_partition(common_users, labels_b_aligned)
        sizes_b = [len(s) for s in comm_b.values()] if comm_b else []
        comm_a = invert_partition(common_users, labels_a_aligned)
        sizes_a = [len(s) for s in comm_a.values()] if comm_a else []

        gini_b = gini_coefficient(sizes_b)
        gini_a = gini_coefficient(sizes_a)

        metrics.append({
            'pair': pair_name,
            'resolution': res,
            'nmi': nmi,
            'ari': ari,
            'vi': vi,
            'n_comms_base': len(comm_b),
            'n_comms_aug': len(comm_a),
            'mean_size_base': np.mean(sizes_b) if sizes_b else 0,
            'mean_size_aug': np.mean(sizes_a) if sizes_a else 0,
            'gini_base': gini_b,
            'gini_aug': gini_a
        })

        # Stability across resolutions within each graph: compute NMI with previous res
        if i > 0:
            prev = resolutions[i-1]
            # base stability
            _, labels_b_prev = get_labels_for_resolution(df_base, prev, node_type='u')
            map_b_prev = dict(zip(names_b, labels_b_prev))
            labels_b_prev_aligned = [map_b_prev.get(u, '-1') for u in common_users]
            nmi_base_prev = normalized_mutual_info_score(labels_b_prev_aligned, labels_b_aligned)
            # aug stability
            _, labels_a_prev = get_labels_for_resolution(df_aug, prev, node_type='u')
            map_a_prev = dict(zip(names_a, labels_a_prev))
            labels_a_prev_aligned = [map_a_prev.get(u, '-1') for u in common_users]
            nmi_aug_prev = normalized_mutual_info_score(labels_a_prev_aligned, labels_a_aligned)

            stability_rows.append({
                'pair': pair_name,
                'resolution': res,
                'nmi_base_prev': nmi_base_prev,
                'nmi_aug_prev': nmi_aug_prev
            })

    metrics_df = pd.DataFrame(metrics)
    # write metrics CSV
    metrics_csv = os.path.join(outdir, f'metrics_{pair_name}.csv')
    metrics_df.to_csv(metrics_csv, index=False)


    if stability_rows:
        stability_df = pd.DataFrame(stability_rows)
        stability_csv = os.path.join(outdir, f'stability_{pair_name}.csv')
        stability_df.to_csv(stability_csv, index=False)
    else:
        stability_df = pd.DataFrame()

    # Plot NMI/ARI/VI across resolutions
    plt.figure(figsize=(8,5))
    plt.plot(metrics_df['resolution'].astype(float), metrics_df['nmi'], marker='o', label='NMI')
    plt.plot(metrics_df['resolution'].astype(float), metrics_df['ari'], marker='s', label='ARI')
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.xlabel('resolution (gamma)')
    plt.ylabel('score')
    plt.title(f'Partition similarity across resolutions: {pair_name}')
    plt.legend()
    plt.grid(alpha=0.2)
    plot_nmi = os.path.join(outdir, f'{plot_prefix}_nmi_{pair_name}.png')
    plt.tight_layout()
    plt.savefig(plot_nmi, dpi=150)
    plt.close()

    # Plot VI separately (since scale differs)
    plt.figure(figsize=(8,4))
    plt.plot(metrics_df['resolution'].astype(float), metrics_df['vi'], marker='o')
    plt.xlabel('resolution (gamma)')
    plt.ylabel('VI (bits)')
    plt.title(f'Variation of Information across resolutions: {pair_name}')
    plt.grid(alpha=0.2)
    plot_vi = os.path.join(outdir, f'{plot_prefix}_vi_{pair_name}.png')
    plt.tight_layout()
    plt.savefig(plot_vi, dpi=150)
    plt.close()

    # Plot community count
    plt.figure(figsize=(8,4))
    plt.plot(metrics_df['resolution'].astype(float), metrics_df['n_comms_base'], marker='o', label='n_comms_base')
    plt.plot(metrics_df['resolution'].astype(float), metrics_df['n_comms_aug'], marker='s', label='n_comms_aug')
    plt.xlabel('resolution (gamma)')
    plt.ylabel('number of communities (users)')
    plt.title(f'Number of communities across resolutions: {pair_name}')
    plt.legend()
    plt.grid(alpha=0.2)
    plot_nc = os.path.join(outdir, f'{plot_prefix}_ncomms_{pair_name}.png')
    plt.tight_layout()
    plt.savefig(plot_nc, dpi=150)
    plt.close()

    return metrics_df, stability_df

# ---------------------------
# Jaccard matching for chosen resolutions
# ---------------------------

def compute_jaccard_for_res(dfA, dfB, res, out_csv, topk=3):
    """
    Compute jaccard best matches for communities between dfA and dfB at resolution res.
    Returns a DataFrame rows: a_comm, a_size, b_comm_best, jaccard_best, ...
    """
    # align only on common users
    usersA = set(dfA.loc[dfA['type'] == 'u', 'name'])
    usersB = set(dfB.loc[dfB['type'] == 'u', 'name'])
    common = sorted(list(usersA & usersB))
    names_a, labels_a = get_labels_for_resolution(dfA, res, node_type='u')
    names_b, labels_b = get_labels_for_resolution(dfB, res, node_type='u')
    map_a = dict(zip(names_a, labels_a))
    map_b = dict(zip(names_b, labels_b))
    labels_a_aligned = [map_a.get(u, '-1') for u in common]
    labels_b_aligned = [map_b.get(u, '-1') for u in common]
    commA = invert_partition(common, labels_a_aligned)
    commB = invert_partition(common, labels_b_aligned)
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

    # load CSVs
    df_R = pd.read_csv(args.rt_csv, dtype=str)
    df_RH = pd.read_csv(args.rth_csv, dtype=str)
    df_P = pd.read_csv(args.rp_csv, dtype=str)
    df_PH = pd.read_csv(args.rph_csv, dtype=str)

    # Ensure column names for resolutions are strings like '0.1'...'1.0'
    resolutions = [f"{x:.1f}" for x in np.arange(0.1, 1.0 + 1e-9, 0.1)]
    # Sanity: ensure these resolution columns exist in all
    for df, name in [(df_R, 'R'), (df_RH, 'R+H'), (df_P, 'P'), (df_PH, 'P+H')]:
        missing = [r for r in resolutions if r not in df.columns]
        if missing:
            raise ValueError(f"Missing resolution columns {missing} in dataframe {name}")

    # Normalize type column values to lowercase
    for df in [df_R, df_RH, df_P, df_PH]:
        df['type'] = df['type'].str.lower()

    # Analyze pairs
    print("Analyzing R vs R+H ...")
    metrics_R_RH, stability_R_RH = analyze_pair(df_R, df_RH, 'R_vs_RH', resolutions, args.outdir, 'plot')

    print("Analyzing P vs P+H ...")
    metrics_P_PH, stability_P_PH = analyze_pair(df_P, df_PH, 'P_vs_PH', resolutions, args.outdir, 'plot')

    # Save combined metrics
    combined_metrics = pd.concat([metrics_R_RH, metrics_P_PH], ignore_index=True)
    combined_metrics.to_csv(os.path.join(args.outdir, 'combined_metrics.csv'), index=False)

    # Jaccard matches for selected resolutions (user-specified or default)
    res_list = args.jaccard_resolutions if args.jaccard_resolutions else ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7','0.8', '0.9','1.0']
    for res in res_list:
        out_jaccard_R = os.path.join(args.outdir, f'jaccard_R_vs_RH_res_{res}.csv')
        compute_jaccard_for_res(df_R, df_RH, res, out_jaccard_R, topk=args.jaccard_topk)
        out_jaccard_P = os.path.join(args.outdir, f'jaccard_P_vs_PH_res_{res}.csv')
        compute_jaccard_for_res(df_P, df_PH, res, out_jaccard_P, topk=args.jaccard_topk)

    # Print quick summary
    print("Summary (metrics head):")
    print(combined_metrics.head())

    print("All results saved to:", args.outdir)
    print("Key files:")
    print(" - combined_metrics.csv")
    print(" - metrics_R_vs_RH.csv, metrics_P_vs_PH.csv")
    print(" - jaccard_*_res_*.csv")
    print(" - several PNG plots (NMI, VI, community counts)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze Leiden partitions across resolutions for R, R+H, P, P+H")
    parser.add_argument('--rt_csv', required=True, help='Retweet-only Leiden CSV')
    parser.add_argument('--rth_csv', required=True, help='Retweet+Hashtag Leiden CSV')
    parser.add_argument('--rp_csv', required=True, help='Reply-only Leiden CSV')
    parser.add_argument('--rph_csv', required=True, help='Reply+Hashtag Leiden CSV')
    parser.add_argument('--outdir', required=True, help='Output directory for CSVs and plots')
    parser.add_argument('--jaccard_resolutions', nargs='*', help='Resolutions (e.g. 0.3 0.5) to compute jaccard matching')
    parser.add_argument('--jaccard_topk', type=int, default=3, help='Top-k matches for jaccard output')
    args = parser.parse_args()
    main(args)
