#!/usr/bin/env python3
"""
analyze_entropy_filtered.py

Computes hashtag entropy for communities, but only for those meeting a minimum
size threshold to focus the analysis on substantive social groups.

Usage:
    python analyze_entropy_filtered.py \\
        --leiden_csv /path/to/nodes_with_communities.csv \\
        --edge_file /path/to/user_hashtag_edges.csv \\
        --outdir /path/to/output_directory \\
        --min_comm_size 11
"""
import os
import math
import time
import pandas as pd
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
import argparse


def shannon_entropy(counts, normalize=True, base=2):
    """Compute (normalized) Shannon entropy from a Counter of counts."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0

    probs = [c / total for c in counts.values() if c > 0]
    entropy = -sum(p * math.log(p, base) for p in probs)

    if normalize:
        unique = len(counts)
        if unique > 1:
            entropy /= math.log(unique, base)
        else:
            # A community with only one unique hashtag has zero entropy (no uncertainty).
            entropy = 0.0
    return entropy


def process_chunk(lines, name_to_type, user_to_comm, sep=","):
    """Process one chunk of edges -> partial counters."""
    comm_to_counts = defaultdict(Counter)
    comm_to_users = defaultdict(set)

    for line in lines:
        try:
            etype, src, dst, weight = line.strip().split(sep)
            etype = int(etype)
            weight = int(float(weight))
        except Exception:
            continue

        if etype != 2:
            continue

        if name_to_type.get(src) == 'u':
            user, hashtag = src, dst
        elif name_to_type.get(dst) == 'u':
            user, hashtag = dst, src
        else:
            # This edge does not involve a user from our Leiden file.
            continue

        comm = user_to_comm.get(user)
        if comm is None:
            continue

        comm_to_counts[comm][hashtag] += weight
        comm_to_users[comm].add(user)

    return comm_to_counts, comm_to_users


def merge_results(partials):
    """Merge list of partial dicts into one."""
    merged_counts = defaultdict(Counter)
    merged_users = defaultdict(set)
    for comm_to_counts, comm_to_users in partials:
        for comm, ctr in comm_to_counts.items():
            merged_counts[comm].update(ctr)
        for comm, users in comm_to_users.items():
            merged_users[comm].update(users)
    return merged_counts, merged_users


def compute_comm_stats(args):
    """Compute stats for a single community (parallelizable)."""
    comm, counts, users, n_hashtag_nodes, normalize = args

    total_usages = sum(counts.values())
    n_unique_hashtags = len(counts)
    entropy = shannon_entropy(counts, normalize=normalize)
    dominant_share = max(counts.values()) / total_usages if total_usages > 0 else 0.0

    return {
        "community": comm,
        "n_users": len(users),
        "n_hashtag_nodes_in_comm": n_hashtag_nodes,
        "n_hashtag_usages_by_users": total_usages,
        "n_unique_hashtags_used": n_unique_hashtags,
        "entropy_bits_normalized" if normalize else "entropy_bits": entropy,
        "dominant_share": dominant_share,
    }


def analyze_entropy_sparse(leiden_csv, edge_file, outdir,
                           min_comm_size=11, resolutions=None, normalize=True,
                           name_col='name', type_col='type',
                           sep=",", chunk_size=1_000_000, n_jobs=None):
    os.makedirs(outdir, exist_ok=True)
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)

    print("Loading Leiden CSV:", leiden_csv)
    df = pd.read_csv(leiden_csv, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    name_to_type = {str(row[name_col]): str(row[type_col]).lower()
                    for _, row in df[[name_col, type_col]].iterrows()}
    print(f"Loaded {len(name_to_type):,} nodes")

    if resolutions is None:
        resolutions = sorted([c for c in df.columns if c.replace('.', '', 1).isdigit()], key=float)
    print("Resolutions detected:", resolutions)

    results_dfs = []

    for res in resolutions:
        print(f"\n--- Processing resolution {res} ---")
        users_df = df[df[type_col].str.lower() == 'u'][[name_col, res]].dropna()
        user_to_comm = dict(zip(users_df[name_col].astype(str), users_df[res].astype(str)))
        print(f" Users at this resolution: {len(user_to_comm):,}")

        # === PARSE EDGES IN PARALLEL ===
        pool = Pool(processes=n_jobs)
        tasks = []
        chunk = []
        with open(edge_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                chunk.append(line)
                if len(chunk) >= chunk_size:
                    tasks.append(pool.apply_async(process_chunk, (chunk, name_to_type, user_to_comm, sep)))
                    print(f"  Dispatched chunk {len(tasks)} ({i+1:,} lines)")
                    chunk = []
            if chunk:
                tasks.append(pool.apply_async(process_chunk, (chunk, name_to_type, user_to_comm, sep)))
                print(f"  Dispatched final chunk {len(tasks)}")
        pool.close()
        
        partials = [task.get() for task in tasks]
        pool.join()
        
        comm_to_counts, comm_to_users = merge_results(partials)
        print(f"  ✔ Merged results from {len(tasks)} chunks into {len(comm_to_counts):,} communities")

        # ======================================================================
        # --- START: NEW FILTERING STEP ---
        # ======================================================================
        print(f"  Applying filter: keeping communities with >= {min_comm_size} users...")
        
        # Identify communities that meet the size threshold
        large_enough_comms = {
            comm for comm, users in comm_to_users.items() 
            if len(users) >= min_comm_size
        }
        
        n_before = len(comm_to_counts)
        print(f"  Found {len(large_enough_comms):,} communities meeting the threshold out of {n_before:,}.")
        
        # ======================================================================
        # --- END: NEW FILTERING STEP ---
        # ======================================================================

        # === PREPARE ARGS FOR COMMUNITY-LEVEL PARALLEL (NOW FILTERED) ===
        mask = df[type_col].str.lower() != 'u'
        comm_to_hashtag_nodes = df.loc[mask, res].astype(str).value_counts().to_dict()

        # Build the argument list, but only for communities that passed the filter
        comm_args = [
            (comm,
             counts,
             comm_to_users.get(comm, set()),
             comm_to_hashtag_nodes.get(comm, 0),
             normalize)
            for comm, counts in comm_to_counts.items()
            if comm in large_enough_comms  # The filtering condition
        ]

        n_comms = len(comm_args)
        if n_comms == 0:
            print("  No communities met the size threshold. Skipping to next resolution.")
            continue
            
        print(f"  Starting entropy stats for {n_comms:,} filtered communities using {n_jobs} workers...")

        pool = Pool(processes=n_jobs)
        results = []
        for i, result in enumerate(pool.imap_unordered(compute_comm_stats, comm_args, chunksize=500), 1):
            results.append(result)
            if i % 1000 == 0 or i == n_comms:
                print(f"    Progress: {i:,}/{n_comms:,} communities ({100*i/n_comms:.1f}%)")
                sys.stdout.flush()
        pool.close()
        pool.join()

        stats_df = pd.DataFrame(results)
        stats_df['resolution'] = res

        results_dfs.append(stats_df)
        print(f"  ✔ Finished entropy stats for resolution {res}")

    if not results_dfs:
        print("\nNo data was processed. Exiting.")
        return None, None

    out_df = pd.concat(results_dfs, ignore_index=True)
    
    # Update output filename to reflect filtering
    base_filename = os.path.basename(leiden_csv).replace('.csv', '')
    out_csv = os.path.join(outdir, f'entropy_reply_{base_filename}_filtered_gt{min_comm_size-1}.csv')
    out_df.to_csv(out_csv, index=False)
    print("\nWrote filtered results to:", out_csv)

    return out_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hashtag entropy for Leiden communities, filtering by size.")
    parser.add_argument('--leiden_csv', required=True, help='Path to the nodes_with_communities.csv file from Leiden.')
    parser.add_argument('--edge_file', required=True, help='Path to the user-hashtag edge list file.')
    parser.add_argument('--outdir', required=True, help='Output directory for the resulting CSV file.')
    parser.add_argument('--min_comm_size', type=int, default=11, help='Minimum number of users for a community to be included (default: 11 for >10 users).')
    parser.add_argument('--sep', default=',', help='Separator for the edge file (default: ",").')
    parser.add_argument('--chunk_size', type=int, default=1_000_000, help='Number of lines to process per chunk.')
    parser.add_argument('--n_jobs', type=int, default=30, help='Number of parallel processes to use.')
    args = parser.parse_args()

    analyze_entropy_sparse(
        leiden_csv=args.leiden_csv,
        edge_file=args.edge_file,
        outdir=args.outdir,
        min_comm_size=args.min_comm_size,
        sep=args.sep,
        chunk_size=args.chunk_size,
        n_jobs=args.n_jobs
    )
