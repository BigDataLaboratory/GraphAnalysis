import os
import math
import time
import pandas as pd
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count


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
            entropy = 0.0
    return entropy


def process_chunk(lines, name_to_type, user_to_comm, sep="\t"):
    """Process one chunk of edges → partial counters."""
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

        if name_to_type.get(src) == 'u' and name_to_type.get(dst) == 'h':
            user, hashtag = src, dst
        elif name_to_type.get(dst) == 'u' and name_to_type.get(src) == 'h':
            user, hashtag = dst, src
        else:
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
                           resolutions=None, normalize=True,
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
        users = df[df[type_col].str.lower() == 'u'][[name_col, res]].dropna()
        user_to_comm = dict(zip(users[name_col].astype(str), users[res].astype(str)))
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

        # === PREPARE ARGS FOR COMMUNITY-LEVEL PARALLEL ===
        mask = df[type_col].str.lower() != 'u'
        comm_to_hashtag_nodes = df.loc[mask, res].astype(str).value_counts().to_dict()

        pool = Pool(processes=n_jobs)
        comm_args = [
            (comm,
             counts,
             comm_to_users.get(comm, set()),
             comm_to_hashtag_nodes.get(comm, 0),
             normalize)
            for comm, counts in comm_to_counts.items()
        ]

        n_comms = len(comm_args)
        print(f"  Starting entropy stats for {n_comms:,} communities "
              f"using {n_jobs} workers...")

        pool = Pool(processes=n_jobs)
        results = []
        for i, result in enumerate(pool.imap_unordered(compute_comm_stats, comm_args, chunksize=500), 1):
            results.append(result)
            if i % 10000 == 0 or i == n_comms:
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
    out_csv = os.path.join(outdir, 'reply_entropy_per_community_per_resolution_sparse.csv')
    out_df.to_csv(out_csv, index=False)
    print("\nWrote", out_csv)

    return out_df


if __name__ == "__main__":
    leiden_csv = "/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/reply_hashtags/54121e227ef711f0b05c08f1eaf4fe18/nodes_with_communities.csv"
    edge_file = "/ipazianas/pasquini/output_graph_analysis/feb-aug_2022/2f019fee5d7411f0be9308f1eaf4fe18/graph/user_hashtag"
    outdir = "/home/pasquini/My_nas/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/entropy_retweet"
    analyze_entropy_sparse(leiden_csv, edge_file, outdir,
                             sep=",", chunk_size=1_000_000, n_jobs=25)