#!/usr/bin/env python3
"""
analyze_centrality_batch.py

Performs a centrality analysis in parallel for a list of candidate echo chambers,
quantitatively assessing the centralization of each one. This version correctly
uses edge weights to calculate centrality scores.

Usage:
    python analyze_centrality_batch.py \\
        --candidates_csv /path/to/echo_chamber_candidates.csv \\
        --leiden_csv /path/to/retweet_leiden.csv \\
        --retweet_edges /path/to/retweet_edges.csv \\
        --reply_edges /path/to/reply_edges.csv \\
        --outdir centrality_analysis \\
        --n_jobs 8
"""
import argparse
import pandas as pd
import os
import networkx as nx
import numpy as np
from multiprocessing import Pool, cpu_count

def get_community_members(leiden_df, community_id, resolution):
    """
    Gets the set of user names for a specific community ID and resolution,
    handling potential data type mismatches.
    """
    res_str = str(resolution)
    comm_id_str = str(community_id)
    
    try:
        cleaned_comm_ids = pd.to_numeric(leiden_df[res_str], errors='coerce').astype('Int64').astype(str)
    except Exception:
        cleaned_comm_ids = leiden_df[res_str].astype(str)

    members = leiden_df[
        (leiden_df['type'].str.lower() == 'u') &
        (cleaned_comm_ids == comm_id_str)
    ]['name'].tolist()
    
    return set(members)

def gini_coefficient(x):
    """Compute Gini coefficient of a numpy array."""
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(x == x[0]):
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)) if np.sum(x) != 0 else 0.0

def analyze_subgraph_centrality(members, edge_file, graph_type, community_id, resolution, outdir, sep=','):
    """
    Builds a weighted community subgraph, calculates relevant centrality metrics,
    and returns a summary.
    """
    G = nx.DiGraph()
    G.add_nodes_from(members)
    
    with open(edge_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                parts = line.strip().split(sep)
                if len(parts) < 3: continue # Ensure weight is present
                source, target, weight_str = parts[1], parts[2], parts[3]
                weight = float(weight_str)
                if source in members and target in members:
                    # Add the edge with its weight
                    G.add_edge(source, target, weight=weight)
            except Exception:
                continue

    if G.number_of_edges() == 0:
        return None, None

    if graph_type == 'Retweet':
        # Use the 'weight' attribute in centrality calculations
        pagerank_scores = nx.pagerank(G, weight='weight')
        in_degree_scores = {n: d for n, d in G.in_degree(weight='weight')}
        centrality_data = pd.DataFrame({
            'user': list(G.nodes()),
            'pagerank': [pagerank_scores.get(n, 0) for n in G.nodes()],
            'in_degree': [in_degree_scores.get(n, 0) for n in G.nodes()]
        }).sort_values(by='pagerank', ascending=False)
        
        gini_pagerank = gini_coefficient(centrality_data['pagerank'].values)
        gini_in_degree = gini_coefficient(centrality_data['in_degree'].values)
        summary = {'gini_pagerank': gini_pagerank, 'gini_in_degree': gini_in_degree}

    elif graph_type == 'Reply':
        # Use the 'weight' attribute for betweenness centrality
        betweenness_scores = nx.betweenness_centrality(G, weight='weight')
        centrality_data = pd.DataFrame({
            'user': list(G.nodes()),
            'betweenness': [betweenness_scores.get(n, 0) for n in G.nodes()]
        }).sort_values(by='betweenness', ascending=False)
        
        gini_betweenness = gini_coefficient(centrality_data['betweenness'].values)
        summary = {'gini_betweenness': gini_betweenness}
    else:
        return None, None

    outfile = os.path.join(outdir, f"centrality_{graph_type.lower()}_comm{community_id}_res{resolution}.csv")
    centrality_data.to_csv(outfile, index=False)
    
    return summary, centrality_data

def process_candidate(args_tuple):
    """
    Worker function to process a single candidate community. This function
    is designed to be called by a multiprocessing Pool.
    """
    candidate, leiden_df, retweet_edges_path, reply_edges_path, outdir, sep = args_tuple
    
    comm_id = str(candidate['source_comm_id'])
    res = str(candidate['resolution'])
    
    print(f"--- Analyzing Candidate: Community '{comm_id}' at Resolution '{res}' ---")
    
    members = get_community_members(leiden_df, comm_id, res)
    
    if not members:
        print(f"  > Warning: Could not find members for comm {comm_id}, res {res}. Skipping.")
        return None
    
    rt_summary, _ = analyze_subgraph_centrality(members, retweet_edges_path, "Retweet", comm_id, res, outdir, sep=sep)
    reply_summary, _ = analyze_subgraph_centrality(members, reply_edges_path, "Reply", comm_id, res, outdir, sep=sep)
    
    summary_row = {'community_id': comm_id, 'resolution': res}
    if rt_summary:
        summary_row.update(rt_summary)
    if reply_summary:
        summary_row.update(reply_summary)
        
    return summary_row

def main(args):
    """
    Main function to orchestrate the parallel centrality analysis.
    """
    os.makedirs(args.outdir, exist_ok=True)
    
    n_jobs = args.n_jobs if args.n_jobs else max(1, cpu_count() - 1)

    try:
        print("Loading Leiden CSV:", args.leiden_csv)
        df_leiden = pd.read_csv(args.leiden_csv, dtype=str, keep_default_na=False)
        
        print("Loading Candidates CSV:", args.candidates_csv)
        df_candidates = pd.read_csv(args.candidates_csv, dtype=str, keep_default_na=False)
    except FileNotFoundError as e:
        print(f"Error: {e}. Exiting.")
        return
        
    print(f"\nFound {len(df_candidates)} candidates to analyze. Starting batch processing with {n_jobs} workers...")

    # Create a list of arguments for each parallel task
    tasks = [
        (row, df_leiden, args.retweet_edges, args.reply_edges, args.outdir, args.sep)
        for _, row in df_candidates.iterrows()
    ]

    # Use a multiprocessing Pool to execute the tasks in parallel
    with Pool(processes=n_jobs) as pool:
        # imap_unordered is used to get results as they complete, which is good for progress
        all_summaries = [result for result in pool.imap_unordered(process_candidate, tasks) if result is not None]

    if not all_summaries:
        print("\nNo candidates were successfully analyzed.")
        return

    # --- Create and Save Final Summary DataFrame ---
    summary_df = pd.DataFrame(all_summaries)
    # Sort for consistent output
    summary_df = summary_df.sort_values(by=['resolution', 'community_id'])
    
    summary_outfile = os.path.join(args.outdir, "centrality_summary_all_candidates.csv")
    summary_df.to_csv(summary_outfile, index=False)
    
    print("\n✅ Batch analysis complete.")
    print(f"  > Comprehensive summary saved to: {summary_outfile}")
    print("\n--- Centrality Inequality Summary (Gini Coefficient) ---")
    print(summary_df.to_string(index=False))
    print("---------------------------------------------------------")
    print("Higher Gini = More unequal distribution (more centralized)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze centrality in parallel for a list of candidate communities.")
    parser.add_argument('--candidates_csv', required=True, help='Path to the CSV file listing candidate echo chambers.')
    parser.add_argument('--leiden_csv', required=True, help='Path to the source (retweet) Leiden CSV.')
    parser.add_argument('--retweet_edges', required=True, help='Path to the user-user retweet edge file.')
    parser.add_argument('--reply_edges', required=True, help='Path to the user-user reply edge file.')
    parser.add_argument('--outdir', required=True, help='Directory to save the centrality score CSVs.')
    parser.add_argument('--sep', default=',', help='Separator for the edge files.')
    parser.add_argument('--n_jobs', type=int, default=3, help='Number of parallel processes to use (defaults to CPU count - 1).')
    args = parser.parse_args()

    main(args)

