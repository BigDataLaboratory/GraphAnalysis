#!/usr/bin/env python3
"""
find_counterparts.py

For each community in a source graph (e.g., retweets), this script finds the
community in a target graph (e.g., replies) that has the highest Jaccard
similarity. It automatically runs this analysis for all available resolution
parameters and saves the combined results to a single file.

Usage:
    python find_counterparts.py \\
        --source_csv /path/to/retweet_leiden.csv \\
        --target_csv /path/to/reply_leiden.csv \\
        --min_comm_size 11 \\
        --outfile /path/to/all_jaccard_matches.csv
"""
import argparse
import pandas as pd
from collections import defaultdict

def get_filtered_communities(df, res_col, min_size):
    """
    Creates a dictionary of communities {comm_id: {user_set}} and filters
    it to include only communities meeting the minimum size.
    """
    communities = defaultdict(set)
    users_df = df[df['type'].str.lower() == 'u']
    for _, row in users_df.iterrows():
        # Ensure community IDs are treated as strings to avoid type issues
        communities[str(row[res_col])].add(row['name'])
    
    n_before = len(communities)
    filtered_communities = {
        comm_id: users for comm_id, users in communities.items()
        if len(users) >= min_size
    }
    n_after = len(filtered_communities)
    print(f"  - Kept {n_after:,} communities out of {n_before:,} (size >= {min_size})")
    
    return filtered_communities

def find_best_jaccard_matches(source_comms, target_comms):
    """
    For each community in source_comms, find the best match in target_comms.
    """
    matches = []
    source_items = list(source_comms.items())
    
    for i, (source_id, source_users) in enumerate(source_items):
        best_match = {'target_id': None, 'jaccard_score': -1.0}
        
        for target_id, target_users in target_comms.items():
            intersection = len(source_users.intersection(target_users))
            if intersection == 0:
                continue
            
            union = len(source_users.union(target_users))
            jaccard = intersection / union if union > 0 else 0.0
            
            if jaccard > best_match['jaccard_score']:
                best_match['target_id'] = target_id
                best_match['jaccard_score'] = jaccard
        
        if (i + 1) % 500 == 0 or (i + 1) == len(source_items):
             print(f"    Progress: Matched {i+1}/{len(source_items)} source communities...")

        if best_match['target_id'] is not None:
            matches.append({
                'source_comm_id': source_id,
                'source_comm_size': len(source_users),
                'best_target_comm_id': best_match['target_id'],
                'best_target_comm_size': len(target_comms.get(best_match['target_id'], set())),
                'best_jaccard_score': best_match['jaccard_score']
            })
            
    return pd.DataFrame(matches)

def main(args):
    print("Loading source CSV:", args.source_csv)
    df_source = pd.read_csv(args.source_csv, dtype=str, keep_default_na=False)
    
    print("Loading target CSV:", args.target_csv)
    df_target = pd.read_csv(args.target_csv, dtype=str, keep_default_na=False)

    # Automatically detect resolution columns from the source dataframe
    resolutions = sorted([c for c in df_source.columns if c.replace('.', '', 1).isdigit()], key=float)
    if not resolutions:
        raise ValueError("No resolution columns (e.g., '0.1', '0.2') found in the source CSV.")
    print(f"\nFound resolutions to analyze: {resolutions}")

    all_matches = []

    for res in resolutions:
        print(f"\n--- Analyzing Resolution: {res} ---")
        
        print("  Processing source communities...")
        source_comms = get_filtered_communities(df_source, res, args.min_comm_size)
        
        print("  Processing target communities...")
        target_comms = get_filtered_communities(df_target, res, args.min_comm_size)

        if not source_comms or not target_comms:
            print(f"  Warning: No communities of sufficient size found for resolution {res}. Skipping.")
            continue

        print("  Finding best Jaccard matches...")
        matches_df = find_best_jaccard_matches(source_comms, target_comms)
        
        if not matches_df.empty:
            matches_df['resolution'] = res
            all_matches.append(matches_df)

    if not all_matches:
        print("\nNo matches found across any resolution. The output file will not be created.")
        return

    # Combine all results into a single DataFrame
    final_df = pd.concat(all_matches, ignore_index=True)

    # Sort by resolution and then by Jaccard score
    final_df = final_df.sort_values(by=['resolution', 'best_jaccard_score'], ascending=[True, False])
    
    final_df.to_csv(args.outfile, index=False)
    print(f"\n✅ Successfully saved all matches to: {args.outfile}")
    
    print("\n--- Top 10 Matches (Resolution 0.1) ---")
    print(final_df[final_df['resolution'] == '0.1'].head(10).to_string(index=False))
    print("------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find counterpart communities across all resolutions using Jaccard similarity.")
    parser.add_argument('--source_csv', required=True, help='Path to the source Leiden CSV (e.g., retweets).')
    parser.add_argument('--target_csv', required=True, help='Path to the target Leiden CSV (e.g., replies).')
    parser.add_argument('--min_comm_size', type=int, default=11, help='Minimum user count for a community to be included.')
    parser.add_argument('--outfile', required=True, help='Path to save the combined output CSV of all matches.')
    args = parser.parse_args()
    
    main(args)

