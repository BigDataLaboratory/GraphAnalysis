#!/usr/bin/env python3
"""
qualitative_analysis.py

Performs a qualitative analysis of echo chamber candidates by identifying and
translating their most frequently used hashtags.

Usage:
    python qualitative_analysis.py \\
        --candidates_csv /path/to/echo_chamber_candidates.csv \\
        --leiden_csv /path/to/retweet_leiden.csv \\
        --edge_file /path/to/user_hashtag_edges.csv \\
        --hashtag_map_csv /path/to/hashtag_map.csv \\
        --outdir /path/to/qualitative_reports \\
        --top_n 20
"""
import argparse
import pandas as pd
from collections import defaultdict, Counter
import os

def load_hashtag_map(filepath, sep='\t'):
    """Loads the hashtag hash-to-text mapping into a dictionary."""
    print(f"Loading hashtag map from: {filepath}")
    try:
        df = pd.read_csv(filepath, sep=sep, header=None, names=['text', 'hash', 'type'], dtype=str)
        hashtag_map = dict(zip(df['hash'], df['text']))
        print(f"  > Loaded {len(hashtag_map):,} hashtag mappings.")
        return hashtag_map
    except FileNotFoundError:
        print(f"  > Warning: Hashtag map file not found. Output will only contain hashes.")
        return {}
    except Exception as e:
        print(f"  > Error loading hashtag map: {e}")
        return {}

def get_community_members(leiden_df, community_id, resolution):
    """Gets the set of user names for a specific community ID and resolution."""
    # This function now expects clean string inputs for community_id and resolution
    members = leiden_df[
        (leiden_df['type'].str.lower() == 'u') &
        (leiden_df[resolution] == community_id)
    ]['name'].tolist()
    
    return set(members)

def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # --- 1. Load all necessary files ---
    print("Loading input files...")
    try:
        df_candidates = pd.read_csv(args.candidates_csv)
        # We use the source (retweet) leiden CSV to get the member lists
        df_leiden = pd.read_csv(args.leiden_csv, dtype=str, keep_default_na=False)
    except FileNotFoundError as e:
        print(f"Error loading critical file: {e}. Exiting.")
        return

    hashtag_map = load_hashtag_map(args.hashtag_map_csv, sep=args.sep)
    
    # --- 2. Iterate through each candidate echo chamber ---
    print(f"\nFound {len(df_candidates)} candidate echo chambers to analyze.")
    
    for index, candidate in df_candidates.iterrows():
        # --- START: ROBUST TYPE CONVERSION ---
        # This is the key change to prevent matching errors.
        # It ensures resolution is a string that matches column names (e.g., '0.1').
        res_str = f"{candidate['resolution']:.1f}"
        
        # It ensures community ID is a clean integer string (e.g., '881') to avoid
        # float issues like '881.0' which would fail a string comparison against '881'.
        try:
            comm_id_str = str(int(float(candidate['source_comm_id'])))
        except (ValueError, TypeError):
            print(f"  > Warning: Could not parse community ID '{candidate['source_comm_id']}'. Skipping.")
            continue
        # --- END: ROBUST TYPE CONVERSION ---

        jaccard = candidate['best_jaccard_score']
        rt_entropy = candidate['rt_entropy']
        
        print(f"\n--- Analyzing Candidate {index+1}/{len(df_candidates)} ---")
        print(f"  Resolution: {res_str}, Community ID: {comm_id_str}, Jaccard: {jaccard:.3f}, RT Entropy: {rt_entropy:.3f}")

        # --- 3. Get the list of users for this community ---
        members = get_community_members(df_leiden, comm_id_str, res_str)
        if not members:
            print("  > Warning: Could not find any members for this community. Skipping.")
            continue
        print(f"  > Found {len(members):,} members.")

        # --- 4. Find all hashtags used by these members ---
        hashtag_counts = Counter()
        print("  > Scanning edge file for hashtag usage...")
        with open(args.edge_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    etype, src, dst, weight = line.strip().split(args.sep)
                    if int(etype) != 2:
                        continue
                        
                    user, hashtag = (None, None)
                    if src in members:
                        user, hashtag = src, dst
                    elif dst in members:
                        user, hashtag = dst, src
                    
                    if user:
                        hashtag_counts[hashtag] += int(float(weight))
                except Exception:
                    continue
        
        if not hashtag_counts:
            print("  > No hashtag usage found for members of this community. Skipping.")
            continue

        # --- 5. Translate hashes and create a report ---
        report_data = []
        for hashtag_hash, count in hashtag_counts.most_common(args.top_n):
            hashtag_text = hashtag_map.get(hashtag_hash, "N/A")
            report_data.append({
                'rank': len(report_data) + 1,
                'hashtag_text': hashtag_text,
                'hashtag_hash': hashtag_hash,
                'count': count
            })
            
        df_report = pd.DataFrame(report_data)
        
        # Save the report to a specific file for this community
        outfile = os.path.join(args.outdir, f"candidate_res{res_str}_comm{comm_id_str}.csv")
        df_report.to_csv(outfile, index=False)
        print(f"  ✔ Saved top {args.top_n} hashtags report to: {outfile}")

    print("\n✅ All candidates analyzed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform qualitative analysis on echo chamber candidates.")
    parser.add_argument('--candidates_csv', required=True, help='Path to the final echo chamber candidates CSV.')
    parser.add_argument('--leiden_csv', required=True, help='Path to the source (retweet) Leiden CSV to get community members.')
    parser.add_argument('--edge_file', required=True, help='Path to the user-hashtag edge list file.')
    parser.add_argument('--hashtag_map_csv', required=True, help='Path to the CSV mapping hashtag hashes to text.')
    parser.add_argument('--outdir', required=True, help='Directory to save the individual community reports.')
    parser.add_argument('--top_n', type=int, default=500, help='Number of top hashtags to report for each community.')
    parser.add_argument('--sep', default=',', help='Separator for the hashtag map and edge files (default: tab).')
    args = parser.parse_args()

    main(args)

