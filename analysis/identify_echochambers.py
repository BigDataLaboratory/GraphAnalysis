#!/usr/bin/env python3
"""
identify_echo_chambers.py

Applies a three-step filtering process to identify strong echo chamber candidates.
It uses structural overlap (Jaccard) and thematic coherence (entropy) to find
communities that are insular in both their structure and conversation.

Usage:
    python identify_echo_chambers.py \\
        --matches_csv /path/to/all_jaccard_matches.csv \\
        --rt_entropy_csv /path/to/retweet_entropy_filtered.csv \\
        --reply_entropy_csv /path/to/reply_entropy_filtered.csv \\
        --outfile /path/to/echo_chamber_candidates.csv \\
        --jaccard_threshold 0.4 \\
        --entropy_quantile 0.25
"""
import argparse
import pandas as pd

def identify_echo_chambers(matches_csv, rt_entropy_csv, reply_entropy_csv,
                           outfile, jaccard_threshold, entropy_quantile):
    """
    Loads Jaccard matches and entropy data to perform a three-step filter.
    """
    print("Loading data...")
    try:
        df_matches = pd.read_csv(matches_csv)
        df_rt_entropy = pd.read_csv(rt_entropy_csv)
        df_reply_entropy = pd.read_csv(reply_entropy_csv)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please check your file paths.")
        return

    print(f"Initial number of potential matches: {len(df_matches):,}")

    # --- Step 1: Filter for Structural Overlap ---
    print(f"\n--- Step 1: Filtering for high structural overlap ---")
    print(f"Keeping matches with Jaccard score > {jaccard_threshold}")
    
    structurally_insular = df_matches[df_matches['best_jaccard_score'] > jaccard_threshold].copy()
    
    n_after_step1 = len(structurally_insular)
    print(f"  > Found {n_after_step1:,} structurally insular communities.")
    if n_after_step1 == 0:
        print("No communities passed the structural filter. Exiting.")
        return

    # --- Step 2: Filter for Amplification Coherence (Low Retweet Entropy) ---
    print(f"\n--- Step 2: Filtering for high amplification coherence (low retweet entropy) ---")
    
    # Calculate the entropy threshold based on the specified quantile
    entropy_col = 'entropy_bits_normalized'
    rt_entropy_threshold = df_rt_entropy[entropy_col].quantile(entropy_quantile)
    print(f"Using retweet entropy threshold (bottom {entropy_quantile*100}%): {rt_entropy_threshold:.4f}")

    # Merge with retweet entropy data
    # Ensure column types are consistent for merging
    structurally_insular['source_comm_id'] = structurally_insular['source_comm_id'].astype(str)
    df_rt_entropy['community'] = df_rt_entropy['community'].astype(str)
    
    merged_step2 = pd.merge(
        structurally_insular,
        df_rt_entropy[['community', 'resolution', entropy_col]],
        how='inner',
        left_on=['source_comm_id', 'resolution'],
        right_on=['community', 'resolution']
    ).rename(columns={entropy_col: 'rt_entropy'})
    
    # Apply the entropy filter
    amplification_focused = merged_step2[merged_step2['rt_entropy'] < rt_entropy_threshold].copy()
    
    n_after_step2 = len(amplification_focused)
    print(f"  > Found {n_after_step2:,} communities that are also thematically focused in amplification.")
    if n_after_step2 == 0:
        print("No communities passed the amplification filter. Exiting.")
        return

    # --- Step 3: Filter for Conversational Coherence (Low Reply Entropy) ---
    print(f"\n--- Step 3: Filtering for high conversational coherence (low reply entropy) ---")

    # Calculate the reply entropy threshold
    reply_entropy_threshold = df_reply_entropy[entropy_col].quantile(entropy_quantile)
    print(f"Using reply entropy threshold (bottom {entropy_quantile*100}%): {reply_entropy_threshold:.4f}")

    # Merge with reply entropy data
    amplification_focused['best_target_comm_id'] = amplification_focused['best_target_comm_id'].astype(str)
    df_reply_entropy['community'] = df_reply_entropy['community'].astype(str)
    
    merged_step3 = pd.merge(
        amplification_focused,
        df_reply_entropy[['community', 'resolution', entropy_col]],
        how='inner',
        left_on=['best_target_comm_id', 'resolution'],
        right_on=['community', 'resolution']
    ).rename(columns={entropy_col: 'reply_entropy'})

    # Apply the final entropy filter
    final_candidates = merged_step3[merged_step3['reply_entropy'] < reply_entropy_threshold].copy()
    
    n_final = len(final_candidates)
    print(f"  > Found {n_final:,} final echo chamber candidates that are also thematically focused in conversation.")

    # --- Final Output ---
    if n_final > 0:
        # Clean up and reorder columns for clarity
        final_candidates = final_candidates.drop(columns=['community_x', 'community_y'])
        final_candidates = final_candidates.sort_values(by=['resolution', 'best_jaccard_score'], ascending=[True, False])
        
        final_candidates.to_csv(outfile, index=False)
        print(f"\n✅ Successfully saved {n_final} echo chamber candidates to: {outfile}")
        
        print("\n--- Top Final Candidates ---")
        print(final_candidates.head(10).to_string(index=False))
        print("----------------------------")
    else:
        print("\nNo communities survived all three filters. No output file was created.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify echo chamber candidates using a three-step filter.")
    parser.add_argument('--matches_csv', required=True, help='Path to the Jaccard matches CSV (output of find_counterparts.py).')
    parser.add_argument('--rt_entropy_csv', required=True, help='Path to the filtered retweet entropy CSV.')
    parser.add_argument('--reply_entropy_csv', required=True, help='Path to the filtered reply entropy CSV.')
    parser.add_argument('--outfile', required=True, help='Path to save the final list of echo chamber candidates.')
    parser.add_argument('--jaccard_threshold', type=float, default=0.4, help='Minimum Jaccard score for structural overlap (e.g., 0.3).')
    parser.add_argument('--entropy_quantile', type=float, default=0.25, help='Quantile for entropy threshold (e.g., 0.25 for bottom 25%).')
    args = parser.parse_args()

    identify_echo_chambers(
        matches_csv=args.matches_csv,
        rt_entropy_csv=args.rt_entropy_csv,
        reply_entropy_csv=args.reply_entropy_csv,
        outfile=args.outfile,
        jaccard_threshold=args.jaccard_threshold,
        entropy_quantile=args.entropy_quantile
    )
