#!/usr/bin/env python3
"""
analyze_jaccard_trends.py

Analyzes the trend of Jaccard similarity between retweet and reply communities
across all resolution parameters.

This script takes the output from `find_counterparts.py` and calculates summary
statistics (mean, median, max) for the Jaccard scores at each resolution,
visualizing the results to show how structural overlap changes with scale.

Usage:
    python analyze_jaccard_trends.py \\
        --input_csv /path/to/all_jaccard_matches.csv \\
        --outdir /path/to/trend_analysis_results
"""
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main(args):
    """
    Main function to load data, perform trend analysis, and save outputs.
    """
    os.makedirs(args.outdir, exist_ok=True)

    # --- 1. Load the comprehensive Jaccard match data ---
    try:
        print(f"Loading Jaccard match data from: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_csv}'.")
        print("Please run the `find_counterparts.py` script first to generate this file.")
        return

    # Ensure resolution is treated as a numeric type for proper sorting and plotting
    df['resolution'] = pd.to_numeric(df['resolution'])
    
    print(f"  > Loaded {len(df):,} community matches across {df['resolution'].nunique()} resolutions.")

    # --- 2. Group by resolution and calculate summary statistics ---
    print("\nCalculating trend statistics for Jaccard scores by resolution...")
    
    # The core of the trend analysis: group by resolution and aggregate key stats.
    # - mean: The average structural overlap.
    # - median: The "typical" structural overlap, resistant to outliers.
    # - max: The Jaccard score of the single most-aligned community pair.
    # - count: The number of substantive communities found at that resolution.
    summary_df = df.groupby('resolution')['best_jaccard_score'].agg(['mean', 'median', 'max', 'count']).reset_index()
    summary_df = summary_df.rename(columns={'count': 'n_matched_communities'})

    # Save the summary statistics to a new CSV file
    summary_csv_path = os.path.join(args.outdir, 'jaccard_trends_by_resolution.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"  > Analysis complete. Saved summary statistics to: {summary_csv_path}")
    print("\n--- Summary of Jaccard Score Trends ---")
    print(summary_df.to_string(index=False))
    print("---------------------------------------")

    # --- 3. Visualize the trends ---
    print("\nGenerating trend plot...")
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    plt.plot(summary_df['resolution'], summary_df['mean'], marker='o', linestyle='-', label='Mean Jaccard Score (Average Overlap)')
    plt.plot(summary_df['resolution'], summary_df['median'], marker='s', linestyle='--', label='Median Jaccard Score (Typical Overlap)')
    plt.plot(summary_df['resolution'], summary_df['max'], marker='^', linestyle=':', color='green', label='Max Jaccard Score (Strongest Echo Chamber)')
    
    plt.title('Trend of Structural Overlap (Jaccard) vs. Community Resolution')
    plt.xlabel('Leiden Resolution Parameter (Gamma)')
    plt.ylabel('Jaccard Similarity Score')
    plt.xticks(summary_df['resolution'])
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.ylim(bottom=0)
    
    plot_path = os.path.join(args.outdir, 'jaccard_trends_plot.png')
    plt.savefig(plot_path, dpi=300)
    
    print(f"  > Successfully saved trend plot to: {plot_path}")
    print("\n✅ Analysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trends in Jaccard similarity across resolutions.")
    parser.add_argument('--input_csv', required=True, help='Path to the combined Jaccard match CSV file (output of find_counterparts.py).')
    parser.add_argument('--outdir', required=True, help='Directory to save the summary CSV and plot.')
    args = parser.parse_args()

    main(args)
