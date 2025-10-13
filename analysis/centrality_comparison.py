#!/usr/bin/env python3
"""
plot_centrality_comparison.py

Creates a scatter plot to compare the centralization (Gini coefficient) of the
retweet and reply layers for a list of candidate echo chambers.

Usage:
    python plot_centrality_comparison.py \\
        --input_csv /path/to/centrality_summary_all_candidates.csv \\
        --outfile centrality_comparison_plot.png
"""
import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main(args):
    """
    Main function to load the centrality summary and generate the scatter plot.
    """
    os.makedirs(os.path.dirname(args.outfile) or '.', exist_ok=True)

    # --- 1. Load the centrality summary data ---
    try:
        print(f"Loading centrality summary from: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{args.input_csv}'.")
        print("Please run the `analyze_centrality_batch.py` script first to generate this file.")
        return

    # Drop any rows where centrality could not be computed
    df = df.dropna(subset=[args.x_axis, args.y_axis])
    
    if df.empty:
        print("No valid data to plot after removing missing values.")
        return

    print(f"  > Plotting {len(df)} candidate communities.")

    # --- 2. Generate the scatter plot ---
    print("\nGenerating scatter plot...")
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 10))
    
    # Create the scatter plot
    sns.scatterplot(
        data=df,
        x=args.x_axis,
        y=args.y_axis,
        alpha=0.7,
        label='Candidate Echo Chambers'
    )
    
    # Add a y=x reference line
    max_val = max(df[args.x_axis].max(), df[args.y_axis].max()) * 1.1
    min_val = min(df[args.x_axis].min(), df[args.y_axis].min()) * 0.9
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x (Equal Centralization)')

    # --- 3. Final Touches ---
    #plt.title('Comparison of Structural Centralization in Echo Chambers', fontsize=16)
    plt.xlabel(f'Amplification Centralization (Gini Pagerank)', fontsize=12)
    plt.ylabel(f'Conversational Centralization (Gini Betweenness)', fontsize=12)
    plt.legend()
    plt.axis('equal') # Ensure the x and y axes have the same scale for accurate comparison
    plt.grid(True, which="both", ls="--")
    
    # Save the plot
    plt.savefig(args.outfile, dpi=300)
    
    print(f"  > Successfully saved plot to: {args.outfile}")
    print("\n✅ Analysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a comparison of community centrality structures.")
    parser.add_argument('--input_csv', required=True, help='Path to the centrality_summary_all_candidates.csv file.')
    parser.add_argument('--outfile', required=True, help='Path to save the output PNG plot.')
    parser.add_argument('--x_axis', default='gini_pagerank', help='Column to use for the X-axis (e.g., gini_pagerank or gini_in_degree).')
    parser.add_argument('--y_axis', default='gini_betweenness', help='Column to use for the Y-axis (e.g., gini_betweenness).')
    args = parser.parse_args()

    main(args)
