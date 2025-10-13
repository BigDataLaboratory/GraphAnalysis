#!/usr/bin/env python3
"""
summarize_entropy.py

Takes the detailed, filtered entropy data as input and generates summary statistics,
plots, and a list of top candidate echo chambers.

Usage:
    python summarize_entropy.py \\
        --input_csv /path/to/entropy_..._filtered.csv \\
        --outdir /path/to/summary_output_directory \\
        --scatter_res 0.5 \\
        --top_n 100
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_visualize_entropy(input_csv, outdir, scatter_res, top_n):
    """
    Main function to load data, compute summaries, and generate plots.
    """
    os.makedirs(outdir, exist_ok=True)
    print(f"Loading filtered entropy data from: {input_csv}")
    
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv}")
        return

    entropy_col = 'entropy_bits_normalized'
    if entropy_col not in df.columns:
        print(f"Error: Expected column '{entropy_col}' not found in the input CSV.")
        return

    # --- 1. Summary Statistics per Resolution ---
    print("Calculating summary statistics per resolution...")
    summary_df = df.groupby('resolution')[entropy_col].agg(['mean', 'median', 'count']).reset_index()
    summary_df.rename(columns={'count': 'n_communities'}, inplace=True)
    
    summary_csv_path = os.path.join(outdir, 'reply_entropy_summary_by_resolution.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"  ✔ Summary statistics saved to: {summary_csv_path}")
    print("\n--- Summary ---")
    print(summary_df)
    print("---------------")

    # --- 2. Plot 1: Mean/Median Entropy vs. Resolution ---
    print("Generating Plot 1: Mean/Median Entropy vs. Resolution...")
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    plt.plot(summary_df['resolution'], summary_df['mean'], marker='o', linestyle='-', label='Mean Entropy')
    plt.plot(summary_df['resolution'], summary_df['median'], marker='s', linestyle='--', label='Median Entropy')
    plt.title('Mean and Median Community Entropy Across Resolutions')
    plt.xlabel('Leiden Resolution Parameter')
    plt.ylabel('Normalized Entropy')
    plt.legend()
    plt.grid(True, which="both", ls="--", c='0.7')
    plot1_path = os.path.join(outdir, 'reply_mean_median_entropy_vs_resolution.png')
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"  ✔ Plot saved to: {plot1_path}")

    # --- 3. Plot 2: Entropy Distribution (Box Plot) ---
    print("Generating Plot 2: Entropy Distribution vs. Resolution...")
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='resolution', y=entropy_col, data=df, palette="coolwarm")
    plt.title('Distribution of Community Entropy Across Resolutions')
    plt.xlabel('Leiden Resolution Parameter')
    plt.ylabel('Normalized Entropy')
    plot2_path = os.path.join(outdir, 'reply_entropy_distribution_boxplot.png')
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"  ✔ Plot saved to: {plot2_path}")

    # --- 4. Plot 3: Community Size vs. Entropy Scatter Plot ---
    print(f"Generating Plot 3: Size vs. Entropy for resolution...")

    resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for res in resolutions:
        df_res = df[df['resolution'] == res].copy()
        
        if df_res.empty:
            print(f"  - Warning: No data found for resolution {res}. Skipping scatter plot.")
        else:
            plt.figure(figsize=(12, 8))
            # Use a subset for plotting if the data is too large to render clearly
            plot_df = df_res.sample(n=min(50000, len(df_res)), random_state=42)
            
            scatter = sns.scatterplot(
                x='n_users',
                y=entropy_col,
                data=plot_df,
                alpha=0.5,
                edgecolor=None
            )
            scatter.set_xscale('log')
            plt.title(f'Community Size vs. Thematic Focus (Entropy) at Resolution {res}')
            plt.xlabel('Number of Users in Community (Log Scale)')
            plt.ylabel('Normalized Entropy')
            plt.grid(True, which="both", ls="--", c='0.7')
            plot3_path = os.path.join(outdir, f'reply_size_vs_entropy_scatter_res_{res}.png')
            plt.savefig(plot3_path, dpi=300)
            plt.close()
            print(f"  ✔ Plot saved to: {plot3_path}")

        # --- 5. Identify and Save Top Candidate Echo Chambers ---
        print(f"Identifying top {top_n} candidate echo chambers for resolution {res}...")
        if not df_res.empty:
            top_candidates = df_res.sort_values(by=entropy_col, ascending=True).head(top_n)
            candidates_csv_path = os.path.join(outdir, f'reply_top_{top_n}_candidates_res_{res}.csv')
            top_candidates.to_csv(candidates_csv_path, index=False)
            print(f"  ✔ Top candidates saved to: {candidates_csv_path}")

    print("\n✅ All analyses complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize and visualize filtered community entropy results.")
    parser.add_argument('--input_csv', required=True, help='Path to the filtered entropy CSV file (output from analyze_entropy_filtered.py).')
    parser.add_argument('--outdir', required=True, help='Directory to save summary CSVs and plots.')
    parser.add_argument('--scatter_res', type=float, default=0.5, help='Resolution to use for the detailed size vs. entropy scatter plot.')
    parser.add_argument('--top_n', type=int, default=100, help='Number of top candidate echo chambers (lowest entropy) to save.')
    args = parser.parse_args()

    analyze_and_visualize_entropy(
        input_csv=args.input_csv,
        outdir=args.outdir,
        scatter_res=args.scatter_res,
        top_n=args.top_n
    )
