import pandas as pd
import numpy as np
import math
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Function Definitions ---

def variation_of_information(labels1, labels2):
    """Compute Variation of Information (VI) in bits."""
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    mi = mutual_info_score(labels1, labels2)
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        return -np.sum([p * math.log(p, 2) for p in probs if p > 0])
    h1 = entropy(labels1)
    h2 = entropy(labels2)
    vi = h1 + h2 - 2 * mi
    return vi

# --- 1. Configuration ---
# Update these file paths to match your filenames
retweet_file = '/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/only_retweet/e70d30d47de611f0845408f1eaf4fe18/nodes_with_communities.csv'
reply_file = '/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/only_response/21271f7c7e7411f09d5308f1eaf4fe18/nodes_with_communities.csv'
output_csv_file = '/scratch/pasquini/result_wc/retweet_vs_reply_community_similarity_filtered.csv'
output_plot_file = "/scratch/pasquini/result_wc/community_similarity_plot_filtered.png"

# Set the minimum size for a community to be included in the analysis
MIN_COMM_SIZE = 11  # This means we keep communities with > 10 users

# List of resolution columns to compare
resolutions = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

# --- 2. Load and Prepare Data ---
print("Loading data...")
df_rt = pd.read_csv(retweet_file, dtype=str, keep_default_na=False)
df_reply = pd.read_csv(reply_file, dtype=str, keep_default_na=False)

# Standardize 'type' column and filter for users
for df in [df_rt, df_reply]:
    if 'type' in df.columns:
        df['type'] = df['type'].str.lower()
        df.drop(df[df['type'] != 'u'].index, inplace=True)

print(f"Found {len(df_rt)} users in the retweet graph.")
print(f"Found {len(df_reply)} users in the reply graph.")

# --- 3. Align Users ---
# Merge the dataframes to get aligned community assignments for common users
merged_df = pd.merge(
    df_rt,
    df_reply,
    on='name',
    suffixes=('_rt', '_reply')
)
print(f"Comparing {len(merged_df)} users common to both graphs.")

# --- 4. Calculate Similarity Metrics with Filtering ---
results = []
print(f"\nStarting analysis, excluding communities with <= {MIN_COMM_SIZE - 1} users...")

for res in resolutions:
    print(f"--- Processing resolution {res} ---")
    
    # Get the original community assignments for this resolution
    partition_rt = merged_df[f'{res}_rt']
    partition_reply = merged_df[f'{res}_reply']
    
    # --- Filtering Step for Retweet Communities ---
    # Calculate the size of each community
    comm_sizes_rt = partition_rt.value_counts()
    # Identify the labels of communities that are too small
    small_comms_rt = comm_sizes_rt[comm_sizes_rt < MIN_COMM_SIZE].index
    # Create a new partition where users in small communities are reassigned to a 'noise' label (-1)
    partition_rt_filtered = partition_rt.apply(lambda x: '-1' if x in small_comms_rt else x)
    
    # --- Filtering Step for Reply Communities ---
    comm_sizes_reply = partition_reply.value_counts()
    small_comms_reply = comm_sizes_reply[comm_sizes_reply < MIN_COMM_SIZE].index
    partition_reply_filtered = partition_reply.apply(lambda x: '-1' if x in small_comms_reply else x)
    
    print(f"  Retweet: Kept {len(comm_sizes_rt) - len(small_comms_rt)} communities out of {len(comm_sizes_rt)}.")
    print(f"  Reply:   Kept {len(comm_sizes_reply) - len(small_comms_reply)} communities out of {len(comm_sizes_reply)}.")
    
    # Calculate the metrics on the NEW, FILTERED partitions
    nmi = normalized_mutual_info_score(partition_rt_filtered, partition_reply_filtered)
    ari = adjusted_rand_score(partition_rt_filtered, partition_reply_filtered)
    vi = variation_of_information(partition_rt_filtered, partition_reply_filtered)

    results.append({
        'resolution': float(res),
        'nmi': nmi,
        'ari': ari,
        'vi': vi
    })

# --- 5. Save and Display Results ---
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_file, index=False)
print(f"\n✅ Filtered results saved to {output_csv_file}")
print("\n--- Filtered Similarity Results ---")
print(results_df.to_string(index=False))


# --- 6. Visualization ---
try:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=results_df, x='resolution', y='nmi', marker='o', label='NMI')
    sns.lineplot(data=results_df, x='resolution', y='ari', marker='s', label='ARI')
    
    plt.title(f'Similarity between Retweet and Reply Communities (Users > {MIN_COMM_SIZE - 1})')
    plt.xlabel('Leiden Resolution Parameter')
    plt.ylabel('Similarity Score')
    plt.ylim(0, 1)
    plt.legend()
    
    plt.savefig(output_plot_file, dpi=300)
    print(f"✅ Plot saved to {output_plot_file}")
    
    # plt.show() # Uncomment if you want to display the plot interactively
except ImportError:
    print("\nTo visualize results, please install matplotlib and seaborn: pip install matplotlib seaborn")
