import pandas as pd
import numpy as np
import math
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, mutual_info_score

def variation_of_information(labels1, labels2):
    """Compute Variation of Information (VI) in bits."""
    # Ensure integer labels starting at 0 for bincount
    labels1 = np.asarray(labels1)
    labels2 = np.asarray(labels2)
    mi = mutual_info_score(labels1, labels2)
    def entropy(labels):
        vals, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        # entropy base 2 (bits)
        return -np.sum([p * math.log(p, 2) for p in probs if p > 0])
    h1 = entropy(labels1)
    h2 = entropy(labels2)
    vi = h1 + h2 - 2 * mi
    return vi

# --- 1. Configuration ---
# Update these file paths to match your filenames
retweet_file = '/home/pasquini/My_nas/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/only_retweet/e70d30d47de611f0845408f1eaf4fe18/nodes_with_communities.csv'
reply_file = '/home/pasquini/My_nas/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/only_response/21271f7c7e7411f09d5308f1eaf4fe18/nodes_with_communities.csv'

# List of resolution columns to compare
resolutions = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

# --- 2. Load and Prepare Data ---
print("Loading data...")
# Load the datasets
df_rt = pd.read_csv(retweet_file)
df_reply = pd.read_csv(reply_file)

# Filter for nodes of type 'user' if hashtags are present
# This ensures we only compare the clustering of users
if 'type' in df_rt.columns:
    df_rt = df_rt[df_rt['type'] == 'u'].copy()
if 'type' in df_reply.columns:
    df_reply = df_reply[df_reply['type'] == 'u'].copy()
    
print(f"Found {len(df_rt)} users in the retweet graph.")
print(f"Found {len(df_reply)} users in the reply graph.")

# --- 3. Align Users ---
# Merge the two dataframes on the user's name (or id).
# This is a CRUCIAL step. It ensures:
#   a) We only compare users present in BOTH graphs (the intersection).
#   b) The community lists are perfectly aligned for comparison.
merged_df = pd.merge(
    df_rt, 
    df_reply, 
    on='name', 
    suffixes=('_rt', '_reply')
)

print(f"Comparing {len(merged_df)} users common to both graphs.")

# --- 4. Calculate Similarity Metrics ---
results = []

for res in resolutions:
    # Get the community assignments (partitions) for this resolution
    # The column names will be like '0.1_rt' and '0.1_reply' after the merge
    partition_rt = merged_df[f'{res}_rt']
    partition_reply = merged_df[f'{res}_reply']
    
    # Calculate the metrics
    nmi = normalized_mutual_info_score(partition_rt, partition_reply)
    ari = adjusted_rand_score(partition_rt, partition_reply)
    vi = variation_of_information(partition_rt, partition_reply)

    results.append({
        'resolution': float(res),
        'nmi': nmi,
        'ari': ari,
        'vi': vi
    })

# Convert results to a DataFrame for easy viewing
results_df = pd.DataFrame(results)

results_df.to_csv('/home/pasquini/My_nas/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/result/retweet_vs_reply_community_similarity.csv', index=False)

# --- 6. (Optional) Visualization ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=results_df, x='resolution', y='nmi', marker='o', label='NMI')
    sns.lineplot(data=results_df, x='resolution', y='ari', marker='o', label='ARI')
    
    plt.title('Similarity between Retweet and Reply Communities')
    plt.xlabel('Leiden Resolution Parameter')
    plt.ylabel('Similarity Score')
    plt.ylim(0, 1)
    plt.legend()
    
    # Save the plot to an image file before showing it
    plt.savefig("/home/pasquini/My_nas/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/result/community_similarity_plot.png", dpi=300) # <-- ADDED LINE
    print(f"✅ Plot saved to {output_plot_file}")
    
    plt.show()

except ImportError:
    print("\nTo visualize results, please install matplotlib and seaborn: pip install matplotlib seaborn")