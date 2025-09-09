import pandas as pd
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# CONFIG
# -------------------------
FILES = {
    "R": "/home/pasquini/My_nas/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/only_retweet/e70d30d47de611f0845408f1eaf4fe18/nodes_with_communities.csv",
    "R+H": "/home/pasquini/My_nas/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/retweet_hashtags/0af23d747eae11f0b7b308f1eaf4fe18/nodes_with_communities.csv",
    "P": "/home/pasquini/My_nas/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/only_response/21271f7c7e7411f09d5308f1eaf4fe18/nodes_with_communities.csv",
    "P+H": "/home/pasquini/My_nas/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/reply_hashtags/54121e227ef711f0b05c08f1eaf4fe18/nodes_with_communities.csv"
}

PAIRS = [
    ("R", "R+H"),
    ("P", "P+H"),
    ("R", "P"),
    ("R+H", "P+H")
]

RESOLUTIONS = [str(round(x,1)) for x in np.arange(0.1, 1.1, 0.1)]
TOP_K = 500

# -------------------------
# FUNZIONI
# -------------------------
def get_top_communities(df, res, k=500, node_type="u"):
    sub = df[df["type"] == node_type].copy()
    groups = sub.groupby(res)["id"].apply(set).to_dict()
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    return dict(sorted_groups[:k])

def jaccard(set1, set2):
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter/union if union > 0 else 0

def best_match_jaccards(comms1, comms2):
    scores = []
    for cid1, nodes1 in comms1.items():
        best = 0
        for cid2, nodes2 in comms2.items():
            score = jaccard(nodes1, nodes2)
            if score > best:
                best = score
        scores.append(best)
    return scores

# -------------------------
# MAIN
# -------------------------
dfs = {name: pd.read_csv(path) for name, path in FILES.items()}
all_results = []

for g1, g2 in PAIRS:
    print(f"\n=== Processing {g1} vs {g2} ===")
    df1, df2 = dfs[g1], dfs[g2]

    for res in tqdm(RESOLUTIONS, desc=f"{g1} vs {g2}"):
        comms1 = get_top_communities(df1, res, k=TOP_K)
        comms2 = get_top_communities(df2, res, k=TOP_K)

        scores1 = best_match_jaccards(comms1, comms2)
        scores2 = best_match_jaccards(comms2, comms1)

        all_results.append({
            "pair": f"{g1} vs {g2}",
            "resolution": float(res),
            "mean_jaccard_1to2": np.mean(scores1) if scores1 else np.nan,
            "mean_jaccard_2to1": np.mean(scores2) if scores2 else np.nan,
            "median_jaccard_1to2": np.median(scores1) if scores1 else np.nan,
            "median_jaccard_2to1": np.median(scores2) if scores2 else np.nan,
            "max_jaccard_1to2": np.max(scores1) if scores1 else np.nan,
            "max_jaccard_2to1": np.max(scores2) if scores2 else np.nan,
            "n_comms_1": len(comms1),
            "n_comms_2": len(comms2)
        })

# salva tutto
output = f"/home/pasquini/My_nas/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/result/jaccard_top{TOP_K}_all_pairs.csv"
dfres = pd.DataFrame(all_results)
dfres.to_csv(output, index=False)
print(f"\n✅ Saved results to {output}")

# -------------------------
# PLOT LINEE
# -------------------------
plt.figure(figsize=(12,6))
for pair in dfres["pair"].unique():
    subset = dfres[dfres["pair"]==pair]
    plt.plot(subset["resolution"], subset["mean_jaccard_1to2"], marker="o", label=f"{pair} (1→2)")
    plt.plot(subset["resolution"], subset["mean_jaccard_2to1"], marker="x", linestyle="--", label=f"{pair} (2→1)")
plt.xlabel("Resolution parameter")
plt.ylabel("Mean Jaccard (best match, top500)")
plt.title("Jaccard overlap vs resolution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/home/pasquini/My_nas/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/result/jaccard_lineplots.png", dpi=200)
plt.close()

# -------------------------
# HEATMAP (solo mean 1→2)
# -------------------------
pivot = dfres.pivot_table(index="pair", columns="resolution", values="mean_jaccard_1to2")
plt.figure(figsize=(12,6))
sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
plt.title("Mean Jaccard (1→2) for each pair and resolution")
plt.xlabel("Resolution")
plt.ylabel("Pair")
plt.tight_layout()
plt.savefig("/home/pasquini/My_nas/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/result/jaccard_heatmap.png", dpi=200)
plt.close()
print("✅ Plots saved: jaccard_lineplots.png, jaccard_heatmap.png")

# -------------------------
# HEATMAP (media simmetrica 1↔2)
# -------------------------
dfres["mean_jaccard_sym"] = 0.5 * (dfres["mean_jaccard_1to2"] + dfres["mean_jaccard_2to1"])

pivot = dfres.pivot_table(index="pair", columns="resolution", values="mean_jaccard_sym")

plt.figure(figsize=(12,6))
sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
plt.title("Symmetric Mean Jaccard (1↔2) for each pair and resolution")
plt.xlabel("Resolution")
plt.ylabel("Pair")
plt.tight_layout()
plt.savefig("/home/pasquini/My_nas/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/result/jaccard_heatmap_symmetric.png", dpi=200)
plt.close()

print("✅ Heatmap saved: jaccard_heatmap_symmetric.png")