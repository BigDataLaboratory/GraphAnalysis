import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

prefix = "/ipazianas/pasquini/output_graph_analysis/communities_leiden/feb-aug_2022/cpm/result/images"

# Data for retweet-hashtag graph
retweet_data = """resolution,mean,median,count
0.1,0.616161,0.959148,1676619
0.2,0.621457,0.960629,1835469
0.3,0.623450,0.960964,1935863
0.4,0.625079,0.960964,1964363
0.5,0.622191,0.960964,2059527
0.6,0.625463,0.960964,2096001
0.7,0.626091,0.960964,2102992
0.8,0.626716,0.960964,2112357
0.9,0.627128,0.960964,2116631
1.0,0.627856,0.960964,2136638
"""

# Data for reply-hashtag graph
reply_data = """resolution,mean,median,count
0.1,0.620479,0.960964,1796119
0.2,0.624313,0.960964,1912625
0.3,0.624746,0.960964,1982371
0.4,0.626971,0.961101,2006539
0.5,0.625295,0.960964,2043499
0.6,0.626505,0.960964,2064851
0.7,0.627089,0.960964,2072580
0.8,0.627604,0.960964,2082257
0.9,0.627901,0.960964,2087099
1.0,0.627979,0.960964,2103517
"""

# Read the data into two separate DataFrames
df_retweet = pd.read_csv(io.StringIO(retweet_data))
df_reply = pd.read_csv(io.StringIO(reply_data))
sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(19, 12))
# Plotting data from both dataframes
plt.plot(df_retweet['resolution'], df_retweet['mean'], marker='o', linestyle='-', label='RT - Mean')
plt.plot(df_retweet['resolution'], df_retweet['median'], marker='o', linestyle='--', label='RT - Median')
plt.plot(df_reply['resolution'], df_reply['mean'], marker='x', linestyle='-', label='R - Mean')
plt.plot(df_reply['resolution'], df_reply['median'], marker='x', linestyle='--', label='R - Median')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# Add titles and labels
#plt.title('Entropy for Retweet and Reply Hashtag Graphs')
plt.xlabel('Resolution')
plt.ylabel('Entropy Value')
plt.legend()
plt.grid(True)

# Save the figure
plt.savefig(f'{prefix}/entropy_retweet_vs_reply.png')


print("Figure saved as entropy_mean_median.png")