from datetime import datetime
from enum import Enum

import networkx as nx
import numpy as np
import pandas as pd

c = pd.read_csv("/Users/danielepasquini/Downloads/fe7d6beaed3511efa86708f1eaf4fe18/communities_rp_0.5.csv", sep=",", header=0, index_col=0)
hashtag = pd.read_csv("/Users/danielepasquini/Downloads/d4841f8aed3211efa71908f1eaf4fe18/hashtag", header=None, sep=",")
hashtag.rename(columns={0:'original', 1:"name", 2:"type"}, inplace=True)
grouped = c.groupby(by=["0.5"]).size()
selected = c.loc[c['0.5'].isin([0])].reset_index(drop=True)

hashtag['name'] = hashtag["name"].astype(str)

ht_resolved_0 = selected[(selected['type']=='h') & (selected['0.5'] == 0)].merge(hashtag, on="name", how="inner").drop_duplicates()
ht_resolved_1 = selected[(selected['type']=='h') & (selected['0.5'] == 1)].merge(hashtag, on="name", how="inner").drop_duplicates()
ht_resolved_2 = selected[(selected['type']=='h') & (selected['0.5'] == 2)].merge(hashtag, on="name", how="inner").drop_duplicates()
ht_resolved_3 = selected[(selected['type']=='h') & (selected['0.5'] == 3)].merge(hashtag, on="name", how="inner").drop_duplicates()
ht_resolved_4 = selected[(selected['type']=='h') & (selected['0.5'] == 4)].merge(hashtag, on="name", how="inner").drop_duplicates()
ht_resolved_5 = selected[(selected['type']=='h') & (selected['0.5'] == 5)].merge(hashtag, on="name", how="inner").drop_duplicates()
ht_resolved_6 = selected[(selected['type']=='h') & (selected['0.5'] == 6)].merge(hashtag, on="name", how="inner").drop_duplicates()
ht_resolved_7 = selected[(selected['type']=='h') & (selected['0.5'] == 7)].merge(hashtag, on="name", how="inner").drop_duplicates()
ht_resolved_8 = selected[(selected['type']=='h') & (selected['0.5'] == 8)].merge(hashtag, on="name", how="inner").drop_duplicates()
ht_resolved_9 = selected[(selected['type']=='h') & (selected['0.5'] == 9)].merge(hashtag, on="name", how="inner").drop_duplicates()
joined = selected.merge(hashtag, on="name", how="inner")

joined[(joined['0.5'] == 0) & (joined['type_x'] == 'h')]

node_community_dict = selected.set_index('name')['0.5'].to_dict()
selected[['name', '0.5']].to_csv("/Users/danielepasquini/Downloads/nodes_id.csv", header=True, index=False, sep=",")
community_mapping = {np.float64(k): v for k, v in node_community_dict.items()}

selected_list = selected['name'].values.tolist()
selected_list = np.float64(selected_list)

user_hashtag = pd.read_csv("/Users/danielepasquini/Downloads/d4841f8aed3211efa71908f1eaf4fe18/graph/user_hashtag.csv", sep=",", header=None)
retweet = pd.read_csv("/Users/danielepasquini/Downloads/d4841f8aed3211efa71908f1eaf4fe18/graph/retweet", sep=",", header=None)
response = pd.read_csv("/Users/danielepasquini/Downloads/d4841f8aed3211efa71908f1eaf4fe18/graph/response", sep=",", header=None)
mention = pd.read_csv("/Users/danielepasquini/Downloads/d4841f8aed3211efa71908f1eaf4fe18/graph/mention", sep=",", header=None)
hashtag_cooccurrences = pd.read_csv("/Users/danielepasquini/Downloads/d4841f8aed3211efa71908f1eaf4fe18/graph/hashtag_cooccurrences", sep=",", header=None)

g = pd.concat([user_hashtag, retweet, response, mention, hashtag_cooccurrences]).reset_index(drop=True)
g.rename(columns={0:'type', 1:"src", 2:"dst", 3:'weight'}, inplace=True)

# Assign community columns (default to None if not found)
g['src_community'] = g['src'].map(community_mapping)
g['dst_community'] = g['dst'].map(community_mapping)


filtered_g = g[g['src'].isin(selected_list) & g['dst'].isin(selected_list)].reset_index(drop=True).copy()

filtered_g_response = filtered_g.loc[filtered_g['type'] == 4]

filtered_g[['src', 'dst', 'weight']].to_csv("/Users/danielepasquini/Downloads/graph_filtered.csv", header=True, sep=",", index=False)
filtered_g.loc[filtered_g['type'] == 4][['src', 'dst', 'weight']].to_csv("/Users/danielepasquini/Downloads/graph_filtered.csv", header=True, sep=",", index=False)

import matplotlib.pyplot  as plt

G = nx.DiGraph()

# Add edges with attributes
for _, row in filtered_g_response.iterrows():
    G.add_edge(row["src"], row["dst"], weight=row["weight"],
               community_src=row["src_community"],
               community_dst=row["dst_community"])

in_degree_weights = dict(G.in_degree(weight='weight'))
out_degree_weights = dict(G.out_degree(weight='weight'))
pagerank_scores = nx.pagerank(G, weight='weight')
density = nx.density(G)

G = nx.DiGraph()

# Add edges with attributes
for _, row in filtered_g.iterrows():
    G.add_edge(row["src"], row["dst"], weight=row["weight"],
               community_src=row["src_community"],
               community_dst=row["dst_community"])

# Assign positions based on communities
communities = set(filtered_g["src_community"]).union(set(filtered_g["dst_community"]))
community_pos = {int(c): (i * 2, 0) for i, c in enumerate(communities)}
pos = {node: community_pos[filtered_g[filtered_g['src'] == node]['src_community'].values[0]] for node in G.nodes}

# Draw the graph
edges = G.edges(data=True)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=12)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in edges})

plt.show()


# Draw the graph
pos = nx.spring_layout(G)
edges = G.edges(data=True)
nx.draw_networkx(G, pos, with_labels=False, node_color='lightblue', edge_color='gray', node_size=20, font_size=12)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['weight'] for u, v, d in edges})

plt.show()


from Utils.Utils import Utils

vivi = pd.read_json("/Users/danielepasquini/Downloads/vivi.json")
vivi = vivi[['id', 'in_reply_to_status_id', 'in_reply_to_user_id', 'retweeted_status', 'user', 'hashtagEntities', 'created_at', 'userMentionEntities']]
vivi['created_at'] = vivi['created_at'].apply(lambda x: x.get('$date') if isinstance(x, dict) else None)
vivi['retweeted_status.created_at'] = vivi['retweeted_status'].apply(lambda x: x.get('created_at').get('$date') if isinstance(x, dict) else None)
vivi['retweeted_status.id'] = vivi['retweeted_status'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
vivi['retweeted_status.user.id'] = vivi['retweeted_status'].apply(lambda x: x.get('user', None).get('id', None) if isinstance(x, dict) else None)
vivi['user.id'] = vivi['user'].apply(lambda x: x.get('id') if isinstance(x, dict) else None)
vivi['retweeted_status.user.screen_name'] = vivi['retweeted_status'].apply(lambda x: x.get('user', None).get('screen_name', None) if isinstance(x, dict) else None)
vivi['user.screen_name'] = vivi['user'].apply(lambda x: x.get('screen_name', None) if isinstance(x, dict) else None)
vivi.drop(columns=['user', 'retweeted_status'], inplace=True)
data = vivi.to_dict(orient='records')

o = []
m = set()
i = 0
l = len(data)
for d in data:
    n_user_id = Utils.hash(d['user.id'])
    m.add((d['user.id'], n_user_id, 0))
    print("doing {} / {}".format(i, l))

    weight = 1

    if d.get('retweeted_status.id', None) is not None:
        relationship_u_rt = 0
        n_rt_user_id = Utils.hash(d['retweeted_status.user.id'])
        e_rt = n_user_id, n_rt_user_id, weight, relationship_u_rt
        o.append(e_rt)
        m.add((d['retweeted_status.user.id'], n_rt_user_id, 1))

    if d.get('retweeted_status.id', None) is not None:
        relationship_t_rt = 1
        n_tweet_id = Utils.hash(d['id'])
        n_rt_tweet_id = Utils.hash(d['retweeted_status.id'])
        a_created_at_tweet = datetime.strptime(d['created_at'], "%Y-%m-%dT%H:%M:%SZ").timestamp()
        a_created_at_rt = datetime.strptime(d['retweeted_status.created_at'], "%Y-%m-%dT%H:%M:%SZ").timestamp() if d['retweeted_status.created_at'] is not None else None
        e_tweet_retweet = (
            n_tweet_id, n_rt_tweet_id, weight, (a_created_at_tweet, a_created_at_rt), relationship_t_rt)
        o.append(e_tweet_retweet)
        m.add((d['id'], n_tweet_id, 2))
        m.add((d['retweeted_status.id'], n_rt_tweet_id, 2))

    if d.get('hashtagEntities', None) is not None:
        relationship = 2
        n_ht = d['hashtagEntities'].lower().split('|') if isinstance(d['hashtagEntities'], str) else []
        ht = [(n_user_id, Utils.compute_hash(x), weight, relationship) for x in n_ht]
        o.extend(ht)
        for x in n_ht:
            m.add((x, Utils.compute_hash(x), 3))

    if d.get('hashtagEntities', None) is not None:
        relationship = 3
        ht_combinations = Utils.combinations_list(d['hashtagEntities'].lower().split('|')) if isinstance(
            d['hashtagEntities'], str) else []
        e_hts_natural = [(x[0], x[1], weight, relationship) for x in ht_combinations]
        e_hts_inverse = [(x[1], x[0], weight, relationship) for x in ht_combinations]
        o.extend(e_hts_natural)
        o.extend(e_hts_inverse)

    if d.get('in_reply_to_user_id', -1) != -1:
        relationship = 4
        n_reply_user_id = Utils.hash(d['in_reply_to_user_id'])
        e_reply = n_user_id, n_reply_user_id, weight, relationship
        o.append(e_reply)

    if d.get('userMentionEntities', None) is not None:
        relationship = 5
        n_mentions = d['userMentionEntities'].lower().split('|') if isinstance(d['userMentionEntities'],
                                                                               str) else []
        e_mentions = [(n_user_id, Utils.compute_hash(x), weight, relationship) for x in n_mentions]
        o.extend(e_mentions)
    i+=1

i = 0
intermediate_result = {}
intermediate_map = set()
for item in o:
    print("doing {} / {}".format(i, l))
    key = (item[0], item[1], item[-1])
    if item[-1] != 1:
        if key not in intermediate_result:
            intermediate_result[key] = 0
        intermediate_result[key] += item[2]  # Sum the third element
    else:
        intermediate_result[key] = item[2:-1]
    i+=1

i = 0
l = len(m)
for item in m:
    print("doing {} / {}".format(i, l))
    intermediate_map.add(item)
    i+=1


class GraphType(Enum):
    retweet = 0
    tweet_retweet = 1
    user_hashtag = 2
    hashtag_cooccurrences = 3
    response = 4
    mention = 5


class MapType(Enum):
    user_id = 0
    user_retweeted_id = 1
    tweet_id = 2
    hashtag = 3

result_graph = {GraphType(0).name: [], GraphType(1).name: [], GraphType(2).name: [], GraphType(3).name: [],
                GraphType(4).name: [], GraphType(5).name: []}
result_map = {MapType(0).name: [], MapType(1).name: [], MapType(2).name: [], MapType(3).name: []}

i = 0
l = len(intermediate_result)
for k, v in intermediate_result.items():
    print("{}/{}".format(i, l))
    result_graph[GraphType(k[2]).name].append((k[2], k[0], k[1], v)) if k[2] != 1 else result_graph[
        GraphType(k[2]).name].append((k[2], k[0], k[1], v[0], v[1]))
    i+=1

for e in intermediate_map:
    result_map[MapType(e[2]).name].append(e)

# if __name__ == '__main__':
#     import modin.pandas as pd
#     t = pd.read_csv("/home/pasquini/My_nas/graph_15mar.csv", sep=",")
#     t.dropna(axis=0, subset=['retweeted_status.id'], inplace=True)
#     rt = t.loc[:, ['retweeted_status.id', 'id', 'created_at', 'retweeted_status.created_at']]
#     rt_grouped = rt.groupby(['retweeted_status.id']).size()
#     rt_grouped.to_csv("/home/pasquini/My_nas/rt_grouped_15_mar.csv", index=False)
#     rt.to_csv("/home/pasquini/My_nas/rt_total_15_mar.csv", index=False)
#
#
# import mmh3
# import pandas as pd
# clusters = pd.read_csv("/ipazianas/pasquini/analisi_iezzi/cluster_wo_hash.csv", sep=",")
# g = pd.read_csv("/ipazianas/pasquini/analisi_iezzi/graph_15mar.csv", sep=",", low_memory=False, lineterminator='\n')
#
# def hash(x):
#     """
#     Compute the not signed hash of the input element
#     """
#     if x is not None:
#         return mmh3.hash64(str(x), 0)[0]
#     else:
#         return 0
#
# g["user.id_hash"] = g["user.id"].apply(hash)
# g["retweeted_status.user.id_hash"] = g["retweeted_status.user.id"].apply(hash)
# j = g.merge(clusters, left_on="user.id_hash", right_on="node_hash")
# lst_col = "hashtagEntities"
# x = j.assign(**{lst_col:j[lst_col].str.split('|')})
# df = x.explode("hashtagEntities")
# df["hashtagEntities"] = df["hashtagEntities"].apply(lambda x: x.lower() if isinstance(x, str) else None)
# df["hashtagEntities_hash3"] = df["hashtagEntities"].apply(hash)
# dff = df.merge(clusters, left_on="hashtagEntities_hash3", right_on="node_hash", how="left")
# dff = dff[dff["in_reply_to_status_id"] == -1]