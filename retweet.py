import numpy as np

if __name__ == '__main__':
    import modin.pandas as pd
    t = pd.read_csv("/home/pasquini/My_nas/graph_15mar.csv", sep=",")
    t.dropna(axis=0, subset=['retweeted_status.id'], inplace=True)
    rt = t.loc[:, ['retweeted_status.id', 'id', 'created_at', 'retweeted_status.created_at']]
    rt_grouped = rt.groupby(['retweeted_status.id']).size()
    rt_grouped.to_csv("/home/pasquini/My_nas/rt_grouped_15_mar.csv", index=False)
    rt.to_csv("/home/pasquini/My_nas/rt_total_15_mar.csv", index=False)


import mmh3
import pandas as pd
clusters = pd.read_csv("/ipazianas/pasquini/analisi_iezzi/cluster_wo_hash.csv", sep=",")
g = pd.read_csv("/ipazianas/pasquini/analisi_iezzi/graph_15mar.csv", sep=",", low_memory=False, lineterminator='\n')

def hash(x):
    """
    Compute the not signed hash of the input element
    """
    if x is not None:
        return mmh3.hash64(str(x), 0)[0]
    else:
        return 0

g["user.id_hash"] = g["user.id"].apply(hash)
g["retweeted_status.user.id_hash"] = g["retweeted_status.user.id"].apply(hash)
j = g.merge(clusters, left_on="user.id_hash", right_on="node_hash")
lst_col = "hashtagEntities"
x = j.assign(**{lst_col:j[lst_col].str.split('|')})
df = x.explode("hashtagEntities")
df["hashtagEntities"] = df["hashtagEntities"].apply(lambda x: x.lower() if isinstance(x, str) else None)
df["hashtagEntities_hash3"] = df["hashtagEntities"].apply(hash)
dff = df.merge(clusters, left_on="hashtagEntities_hash3", right_on="node_hash", how="left")
dff = dff[dff["in_reply_to_status_id"] == -1]