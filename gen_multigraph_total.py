#import pandas as pd
import modin.pandas as pd
import ray
import mmh3
import gen_map_hash
from itertools import combinations
from pymongo import MongoClient
#from parallel_pandas import ParallelPandas
import time

class Multigraph:

    def hash(self, x):
        """
        Compute the not signed hash of the input element
        """

        return mmh3.hash64(str(x), 0)[0]

    def compute_hash(self, x):
        """
        Compute the not signed hash of the input element
        """
        
        if x is not None:
            return mmh3.hash64(x, 0)[0]

    def combinations_list(self, x):
        """
        Create all the possible combinations within hashtag in the same tweet, using their hashes
        """

        if x is not None:
            hashed = []
            for ht in x:
                hashed.append(mmh3.hash64(ht, 0)[0])
            hashed.sort()
            return list(combinations(hashed, 2))

    def gen_multigraph(self, df_list):
        """
        Give a list of dataframes that represent different edgelists, build a single dataframe concat each dataframe
        in the input list
        :param df_list: list of dataframes that represent different edgelists
        :return: single dataframe concat each dataframe in the input list
        """
        # Concatenazione dei DataFrame
        result_df = pd.concat(df_list, ignore_index=True)
        return result_df

    def relationship_retweet(self, tweets):
        # DataFrame e_rt (retweet)
        e_rt_src = tweets['user.id']
        e_rt_dst = tweets['retweeted_status.user.id']
        e_rt = pd.DataFrame({'src': e_rt_src.apply(self.hash), 'dst': e_rt_dst.apply(self.hash)})

        e_rt.dropna()
        
        e_rt['weight'] = 1
        #e_rt = e_rt.groupby(['src', 'dst'])['weight'].apply(lambda x: x.count()).reset_index()
            # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style 
            # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
            # una colonna 'size' e restituendo effettivamente un dataframe.
        e_rt = e_rt.groupby(['src', 'dst'], as_index=False).size().rename(columns={'size':'weight'})
        e_rt['relationship'] = 'retweet'

        return e_rt

    def relationship_hashtag(self, tweets):
        # DataFrame e_ht (hashtag)
        e_ht_src = tweets['user.id']
        e_ht_dst = tweets['hashtagEntities'].apply(lambda x: x.lower().split('|') if isinstance(x, str) else [])
        e_ht = pd.DataFrame({'src': e_ht_src, 'dst': e_ht_dst})

        # Rimozione NaN, altrimenti TypeError dato che vengono considerati come Float
        e_ht = e_ht.explode('dst').dropna()

        e_ht['dst'] = e_ht['dst'].apply(self.compute_hash)
        
        e_ht['weight'] = 1
        #e_ht = e_ht.groupby(['src', 'dst'])['weight'].apply(lambda x: x.count()).reset_index()
            # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style 
            # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
            # una colonna 'size' e restituendo effettivamente un dataframe.
        e_ht = e_ht.groupby(['src', 'dst'], as_index=False).size().rename(columns={'size':'weight'})
        
        e_ht['relationship'] = 'hashtag'
        
        return e_ht

    def relationship_cooccurences(self, tweets):
        # DataFrame e_ht_ht (cooccurrences)
        ht_ht_src = tweets['user.id']
        ht_ht_dst = tweets['hashtagEntities'].apply(lambda x: self.combinations_list(x.lower().split('|')) if isinstance(x, str) else [])
        e_ht_ht = pd.DataFrame({'src': ht_ht_src, 'dst': ht_ht_dst})

        # Rimozione NaN, altrimenti TypeError dato che vengono considerati come Float
        e_ht_ht = e_ht_ht.explode('dst').dropna()

        e_ht_ht['weight'] = 1
        #e_ht_ht = e_ht_ht.groupby(['src', 'dst'])['weight'].apply(lambda x: x.count()).reset_index()

            # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style 
            # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
            # una colonna 'size' e restituendo effettivamente un dataframe.
        e_ht_ht = e_ht_ht.groupby(['src', 'dst'], as_index=False).size().rename(columns={'size':'weight'})

        e_ht_ht['relationship'] = 'cooccurrences'

        return e_ht_ht

    def relationship_responses(self, tweets):
        """
            Dataframe that lists the responses number among users.
            Users on 'src' column replie to users on 'dst' column.

            :return: single dataframe
        """
        # DataFrame e_rp (reply)
        e_rp_src = tweets['user.id']
        e_rp_dst = tweets['in_reply_to_user_id']

        e_rp = pd.DataFrame({'src': e_rp_src.apply(self.hash), 'dst': e_rp_dst})

        # Filtro i valori -1, ovvero quegli utenti che non hanno risposto a nessuno. In seguito eseguo la funzione hash
        e_rp['dst'] = e_rp[e_rp['dst'] != -1]['dst'].apply(self.hash)

        e_rp['weight'] = 1
        #e_rp = e_rp.groupby(['src', 'dst'])['weight'].apply(lambda x: x.count()).reset_index()

            # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style 
            # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
            # una colonna 'size' e restituendo effettivamente un dataframe.
        e_rp = e_rp.groupby(['src', 'dst'], as_index=False).size().rename(columns={'size':'weight'})

        e_rp['relationship'] = 'reply'

        return e_rp

    def relationship_mentions(self, tweets):
        """
            Dataframe that lists the mentions number among users.
            Users on 'src' column mention users on 'dst' column.

            :return: single dataframe
        """
        # DataFrame e_mt (mentions)
        e_mt_src = tweets['user.id']
        e_mt_dst = tweets['userMentionEntities'].apply(lambda x: x.lower().split('|') if isinstance(x, str) else [])
        e_mt = pd.DataFrame({'src': e_mt_src, 'dst': e_mt_dst})

        # Rimozione NaN, altrimenti TypeError dato che vengono considerati come Float
        e_mt = e_mt.explode('dst').dropna()

        e_mt['dst'] = e_mt['dst'].apply(self.compute_hash)

        e_mt['weight'] = 1
        #e_mt = e_mt.groupby(['src', 'dst'])['weight'].apply(lambda x: x.count()).reset_index()
        
            # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style 
            # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
            # una colonna 'size' e restituendo effettivamente un dataframe.
        e_mt = e_mt.groupby(['src', 'dst'], as_index=False).size().rename(columns={'size':'weight'})
        
        e_mt['relationship'] = 'mention'
        
        return e_mt


#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', 50)
#pd.set_option('display.max_rows', None)

#ParallelPandas.initialize(n_cpu=4, split_factor=4, disable_pr_bar=False)

#ray.init()

# Connessione a MongoDB
mongo = MongoClient("mongodb://localhost:27017/")
db = mongo.twitter
collection = db.dati

# Estrazione dei dati da MongoDB
cursor = collection.find()#.limit(250)

# Dataframe normalizzato
tweets_norm = pd.json_normalize(cursor)

multigraph_instance = Multigraph()
map_instance = gen_map_hash.Map()

# Creazione dei grafi
start = time.time()

retweet = multigraph_instance.relationship_retweet(tweets_norm)

hashtag = multigraph_instance.relationship_hashtag(tweets_norm)
cooccurrences = multigraph_instance.relationship_cooccurences(tweets_norm)
reply = multigraph_instance.relationship_responses(tweets_norm)
mention = multigraph_instance.relationship_mentions(tweets_norm)

# Creazione delle map
#user_map = map_instance.user_id_hashtable(tweets_norm)
#hashtag_map = map_instance.hashtag_hashtable(tweets_norm)
#retweet_user_map = map_instance.user_id_retweet_hashtable(tweets_norm)
#user_screen_name_map = map_instance.user_screen_name_hashtable(tweets_norm)

result = multigraph_instance.gen_multigraph([retweet, hashtag, cooccurrences, reply, mention])
end = time.time()
result = end - start
print(f'Time took: {result} s.')
# Visualizzazione del risultato
#print(retweet)
