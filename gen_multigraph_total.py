import pandas as pd
import mmh3
from itertools import combinations
from pymongo import MongoClient

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
        e_rt['relationship'] = 'retweet'

    def relationship_hashtag(self, tweets):
        # DataFrame e_ht (hashtag)
        e_ht_src = tweets['user.id']
        e_ht_dst = tweets['hashtagEntities'].apply(lambda x: x.lower().split('|') if isinstance(x, str) else [])    # Applica una funzione anonima che fa lo split se il valore è una stringa
        e_ht = pd.DataFrame({'src': e_ht_src, 'dst': e_ht_dst})
        e_ht = e_ht.explode('dst').dropna()     # Rimozione NaN, altrimenti TypeError dato che vengono considerati come Float
        e_ht['dst'] = e_ht['dst'].apply(self.compute_hash)
        e_ht['relationship'] = 'hashtag'

    def relationship_cooccurences(self, tweets):
        # DataFrame e_ht_ht (cooccurrences)
        ht_ht_src = tweets['user.id']
        ht_ht_dst = tweets['hashtagEntities'].apply(lambda x: self.combinations_list(x.lower().split('|')) if isinstance(x, str) else [])   # Applica una funzione anonima che fa lo split se il valore è una stringa
        e_ht_ht = pd.DataFrame({'src': ht_ht_src, 'dst': ht_ht_dst})
        e_ht_ht = e_ht_ht.explode('dst').dropna()
        e_ht_ht['relationship'] = 'cooccurrences'

    def relationship_responses(self, tweets):
        pass

    def relationship_mentions(self, tweets):
        pass


#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', 50)
pd.set_option('display.max_rows', None)

# Connessione a MongoDB
mongo = MongoClient("mongodb://localhost:27017/")
db = mongo.twitter
collection = db.dati

# Estrazione dei dati da MongoDB
cursor = collection.find().limit(500)

# Dataframe normalizzato
tweets_norm = pd.json_normalize(cursor)

instance = Multigraph()

# Creazione del risultato finale
result = instance.gen_multigraph_total(tweets_norm)

# Visualizzazione del risultato
print(result)
