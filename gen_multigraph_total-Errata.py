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
        print(x, type(x))
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

    def gen_multigraph_total(self, tweets):

        # DataFrame e_rt (retweet)
        e_rt_src = tweets['user.id']
        e_rt_dst = tweets['retweeted_status.user.id']
        e_rt = pd.DataFrame({'src': e_rt_src.apply(self.hash), 'dst': e_rt_dst.apply(self.hash)})
        e_rt['weight'] = 1
        e_rt['relationship'] = 'retweet'

        # DataFrame e_ht (hashtag)
        e_ht_src = tweets['user.id']
        e_ht_dst = tweets['hashtagEntities'].apply(lambda x: x.split('|') if isinstance(x, str) else [])    # Applica una funzione anonima che fa lo split se il valore è una stringa 
        e_ht_dst = e_ht_dst.explode()#.apply(self.compute_hash)
        e_ht = pd.DataFrame({'src': e_ht_src, 'dst': e_ht_dst})
        e_ht = e_ht[e_ht['dst'].notnull() & (e_ht['dst'] != '')]    # Filtraggio dei valori nulli e stringhe vuote
        e_ht['weight'] = 1
        e_ht['relationship'] = 'hashtag'

        # DataFrame e_ht_ht (cooccurrences)
        ht_ht_src = tweets['user.id']
        ht_ht_dst = tweets['hashtagEntities'].apply(lambda x: self.combinations_list(x.split('|')) if isinstance(x, str) else [])   # Applica una funzione anonima che fa lo split se il valore è una stringa 
        ht_ht_dst = ht_ht_dst.explode()
        e_ht_ht = pd.DataFrame({'src': ht_ht_src, 'dst': ht_ht_dst})
        e_ht_ht = e_ht_ht[e_ht_ht['dst'].notnull() & (e_ht_ht['dst'] != '')]    # Filtraggio dei valori nulli e stringhe vuote
        e_ht_ht['weight'] = 1
        e_ht_ht['relationship'] = 'cooccurrences'

        # Concatenazione dei DataFrame
        result_df = pd.concat([e_rt, e_ht, e_ht_ht], ignore_index=True)
    
        return result_df
        

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
