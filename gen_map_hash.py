import pandas as pd
import mmh3
from parallel_pandas import ParallelPandas

class Map:

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
    
    def user_id_hashtable(self, tweets):
        id = tweets['user.id']
        user_hash = pd.DataFrame({'user_id': id, 'hash': id.apply(self.hash)})

        user_hash.dropna()

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style 
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        user_hash = user_hash.groupby(['user_id', 'hash'], as_index=False).size().rename(columns={'size':'weight'})

        return user_hash
    
    def hashtag_hashtable(self, tweets):
        hashtag = tweets['hashtagEntities'].apply(lambda x: x.lower().split('|') if isinstance(x, str) else [])
        hashtag_hash = pd.DataFrame({'hashtag': hashtag, 'hash': hashtag})

        hashtag_hash = hashtag_hash.explode(['hashtag', 'hash']).dropna()
        hashtag_hash['hash'] = hashtag_hash['hash'].apply(self.compute_hash)

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style 
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        hashtag_hash = hashtag_hash.groupby(['hashtag', 'hash'], as_index=False).size().rename(columns={'size':'weight'})

        return hashtag_hash

    def user_id_retweet_hashtable(self, tweets):
        id = tweets['retweeted_status.user.id']
        retweet_user_hash = pd.DataFrame({'user_id_retweet': id, 'hash': id.apply(self.hash)})

        retweet_user_hash.dropna()

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style 
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        retweet_user_hash = retweet_user_hash.groupby(['user_id_retweet', 'hash'], as_index=False).size().rename(columns={'size':'weight'})

        return retweet_user_hash
    
    def user_screen_name_hashtable(self, tweets):
        name = tweets['user.screen_name'].apply(lambda x: x.lower() if isinstance(x, str) else [])
        user_screen_name_hash = pd.DataFrame({'user_screen_name': name, 'hash': name.apply(self.compute_hash)})

        user_screen_name_hash.dropna()

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style 
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        user_screen_name_hash = user_screen_name_hash.groupby(['user_screen_name', 'hash'], as_index=False).size().rename(columns={'size':'weight'})

        return user_screen_name_hash