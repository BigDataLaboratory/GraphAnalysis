import logging
import pandas as pd
from algorithms.Utils import Utils

class Map:
    logger = logging.getLogger('Map')

    def __init__(self, tweets):
        self.tweets = tweets

    def user_id_hashtable(self):
        logging.info('Generating user hashtable')
        start = time.time()
        id = self.tweets['user.id']
        user_hash = pd.DataFrame({'user_id': id, 'hash': id.apply(Utils.hash)})

        user_hash.dropna()

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        user_hash = user_hash.groupby(['user_id', 'hash'], as_index=False).size().rename(columns={'size': 'weight'})

        end = time.time()
        self.logger.info('User hashtable generation execution time: %5.2fs' % (end - start))

        user_hash.to_csv('./user_hashtable.csv', index=False)

        return user_hash

    def hashtag_hashtable(self):
        self.logger.info('Generating hashtag hashtable')
        start = time.time()
        hashtag = self.tweets['hashtagEntities'].apply(lambda x: x.lower().split('|') if isinstance(x, str) else [])

        hashtag_exploded = hashtag.explode(ignore_index=True).dropna(ignore_index=True)
        hashtag_hash = hashtag_exploded.apply(Utils.compute_hash)
        hashtag_hash_df = pd.DataFrame({'hashtag': hashtag_exploded, 'node_hash': hashtag_hash})

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        hashtag_hash_df = hashtag_hash_df.groupby(['hashtag', 'node_hash'], as_index=False).size().rename(
            columns={'size': 'weight'})

        end = time.time()
        self.logger.info('Hashtag hashtable generation execution time: %5.2fs' % (end - start))

        return hashtag_hash

    def user_id_retweet_hashtable(self):
        self.logger.info('Generating user_id_retweet hashtable')
        start = time.time()
        id = self.tweets['retweeted_status.user.id']
        retweet_user_hash = pd.DataFrame({'user_id_retweet': id, 'hash': id.apply(Utils.hash)})

        retweet_user_hash.dropna()

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        retweet_user_hash = retweet_user_hash.groupby(['user_id_retweet', 'hash'], as_index=False).size().rename(
            columns={'size': 'weight'})

        end = time.time()
        self.logger.info('User_id_retweet hastable generation execution time: %5.2fs' % (end - start))

        return retweet_user_hash

    def user_screen_name_hashtable(self):
        self.logger.info('Generating user_screen_name hashtable')
        start = time.time()
        name = self.tweets['user.screen_name'].apply(lambda x: x.lower() if isinstance(x, str) else [])
        user_screen_name_hash = pd.DataFrame({'user_screen_name': name, 'hash': name.apply(Utils.compute_hash)})

        user_screen_name_hash.dropna()

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        user_screen_name_hash = user_screen_name_hash.groupby(['user_screen_name', 'hash'],
                                                              as_index=False).size().rename(columns={'size': 'weight'})

        end = time.time()
        self.logger.info('User_screen_name hastable generation execution time: %5.2fs' % (end - start))

        return user_screen_name_hash
