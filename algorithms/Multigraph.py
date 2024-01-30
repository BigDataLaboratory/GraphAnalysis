import pandas as pd
import logging
import time

from algorithms.Utils import Utils

class Multigraph:

    logger = logging.getLogger('Multigraph')

    def __init__(self, tweets=None):
        self.tweets = tweets

    def gen_multigraph(self, df_list):
        """
        Give a list of dataframes that represent different edgelists, build a single dataframe concat each dataframe
        in the input list
        :param df_list: list of dataframes that represent different edgelists
        :return: single dataframe concat each dataframe in the input list
        """
        self.logger.info('Concatenating graphs')
        start = time.time()

        # Concatenazione dei DataFrame
        result_df = pd.concat(df_list, ignore_index=True)

        end = time.time()
        self.logger.info('Concatenation execution time: %5.2fs' % (end - start))

        return result_df

    def relationship_retweet(self):

        self.logger.info('Generating retweet graph')
        start = time.time()
        # DataFrame e_rt (retweet)
        e_rt_src = self.tweets['user.id']
        e_rt_dst = self.tweets['retweeted_status.user.id']
        e_rt = pd.DataFrame({'src': e_rt_src.apply(Utils.hash), 'dst': e_rt_dst.apply(Utils.hash)})

        e_rt.dropna(ignore_index=True)

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        e_rt = e_rt.groupby(['src', 'dst'], as_index=False).size().rename(columns={'size': 'weight'})

        e_rt['relationship'] = 'retweet'

        end = time.time()
        self.logger.info('Retweet graph generation execution time: %5.2fs' % (end - start))

        return e_rt

    def relationship_hashtag(self):

        self.logger.info('Generating hashtag graph')
        start = time.time()
        # DataFrame e_ht (hashtag)
        e_ht_src = self.tweets['user.id']
        e_ht_dst = self.tweets['hashtagEntities'].apply(lambda x: x.lower().split('|') if isinstance(x, str) else [])
        e_ht = pd.DataFrame({'src': e_ht_src.apply(Utils.hash), 'dst': e_ht_dst})

        # Rimozione NaN, altrimenti TypeError dato che vengono considerati come Float
        e_ht = e_ht.explode('dst').dropna(ignore_index=True)

        e_ht['dst'] = e_ht['dst'].apply(Utils.compute_hash)

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        e_ht = e_ht.groupby(['src', 'dst'], as_index=False).size().rename(columns={'size': 'weight'})

        e_ht['relationship'] = 'hashtag'

        end = time.time()
        self.logger.info('Hashtag graph generation execution time: %5.2fs' % (end - start))

        return e_ht

    def relationship_cooccurences(self):

        self.logger.info('Generating cooccurences graph')
        start = time.time()
        # DataFrame e_ht_ht (cooccurrences)
        ht_ht_src = self.tweets['user.id']
        ht_ht_dst = self.tweets['hashtagEntities'].apply(lambda x: Utils.combinations_list(x.lower().split('|')) if isinstance(x, str) else [])
        e_ht_ht = pd.DataFrame({'src': ht_ht_src.apply(Utils.hash), 'dst': ht_ht_dst})

        # Rimozione NaN, altrimenti TypeError dato che vengono considerati come Float
        e_ht_ht = e_ht_ht.explode('dst').dropna(ignore_index=True)
        e_ht_ht = e_ht_ht.explode('dst')

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        e_ht_ht = e_ht_ht.groupby(['src', 'dst'], as_index=False).size().rename(columns={'size': 'weight'})

        e_ht_ht['relationship'] = 'cooccurrences'

        end = time.time()
        self.logger.info('Cooccurences graph generation execution time: %5.2fs' % (end - start))

        return e_ht_ht

    def relationship_responses(self):
        """
            Dataframe that lists the responses number among users.
            Users on 'src' column replie to users on 'dst' column.

            :return: single dataframe
        """
        self.logger.info('Generating reply graph')
        start = time.time()
        # DataFrame e_rp (reply)
        e_rp_src = self.tweets['user.id']
        e_rp_dst = self.tweets['in_reply_to_user_id']

        e_rp = pd.DataFrame({'src': e_rp_src.apply(Utils.hash), 'dst': e_rp_dst})

        # Filtro i valori -1, ovvero quegli utenti che non hanno risposto a nessuno. In seguito eseguo la funzione hash
        e_rp['dst'] = e_rp[e_rp['dst'] != -1]['dst'].apply(Utils.hash)

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        e_rp = e_rp.groupby(['src', 'dst'], as_index=False).size().rename(columns={'size': 'weight'})

        e_rp['relationship'] = 'reply'

        end = time.time()
        self.logger.info('Reply graph generation execution time: %5.2fs' % (end - start))

        return e_rp

    def relationship_mentions(self):
        """
            Dataframe that lists the mentions number among users.
            Users on 'src' column mention users on 'dst' column.

            :return: single dataframe
        """
        self.logger.info('Generating mentions graph')
        start = time.time()
        # DataFrame e_mt (mentions)
        e_mt_src = self.tweets['user.id']
        e_mt_dst = self.tweets['userMentionEntities'].apply(lambda x: x.lower().split('|') if isinstance(x, str) else [])
        e_mt = pd.DataFrame({'src': e_mt_src.apply(Utils.hash), 'dst': e_mt_dst})

        # Rimozione NaN, altrimenti TypeError dato che vengono considerati come Float
        e_mt = e_mt.explode('dst').dropna(ignore_index=True)

        e_mt['dst'] = e_mt['dst'].apply(Utils.compute_hash)

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        e_mt = e_mt.groupby(['src', 'dst'], as_index=False).size().rename(columns={'size': 'weight'})

        e_mt['relationship'] = 'mention'

        end = time.time()
        self.logger.info('Mention graph generation execution time: %5.2fs' % (end - start))

        return e_mt
