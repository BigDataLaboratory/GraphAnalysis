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
        logging.info('Generating user hashtable')
        start = time.time()
        id = tweets['user.id']
        user_hash = pd.DataFrame({'user_id': id, 'hash': id.apply(self.hash)})

        user_hash.dropna()

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        user_hash = user_hash.groupby(['user_id', 'hash'], as_index=False).size().rename(columns={'size': 'weight'})

        end = time.time()
        logging.info('User hashtable generation execution time: %5.2fs' % (end - start))

        user_hash.to_csv('./user_hashtable.csv', index=False)

        return user_hash

    def hashtag_hashtable(self, tweets):
        logging.info('Generating hashtag hashtable')
        start = time.time()
        hashtag = tweets['hashtagEntities'].apply(lambda x: x.lower().split('|') if isinstance(x, str) else [])

        hashtag_exploded = hashtag.explode(ignore_index=True).dropna(ignore_index=True)
        hashtag_hash = hashtag_exploded.apply(self.compute_hash)
        hashtag_hash_df = pd.DataFrame({'hashtag': hashtag_exploded, 'node_hash': hashtag_hash})

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        hashtag_hash_df = hashtag_hash_df.groupby(['hashtag', 'node_hash'], as_index=False).size().rename(
            columns={'size': 'weight'})

        end = time.time()
        logging.info('Hashtag hashtable generation execution time: %5.2fs' % (end - start))

        hashtag_hash_df.to_csv('./resources/hashtag_hashtable.csv')

        return hashtag_hash

    def user_id_retweet_hashtable(self, tweets):
        logging.info('Generating user_id_retweet hashtable')
        start = time.time()
        id = tweets['retweeted_status.user.id']
        retweet_user_hash = pd.DataFrame({'user_id_retweet': id, 'hash': id.apply(self.hash)})

        retweet_user_hash.dropna()

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        retweet_user_hash = retweet_user_hash.groupby(['user_id_retweet', 'hash'], as_index=False).size().rename(
            columns={'size': 'weight'})

        end = time.time()
        logging.info('User_id_retweet hastable generation execution time: %5.2fs' % (end - start))

        retweet_user_hash.to_csv('./user_id_retweet_hashtable.csv')

        return retweet_user_hash

    def user_screen_name_hashtable(self, tweets):
        logging.info('Generating user_screen_name hashtable')
        start = time.time()
        name = tweets['user.screen_name'].apply(lambda x: x.lower() if isinstance(x, str) else [])
        user_screen_name_hash = pd.DataFrame({'user_screen_name': name, 'hash': name.apply(self.compute_hash)})

        user_screen_name_hash.dropna()

        # Restituisce un DataFrame: as_index=False imposta il raggruppamento in SQL-style
        # ed in combinazione con size() conta il numero di righe per ogni gruppo aggiungendo
        # una colonna 'size' e restituendo effettivamente un dataframe.
        user_screen_name_hash = user_screen_name_hash.groupby(['user_screen_name', 'hash'],
                                                              as_index=False).size().rename(columns={'size': 'weight'})

        end = time.time()
        logging.info('User_screen_name hastable generation execution time: %5.2fs' % (end - start))

        user_screen_name_hash.to_csv('./user_screen_name_hashtable.csv')

        return user_screen_name_hash
