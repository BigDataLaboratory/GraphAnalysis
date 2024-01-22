import pandas as pd
import mmh3
import algorithms.Map
import logging
from itertools import combinations
from pymongo import MongoClient
import time

from algorithms.Multigraph import Multigraph

logging.basicConfig(filename='./Scrivania/logs.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level='INFO')

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_colwidth', 50)


if __name__ == 'main':
    # Connessione a MongoDB
    mongo = MongoClient("mongodb://localhost:27017/")
    db = mongo.twitter
    collection = db.dati

    # Estrazione dei dati da MongoDB
    cursor = collection.find()#.limit(250)

    # Dataframe normalizzato
    tweets_norm = pd.json_normalize(cursor)

    multigraph_instance = Multigraph()
    map_instance = algorithms.Map()

    # Creazione dei grafi
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
    #print(f'Time took: {result} s.')
    # Visualizzazione del risultato
    #print(retweet)
