from algorithms.Map import Map
from algorithms.Utils import Utils
from algorithms.Multigraph import Multigraph
from pymongo import MongoClient

import pandas as pd
import logging

logging.basicConfig(filename='./Scrivania/logs.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level='INFO')

pd.set_option('display.max_columns', None)


def get_mongo_conf(configs):
    l = []
    for k, v in configs.items():
        l.append((k, v))
    return l


def get_properties():
    import json
    with open('./properties/prop.json') as f:
        properties = json.load(f)
    return properties


if __name__ == 'main':

    prop = get_properties()

    if prop["input"]["type"] == "mongo":
        mongo_config = prop["input"]["conf"]
        # Connessione a MongoDB
        mongo = MongoClient(mongo_config["connection.uri"])
        db = mongo.twitter
        collection = db.dati
    else:
        pass

    # Estrazione dei dati da MongoDB
    cursor = collection.find()  # .limit(250)

    # Dataframe normalizzato
    tweets_norm = pd.json_normalize(cursor)

    multigraph_instance = Multigraph(tweets_norm)
    map_instance = Map(tweets_norm)

    # Creazione dei grafi
    retweet = multigraph_instance.relationship_retweet()
    hashtag = multigraph_instance.relationship_hashtag()
    cooccurrences = multigraph_instance.relationship_cooccurences()
    reply = multigraph_instance.relationship_responses()
    mention = multigraph_instance.relationship_mentions()

    # Creazione delle map
    # user_map = map_instance.user_id_hashtable(tweets_norm)
    # Utils.persist_to_file(user_map, prop["output"]["path"]+prop["output"]["map_file_name_prefix"]+"user")
    # hashtag_map = map_instance.hashtag_hashtable(tweets_norm)
    # Utils.persist_to_file(hashtag_map, prop["output"]["path"]+prop["output"]["map_file_name_prefix"]+"hashtag_hash")
    # retweet_user_map = map_instance.user_id_retweet_hashtable(tweets_norm)
    # Utils.persist_to_file(retweet_user_map, prop["output"]["path"]+prop["output"]["map_file_name_prefix"]+"user_id_retweet")
    # user_screen_name_map = map_instance.user_screen_name_hashtable(tweets_norm)
    # Utils.persist_to_file(user_screen_name_map, prop["output"]["path"]+prop["output"]["map_file_name_prefix"]+"user_screen_name")

    result = multigraph_instance.gen_multigraph([retweet, hashtag, cooccurrences, reply, mention])
    Utils.persist_to_file(result, prop["output"]["path"]+prop["graph_file_name"]["multigraph"])
