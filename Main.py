from algorithms.Multigraph import Multigraph
from algorithms.Map import Map
from algorithms.LeidenClustering import Leiden
from pymongo import MongoClient
from pathlib import Path
from algorithms.Utils import Utils

import pandas as pd
import logging
import os
    
os.chdir(Path(__file__).parent)

logging.basicConfig(filename='./logs/logs.log',
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

if __name__ == '__main__':
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
    cursor = collection.find()#.limit(250)

    # Dataframe normalizzato
    tweets_norm = pd.json_normalize(cursor)

    multigraph_instance = Multigraph(tweets_norm)
    map_instance = Map(tweets_norm)
    leiden_instance = Leiden()

    # Creazione dei grafi
    retweet = multigraph_instance.relationship_retweet()
    hashtag = multigraph_instance.relationship_hashtag()
    cooccurrences = multigraph_instance.relationship_cooccurences()
    reply = multigraph_instance.relationship_responses()
    mention = multigraph_instance.relationship_mentions()

    # Creazione delle map
    # user_map = map_instance.user_id_hashtable()
    # Utils.persist_to_file(user_map, prop["output"]["path"]+prop["output"]["map_file_name_prefix"]+"user")
    # hashtag_map = map_instance.hashtag_hashtable()
    # Utils.persist_to_file(hashtag_map, prop["output"]["path"]+prop["output"]["map_file_name_prefix"]+"hashtag")
    # retweet_user_map = map_instance.user_id_retweet_hashtable()
    # Utils.persist_to_file(retweet_user_map, prop["output"]["path"]+prop["output"]["map_file_name_prefix"]+"user_id_retweet")
    # user_screen_name_map = map_instance.user_screen_name_hashtable()
    # Utils.persist_to_file(user_screen_name_map, prop["output"]["path"]+prop["output"]["map_file_name_prefix"]+"user_screen_name")
    # user_screen_name_user_id_map = map_instance.user_screen_name_user_id_hashtable()
    # Utils.persist_to_file(user_screen_name_user_id_map, prop["output"]["path"]+prop["output"]["map_file_name_prefix"]+"user_screen_name_user_id")


    #result = multigraph_instance.gen_multigraph([retweet, hashtag, cooccurrences, reply, mention])
    #Utils.persist_to_file(result, prop["output"]["path"]+prop["graph_file_name"]["multigraph"])
    
    g = leiden_instance.csv_to_igraph(prop["input"]["path"] + 'graph.csv')
    leiden_instance.add_leiden_to_igraph(g)
    df = leiden_instance.get_clusters_as_dataframe(g)
    cl = leiden_instance.hash_to_name(df, prop["output"]["path"] + prop["output"]["map_file_name_prefix"] 
                                      + "user.csv", prop["output"]["path"] + prop["output"]["map_file_name_prefix"]
                                        + "hashtag.csv", prop["output"]["path"] + prop["output"]["map_file_name_prefix"]
                                        + "user_screen_name_user_id.csv")
    # Utils.persist_to_file(cl[0], prop["output"]["path"] + prop["output"]["clusters_file_name_prefix"] + "user")
    # Utils.persist_to_file(cl[1], prop["output"]["path"] + prop["output"]["clusters_file_name_prefix"] + "hashtag")
