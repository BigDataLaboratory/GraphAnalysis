import argparse
from collections import namedtuple
from datetime import datetime

from algorithms.Multigraph import Multigraph
from algorithms.Map import Map
from pathlib import Path
from Utils.Utils import Utils
from algorithms.Leiden import Leiden
from algorithms.RawData import RawData

import pandas as pd
import logging
import os

from algorithms.TopicGenerator import TopicGenerator

os.chdir(Path(__file__).parent)


class GraphAnalysis:
    logger = logging.getLogger('GraphAnalysis')

    def __init__(self, parameters: namedtuple):
        self.parameters = parameters

    def run(self):
        if self.parameters.do_graph_generation:
            raw_data = RawData(self.parameters.source_uri,
                               self.parameters.source_username,
                               self.parameters.source_password,
                               self.parameters.source_auth_source,
                               self.parameters.source_auth_mechanism,
                               self.parameters.source_collection,
                               self.parameters.source_chunk_start_date,
                               self.parameters.source_chunk_end_date,
                               input_type=self.parameters.source_input_type)

            raw_data.connect(self.parameters.source_db_name)

            """
            use it when mongo is available again
            """
            w = {'$or': [{'hashtagEntities': {'$exists': True}}, {'retweeted_status': {'$exists': True}},
                         {'in_reply_to_status_id': {'$exists': True}}]}
            s = {'_id': 0,
                 'id': 1,
                 'in_reply_to_status_id': 1,
                 'in_reply_to_user_id': 1,
                 'retweeted_status.created_at': 1,
                 'retweeted_status.id': 1,
                 'retweeted_status.user.id': 1,
                 'retweeted_status.user.screen_name': 1,
                 'user.id': 1,
                 'user.screen_name': 1,
                 'hashtagEntities': 1,
                 'created_at': 1,
                 'userMentionEntities': 1}

            """
            w = 'hashtagEntities.notnull() | `retweeted_status.id`.notnull() | in_reply_to_status_id.notnull()'
            s = ['id',
                 'in_reply_to_status_id',
                 'in_reply_to_user_id',
                 'retweeted_status.created_at.$date',
                 'retweeted_status.id',
                 'retweeted_status.user.id',
                 'retweeted_status.user.screen_name',
                 'user.id',
                 'user.screen_name',
                 'hashtagEntities',
                 'created_at.$date',
                 'userMentionEntities'
                 ]
            """
            tweets = raw_data.query_data_in_chunks(w, s)

            multigraph_instance = Multigraph(tweets)
            map_instance = Map(tweets)

            # Graph generation
            graph_list = []
            if self.parameters.do_retweet_graph:
                retweet = multigraph_instance.relationship_retweet()
                if retweet is not None:
                    Utils.persist_to_file(retweet, "{}/{}".format(self.parameters.output_graph_path,
                                                                  self.parameters.output_retweet_graph_path))
                    graph_list.append(retweet)
            if self.parameters.do_hashtag_graph:
                hashtag = multigraph_instance.relationship_hashtag()
                if hashtag is not None:
                    Utils.persist_to_file(hashtag, "{}/{}".format(self.parameters.output_graph_path,
                                                                  self.parameters.output_hashtag_graph_path))
                    graph_list.append(hashtag)
            if self.parameters.do_hashtag_cooccurrences_graph:
                cooccurrences = multigraph_instance.relationship_cooccurences()
                if cooccurrences is not None:
                    Utils.persist_to_file(cooccurrences, "{}/{}".format(self.parameters.output_graph_path,
                                                                        self.parameters.output_hashtag_cooccurrences_graph_path))
                    graph_list.append(cooccurrences)
            if self.parameters.do_response_graph:
                reply = multigraph_instance.relationship_responses()
                if reply is not None:
                    Utils.persist_to_file(reply, "{}/{}".format(self.parameters.output_graph_path,
                                                                self.parameters.output_response_graph_path))
                    graph_list.append(reply)
            if self.parameters.do_mention_graph:
                mention = multigraph_instance.relationship_mentions()
                if mention is not None:
                    Utils.persist_to_file(mention, "{}/{}".format(self.parameters.output_graph_path,
                                                                  self.parameters.output_mention_graph_path))
                    graph_list.append(mention)

            # Map generation
            user_map = map_instance.user_id_hashtable()
            Utils.persist_to_file(user_map, "{}/{}_user_id".format(self.parameters.output_graph_path,
                                                                   self.parameters.output_map_prefix))
            hashtag_map = map_instance.hashtag_hashtable()
            if hashtag_map is not None:
                Utils.persist_to_file(hashtag_map, "{}/{}_hashtag".format(self.parameters.output_graph_path,
                                                                          self.parameters.output_map_prefix))
            retweet_user_map = map_instance.user_id_retweet_hashtable()
            if retweet_user_map is not None:
                Utils.persist_to_file(retweet_user_map, "{}/{}_retweeted_user_id".format(self.parameters.output_graph_path,
                                                                                     self.parameters.output_map_prefix))
            user_screen_name_map = map_instance.user_screen_name_hashtable()
            Utils.persist_to_file(user_screen_name_map,
                                  "{}/{}_user_screen_name".format(self.parameters.output_graph_path,
                                                                  self.parameters.output_map_prefix))
            user_screen_name_user_id_map = map_instance.user_screen_name_user_id_hashtable()
            Utils.persist_to_file(user_screen_name_user_id_map,
                                  "{}/{}_user_screen_name_user_id".format(self.parameters.output_graph_path,
                                                                          self.parameters.output_map_prefix))

            # Multigraph generation
            #multigraph = multigraph_instance.gen_multigraph(graph_list)
            #Utils.persist_to_file(multigraph, "{}/{}".format(self.parameters.output_graph_path,
            #                                                self.parameters.output_multi_graph_path))
            multigraph = None

        # Community detection
        if self.parameters.do_community_detection:
            leiden_instance = Leiden()
            g = leiden_instance.csv_to_igraph(
                input_csv_graph_file_path=self.parameters.graph_file_path) if self.parameters.do_read_graph_from_file else leiden_instance.csv_to_igraph(
                dataframe_graph=multigraph)
            leiden_instance.add_leiden_to_igraph(g)
            communities = leiden_instance.get_clusters_as_dataframe(g)
            Utils.persist_to_file(communities, "{}".format(self.parameters.community_output_file_path))

        # Get text data from raw dataset
        if self.parameters.do_get_text:
            # read communities saved on external file
            if self.parameters.do_read_communities_from_file:
                communities = pd.read_csv(self.parameters.community_file_path, sep=",", header=0)
            else:
                raise AttributeError("It's not possible to use communities generated at runtime")
            # read map saved on external file
            if not self.parameters.do_read_maps_from_file:
                raise AttributeError("It's not possible to use maps generated at runtime")
            else:
                user_map = pd.read_csv(self.parameters.user_map, sep=",", header=0)
                retweet_user_map = pd.read_csv(self.parameters.retweet_user_map, sep=",", header=0)
                u_map = pd.concat([user_map[["original", "node_hash"]], retweet_user_map[["original", "node_hash"]]])\
                    .drop_duplicates(ignore_index=True)

            indexes = self.parameters.community_indexes if self.parameters.community_indexes else []
            communities = communities[communities[self.parameters.community_col_name].isin(indexes)]

            merged = communities.merge(u_map, left_on="node_hash",
                                       right_on="node_hash",
                                       how="inner")

            u = merged['original'].tolist()
            u = list(map(int, u))

            raw_data = RawData(self.parameters.source_uri,
                               self.parameters.source_username,
                               self.parameters.source_password,
                               self.parameters.source_auth_source,
                               self.parameters.source_auth_mechanism,
                               input_type=self.parameters.td_input_type)

            raw_data.connect(self.parameters.source_db_name)

            """
            use it when mongo is available again

            pipeline = [
                {
                    '$match': {
                        'user.id': { '$in': u }  # Filter docs based on users list
                    }
                },
                {
                    '$project': {
                        'text': {
                            '$cond': {
                                'if': { '$gt': ['$retweeted_status', None] },  # Check if it is a retweet
                                'then': '$retweeted_status.text',              # If it is a retweet, get the text field retweeted_status
                                'else': '$text'                                # else, get the original text field
                            }
                        },
                        '_id': 0,
                        'user.id': 1,
                        'created_at': 1,
                        'type': {
                            '$cond': {
                                'if': { '$gt': ['$retweeted_status', None] },  # Check if it is a retweet
                                'then': 'normal',                              # If it is a retweet, set "normal" to type
                                'else': 'retweet'                              # else set "retweet"
                            }
                        }
                    }
                }
            ]
            
            results = collection.aggregate(pipeline)
            """

            result = raw_data.query(None, ['text', 'user.id', 'created_at.$date'])
            self.logger.debug("Generated final intermediate result with text data")

        if self.parameters.do_topic_builder:
            # BERT Topic
            topic_generator = TopicGenerator(result["text"].values.tolist())
            tm, t, p = topic_generator.topic_modeling()

            tp = pd.DataFrame(
                {'topic': t,
                 'prob': p
                 })

            text_with_topics = pd.concat([result, tp], axis=1)
            di = tm.get_document_info(result["text"].values.tolist())

            Utils.persist_to_file(text_with_topics, self.parameters.topics_file_path+"1234")
            Utils.persist_to_file(tm.get_topic_info(), self.parameters.topics_file_path)
            Utils.persist_to_file(di, self.parameters.docs_file_path)
            tm.save(self.parameters.model_path, serialization=self.parameters.model_serialization, save_ctfidf=True)


def get_properties(file_path="properties/prop.json"):
    import json
    with open(file_path) as f:
        properties = json.load(f)
    return properties


def set_logger(filepath, level='INFO'):
    logging.basicConfig(filename=filepath,
                        filemode='w',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=level)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Command line args")
    parser.add_argument('--properties', type=str, help='Properties file path')
    args = parser.parse_args()

    if args.properties is not None:
        prop: dict = get_properties(args.properties)
    else:
        prop: dict = get_properties()

    set_logger(prop["log"]["filepath"], prop["log"]["level"])
    do_graph_generation = prop["graph_generation"]["to_execute"]
    do_community_detection = prop["community_detection"]["to_execute"]
    do_get_text = prop["get_users_text"]["to_execute"]
    do_topic_builder = prop["topic_builder"]["to_execute"]

    config = prop["graph_generation"]["parameters"]["input"]["conf"]
    source_input_type = prop["graph_generation"]["parameters"]["input"]["type"]
    source_uri = config["uri"]
    source_username = config["username"]
    source_password = config["password"]
    source_auth_source = config["authName"]
    source_auth_mechanism = config["authMechanism"]
    source_db_name = config["db_name"]
    source_collection = config["collection"]
    source_chunk_start_date = datetime.strptime(config["chunk_start_date"], '%d/%m/%Y')
    source_chunk_end_date = datetime.strptime(config["chunk_end_date"], '%d/%m/%Y')

    do_retweet_graph = prop["graph_generation"]["parameters"]["graph_type"]["retweet"]
    do_hashtag_graph = prop["graph_generation"]["parameters"]["graph_type"]["user_hashtag"]
    do_hashtag_cooccurrences_graph = prop["graph_generation"]["parameters"]["graph_type"]["hashtag_cooccurrences"]
    do_mention_graph = prop["graph_generation"]["parameters"]["graph_type"]["mention"]
    do_response_graph = prop["graph_generation"]["parameters"]["graph_type"]["response"]

    output_graph_path = prop["graph_generation"]["parameters"]["output"]["path"]
    output_retweet_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["retweet"]
    output_response_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["response"]
    output_mention_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["mention"]
    output_hashtag_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["hashtag"]
    output_hashtag_cooccurrences_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"][
        "hashtag_cooccurrences"]
    output_multi_graph_path = prop["graph_generation"]["parameters"]["output"]["graph_file_name"]["multigraph"]
    output_map_prefix = prop["graph_generation"]["parameters"]["output"]["map_file_name_prefix"]

    community_config = prop["community_detection"]["parameters"]
    do_read_graph_from_file = community_config["read_from_file"]
    graph_file_path = community_config["graph_file_path"]
    community_output_file_path = community_config["community_output_file_path"]

    users_text_config = prop["get_users_text"]["parameters"]
    community_indexes = users_text_config["communities"]["indexes"]
    community_col_name = users_text_config["communities"]["community_col_name"]
    do_read_communities_from_file = users_text_config["communities"]["read_communities_from_file"]
    community_file_path = users_text_config["communities"]["community_file_path"]

    do_read_maps_from_file = users_text_config["map_file_path"]["read_maps_from_file"]
    user_map = users_text_config["map_file_path"]["user_map"]
    hashtag_map = users_text_config["map_file_path"]["hashtag_map"]
    retweet_user_map = users_text_config["map_file_path"]["retweet_user_map"]

    text_data_config = users_text_config["input"]["conf"]
    td_input_type = users_text_config["input"]["type"]
    td_uri = text_data_config["uri"]
    td_username = text_data_config["username"]
    td_password = text_data_config["password"]
    td_auth_source = text_data_config["authName"]
    td_auth_mechanism = text_data_config["authMechanism"]
    td_db_name = text_data_config["db_name"]
    td_collection = text_data_config["collection"]

    topic_config = prop["topic_builder"]["parameters"]
    topics_file_path = topic_config["topics_file_path"]
    docs_file_path = topic_config["docs_file_path"]
    model_path = topic_config["model"]["model_path"]
    model_serialization = topic_config["model"]["serialization"]

    Parameters = namedtuple('Parameters', [
        "do_graph_generation",
        "do_community_detection",
        "do_get_text",
        "do_topic_builder",
        "source_input_type",
        "source_uri",
        "source_username",
        "source_password",
        "source_auth_source",
        "source_auth_mechanism",
        "source_db_name",
        "source_collection",
        "source_chunk_start_date",
        "source_chunk_end_date",
        "do_retweet_graph",
        "do_hashtag_graph",
        "do_hashtag_cooccurrences_graph",
        "do_mention_graph",
        "do_response_graph",
        "output_graph_path",
        "output_retweet_graph_path",
        "output_response_graph_path",
        "output_mention_graph_path",
        "output_hashtag_graph_path",
        "output_hashtag_cooccurrences_graph_path",
        "output_multi_graph_path",
        "output_map_prefix",
        "do_read_graph_from_file",
        "graph_file_path",
        "community_output_file_path",
        "community_indexes",
        "community_col_name",
        "do_read_communities_from_file",
        "community_file_path",
        "do_read_maps_from_file",
        "user_map",
        "hashtag_map",
        "retweet_user_map",
        "td_input_type",
        "td_uri",
        "td_username",
        "td_password",
        "td_auth_source",
        "td_auth_mechanism",
        "td_db_name",
        "td_collection",
        "topics_file_path",
        "docs_file_path",
        "model_path",
        "model_serialization"
    ])

    P = Parameters(do_graph_generation,
                   do_community_detection,
                   do_get_text,
                   do_topic_builder,
                   source_input_type,
                   source_uri,
                   source_username,
                   source_password,
                   source_auth_source,
                   source_auth_mechanism,
                   source_db_name,
                   source_collection,
                   source_chunk_start_date,
                   source_chunk_end_date,
                   do_retweet_graph,
                   do_hashtag_graph,
                   do_hashtag_cooccurrences_graph,
                   do_mention_graph,
                   do_response_graph,
                   output_graph_path,
                   output_retweet_graph_path,
                   output_response_graph_path,
                   output_mention_graph_path,
                   output_hashtag_graph_path,
                   output_hashtag_cooccurrences_graph_path,
                   output_multi_graph_path,
                   output_map_prefix,
                   do_read_graph_from_file,
                   graph_file_path,
                   community_output_file_path,
                   community_indexes,
                   community_col_name,
                   do_read_communities_from_file,
                   community_file_path,
                   do_read_maps_from_file,
                   user_map,
                   hashtag_map,
                   retweet_user_map,
                   td_input_type,
                   td_uri,
                   td_username,
                   td_password,
                   td_auth_source,
                   td_auth_mechanism,
                   td_db_name,
                   td_collection,
                   topics_file_path,
                   docs_file_path,
                   model_path,
                   model_serialization
                   )
    graph_analysis = GraphAnalysis(P)
    try:
        graph_analysis.run()
    except Exception as e:
        raise

    """
    cluster_map = ClusterMap()
    clusters = cluster_map.read_cluster_file(prop["output"]["path"] + 'clusters_general.csv')
    maps = cluster_map.read_map_file(
        prop["output"]["path"] + prop["output"]["map_file_name_prefix"] + "hashtag_hash.csv")
    clusters_maps_df = cluster_map.from_hash_to_id(clusters, maps)
    clusters_grouped_df = cluster_map.group_and_count_by_cluster(clusters_maps_df)

    Utils.persist_to_file(clusters_maps_df, prop["output"]["path"] + 'cluster_wo_hash')
    Utils.persist_to_file(clusters_grouped_df, prop["output"]["path"] + 'cluster_grouped_count')


    cl = leiden_instance.hash_to_name(df, prop["output"]["path"] + prop["output"]["map_file_name_prefix"]
                                      + "user.csv", prop["output"]["path"] + prop["output"]["map_file_name_prefix"]
                                        + "hashtag_hash.csv", prop["output"]["path"] + prop["output"]["map_file_name_prefix"]
                                        + "user_screen_name_user_id.csv")
    Utils.persist_to_file(cl[0], prop["output"]["path"] + prop["output"]["clusters_file_name_prefix"] + "user")
    Utils.persist_to_file(cl[1], prop["output"]["path"] + prop["output"]["clusters_file_name_prefix"] + "hashtag")
    """
